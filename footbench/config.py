"""Configuration loading for the footbench pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

CRITERIA = ("soundness", "priors", "code")

PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


@dataclass(frozen=True)
class ModelCfg:
    name: str
    provider: str
    family: str
    supports_temperature: bool = True
    supports_seed: bool = False
    supports_images: bool = True
    base_url: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxCfg:
    image: str = "footbench-sandbox"
    timeout_seconds: int = 60
    memory: str = "1g"
    cpus: int = 1
    pids: int = 256


@dataclass(frozen=True)
class Config:
    root: Path
    candidate_models: tuple[str, ...]
    judge_models: tuple[str, ...]
    n_samples: int
    gen_temperature: float
    judge_temperature: float
    seed: int
    criteria_weights: tuple[float, ...]
    interval_tol: float
    sandbox: SandboxCfg
    generation_max_output_tokens: int
    judging_max_output_tokens: int
    judge_max_format_retries: int
    models: dict[str, ModelCfg]

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def prompt_path(self) -> Path:
        return self.root / "footbench.prompt.md"

    def model(self, name: str) -> ModelCfg:
        return self.models[name]


def load_config(root: Path | None = None) -> Config:
    root = root or Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    raw = yaml.safe_load((root / "config.yaml").read_text())

    models: dict[str, ModelCfg] = {}
    for name, m in (raw.get("models") or {}).items():
        models[name] = ModelCfg(
            name=name,
            provider=m["provider"],
            family=m["family"],
            supports_temperature=m.get("supports_temperature", True),
            supports_seed=m.get("supports_seed", False),
            supports_images=m.get("supports_images", True),
            base_url=m.get("base_url"),
            params=m.get("params") or {},
        )
    for name in list(raw["candidate_models"]) + list(raw["judge_models"]):
        if name not in models:
            raise ValueError(f"model {name!r} has no entry under models: in config.yaml")
        if models[name].provider not in PROVIDER_ENV_KEYS:
            raise ValueError(f"model {name!r} has unknown provider {models[name].provider!r}")

    weights = tuple(float(w) for w in raw["criteria_weights"])
    if len(weights) != len(CRITERIA):
        raise ValueError(f"criteria_weights must have exactly {len(CRITERIA)} entries")

    sb = raw.get("sandbox") or {}
    gen = raw.get("generation") or {}
    judging = raw.get("judging") or {}
    return Config(
        root=root,
        candidate_models=tuple(raw["candidate_models"]),
        judge_models=tuple(raw["judge_models"]),
        n_samples=int(raw["n_samples"]),
        gen_temperature=float(raw["gen_temperature"]),
        judge_temperature=float(raw["judge_temperature"]),
        seed=int(raw["seed"]),
        criteria_weights=weights,
        interval_tol=float(raw["interval_tol"]),
        sandbox=SandboxCfg(
            image=sb.get("image", "footbench-sandbox"),
            timeout_seconds=int(sb.get("timeout_seconds", 60)),
            memory=str(sb.get("memory", "1g")),
            cpus=int(sb.get("cpus", 1)),
            pids=int(sb.get("pids", 256)),
        ),
        generation_max_output_tokens=int(gen.get("max_output_tokens", 64000)),
        judging_max_output_tokens=int(judging.get("max_output_tokens", 16000)),
        judge_max_format_retries=int(judging.get("max_format_retries", 2)),
        models=models,
    )


def missing_api_keys(cfg: Config) -> list[str]:
    """Env-var names required by the configured providers that are not set."""
    needed = {PROVIDER_ENV_KEYS[m.provider] for m in cfg.models.values()}
    return sorted(k for k in needed if not os.environ.get(k))
