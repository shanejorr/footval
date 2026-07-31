"""Configuration loading for the footval pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "zai": "ZAI_API_KEY",
}

# One judge system prompt per pairwise criterion, loaded byte-for-byte.
JUDGE_PROMPT_FILES = {
    "analytical_reasoning": "footval.judge.analytical.prompt.md",
    "intuitive_reasoning": "footval.judge.intuitive.prompt.md",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins on conflict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class ModelCfg:
    name: str
    provider: str
    family: str
    # Provider list price per 1M billable output tokens (None -> cost not computed)
    output_usd_per_mtok: float | None = None
    supports_temperature: bool = True
    supports_seed: bool = False
    base_url: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    # Merged over `params` only when this model judges. Three models sit on both
    # rosters and judge one effort tier above the level they answer at.
    judge_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    root: Path
    candidate_models: tuple[str, ...]
    n_samples: int
    gen_temperature: float
    judge_temperature: float
    seed: int
    interval_tol: float
    generation_max_output_tokens: int
    models: dict[str, ModelCfg]
    pairwise_judges: tuple[str, ...] = ()
    pairwise_both_orders: bool = False
    pairwise_max_output_tokens: int = 32000
    pairwise_max_format_retries: int = 2

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def outputs_data_dir(self) -> Path:
        return self.outputs_dir / "data"

    @property
    def prompt_path(self) -> Path:
        return self.root / "footval.prompt.md"

    def judge_prompt_path(self, criterion: str) -> Path:
        return self.root / JUDGE_PROMPT_FILES[criterion]

    def model(self, name: str) -> ModelCfg:
        return self.models[name]

    def judge_model(self, name: str) -> ModelCfg:
        """The model config as it should be sent when this model is judging."""
        mcfg = self.models[name]
        if not mcfg.judge_params:
            return mcfg
        return replace(mcfg, params=deep_merge(mcfg.params, mcfg.judge_params), judge_params={})


def _default_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "config.yaml").is_file() and (cwd / "footval.prompt.md").is_file():
        return cwd
    return Path(__file__).resolve().parent.parent


def load_config(root: Path | None = None) -> Config:
    root = root or _default_root()
    load_dotenv(root / ".env")
    raw = yaml.safe_load((root / "config.yaml").read_text())

    models: dict[str, ModelCfg] = {}
    for name, m in (raw.get("models") or {}).items():
        models[name] = ModelCfg(
            name=name,
            provider=m["provider"],
            family=m["family"],
            output_usd_per_mtok=(
                float(m["output_usd_per_mtok"]) if m.get("output_usd_per_mtok") else None
            ),
            supports_temperature=m.get("supports_temperature", True),
            supports_seed=m.get("supports_seed", False),
            base_url=m.get("base_url"),
            params=m.get("params") or {},
            judge_params=m.get("judge_params") or {},
        )
    pairwise = raw.get("pairwise") or {}
    pairwise_judges = tuple(pairwise.get("judges") or ())
    for name in [*raw["candidate_models"], *pairwise_judges]:
        if name not in models:
            raise ValueError(f"model {name!r} has no entry under models: in config.yaml")
        if models[name].provider not in PROVIDER_ENV_KEYS:
            raise ValueError(f"model {name!r} has unknown provider {models[name].provider!r}")

    gen = raw.get("generation") or {}
    return Config(
        root=root,
        candidate_models=tuple(raw["candidate_models"]),
        n_samples=int(raw["n_samples"]),
        gen_temperature=float(raw["gen_temperature"]),
        judge_temperature=float(raw["judge_temperature"]),
        seed=int(raw["seed"]),
        interval_tol=float(raw["interval_tol"]),
        generation_max_output_tokens=int(gen.get("max_output_tokens", 64000)),
        models=models,
        pairwise_judges=pairwise_judges,
        pairwise_both_orders=bool(pairwise.get("both_orders", False)),
        pairwise_max_output_tokens=int(pairwise.get("max_output_tokens", 32000)),
        pairwise_max_format_retries=int(pairwise.get("max_format_retries", 2)),
    )


def missing_api_keys(cfg: Config) -> list[str]:
    """Env-var names required by the configured providers that are not set."""
    needed = {PROVIDER_ENV_KEYS[m.provider] for m in cfg.models.values()}
    return sorted(k for k in needed if not os.environ.get(k))
