"""Vendor-specific LLM access. Every SDK import, thinking-parameter name,
image-block shape, and snapshot-ID read lives here and nowhere else.

Four adapters: anthropic, openai (Responses API), gemini (google-genai),
deepseek (OpenAI-compatible chat completions). Per-model `params` from
config.yaml are deep-merged verbatim into the call, so parameter drift is a
config edit, not a code edit.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import httpx

from .config import Config, ModelCfg

_TIMEOUT_S = 1800.0  # max-effort generations can run for many minutes
_RETRY_DELAYS_S = (2, 8, 30)


class TransientAPIError(Exception):
    """Rate limits / 5xx / network — retried with backoff."""


class ModelConfigError(Exception):
    """404 / invalid model or params / auth — recorded; the run continues."""


@dataclass(frozen=True)
class ContentPart:
    kind: Literal["text", "image_png"]
    text: str | None = None
    png_bytes: bytes | None = None


@dataclass(frozen=True)
class LLMRequest:
    model: ModelCfg
    system: str | None
    parts: tuple[ContentPart, ...]
    temperature: float | None
    seed: int | None
    max_output_tokens: int


@dataclass(frozen=True)
class LLMResult:
    text: str
    snapshot_id: str
    request_params: dict[str, Any]  # what was ACTUALLY sent (audit record; no content)
    stop_reason: str | None
    usage: dict[str, Any]
    images_included: bool = field(default=False)


def complete(req: LLMRequest) -> LLMResult:
    """Public entry: dispatch to the vendor adapter with transient-error retries."""
    last: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS_S, None)):
        try:
            return _dispatch(req)
        except TransientAPIError as exc:
            last = exc
            if delay is None:
                break
            print(f"    transient error ({exc}); retry {attempt + 1} in {delay}s")
            time.sleep(delay)
    raise TransientAPIError(f"retries exhausted: {last}") from last


def _dispatch(req: LLMRequest) -> LLMResult:
    images_ok = req.model.supports_images
    if not images_ok:
        req = replace(req, parts=tuple(p for p in req.parts if p.kind == "text"))
    adapter = {
        "anthropic": _anthropic,
        "openai": _openai,
        "gemini": _gemini,
        "deepseek": _deepseek,
    }[req.model.provider]
    result = adapter(req)
    had_images = any(p.kind == "image_png" for p in req.parts)
    return replace(result, images_included=images_ok and had_images)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _audit(kwargs: dict[str, Any], content_keys: tuple[str, ...]) -> dict[str, Any]:
    """Request params minus message content, for the artifact record."""
    return {k: v for k, v in kwargs.items() if k not in content_keys}


# --- Anthropic --------------------------------------------------------------------


def _anthropic(req: LLMRequest) -> LLMResult:
    import anthropic

    client = anthropic.Anthropic(timeout=_TIMEOUT_S, max_retries=3)
    content: list[dict[str, Any]] = []
    for part in req.parts:
        if part.kind == "text":
            content.append({"type": "text", "text": part.text})
        else:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(part.png_bytes).decode(),
                    },
                }
            )
    kwargs: dict[str, Any] = {
        "model": req.model.name,
        "max_tokens": req.max_output_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if req.system:
        kwargs["system"] = req.system
    if req.model.supports_temperature and req.temperature is not None:
        kwargs["temperature"] = req.temperature
    kwargs = _deep_merge(kwargs, req.model.params)
    try:
        # streaming is required at this max_tokens; get_final_message() accumulates
        with client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
    except anthropic.RateLimitError as exc:
        raise TransientAPIError(str(exc)) from exc
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
        raise TransientAPIError(str(exc)) from exc
    except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
            raise TransientAPIError(str(exc)) from exc
        raise ModelConfigError(f"{exc.status_code}: {exc.message}") from exc
    except httpx.HTTPError as exc:
        # mid-stream disconnects surface as raw httpx errors, not SDK types
        raise TransientAPIError(f"stream interrupted: {exc}") from exc
    text = "".join(b.text for b in msg.content if b.type == "text")
    return LLMResult(
        text=text,
        snapshot_id=msg.model,
        request_params=_audit(kwargs, ("messages",)),
        stop_reason=msg.stop_reason,
        usage={
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        },
    )


# --- OpenAI (Responses API) ---------------------------------------------------------


def _openai(req: LLMRequest) -> LLMResult:
    import openai

    client = openai.OpenAI(timeout=_TIMEOUT_S, max_retries=3)
    return _openai_responses(client, req)


def _openai_responses(client: Any, req: LLMRequest) -> LLMResult:
    import openai

    content: list[dict[str, Any]] = []
    for part in req.parts:
        if part.kind == "text":
            content.append({"type": "input_text", "text": part.text})
        else:
            b64 = base64.standard_b64encode(part.png_bytes).decode()
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
    kwargs: dict[str, Any] = {
        "model": req.model.name,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": req.max_output_tokens,
    }
    if req.system:
        kwargs["instructions"] = req.system
    if req.model.supports_temperature and req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.model.supports_seed and req.seed is not None:
        kwargs["seed"] = req.seed
    kwargs = _deep_merge(kwargs, req.model.params)
    try:
        resp = client.responses.create(**kwargs)
    except openai.RateLimitError as exc:
        raise TransientAPIError(str(exc)) from exc
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        raise TransientAPIError(str(exc)) from exc
    except openai.APIStatusError as exc:
        if exc.status_code >= 500:
            raise TransientAPIError(str(exc)) from exc
        raise ModelConfigError(f"{exc.status_code}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise TransientAPIError(f"transport error: {exc}") from exc
    stop = resp.status
    if getattr(resp, "incomplete_details", None):
        stop = f"incomplete: {resp.incomplete_details.reason}"
    usage = {}
    if getattr(resp, "usage", None):
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
    return LLMResult(
        text=resp.output_text or "",
        snapshot_id=resp.model,
        request_params=_audit(kwargs, ("input",)),
        stop_reason=stop,
        usage=usage,
    )


# --- Google Gemini ---------------------------------------------------------------


def _gemini(req: LLMRequest) -> LLMResult:
    from google import genai
    from google.genai import errors, types

    client = genai.Client()  # reads GEMINI_API_KEY
    parts: list[Any] = []
    for part in req.parts:
        if part.kind == "text":
            parts.append(types.Part.from_text(text=part.text))
        else:
            parts.append(types.Part.from_bytes(data=part.png_bytes, mime_type="image/png"))
    config: dict[str, Any] = {"max_output_tokens": req.max_output_tokens}
    if req.system:
        config["system_instruction"] = req.system
    if req.model.supports_temperature and req.temperature is not None:
        config["temperature"] = req.temperature
    if req.model.supports_seed and req.seed is not None:
        config["seed"] = req.seed
    config = _deep_merge(config, req.model.params)
    try:
        gen_config = types.GenerateContentConfig(**config)
    except Exception as exc:  # pydantic rejects unknown fields -> config problem
        raise ModelConfigError(f"invalid Gemini config params: {exc}") from exc
    try:
        resp = client.models.generate_content(
            model=req.model.name,
            contents=[types.Content(role="user", parts=parts)],
            config=gen_config,
        )
    except errors.APIError as exc:
        if exc.code and exc.code >= 500:
            raise TransientAPIError(str(exc)) from exc
        if exc.code == 429:
            raise TransientAPIError(str(exc)) from exc
        raise ModelConfigError(f"{exc.code}: {exc.message}") from exc
    except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
        raise TransientAPIError(f"transport error: {exc}") from exc
    text = resp.text or ""
    stop = None
    if resp.candidates:
        finish = resp.candidates[0].finish_reason
        stop = finish.name if finish is not None else None
    usage = {}
    if resp.usage_metadata is not None:
        usage = {
            "input_tokens": resp.usage_metadata.prompt_token_count,
            "output_tokens": resp.usage_metadata.candidates_token_count,
            "thoughts_tokens": resp.usage_metadata.thoughts_token_count,
        }
    audit_config = dict(config)
    if "system_instruction" in audit_config:
        audit_config["system_instruction"] = "<omitted: judge system prompt>"
    return LLMResult(
        text=text,
        snapshot_id=getattr(resp, "model_version", None) or req.model.name,
        request_params={"model": req.model.name, "config": audit_config},
        stop_reason=stop,
        usage=usage,
    )


# --- DeepSeek (OpenAI-compatible chat completions) -----------------------------------


def _deepseek(req: LLMRequest) -> LLMResult:
    import openai

    client = openai.OpenAI(
        base_url=req.model.base_url,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        timeout=_TIMEOUT_S,
        max_retries=3,
    )
    content: list[dict[str, Any]] | str
    if any(p.kind == "image_png" for p in req.parts):
        content = []
        for part in req.parts:
            if part.kind == "text":
                content.append({"type": "text", "text": part.text})
            else:
                b64 = base64.standard_b64encode(part.png_bytes).decode()
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )
    else:
        content = "\n\n".join(p.text for p in req.parts if p.kind == "text")
    messages: list[dict[str, Any]] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": content})
    kwargs: dict[str, Any] = {
        "model": req.model.name,
        "messages": messages,
        "max_tokens": req.max_output_tokens,
    }
    if req.model.supports_temperature and req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.model.supports_seed and req.seed is not None:
        kwargs["seed"] = req.seed
    kwargs = _deep_merge(kwargs, req.model.params)
    try:
        resp = client.chat.completions.create(**kwargs)
    except openai.RateLimitError as exc:
        raise TransientAPIError(str(exc)) from exc
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        raise TransientAPIError(str(exc)) from exc
    except openai.APIStatusError as exc:
        if exc.status_code >= 500:
            raise TransientAPIError(str(exc)) from exc
        raise ModelConfigError(f"{exc.status_code}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise TransientAPIError(f"transport error: {exc}") from exc
    choice = resp.choices[0]
    usage = {}
    if resp.usage:
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
    return LLMResult(
        text=choice.message.content or "",
        snapshot_id=resp.model,
        request_params=_audit(kwargs, ("messages",)),
        stop_reason=choice.finish_reason,
        usage=usage,
    )


# --- probe ----------------------------------------------------------------------


def probe_cli(cfg: Config, only: list[str] | None = None) -> None:
    """One cheap call per configured model; prints resolved snapshot IDs.

    Run this before spending money: 404s mean the model ID in config.yaml
    needs updating. Thinking params are stripped so the probe stays tiny
    (full params are exercised by the first real generation).
    """
    from . import artifacts

    names = list(dict.fromkeys([*cfg.candidate_models, *cfg.judge_models]))
    if only:
        names = [n for n in names if n in only]
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        mcfg = cfg.model(name)
        probe_model = replace(
            mcfg, params={k: v for k, v in mcfg.params.items() if k != "thinking"}
        )
        req = LLMRequest(
            model=probe_model,
            system=None,
            parts=(ContentPart(kind="text", text="Reply with the single word: OK"),),
            temperature=None,
            seed=None,
            max_output_tokens=2048,
        )
        try:
            res = complete(req)
            results[name] = {"ok": True, "snapshot_id": res.snapshot_id}
            print(f"  {name:<28} OK    snapshot={res.snapshot_id}")
        except Exception as exc:  # noqa: BLE001 — probe must report, never crash
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name:<28} FAIL  {type(exc).__name__}: {exc}")
    store = artifacts.Store(cfg.artifacts_dir)
    meta = store.read_json(store.run_meta_path) or {}
    meta.setdefault("probe", {}).update(results)  # merge: partial probes must not clobber
    store.write_json(store.run_meta_path, meta)
    failed = [n for n, r in results.items() if not r["ok"]]
    if failed:
        print(f"\n{len(failed)} model(s) failed — fix their IDs/params in config.yaml, re-probe.")
    else:
        print("\nall models resolved.")
