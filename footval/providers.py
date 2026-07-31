"""Vendor-specific LLM access. Every SDK import, thinking-parameter name,
image-block shape, and snapshot-ID read lives here and nowhere else.

Four adapters: anthropic, openai (Responses API), gemini (google-genai),
zai (OpenAI-compatible chat completions, for GLM). Per-model `params` from
config.yaml are deep-merged verbatim into the call, so parameter drift is a
config edit, not a code edit.

Prompt caching is expressed on the request, not per call site:
``LLMRequest.cache_part_idxs`` marks the content parts that end a stable
prefix (Anthropic turns them into explicit `cache_control` breakpoints; the
system prompt is always one) and ``cache_key`` becomes OpenAI's
`prompt_cache_key`. Everyone else caches identical prefixes automatically.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import httpx

from .config import Config, ModelCfg, deep_merge

_TIMEOUT_S = 1800.0  # max-effort generations can run for many minutes
_RETRY_DELAYS_S = (2, 8, 30)

# Providers with a 50%-off asynchronous batch endpoint. Anything else has its
# pairwise items run synchronously through complete().
BATCH_PROVIDERS = ("anthropic", "openai", "gemini")


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
    # Indices into `parts` that end a byte-stable prefix worth caching. Anthropic
    # needs these spelled out (max 4 breakpoints per request, and the system
    # prompt takes one); other vendors cache matching prefixes on their own.
    cache_part_idxs: tuple[int, ...] = ()
    # OpenAI `prompt_cache_key`: routes requests sharing a prefix to the same
    # cache. Ignored by every other provider.
    cache_key: str | None = None


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
        "zai": _zai,
    }[req.model.provider]
    result = adapter(req)
    had_images = any(p.kind == "image_png" for p in req.parts)
    return replace(result, images_included=images_ok and had_images)


def _audit(kwargs: dict[str, Any], content_keys: tuple[str, ...]) -> dict[str, Any]:
    """Request params minus message content, for the artifact record."""
    return {k: v for k, v in kwargs.items() if k not in content_keys}


# --- Anthropic --------------------------------------------------------------------


_EPHEMERAL = {"type": "ephemeral"}


def _anthropic_kwargs(req: LLMRequest) -> dict[str, Any]:
    """Messages-API kwargs; also the batch `params` body verbatim.

    Explicit prompt-cache breakpoints go on the system prompt and on every part
    named by ``req.cache_part_idxs``. Anthropic allows at most four, and reads
    the longest matching prefix, so ordering stable content first is what makes
    them pay off.
    """
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
    for idx in req.cache_part_idxs:
        if 0 <= idx < len(content):
            content[idx]["cache_control"] = dict(_EPHEMERAL)
    kwargs: dict[str, Any] = {
        "model": req.model.name,
        "max_tokens": req.max_output_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if req.system:
        kwargs["system"] = [{"type": "text", "text": req.system, "cache_control": dict(_EPHEMERAL)}]
    if req.model.supports_temperature and req.temperature is not None:
        kwargs["temperature"] = req.temperature
    return deep_merge(kwargs, req.model.params)


def _anthropic(req: LLMRequest) -> LLMResult:
    import anthropic

    client = anthropic.Anthropic(timeout=_TIMEOUT_S, max_retries=3)
    kwargs = _anthropic_kwargs(req)
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
    except TypeError as exc:
        # unknown kwarg from merged config params — the SDK rejects it pre-flight
        raise ModelConfigError(f"SDK rejected request params: {exc}") from exc
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


def _responses_kwargs(req: LLMRequest) -> dict[str, Any]:
    """Responses-API kwargs; also the batch JSONL `body` for /v1/responses."""
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
    if req.cache_key:
        kwargs["prompt_cache_key"] = req.cache_key
    return deep_merge(kwargs, req.model.params)


def _openai_responses(client: Any, req: LLMRequest) -> LLMResult:
    import openai

    kwargs = _responses_kwargs(req)
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
    except TypeError as exc:
        raise ModelConfigError(f"SDK rejected request params: {exc}") from exc
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


def _gemini_config(req: LLMRequest) -> dict[str, Any]:
    """GenerateContentConfig as a plain dict; also the per-item batch config."""
    config: dict[str, Any] = {"max_output_tokens": req.max_output_tokens}
    if req.system:
        config["system_instruction"] = req.system
    if req.model.supports_temperature and req.temperature is not None:
        config["temperature"] = req.temperature
    if req.model.supports_seed and req.seed is not None:
        config["seed"] = req.seed
    return deep_merge(config, req.model.params)


def _gemini_contents_dict(req: LLMRequest) -> list[dict[str, Any]]:
    """Dict-form contents for the batch path (text-only by design)."""
    parts = []
    for part in req.parts:
        if part.kind != "text":
            raise ValueError("gemini batch path is text-only; image parts unsupported")
        parts.append({"text": part.text})
    return [{"role": "user", "parts": parts}]


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
    config = _gemini_config(req)
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


# --- Z.ai GLM (OpenAI-compatible chat completions) -----------------------------------


def _chat_kwargs(req: LLMRequest) -> dict[str, Any]:
    """Chat-completions kwargs; z.ai sync and the /v1/chat/completions batch body."""
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
    # OpenAI-only knob; z.ai's compatible endpoint would reject the unknown field.
    if req.cache_key and req.model.provider == "openai":
        kwargs["prompt_cache_key"] = req.cache_key
    return deep_merge(kwargs, req.model.params)


def _zai_call_kwargs(req: LLMRequest) -> dict[str, Any]:
    """Chat kwargs with vendor-specific params moved into ``extra_body``.

    z.ai knobs like ``thinking`` are not OpenAI SDK parameters — passed as
    plain kwargs the SDK raises TypeError before any HTTP call — but
    ``extra_body`` ships them verbatim in the JSON body. The batch JSONL path
    uses `_chat_kwargs` directly (raw JSON, no SDK signature), so only this
    synchronous call site needs the split.
    """
    import inspect

    from openai.resources.chat.completions import Completions

    kwargs = _chat_kwargs(req)
    known = inspect.signature(Completions.create).parameters
    extra = {k: kwargs.pop(k) for k in list(kwargs) if k not in known}
    if extra:
        kwargs["extra_body"] = deep_merge(kwargs.get("extra_body") or {}, extra)
    return kwargs


def _zai(req: LLMRequest) -> LLMResult:
    import openai

    client = openai.OpenAI(
        base_url=req.model.base_url,
        api_key=os.environ.get("ZAI_API_KEY", ""),
        timeout=_TIMEOUT_S,
        max_retries=3,
    )
    kwargs = _zai_call_kwargs(req)
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
    except TypeError as exc:
        raise ModelConfigError(f"SDK rejected request params: {exc}") from exc
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

# Thinking/effort knobs, stripped so a probe call costs a handful of tokens.
_PROBE_STRIP = frozenset(
    {"thinking", "output_config", "reasoning", "thinking_config", "reasoning_effort"}
)


def probe_cli(cfg: Config, only: list[str] | None = None, full: bool = False) -> None:
    """One cheap call per configured model; prints resolved snapshot IDs.

    Run this before spending money: 404s mean the model ID in config.yaml
    needs updating. Thinking/effort params are stripped so the probe stays tiny;
    ``full`` sends them verbatim, which costs a little thinking but catches
    parameter-name drift the stripped probe cannot see.
    """
    from . import artifacts

    names = list(dict.fromkeys([*cfg.candidate_models, *cfg.pairwise_judges]))
    if only:
        names = [n for n in names if n in only]
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        mcfg = cfg.model(name)
        probe_model = mcfg
        if not full:
            probe_model = replace(
                mcfg, params={k: v for k, v in mcfg.params.items() if k not in _PROBE_STRIP}
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


# --- batch APIs --------------------------------------------------------------------
# Anthropic / OpenAI / Gemini offer 50%-discount batch endpoints (BATCH_PROVIDERS);
# z.ai has none, so GLM items run through complete(). All return a uniform per-item
# dict: {custom_id, ok, text, error, snapshot_id, stop_reason, usage}.


@dataclass(frozen=True)
class BatchItem:
    custom_id: str
    req: LLMRequest


def _item_result(custom_id: str | None) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "ok": False,
        "text": None,
        "error": None,
        "snapshot_id": None,
        "stop_reason": None,
        "usage": {},
    }


def anthropic_batch_params(item: BatchItem) -> dict[str, Any]:
    """The batch `params` body — byte-identical to the synchronous kwargs,
    prompt-cache breakpoints included."""
    return _anthropic_kwargs(item.req)


def openai_batch_line(item: BatchItem, endpoint: str) -> dict[str, Any]:
    if endpoint == "/v1/responses":
        body = _responses_kwargs(item.req)
    else:
        body = _chat_kwargs(item.req)
    return {"custom_id": item.custom_id, "method": "POST", "url": endpoint, "body": body}


def gemini_batch_request(item: BatchItem) -> dict[str, Any]:
    return {
        "contents": _gemini_contents_dict(item.req),
        "config": _gemini_config(item.req),
        "metadata": {"custom_id": item.custom_id},
    }


def submit_anthropic_batch(items: list[BatchItem]) -> dict[str, Any]:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic(max_retries=3)
    requests = [
        Request(
            custom_id=item.custom_id,
            params=MessageCreateParamsNonStreaming(**anthropic_batch_params(item)),
        )
        for item in items
    ]
    batch = client.messages.batches.create(requests=requests)
    return {"batch_id": batch.id}


def poll_anthropic_batch(batch_id: str) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    status = "ended" if batch.processing_status == "ended" else "in_progress"
    return {"status": status, "counts": batch.request_counts.model_dump()}


def fetch_anthropic_batch(batch_id: str) -> list[dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic()
    out = []
    for entry in client.messages.batches.results(batch_id):
        rec = _item_result(entry.custom_id)
        if entry.result.type == "succeeded":
            msg = entry.result.message
            rec["ok"] = True
            rec["text"] = "".join(b.text for b in msg.content if b.type == "text")
            rec["snapshot_id"] = msg.model
            rec["stop_reason"] = msg.stop_reason
            rec["usage"] = {
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(
                    msg.usage, "cache_creation_input_tokens", None
                ),
            }
        else:
            detail = getattr(entry.result, "error", None)
            rec["error"] = f"{entry.result.type}: {detail}"
        out.append(rec)
    return out


def submit_openai_batch(items: list[BatchItem], endpoint: str = "/v1/responses") -> dict[str, Any]:
    import io
    import json as _json

    import openai

    client = openai.OpenAI(max_retries=3)
    lines = [_json.dumps(openai_batch_line(item, endpoint)) for item in items]
    buf = io.BytesIO("\n".join(lines).encode())
    upload = client.files.create(file=("footval-pairwise.jsonl", buf), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id, endpoint=endpoint, completion_window="24h"
    )
    return {"batch_id": batch.id, "input_file_id": upload.id, "endpoint": endpoint}


def poll_openai_batch(batch_id: str) -> dict[str, Any]:
    import openai

    client = openai.OpenAI()
    batch = client.batches.retrieve(batch_id)
    mapping = {"completed": "ended", "expired": "ended", "failed": "failed", "cancelled": "failed"}
    out: dict[str, Any] = {
        "status": mapping.get(batch.status, "in_progress"),
        "counts": batch.request_counts.model_dump() if batch.request_counts else {},
    }
    if batch.errors:
        out["errors"] = str(batch.errors)[:2000]
    return out


def _openai_body_result(body: dict[str, Any]) -> dict[str, Any]:
    """Extract text/stop/usage from a raw Responses or chat-completions body."""
    if "output" in body:  # /v1/responses
        text = ""
        for block in body.get("output") or []:
            if block.get("type") == "message":
                for chunk in block.get("content") or []:
                    if chunk.get("type") == "output_text":
                        text += chunk.get("text") or ""
        stop = body.get("status")
        if body.get("incomplete_details"):
            stop = f"incomplete: {(body['incomplete_details'] or {}).get('reason')}"
        usage = body.get("usage") or {}
        return {
            "ok": True,
            "text": text,
            "snapshot_id": body.get("model"),
            "stop_reason": stop,
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
            },
        }
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    return {
        "ok": True,
        "text": (choice.get("message") or {}).get("content") or "",
        "snapshot_id": body.get("model"),
        "stop_reason": choice.get("finish_reason"),
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        },
    }


def fetch_openai_batch(batch_id: str) -> tuple[list[dict[str, Any]], str]:
    """Returns (uniform item results, raw output JSONL for the audit dump)."""
    import json as _json

    import openai

    client = openai.OpenAI()
    batch = client.batches.retrieve(batch_id)
    out: list[dict[str, Any]] = []
    raw = ""
    for file_id in (batch.output_file_id, batch.error_file_id):
        if not file_id:
            continue
        content = client.files.content(file_id).text
        raw += content
        for line in content.splitlines():
            if not line.strip():
                continue
            obj = _json.loads(line)
            rec = _item_result(obj.get("custom_id"))
            resp = obj.get("response") or {}
            if obj.get("error"):
                rec["error"] = str(obj["error"])[:2000]
            elif resp.get("status_code") == 200:
                rec.update(_openai_body_result(resp.get("body") or {}))
            else:
                rec["error"] = f"status {resp.get('status_code')}: {str(resp.get('body'))[:500]}"
            out.append(rec)
    return out, raw


def submit_gemini_batch(model: ModelCfg, items: list[BatchItem]) -> dict[str, Any]:
    from google import genai

    client = genai.Client()
    inlined = [gemini_batch_request(item) for item in items]
    job = client.batches.create(
        model=model.name,
        src={"inlined_requests": inlined},
        config={"display_name": f"footval-pairwise-{model.name}"},
    )
    return {"batch_id": job.name}


_GEMINI_TERMINAL = {
    "JOB_STATE_SUCCEEDED": "ended",
    "JOB_STATE_PARTIALLY_SUCCEEDED": "ended",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLED": "failed",
    "JOB_STATE_EXPIRED": "failed",
}


def poll_gemini_batch(batch_id: str) -> dict[str, Any]:
    from google import genai

    client = genai.Client()
    job = client.batches.get(name=batch_id)
    state = job.state.name if job.state else "UNKNOWN"
    out = {"status": _GEMINI_TERMINAL.get(state, "in_progress"), "state": state}
    if getattr(job, "error", None):
        out["errors"] = str(job.error)[:2000]
    return out


def fetch_gemini_batch(batch_id: str, submitted_custom_ids: list[str]) -> list[dict[str, Any]]:
    """Map results by echoed metadata, falling back to submission order
    (documented as preserved for inline sources)."""
    from google import genai

    client = genai.Client()
    job = client.batches.get(name=batch_id)
    responses = list(job.dest.inlined_responses or []) if job.dest else []
    out = []
    for i, item in enumerate(responses):
        meta = getattr(item, "metadata", None) or {}
        cid = meta.get("custom_id") if isinstance(meta, dict) else None
        if cid is None and i < len(submitted_custom_ids):
            cid = submitted_custom_ids[i]
        rec = _item_result(cid)
        if getattr(item, "error", None):
            rec["error"] = str(item.error)[:2000]
        elif item.response is not None:
            resp = item.response
            try:
                rec["text"] = resp.text or ""
            except Exception:  # .text raises on empty candidates
                rec["text"] = ""
            rec["ok"] = True
            if resp.candidates:
                finish = resp.candidates[0].finish_reason
                rec["stop_reason"] = finish.name if finish is not None else None
            rec["snapshot_id"] = getattr(resp, "model_version", None)
            if resp.usage_metadata is not None:
                rec["usage"] = {
                    "input_tokens": resp.usage_metadata.prompt_token_count,
                    "output_tokens": resp.usage_metadata.candidates_token_count,
                    "thoughts_tokens": resp.usage_metadata.thoughts_token_count,
                    "cached_tokens": resp.usage_metadata.cached_content_token_count,
                }
        else:
            rec["error"] = "empty batch item (no response, no error)"
        out.append(rec)
    return out
