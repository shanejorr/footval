"""Stage 1 — generate candidate responses.

Sends footbench.prompt.md byte-for-byte to each candidate, n_samples times.
Candidate output is NEVER retried or repaired on content grounds — malformed
output is signal. Transport errors are retried inside the provider layer.
Resumable: instances with a stored response.json are skipped.
"""

from __future__ import annotations

import datetime as dt

from . import artifacts, parsing, providers
from .config import Config


def run(cfg: Config, only: list[str] | None = None) -> None:
    prompt = cfg.prompt_path.read_text()
    store = artifacts.Store(cfg.artifacts_dir)
    _record_run_meta(cfg, store)

    generated, skipped, failed = [], [], []
    for model_name in cfg.candidate_models:
        if only and model_name not in only:
            continue
        mcfg = cfg.model(model_name)
        for sample_idx in range(cfg.n_samples):
            iid = artifacts.instance_id(model_name, sample_idx)
            if store.response_path(iid).exists():
                skipped.append(iid)
                continue
            print(f"  generating {iid} ...")
            req = providers.LLMRequest(
                model=mcfg,
                system=None,
                parts=(providers.ContentPart(kind="text", text=prompt),),
                temperature=cfg.gen_temperature,
                seed=cfg.seed,
                max_output_tokens=cfg.generation_max_output_tokens,
            )
            try:
                res = providers.complete(req)
            except (providers.ModelConfigError, providers.TransientAPIError) as exc:
                store.write_json(
                    store.generation_error_path(iid),
                    {
                        "instance_id": iid,
                        "model": model_name,
                        "sample_idx": sample_idx,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "ts": _now(),
                    },
                )
                failed.append((iid, str(exc)))
                print(f"    FAILED: {exc}")
                continue

            parsed, parse_mode = parsing.parse_candidate_json(res.text)
            store.write_json(
                store.response_path(iid),
                {
                    "instance_id": iid,
                    "model": model_name,
                    "sample_idx": sample_idx,
                    "snapshot_id": res.snapshot_id,
                    "ts": _now(),
                    "temperature": cfg.gen_temperature if mcfg.supports_temperature else None,
                    "seed": cfg.seed if mcfg.supports_seed else None,
                    "request_params": res.request_params,
                    "stop_reason": res.stop_reason,
                    "usage": res.usage,
                    "raw_text": res.text,
                    "parse_mode": parse_mode,
                    "parsed_json": parsed,
                },
            )
            script = parsed.get("plot_script") if isinstance(parsed, dict) else None
            if isinstance(script, str):
                store.plot_script_path(iid).write_text(script)
            err_path = store.generation_error_path(iid)
            if err_path.exists():
                err_path.unlink()
            generated.append(iid)
            note = "" if parse_mode == "direct" else f" (parse_mode={parse_mode})"
            print(f"    ok: snapshot={res.snapshot_id} stop={res.stop_reason}{note}")

    print(
        f"generate: {len(generated)} generated, {len(skipped)} already present,"
        f" {len(failed)} failed"
    )
    for iid, msg in failed:
        print(f"  FAILED {iid}: {msg}")
    if failed:
        print(
            "fix model IDs/params in config.yaml and re-run `make generate`"
            " (existing responses are kept)."
        )
    truncated = [
        iid
        for iid in store.instance_ids()
        if (store.load_response(iid) or {}).get("stop_reason") in ("max_tokens", "length")
        or str((store.load_response(iid) or {}).get("stop_reason", "")).startswith("incomplete")
    ]
    if truncated:
        print(f"WARNING: truncated responses (raise generation.max_output_tokens?): {truncated}")


def _record_run_meta(cfg: Config, store: artifacts.Store) -> None:
    meta = store.read_json(store.run_meta_path) or {}
    meta.setdefault("config", {})
    meta["config"].update(
        {
            "candidate_models": list(cfg.candidate_models),
            "n_samples": cfg.n_samples,
            "gen_temperature": cfg.gen_temperature,
            "judge_temperature": cfg.judge_temperature,
            "seed": cfg.seed,
            "interval_tol": cfg.interval_tol,
        }
    )
    meta.setdefault("stage_timestamps", {})["generate"] = _now()
    store.write_json(store.run_meta_path, meta)


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
