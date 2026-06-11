"""Stage 3 — deterministic structural and numeric checks on candidate responses.

Pure functions over the stored response (no I/O in the core), plus a thin
runner. Results are diagnostics fed to judges and stage 6 — never a scored
criterion, and never a reason to repair a response.
"""

from __future__ import annotations

from typing import Any

from . import artifacts
from .config import Config
from .distributions import (
    FAMILIES,
    REQUIRED_PARAMS,
    family_center,
    interval_95,
    is_number,
    validate_params,
)

BUCKETS = ("analytics", "intuition")
TYPES = ("game_management", "offensive", "defensive")
TOP_LEVEL_KEYS = ("quantity", "strategies", "plot_script")

CHECK_NAMES = ("json_valid", "grid_complete", "intervals_ok", "family_ok", "params_reproduce_ok")


def check_instance(parsed: Any, parse_mode: str, interval_tol: float) -> dict[str, Any]:
    """Run all five checks on one parsed response. Pure; safe on any input."""
    detail: dict[str, Any] = {"parse_mode": parse_mode}

    json_valid, json_detail = _check_json(parsed, parse_mode)
    detail["json"] = json_detail
    if not json_valid:
        detail["reason"] = "response JSON invalid; downstream checks fail by definition"
        return {name: False for name in CHECK_NAMES} | {"json_valid": False, "detail": detail}

    strategies = parsed["strategies"]
    grid_complete, grid_detail = _check_grid(strategies)
    detail["grid"] = grid_detail

    strat_details = [_check_strategy(s, i, interval_tol) for i, s in enumerate(strategies)]
    detail["strategies"] = strat_details

    has_strategies = len(strat_details) > 0
    return {
        "json_valid": True,
        "grid_complete": grid_complete,
        "intervals_ok": has_strategies and all(d["interval_sane"] for d in strat_details),
        "family_ok": has_strategies and all(d["family_valid"] for d in strat_details),
        "params_reproduce_ok": has_strategies and all(d["params_ok"] for d in strat_details),
        "detail": detail,
    }


def _check_json(parsed: Any, parse_mode: str) -> tuple[bool, dict[str, Any]]:
    if parse_mode == "failed" or parsed is None:
        return False, {"error": "did not parse as JSON"}
    if not isinstance(parsed, dict):
        return False, {"error": f"top level is {type(parsed).__name__}, not an object"}
    missing = [k for k in TOP_LEVEL_KEYS if k not in parsed]
    if missing:
        return False, {"missing_keys": missing}
    if not isinstance(parsed["strategies"], list):
        return False, {"error": "strategies is not a list"}
    return True, {"missing_keys": []}


def _check_grid(strategies: list[Any]) -> tuple[bool, dict[str, Any]]:
    cells: dict[tuple[str, str], list[int]] = {}
    titles: list[str] = []
    issues: list[str] = []
    for i, s in enumerate(strategies):
        if not isinstance(s, dict):
            issues.append(f"strategy {i} is not an object")
            continue
        bucket, stype = s.get("bucket"), s.get("type")
        if bucket not in BUCKETS or stype not in TYPES:
            issues.append(f"strategy {i} has invalid cell ({bucket!r}, {stype!r})")
        else:
            cells.setdefault((bucket, stype), []).append(i)
        title = s.get("title")
        if isinstance(title, str) and title.strip():
            titles.append(title.strip().lower())
        else:
            issues.append(f"strategy {i} has no usable title")

    expected = {(b, t) for b in BUCKETS for t in TYPES}
    missing_cells = sorted(f"{b}/{t}" for (b, t) in expected - set(cells))
    duplicate_cells = sorted(f"{b}/{t}" for (b, t), idxs in cells.items() if len(idxs) > 1)
    duplicate_titles = sorted({t for t in titles if titles.count(t) > 1})

    ok = (
        len(strategies) == 6
        and not missing_cells
        and not duplicate_cells
        and not duplicate_titles
        and not issues
    )
    return ok, {
        "n_strategies": len(strategies),
        "missing_cells": missing_cells,
        "duplicate_cells": duplicate_cells,
        "duplicate_titles": duplicate_titles,
        "issues": issues,
    }


def _check_strategy(s: Any, idx: int, interval_tol: float) -> dict[str, Any]:
    d: dict[str, Any] = {
        "idx": idx,
        "title": None,
        "cell": None,
        "interval_sane": False,
        "family": None,
        "family_valid": False,
        "params_ok": False,
        "reported": None,
        "recomputed": None,
        "errors": None,
        "tol_abs": None,
        "reason": None,
    }
    if not isinstance(s, dict):
        d["reason"] = "strategy is not an object"
        return d
    d["title"] = s.get("title")
    d["cell"] = f"{s.get('bucket')}/{s.get('type')}"

    prior = s.get("prior") if isinstance(s.get("prior"), dict) else {}
    bs = prior.get("belief_summaries")
    bs = bs if isinstance(bs, dict) else {}
    mpv = bs.get("most_plausible_value")
    low = bs.get("interval_95_low")
    high = bs.get("interval_95_high")
    interval_sane = (
        all(is_number(v) for v in (mpv, low, high)) and low < high and low <= mpv <= high
    )
    d["interval_sane"] = interval_sane
    d["reported"] = {"low": low, "high": high, "most_plausible_value": mpv}

    fam_obj = prior.get("distribution_family")
    family = fam_obj.get("name") if isinstance(fam_obj, dict) else None
    d["family"] = family
    d["family_valid"] = family in FAMILIES
    if not d["family_valid"]:
        d["reason"] = f"invalid or missing distribution family {family!r}"
        return d

    params = prior.get("parameters")
    err = validate_params(family, params)
    if err:
        d["reason"] = err
        return d
    if not interval_sane:
        d["reason"] = "reported interval not sane; reproduction not evaluable"
        return d

    numeric = {k: float(params[k]) for k in REQUIRED_PARAMS[family]}
    rec_low, rec_high = interval_95(family, numeric)
    center = family_center(family, numeric)
    width = high - low
    tol_abs = interval_tol * width
    errors = {
        "low": abs(rec_low - low),
        "high": abs(rec_high - high),
        "center": abs(center - mpv),
    }
    d["recomputed"] = {"low": rec_low, "high": rec_high, "center": center}
    d["errors"] = errors
    d["tol_abs"] = tol_abs
    d["params_ok"] = all(e <= tol_abs for e in errors.values())
    if not d["params_ok"]:
        worst = max(errors, key=errors.get)
        d["reason"] = (
            f"recomputed {worst} deviates from reported by {errors[worst]:.4g}"
            f" (tolerance {tol_abs:.4g} = {interval_tol} x interval width)"
        )
    return d


def run(cfg: Config, only_instance: str | None = None) -> None:
    """Write checks.json for every instance with a stored response."""
    store = artifacts.Store(cfg.artifacts_dir)
    written = skipped = 0
    for iid in store.instance_ids():
        if only_instance and iid != only_instance:
            continue
        if store.checks_path(iid).exists():
            skipped += 1
            continue
        resp = store.load_response(iid)
        if resp is None:
            continue
        report = check_instance(
            resp.get("parsed_json"), resp.get("parse_mode", "failed"), cfg.interval_tol
        )
        store.write_json(store.checks_path(iid), {"instance_id": iid} | report)
        written += 1
    print(f"checks: {written} written, {skipped} already present")
