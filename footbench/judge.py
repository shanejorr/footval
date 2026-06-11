"""Stage 4 — LLM judging.

Each judge scores one anonymized response instance at a time (independent
scoring, never a ranking). Judges receive the original task, the rubric, the
execution result + rendered figures, and the automated check report as ground
truth for mechanical facts. Judge calls MAY be retried on malformed output —
only candidate output is sacred.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from . import artifacts, parsing, providers
from .config import CRITERIA, Config, ModelCfg

_MAX_IMAGES = 8

# Spec section 6.3, verbatim. Do not paraphrase — judges anchor on this text.
# Split per criterion so the pairwise stage can reuse criteria 1-2; RUBRIC
# must recompose byte-identically (golden test pins it).
RUBRIC_SOUNDNESS = """\
**Criterion 1 — Soundness of recommendations**

- **5** — All six picks are genuinely underutilized *and* credibly win-positive; rationales are specific and football-literate; analytics vs. intuition buckets correctly distinguished (intuition picks are not just restated analytics consensus); picks distinct and correctly slotted in their grid cell.
- **3** — Mixed: several solid picks, but some are mainstream rather than underutilized, weakly justified, or misfiled.
- **1** — Most picks are generic/already standard, implausible, duplicative, or misattributed; rationales vague or wrong."""

RUBRIC_PRIORS = """\
**Criterion 2 — Reasonableness of Bayesian priors**

- **5** — Family matches the stated shape/tails and the signed support; most-plausible values and 95% intervals are football-realistic in magnitude (neither overconfident nor absurdly wide); conversions correct; honest treatment of weak cells (near-zero center, wider interval).
- **3** — Generally reasonable but with notable issues: an implausible magnitude or two, an over-tight or over-wide interval, a family that doesn't match the stated shape, or a conversion slip.
- **1** — Largely unreasonable: invalid/wrong family for signed data, badly mis-scaled intervals, inconsistent parameters, or identical boilerplate priors copy-pasted across differing strategies."""

RUBRIC_CODE = """\
**Criterion 3 — Python code quality** (treat the execution result as ground truth for whether it runs)

- **5** — Runs cleanly and produces all six correct plots; code is clear, efficient, parameterized as the prompt requires, uses the correct scipy distribution per family, and does what it claims.
- **3** — Runs but rough: inefficiency, partial plotting, minor mismatch between code and reported parameters, or sloppy structure.
- **1** — Fails to run or produces wrong/empty plots, or does not implement the stated distributions/parameters."""

RUBRIC = (
    f"{RUBRIC_SOUNDNESS}\n\n{RUBRIC_PRIORS}\n\n{RUBRIC_CODE}\n\n"
    "Interpolate scores of 4 and 2 between these anchors."
)

SYSTEM = f"""You are an impartial evaluator scoring one model's response to a fixed task. \
You see exactly one anonymized response per call — score it on its own merits against the \
rubric; you are not ranking it against other responses.

Score three criteria, each as an INTEGER from 1 to 5, using these anchors:

{RUBRIC}

Ground rules:
- The EXECUTION RESULT and AUTOMATED CHECK REPORT provided with the response are ground \
truth for the mechanical facts (does the script run, is the grid complete, do the reported \
parameters reproduce the stated interval). Do not re-litigate them; spend your judgment on \
the subjective quality.
- If rendered plot images are provided, inspect them as evidence for the code criterion.
- A response that failed to parse as JSON, or is missing parts, should be scored low on the \
affected criteria per the anchors — evaluate whatever is present.

Return ONLY a JSON object in exactly this shape, with integer scores and 1-2 sentence \
justifications:

{{"soundness": {{"score": 1, "justification": "..."}}, "priors": {{"score": 1, "justification": "..."}}, "code": {{"score": 1, "justification": "..."}}}}"""

_RETRY_NOTE = (
    "REMINDER: your previous reply was not a valid judgment object. Return ONLY the JSON "
    'object {"soundness": {"score": <int 1-5>, "justification": "..."}, "priors": {...}, '
    '"code": {...}} with integer scores — no prose, no code fences.'
)


def run(
    cfg: Config,
    only_judges: list[str] | None = None,
    only_instance: str | None = None,
    dump_prompt: bool = False,
) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    prompt_md = cfg.prompt_path.read_text()
    judges = [
        cfg.model(name) for name in cfg.judge_models if not only_judges or name in only_judges
    ]

    done = skipped = failed = 0
    for iid in store.instance_ids():
        if only_instance and iid != only_instance:
            continue
        resp = store.load_response(iid)
        if resp is None:
            continue
        for judge in judges:
            if store.judgment_path(iid, judge.name).exists():
                skipped += 1
                continue
            parts, redaction_count = build_judge_parts(store, iid, resp, prompt_md, judge)
            if dump_prompt:
                print(f"\n===== prompt for judge={judge.name} instance={iid} =====")
                print(f"--- system ---\n{SYSTEM}\n")
                for part in parts:
                    if part.kind == "text":
                        print(f"--- text part ---\n{part.text}\n")
                    else:
                        print(f"--- image part ({len(part.png_bytes)} bytes png) ---")
                continue
            print(f"  judging {iid} with {judge.name} ...")
            judgment = judge_instance(cfg, judge, resp, parts)
            judgment["redaction_count"] = redaction_count
            store.write_json(store.judgment_path(iid, judge.name), {"instance_id": iid} | judgment)
            done += 1
            if judgment["scores"] is None:
                failed += 1
                print("    no valid scores after retries")
            else:
                summary = {c: judgment["scores"][c]["score"] for c in CRITERIA}
                print(f"    scores: {summary}")
    if not dump_prompt:
        print(f"judge: {done} judged, {skipped} already present, {failed} without valid scores")


def build_judge_parts(
    store: artifacts.Store,
    iid: str,
    resp: dict[str, Any],
    prompt_md: str,
    judge: ModelCfg,
) -> tuple[list[providers.ContentPart], int]:
    """Assemble the anonymized user content for one judge call.

    Returns the content parts and the number of redactions applied.
    """
    parsed = resp.get("parsed_json")
    if isinstance(parsed, dict | list):
        body = json.dumps(parsed, indent=2, ensure_ascii=False)
        body_label = "The response parsed as JSON; it follows pretty-printed."
    else:
        body = resp.get("raw_text") or ""
        body_label = "The response did NOT parse as JSON; the raw text follows."
    redacted_body, n_redactions = parsing.redact(body)

    exec_result = store.load_exec(iid)
    checks = store.load_checks(iid)

    sections = [
        "=== TASK GIVEN TO THE MODEL ===\n" + prompt_md,
        f"=== RESPONSE UNDER REVIEW ===\n{body_label}\n\n{redacted_body}",
        "=== EXECUTION RESULT (ground truth) ===\n" + _exec_summary(exec_result),
        "=== AUTOMATED CHECK REPORT (ground truth) ===\n" + check_summary(checks),
    ]
    parts = [providers.ContentPart(kind="text", text="\n\n".join(sections))]

    figures = _figure_bytes(store, iid)
    if figures:
        if judge.supports_images:
            parts.append(
                providers.ContentPart(
                    kind="text",
                    text=f"=== RENDERED FIGURES ===\n{len(figures)} figure(s) attached as images.",
                )
            )
            for png in figures:
                parts.append(providers.ContentPart(kind="image_png", png_bytes=png))
        else:
            parts.append(
                providers.ContentPart(
                    kind="text",
                    text=(
                        "=== RENDERED FIGURES ===\nRendered images are unavailable to you; "
                        "rely on the execution report for what the script produced."
                    ),
                )
            )
    return parts, n_redactions


def judge_instance(
    cfg: Config,
    judge: ModelCfg,
    resp: dict[str, Any],
    parts: list[providers.ContentPart],
) -> dict[str, Any]:
    candidate_model = resp.get("model")
    base = {
        "judge": judge.name,
        "judge_family": judge.family,
        "is_self": judge.name == candidate_model,
        "images_included": False,
        "ts": _now(),
        "request_params": None,
        "judge_snapshot_id": None,
        "format_retries": 0,
        "scores": None,
        "raw_text": None,
        "error": None,
    }
    attempt_parts = list(parts)
    for attempt in range(cfg.judge_max_format_retries + 1):
        req = providers.LLMRequest(
            model=judge,
            system=SYSTEM,
            parts=tuple(attempt_parts),
            temperature=cfg.judge_temperature,
            seed=cfg.seed,
            max_output_tokens=cfg.judging_max_output_tokens,
        )
        try:
            res = providers.complete(req)
        except (providers.ModelConfigError, providers.TransientAPIError) as exc:
            base["error"] = f"{type(exc).__name__}: {exc}"
            return base
        base["request_params"] = res.request_params
        base["judge_snapshot_id"] = res.snapshot_id
        base["images_included"] = res.images_included
        base["raw_text"] = res.text
        base["format_retries"] = attempt
        scores = _parse_scores(res.text)
        if scores is not None:
            base["scores"] = scores
            return base
        base["error"] = "judge output did not match the required score schema"
        attempt_parts = list(parts) + [providers.ContentPart(kind="text", text=_RETRY_NOTE)]
    return base


def _parse_scores(text: str) -> dict[str, Any] | None:
    parsed, _mode = parsing.parse_candidate_json(text)
    if not isinstance(parsed, dict):
        return None
    out: dict[str, Any] = {}
    for crit in CRITERIA:
        entry = parsed.get(crit)
        if not isinstance(entry, dict):
            return None
        score = entry.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            return None
        justification = entry.get("justification")
        if not isinstance(justification, str) or not justification.strip():
            return None
        out[crit] = {"score": score, "justification": justification.strip()}
    return out


def _exec_summary(exec_result: dict[str, Any] | None) -> str:
    if exec_result is None:
        return "The plot script was not executed (no execution record)."
    lines = [
        f"ran_ok: {exec_result.get('ran_ok')}",
        f"exit_code: {exec_result.get('exit_code')}",
        f"timed_out: {exec_result.get('timed_out')}",
        f"figures_produced: {exec_result.get('n_figures')}",
    ]
    if exec_result.get("detail"):
        lines.append(f"detail: {exec_result['detail']}")
    stderr = (exec_result.get("stderr") or "").strip()
    if stderr:
        lines.append(f"stderr (tail):\n{stderr[-2000:]}")
    return "\n".join(lines)


def check_summary(checks: dict[str, Any] | None) -> str:
    if checks is None:
        return "No automated check report available."
    summary: dict[str, Any] = {
        name: checks.get(name)
        for name in (
            "json_valid",
            "grid_complete",
            "intervals_ok",
            "family_ok",
            "params_reproduce_ok",
        )
    }
    detail = checks.get("detail") or {}
    summary["parse_mode"] = detail.get("parse_mode")
    grid = detail.get("grid") or {}
    summary["grid_issues"] = {
        k: grid.get(k)
        for k in ("n_strategies", "missing_cells", "duplicate_cells", "duplicate_titles", "issues")
    }
    strategies = []
    for s in detail.get("strategies") or []:
        strategies.append(
            {
                "idx": s.get("idx"),
                "title": s.get("title"),
                "cell": s.get("cell"),
                "interval_sane": s.get("interval_sane"),
                "family": s.get("family"),
                "family_valid": s.get("family_valid"),
                "params_reproduce": s.get("params_ok"),
                "reason": s.get("reason"),
            }
        )
    summary["per_strategy"] = strategies
    return json.dumps(summary, indent=2, ensure_ascii=False)


def _figure_bytes(store: artifacts.Store, iid: str) -> list[bytes]:
    figdir = store.figures_dir(iid)
    if not figdir.exists():
        return []
    out = []
    for path in sorted(figdir.glob("*.png"))[:_MAX_IMAGES]:
        out.append(path.read_bytes())
    return out


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
