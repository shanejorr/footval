"""Pairwise comparison stage.

Every unordered pair of candidate response instances is compared by a panel of
judges on criteria 1-2 only (soundness, priors) — forced A/B choice per
criterion, text-only (no exec results or figures). Cost-optimized: batch APIs
for Anthropic/OpenAI/Gemini (50% off), synchronous calls for DeepSeek (no
batch API), and cache-friendly byte-stable prompt prefixes everywhere.

Lifecycle: submit -> status -> collect -> (submit again for stragglers) -> csv.
`outstanding = expected - terminal verdict files`, so re-running any
subcommand is always safe and resubmission IS the retry mechanism.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
import re
from itertools import combinations
from typing import Any

from . import artifacts, parsing, providers
from .config import Config, ModelCfg

CRITERIA_PAIRWISE = ("soundness", "priors")

# Content part index of the candidate-A bundle (Anthropic cache breakpoint).
CACHE_PART_IDX = 1

# Quality descriptions for the two compared criteria. Stated as stronger-vs-weaker
# prose (no numeric scale) so the comparison is a pure forced choice.
RUBRIC_SOUNDNESS = """\
**Criterion 1 — Soundness of recommendations**

- A **stronger** response: all six picks are genuinely underutilized *and* credibly win-positive; rationales are specific and football-literate; analytics vs. intuition buckets are correctly distinguished (intuition picks are not just restated analytics consensus); picks are distinct and correctly slotted in their grid cell.
- A **weaker** response: picks are generic/already standard, implausible, duplicative, or misattributed; rationales are vague or wrong; or buckets are misfiled (mainstream ideas labeled underutilized, or restated analytics passed off as intuition)."""

RUBRIC_PRIORS = """\
**Criterion 2 — Reasonableness of Bayesian priors**

Judge the *substance* of the beliefs first; treat parameter-reproduction accuracy as a secondary, mechanical matter.

- A **stronger** response (primary signals): effect-size magnitudes are football-plausible (no single strategy swinging the per-game scoring margin by several points); the six cells are genuinely differentiated rather than copy-pasted; intervals are neither overconfident nor absurdly wide; weak cells are handled honestly (near-zero center, wider interval); and the distribution family's shape matches the stated belief (a real left tail where downside is claimed, signed support where the effect can go either way).
- A **weaker** response (primary signals): grossly implausible or over-tight/over-wide magnitudes; identical boilerplate priors reused across differing strategies; an invalid family for signed data; or a family whose shape contradicts the stated belief (for example a symmetric family used to sidestep a claimed asymmetric downside).
- **Secondary (tiebreaker only):** whether the reported parameters reproduce the stated interval. A reproduction failure is a conversion slip — the parameters and the stated interval disagree — not evidence that the underlying belief is unreasonable. Do not reward playing it safe: a response that reaches for a richer family to represent honest asymmetric downside and gets a tail quantile slightly off should not automatically lose to one that avoided the shape with a symmetric family. Decide which *intended* belief is better first, and let reproduction accuracy break the tie only when the two responses are otherwise comparable on substance."""

SYSTEM = f"""You are an impartial evaluator comparing two anonymized responses ("Response A" and \
"Response B") to the same fixed task. For each criterion below, decide which response is BETTER. \
This is a forced choice: you must answer "A" or "B" for each criterion; ties are not allowed.

{RUBRIC_SOUNDNESS}

{RUBRIC_PRIORS}

Use these descriptions of a stronger versus weaker response as the definition of quality; pick \
the response that is better on that criterion.

Ground rules:
- The AUTOMATED CHECK REPORT attached to each response is ground truth for mechanical facts \
(grid completeness, interval sanity, family validity, whether reported parameters reproduce the \
stated interval). Do not re-litigate it; spend your judgment on the subjective quality.
- Weight a parameter-reproduction failure as a mechanical conversion error, not a substantive \
flaw: it should not by itself decide the priors criterion when one response's beliefs are clearly \
more reasonable. Let reproduction accuracy break the tie only when the two responses are \
otherwise comparable on substance.
- Weigh distinct *kinds* of problems, not the raw count of failing cells. Several cells failing \
for the same underlying reason (for example one wrong quantile applied to every skew-normal cell) \
is one defect, not many.
- A response that failed to parse as JSON or is missing parts should generally lose the affected \
criterion — evaluate whatever is present.
- Judge each criterion independently; the same response need not win both.

Return ONLY a JSON object in exactly this shape, with 1-2 sentence justifications:

{{"soundness": {{"winner": "A", "justification": "..."}}, "priors": {{"winner": "A", "justification": "..."}}}}"""

_INSTRUCTION = (
    "Compare Response A and Response B on the two criteria and return only the JSON verdict object."
)

_RETRY_NOTE = (
    "REMINDER: your previous reply was not a valid verdict object. Return ONLY the JSON object "
    '{"soundness": {"winner": "A" or "B", "justification": "..."}, "priors": {"winner": "A" or '
    '"B", "justification": "..."}} — no prose, no code fences, no other keys.'
)

_CID_RE = re.compile(r"^p(\d{2,4})-(ab|ba)$")

_POLLERS = {
    "anthropic": "poll_anthropic_batch",
    "openai": "poll_openai_batch",
    "gemini": "poll_gemini_batch",
}


# --- pair enumeration & ids -----------------------------------------------------


def custom_id_for(pair_idx: int, order: str) -> str:
    return f"p{pair_idx:02d}-{order}"


def parse_custom_id(cid: str) -> tuple[int, str]:
    m = _CID_RE.match(cid)
    if not m:
        raise ValueError(f"bad pairwise custom_id {cid!r}")
    return int(m.group(1)), m.group(2)


def _assign_a(seed: int, iid_lo: str, iid_hi: str) -> str:
    """Deterministic per-pair A/B coin flip, independent of enumeration order."""
    rng = random.Random(f"{seed}:{iid_lo}:{iid_hi}")
    return iid_lo if rng.random() < 0.5 else iid_hi


def enumerate_comparisons(cfg: Config, instance_ids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for idx, (lo, hi) in enumerate(combinations(sorted(instance_ids), 2)):
        first = _assign_a(cfg.seed, lo, hi)
        second = hi if first == lo else lo
        orders = ("ab", "ba") if cfg.pairwise_both_orders else ("ab",)
        for order in orders:
            ia, ib = (first, second) if order == "ab" else (second, first)
            out[custom_id_for(idx, order)] = {
                "pair_idx": idx,
                "order": order,
                "instance_a": ia,
                "instance_b": ib,
                "model_a": artifacts.model_from_instance_id(ia),
                "model_b": artifacts.model_from_instance_id(ib),
            }
    return out


# --- manifest --------------------------------------------------------------------


def build_manifest(cfg: Config, store: artifacts.Store) -> dict[str, Any]:
    iids = [i for i in store.instance_ids() if store.response_path(i).exists()]
    return {
        "seed": cfg.seed,
        "both_orders": cfg.pairwise_both_orders,
        "judges": list(cfg.pairwise_judges),
        "instances": sorted(iids),
        "comparisons": enumerate_comparisons(cfg, iids),
    }


def load_or_write_manifest(cfg: Config, store: artifacts.Store) -> dict[str, Any]:
    """Regenerate deterministically; hard-fail on contradiction with the stored
    manifest (seed/candidate drift would silently mismatch verdict files)."""
    fresh = build_manifest(cfg, store)
    existing = store.read_json(store.pairwise_manifest_path)
    if existing is not None:
        if existing.get("seed") != fresh["seed"]:
            raise SystemExit(
                "pairwise manifest drift on 'seed' — existing verdicts would be "
                "mismatched. Move artifacts/pairwise/ aside to restart."
            )
        # Pure extension (new instances sorting after the existing ones) keeps
        # every stored custom_id pointing at the same pair; anything that
        # renumbers or reassigns an existing comparison is a hard failure.
        for cid, entry in existing.get("comparisons", {}).items():
            if cid in fresh["comparisons"] and fresh["comparisons"][cid] != entry:
                raise SystemExit(
                    f"pairwise manifest drift on comparison {cid!r} — existing verdicts "
                    "would be mismatched. Move artifacts/pairwise/ aside to restart."
                )
            if cid not in fresh["comparisons"]:
                raise SystemExit(
                    f"stored comparison {cid!r} no longer exists (instance removed?) — "
                    "Move artifacts/pairwise/ aside to restart."
                )
    if existing is None or existing.get("comparisons") != fresh["comparisons"]:
        store.write_json(store.pairwise_manifest_path, {**fresh, "ts": _now()})
    return fresh


# --- prompt construction -----------------------------------------------------------


def check_summary(checks: dict[str, Any] | None) -> str:
    """Compact, judge-facing summary of one instance's automated check report.

    Ground-truth context handed to judges so they don't re-derive mechanical facts.
    """
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
    raw_strategies = detail.get("strategies") or []
    strategies = []
    for s in raw_strategies:
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
    summary["params_reproduce_summary"] = _repro_summary(raw_strategies)
    return json.dumps(summary, indent=2, ensure_ascii=False)


def _repro_summary(strategies: list[dict[str, Any]]) -> str:
    """One-line, common-cause roll-up of parameter-reproduction failures.

    Presents a single repeated conversion mistake as one issue rather than N
    independent failures, so the judge does not tally correlated cells. Neutral
    ground-truth context, not a verdict.
    """
    n = len(strategies)
    if n == 0:
        return "no strategies to check"
    fails = [s for s in strategies if s.get("params_ok") is False]
    if not fails:
        return f"all {n} cells reproduce their stated interval"
    cells = [s.get("idx") for s in fails]
    families = {s.get("family") for s in fails}
    note = ""
    if len(fails) > 1 and len(families) == 1:
        fam = next(iter(families))
        note = (
            f"; every failing cell uses the same family ({fam}), so this most likely "
            "reflects one repeated conversion mistake rather than several distinct errors"
        )
    return (
        f"{len(fails)} of {n} cells' reported parameters do not reproduce the stated "
        f"interval (cells {cells}){note}"
    )


def build_bundle(store: artifacts.Store, iid: str, label: str) -> tuple[str, int]:
    """One candidate's full anonymized package. Deterministic from on-disk
    artifacts, so every call sharing this (iid, label) gets identical bytes —
    the property provider-side prefix caching depends on."""
    resp = store.load_response(iid) or {}
    parsed = resp.get("parsed_json")
    if isinstance(parsed, dict | list):
        body = json.dumps(parsed, indent=2, ensure_ascii=False)
        body_label = f"Response {label} parsed as JSON; it follows pretty-printed."
    else:
        body = resp.get("raw_text") or ""
        body_label = f"Response {label} did NOT parse as JSON; the raw text follows."
    text = (
        f"=== RESPONSE {label} ===\n{body_label}\n\n{body}\n\n"
        f"=== AUTOMATED CHECK REPORT FOR RESPONSE {label} (ground truth) ===\n"
        f"{check_summary(store.load_checks(iid))}"
    )
    return parsing.redact(text)


def build_parts(
    prompt_md: str, bundle_a: str, bundle_b: str, retry: bool
) -> tuple[providers.ContentPart, ...]:
    parts = [
        providers.ContentPart(kind="text", text="=== TASK GIVEN TO BOTH MODELS ===\n" + prompt_md),
        providers.ContentPart(kind="text", text=bundle_a),  # parts[CACHE_PART_IDX]
        providers.ContentPart(kind="text", text=bundle_b),
        providers.ContentPart(kind="text", text=_INSTRUCTION),
    ]
    if retry:
        parts.append(providers.ContentPart(kind="text", text=_RETRY_NOTE))
    return tuple(parts)


def build_request(
    cfg: Config, judge_cfg: ModelCfg, parts: tuple[providers.ContentPart, ...]
) -> providers.LLMRequest:
    return providers.LLMRequest(
        model=judge_cfg,
        system=SYSTEM,
        parts=parts,
        temperature=cfg.judge_temperature,
        seed=cfg.seed,
        max_output_tokens=cfg.pairwise_output_cap,
    )


class _BundleCache:
    def __init__(self, store: artifacts.Store):
        self.store = store
        self._cache: dict[tuple[str, str], tuple[str, int]] = {}

    def get(self, iid: str, label: str) -> tuple[str, int]:
        key = (iid, label)
        if key not in self._cache:
            self._cache[key] = build_bundle(self.store, iid, label)
        return self._cache[key]


# --- state -------------------------------------------------------------------------


def expected_items(manifest: dict, judges: list[str]) -> list[tuple[str, str]]:
    cids = sorted(manifest["comparisons"], key=parse_custom_id)
    return [(j, cid) for j in judges for cid in cids]


def outstanding_items(
    store: artifacts.Store, manifest: dict, judges: list[str]
) -> list[tuple[str, str]]:
    return [
        (j, cid)
        for (j, cid) in expected_items(manifest, judges)
        if not store.pairwise_verdict_path(j, cid).exists()
    ]


def _load_attempts(store: artifacts.Store) -> dict[str, dict[str, int]]:
    return store.read_json(store.pairwise_attempts_path) or {}


# --- verdicts ------------------------------------------------------------------------


def parse_verdicts(text: str) -> dict[str, Any] | None:
    parsed, _mode = parsing.parse_candidate_json(text)
    if not isinstance(parsed, dict):
        return None
    out: dict[str, Any] = {}
    for crit in CRITERIA_PAIRWISE:
        entry = parsed.get(crit)
        if not isinstance(entry, dict):
            return None
        winner = entry.get("winner")
        if winner not in ("A", "B"):
            return None
        justification = entry.get("justification")
        if not isinstance(justification, str) or not justification.strip():
            return None
        out[crit] = {"winner": winner, "justification": justification.strip()}
    return out


def _verdict_record(
    cfg: Config,
    manifest: dict,
    judge_name: str,
    cid: str,
    *,
    via: str,
    verdicts: dict[str, Any] | None,
    raw_text: str | None = None,
    error: str | None = None,
    snapshot_id: str | None = None,
    stop_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    attempts: int = 0,
    batch_id: str | None = None,
    redactions: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    entry = manifest["comparisons"][cid]
    jcfg = cfg.model(judge_name)
    return {
        "custom_id": cid,
        "judge": judge_name,
        "judge_family": jcfg.family,
        **entry,
        "is_self_a": judge_name == entry["model_a"],
        "is_self_b": judge_name == entry["model_b"],
        "via": via,
        "batch_id": batch_id,
        "judge_snapshot_id": snapshot_id,
        "stop_reason": stop_reason,
        "usage": usage or {},
        "format_attempts": attempts,
        "redaction_counts": list(redactions),
        "verdicts": verdicts,
        "raw_text": raw_text,
        "error": error,
        "ts": _now(),
    }


# --- subcommands -----------------------------------------------------------------------


def estimate(cfg: Config, only_judges: list[str] | None = None) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    manifest = build_manifest(cfg, store)  # in-memory; no writes
    judges = _select_judges(cfg, only_judges)
    bundles = _BundleCache(store)
    prompt_md = cfg.prompt_path.read_text()
    shared_chars = len(SYSTEM) + len(prompt_md) + len(_INSTRUCTION)
    print(
        f"pairs: {len({e['pair_idx'] for e in manifest['comparisons'].values()})}, "
        f"comparisons per judge: {len(manifest['comparisons'])}, "
        f"both_orders: {manifest['both_orders']}"
    )
    grand_tokens = 0
    for judge_name in judges:
        outstanding = outstanding_items(store, manifest, [judge_name])
        total_chars = 0
        for _j, cid in outstanding:
            entry = manifest["comparisons"][cid]
            a_text, _ = bundles.get(entry["instance_a"], "A")
            b_text, _ = bundles.get(entry["instance_b"], "B")
            total_chars += shared_chars + len(a_text) + len(b_text)
        est_tokens = int(total_chars / 4)
        grand_tokens += est_tokens
        provider = cfg.model(judge_name).provider
        batchable = "batch 50% off" if provider != "deepseek" else "sync (no batch API)"
        print(
            f"  {judge_name:<22} {len(outstanding):>3} outstanding  "
            f"~{est_tokens / 1e6:.2f}M input tokens  [{provider}: {batchable}]"
        )
    print(
        f"total estimated input: ~{grand_tokens / 1e6:.2f}M tokens (before caching/batch discounts)"
    )


def submit(cfg: Config, only_judges: list[str] | None = None, dump_prompt: bool = False) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    manifest = load_or_write_manifest(cfg, store)
    judges = _select_judges(cfg, only_judges)
    attempts = _load_attempts(store)
    prompt_md = cfg.prompt_path.read_text()
    bundles = _BundleCache(store)
    batches_log = store.read_json(store.pairwise_batches_path) or []

    for judge_name in judges:
        jcfg = cfg.model(judge_name)
        cids = [cid for _j, cid in outstanding_items(store, manifest, [judge_name])]
        if not cids:
            print(f"  {judge_name}: nothing outstanding")
            continue
        # adjacency by candidate-A maximizes provider-side prefix-cache hits
        cids.sort(key=lambda c: (manifest["comparisons"][c]["instance_a"], parse_custom_id(c)))
        items = []
        for cid in cids:
            entry = manifest["comparisons"][cid]
            a_text, _ = bundles.get(entry["instance_a"], "A")
            b_text, _ = bundles.get(entry["instance_b"], "B")
            retry = attempts.get(judge_name, {}).get(cid, 0) > 0
            parts = build_parts(prompt_md, a_text, b_text, retry)
            req = build_request(cfg, jcfg, parts)
            items.append(providers.BatchItem(custom_id=cid, req=req, cache_part_idx=CACHE_PART_IDX))
        if dump_prompt:
            item = items[0]
            print(f"\n===== pairwise prompt: judge={judge_name} custom_id={item.custom_id} =====")
            print(f"--- system ---\n{item.req.system}\n")
            for part in item.req.parts:
                print(f"--- text part ({len(part.text)} chars) ---\n{part.text[:1500]}\n[...]\n")
            return
        if jcfg.provider == "deepseek":
            print(
                f"  {judge_name}: {len(cids)} outstanding — no batch API; "
                "run `make pairwise-deepseek`"
            )
            continue
        endpoint = _openai_endpoint(batches_log) if jcfg.provider == "openai" else None
        print(f"  submitting {len(items)} comparisons for {judge_name} ({jcfg.provider}) ...")
        try:
            if jcfg.provider == "anthropic":
                info = providers.submit_anthropic_batch(items)
            elif jcfg.provider == "openai":
                info = providers.submit_openai_batch(items, endpoint=endpoint)
            elif jcfg.provider == "gemini":
                info = providers.submit_gemini_batch(jcfg, items)
            else:  # pragma: no cover — config validation prevents this
                raise SystemExit(f"unknown provider {jcfg.provider!r}")
        except Exception as exc:  # noqa: BLE001 — record and keep items outstanding
            print(f"    SUBMIT FAILED ({type(exc).__name__}): {exc}")
            continue
        batches_log.append(
            {
                "provider": jcfg.provider,
                "judge": judge_name,
                **info,
                "custom_ids": [i.custom_id for i in items],
                "submitted_at": _now(),
                "state": "submitted",
            }
        )
        store.write_json(store.pairwise_batches_path, batches_log)
        print(f"    batch_id: {info['batch_id']}")
    print("submit done — `make pairwise-status` to poll, `make pairwise-deepseek` for sync items.")


def _openai_endpoint(batches_log: list[dict]) -> str:
    """Fall back to chat completions if a /v1/responses batch failed wholesale."""
    for b in batches_log:
        if (
            b["provider"] == "openai"
            and b.get("state") == "failed"
            and (b.get("endpoint") == "/v1/responses")
        ):
            return "/v1/chat/completions"
    return "/v1/responses"


def run_deepseek(cfg: Config, only_judges: list[str] | None = None) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    manifest = load_or_write_manifest(cfg, store)
    judges = [j for j in _select_judges(cfg, only_judges) if cfg.model(j).provider == "deepseek"]
    if not judges:
        print("no deepseek judges configured")
        return
    prompt_md = cfg.prompt_path.read_text()
    bundles = _BundleCache(store)
    for judge_name in judges:
        jcfg = cfg.model(judge_name)
        cids = [cid for _j, cid in outstanding_items(store, manifest, [judge_name])]
        # sequential same-A calls hit DeepSeek's automatic server-side prefix cache
        cids.sort(key=lambda c: (manifest["comparisons"][c]["instance_a"], parse_custom_id(c)))
        print(f"  {judge_name}: {len(cids)} comparisons (sync)")
        for cid in cids:
            entry = manifest["comparisons"][cid]
            a_text, a_n = bundles.get(entry["instance_a"], "A")
            b_text, b_n = bundles.get(entry["instance_b"], "B")
            verdicts = None
            res = None
            error = None
            attempt = 0
            for attempt in range(cfg.judge_max_format_retries + 1):
                parts = build_parts(prompt_md, a_text, b_text, retry=attempt > 0)
                try:
                    res = providers.complete(build_request(cfg, jcfg, parts))
                except (providers.ModelConfigError, providers.TransientAPIError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break
                verdicts = parse_verdicts(res.text)
                if verdicts is not None:
                    break
                error = "judge output did not match the required verdict schema"
            record = _verdict_record(
                cfg,
                manifest,
                judge_name,
                cid,
                via="sync",
                verdicts=verdicts,
                raw_text=res.text if res else None,
                error=None if verdicts else error,
                snapshot_id=res.snapshot_id if res else None,
                stop_reason=res.stop_reason if res else None,
                usage=res.usage if res else None,
                attempts=attempt,
                redactions=(a_n, b_n),
            )
            store.write_json(store.pairwise_verdict_path(judge_name, cid), record)
            if verdicts:
                summary = {c: verdicts[c]["winner"] for c in CRITERIA_PAIRWISE}
                print(f"    {cid}: {summary}")
            else:
                print(f"    {cid}: NO VALID VERDICT ({error})")


def status(cfg: Config) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    batches = store.read_json(store.pairwise_batches_path) or []
    if not batches:
        print("no batches submitted yet")
        return
    for b in batches:
        if b["state"] in ("collected", "failed"):
            print(f"  {b['judge']:<22} {b['batch_id']}: {b['state']}")
            continue
        poll = getattr(providers, _POLLERS[b["provider"]])(b["batch_id"])
        line = f"  {b['judge']:<22} {b['batch_id']}: {poll['status']}"
        if poll.get("counts"):
            line += f"  {poll['counts']}"
        if poll.get("state"):
            line += f"  ({poll['state']})"
        if poll.get("errors"):
            line += f"\n      errors: {poll['errors'][:300]}"
        print(line)


def collect(cfg: Config) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    manifest = load_or_write_manifest(cfg, store)
    attempts = _load_attempts(store)
    batches = store.read_json(store.pairwise_batches_path) or []
    changed = False
    for b in batches:
        if b["state"] in ("collected", "failed"):
            continue
        poll = getattr(providers, _POLLERS[b["provider"]])(b["batch_id"])
        if poll["status"] == "in_progress":
            print(f"  {b['judge']}: still running")
            continue
        if poll["status"] == "failed":
            print(f"  {b['judge']}: batch FAILED — {poll.get('errors', 'no detail')}")
            b["state"] = "failed"
            b["errors"] = poll.get("errors")
            changed = True
            continue
        if b["provider"] == "anthropic":
            results = providers.fetch_anthropic_batch(b["batch_id"])
            raw = json.dumps(results, indent=2, default=str)
        elif b["provider"] == "openai":
            results, raw = providers.fetch_openai_batch(b["batch_id"])
        else:
            results = providers.fetch_gemini_batch(b["batch_id"], b["custom_ids"])
            raw = json.dumps(results, indent=2, default=str)
        raw_name = f"{b['judge']}__{b['batch_id'].replace('/', '_')}.jsonl"
        store.pairwise_raw_dir.mkdir(parents=True, exist_ok=True)
        (store.pairwise_raw_dir / raw_name).write_text(raw)

        written = retried = 0
        for rec in results:
            cid = rec.get("custom_id")
            if cid not in manifest["comparisons"]:
                print(f"    WARNING: result for unknown custom_id {cid!r}")
                continue
            vpath = store.pairwise_verdict_path(b["judge"], cid)
            if vpath.exists():
                continue
            verdicts = parse_verdicts(rec["text"] or "") if rec["ok"] else None
            prior_attempts = attempts.get(b["judge"], {}).get(cid, 0)
            if verdicts is not None:
                record = _verdict_record(
                    cfg,
                    manifest,
                    b["judge"],
                    cid,
                    via="batch",
                    verdicts=verdicts,
                    raw_text=rec["text"],
                    snapshot_id=rec["snapshot_id"],
                    stop_reason=rec["stop_reason"],
                    usage=rec["usage"],
                    attempts=prior_attempts,
                    batch_id=b["batch_id"],
                )
                store.write_json(vpath, record)
                written += 1
                continue
            n = attempts.setdefault(b["judge"], {})
            n[cid] = n.get(cid, 0) + 1
            if n[cid] > cfg.judge_max_format_retries:
                record = _verdict_record(
                    cfg,
                    manifest,
                    b["judge"],
                    cid,
                    via="batch",
                    verdicts=None,
                    raw_text=rec.get("text"),
                    error=rec.get("error") or "malformed verdict after retries",
                    snapshot_id=rec.get("snapshot_id"),
                    stop_reason=rec.get("stop_reason"),
                    usage=rec.get("usage"),
                    attempts=n[cid],
                    batch_id=b["batch_id"],
                )
                store.write_json(vpath, record)
                written += 1
            else:
                retried += 1
        store.write_json(store.pairwise_attempts_path, attempts)
        b["state"] = "collected"
        b["collected_counts"] = {"written": written, "needs_resubmit": retried}
        changed = True
        print(f"  {b['judge']}: {written} verdicts written, {retried} need resubmission")
    if changed:
        store.write_json(store.pairwise_batches_path, batches)
    outstanding = outstanding_items(store, manifest, list(cfg.pairwise_judges))
    if outstanding:
        by_judge: dict[str, int] = {}
        for j, _cid in outstanding:
            by_judge[j] = by_judge.get(j, 0) + 1
        print(f"outstanding: {by_judge} — run `make pairwise-submit` (and/or pairwise-deepseek)")
    else:
        print("all verdicts collected — run `make pairwise-csv`")


def write_csv(cfg: Config) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    manifest = load_or_write_manifest(cfg, store)
    cids = sorted(manifest["comparisons"], key=parse_custom_id)
    out_path = cfg.outputs_data_dir / "pairwise_results.csv"
    cfg.outputs_data_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    wins: dict[tuple[str, str], int] = {}
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["judge", "model_a", "model_b", "criterion", "winner"])
        for judge_name in cfg.pairwise_judges:
            for cid in cids:
                verdict = store.load_pairwise_verdict(judge_name, cid)
                entry = manifest["comparisons"][cid]
                if verdict is None:
                    missing += 1
                    continue
                for crit in CRITERIA_PAIRWISE:
                    winner = ""
                    if verdict.get("verdicts"):
                        side = verdict["verdicts"][crit]["winner"]
                        winner = entry["model_a"] if side == "A" else entry["model_b"]
                        wins[(winner, crit)] = wins.get((winner, crit), 0) + 1
                    writer.writerow([judge_name, entry["model_a"], entry["model_b"], crit, winner])
    print(f"wrote {out_path}")
    if missing:
        print(f"WARNING: {missing} comparisons have no verdict file yet (rows omitted)")
    for crit in CRITERIA_PAIRWISE:
        ranked = sorted(((m, n) for (m, c), n in wins.items() if c == crit), key=lambda x: -x[1])
        print(f"win counts [{crit}]: " + ", ".join(f"{m}={n}" for m, n in ranked))


def _select_judges(cfg: Config, only: list[str] | None) -> list[str]:
    if not cfg.pairwise_judges:
        raise SystemExit("no pairwise.judges configured in config.yaml")
    return [j for j in cfg.pairwise_judges if not only or j in only]


def run_cli(
    cfg: Config,
    subcommand: str | None,
    only: list[str] | None = None,
    dump_prompt: bool = False,
) -> None:
    subs = {
        "estimate": lambda: estimate(cfg, only),
        "submit": lambda: submit(cfg, only, dump_prompt),
        "deepseek": lambda: run_deepseek(cfg, only),
        "status": lambda: status(cfg),
        "collect": lambda: collect(cfg),
        "csv": lambda: write_csv(cfg),
    }
    if subcommand not in subs:
        raise SystemExit(f"usage: python -m footbench pairwise {{{' | '.join(subs)}}}")
    subs[subcommand]()


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
