"""One-off: rebuild the pairwise manifest for a changed candidate roster while
REUSING existing verdicts for every pair that did not change.

Context: pairwise `custom_id`s are positional (`p{pair_idx}-{order}` from
`enumerate(combinations(sorted(instance_ids), 2))`), so swapping a candidate
renumbers pairs and trips the drift guard in `load_or_write_manifest`. The only
sanctioned recovery is "move artifacts/pairwise aside and re-run everything".
That would waste API calls AND perturb the other models' already-published
verdicts (judges are non-deterministic even at temp 0).

Instead: after retiring the old candidate's instance from artifacts/instances/
and generating the new one, this script rebuilds a fresh manifest and pre-seeds
verdict files for every pair NOT involving the new model, copied from a backup of
the previous pairwise state. Only the new model's pairs are left outstanding, so
`make pairwise-submit` judges just those.

Correctness: the A/B coin flip (`_assign_a`) depends only on
(seed, instance_lo, instance_hi) — not on pair_idx — so for any unchanged pair
`instance_a`/`instance_b` are identical across the old and new manifests. Verdicts
are keyed by (judge, instance_a, instance_b) and rewritten under the new custom_id;
`write_csv` maps A/B -> model via the manifest entry, so the winner is preserved.

Usage (run from the footval/ root, AFTER backing up artifacts/pairwise):
    cp -r artifacts/pairwise artifacts/pairwise.bak
    uv run python scripts/reseed_pairwise.py --new-model claude-sonnet-5 --backup pairwise.bak
"""

from __future__ import annotations

import argparse
import shutil
import sys

from footval import artifacts, pairwise
from footval.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--new-model",
        required=True,
        help="candidate model name whose pairs should stay OUTSTANDING (e.g. claude-sonnet-5)",
    )
    ap.add_argument(
        "--backup",
        default="pairwise.bak",
        help="name of the backup dir under artifacts/ holding the prior pairwise state",
    )
    args = ap.parse_args()

    cfg = load_config()
    store = artifacts.Store(cfg.artifacts_dir)
    new_model = args.new_model
    judges = list(cfg.pairwise_judges)

    backup_dir = cfg.artifacts_dir / args.backup
    backup_verdicts = backup_dir / "verdicts"
    if not backup_verdicts.is_dir():
        sys.exit(
            f"backup verdicts not found: {backup_verdicts} (did you copy artifacts/pairwise first?)"
        )

    # 1) Confirm the new model's response exists (it must be in the fresh manifest).
    new_iid = artifacts.instance_id(new_model, 0)
    if not store.response_path(new_iid).exists():
        sys.exit(f"no response for {new_iid}; generate the new model before reseeding")

    # 2) Index OLD verdicts by (judge, instance_a, instance_b) -> record.
    old: dict[tuple[str, str, str], dict] = {}
    for judge_dir in sorted(backup_verdicts.iterdir()):
        if not judge_dir.is_dir():
            continue
        judge = judge_dir.name
        for vf in judge_dir.glob("*.json"):
            rec = store.read_json(vf)
            if rec is None:
                continue
            old[(judge, rec["instance_a"], rec["instance_b"])] = rec
    print(f"indexed {len(old)} old verdict records from {backup_verdicts}")

    # 3) Safety: never clobber verdicts already collected for the NEW model.
    live_verdicts = store.pairwise_dir / "verdicts"
    if live_verdicts.is_dir():
        for jd in live_verdicts.iterdir():
            if not jd.is_dir():
                continue
            for vf in jd.glob("*.json"):
                rec = store.read_json(vf)
                if rec and new_model in (rec.get("model_a"), rec.get("model_b")):
                    sys.exit(
                        f"live verdict already involves {new_model} ({vf}); "
                        "refusing to clobber collected results — remove them manually to re-seed."
                    )

    # 4) Clear live pairwise state so a fresh manifest can be written cleanly.
    for name in ("manifest.json", "batches.json", "attempts.json"):
        (store.pairwise_dir / name).unlink(missing_ok=True)
    for sub in ("verdicts", "raw"):
        shutil.rmtree(store.pairwise_dir / sub, ignore_errors=True)

    # 5) Fresh manifest for the new roster (disk-driven: reads artifacts/instances/).
    manifest = pairwise.load_or_write_manifest(cfg, store)
    comps = manifest["comparisons"]
    pairs = len({e["pair_idx"] for e in comps.values()})
    instances = manifest["instances"]
    print(
        f"fresh manifest: {len(instances)} instances, {pairs} pairs, {len(comps)} comparisons/judge"
    )
    if new_iid not in instances:
        sys.exit(f"{new_iid} not in fresh manifest instances — aborting")

    # 6) Pre-seed every (judge, cid) that does NOT involve the new model.
    seeded = 0
    outstanding_new = 0
    missing: list[tuple[str, str]] = []
    for judge, cid in pairwise.expected_items(manifest, judges):
        entry = comps[cid]
        involves_new = new_model in (entry["model_a"], entry["model_b"])
        if involves_new:
            outstanding_new += 1
            continue
        key = (judge, entry["instance_a"], entry["instance_b"])
        src = old.get(key)
        if src is None:
            missing.append((judge, cid))
            continue
        record = pairwise._verdict_record(
            cfg,
            manifest,
            judge,
            cid,
            via="reused",
            verdicts=src.get("verdicts"),
            raw_text=src.get("raw_text"),
            error=src.get("error"),
            snapshot_id=src.get("judge_snapshot_id"),
            stop_reason=src.get("stop_reason"),
            usage=src.get("usage"),
            attempts=int(src.get("format_attempts") or 0),
            batch_id=src.get("batch_id"),
            redactions=tuple(src.get("redaction_counts") or (0, 0)),
        )
        store.write_json(store.pairwise_verdict_path(judge, cid), record)
        seeded += 1

    if missing:
        sys.exit(
            f"FAILED: {len(missing)} non-{new_model} (judge,cid) pairs had no matching "
            f"old verdict; first few: {missing[:5]}"
        )

    print(
        f"pre-seeded {seeded} reused verdicts; {outstanding_new} (judge,cid) left "
        f"OUTSTANDING for {new_model}"
    )
    still = pairwise.outstanding_items(store, manifest, judges)
    by_judge: dict[str, int] = {}
    for j, _cid in still:
        by_judge[j] = by_judge.get(j, 0) + 1
    print(f"outstanding by judge: {by_judge}  (total {len(still)})")
    print("next: make pairwise-estimate  ->  make pairwise-submit + make pairwise-sync")


if __name__ == "__main__":
    main()
