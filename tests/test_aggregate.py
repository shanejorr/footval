import json

import pytest
from conftest import jrow, jrows, make_cfg

from footbench import aggregate
from footbench.artifacts import Store

FAMILIES = {"mA": "anthropic", "mB": "openai", "j1": "anthropic", "j2": "openai"}


def two_model_rows():
    return (
        jrows("mA", "j1", 4, 3, 5, judge_family="anthropic")
        + jrows("mA", "j2", 2, 3, 3)
        + jrows("mB", "j1", 5, 5, 5, judge_family="anthropic")
        + jrows("mB", "j2", 1, 1, 1)
    )


def by_model_criterion(model_scores):
    return {(r["model"], r["criterion"]): r for r in model_scores}


def test_hand_computed_means_composite_and_ranks(tmp_path):
    cfg = make_cfg(tmp_path, ["mA", "mB"], ["j1", "j2"], FAMILIES)
    tables = aggregate.build_tables(cfg, two_model_rows())
    ms = by_model_criterion(tables["model_scores"])

    assert ms[("mA", "soundness")]["mean_score"] == pytest.approx(3.0)
    assert ms[("mA", "priors")]["mean_score"] == pytest.approx(3.0)
    assert ms[("mA", "code")]["mean_score"] == pytest.approx(4.0)
    assert ms[("mA", "composite")]["mean_score"] == pytest.approx(10 / 3, abs=1e-4)
    assert ms[("mB", "composite")]["mean_score"] == pytest.approx(3.0)

    assert ms[("mA", "composite")]["rank"] == 1
    assert ms[("mB", "composite")]["rank"] == 2
    # soundness means tie at 3.0 -> both share min rank 1
    assert ms[("mA", "soundness")]["rank"] == 1
    assert ms[("mB", "soundness")]["rank"] == 1
    assert ms[("mA", "code")]["rank"] == 1
    assert ms[("mB", "code")]["rank"] == 2
    assert ms[("mA", "soundness")]["n_judgments"] == 2


def test_non_equal_weights(tmp_path):
    cfg = make_cfg(tmp_path, ["mA", "mB"], ["j1", "j2"], FAMILIES, weights=(2.0, 1.0, 1.0))
    tables = aggregate.build_tables(cfg, two_model_rows())
    ms = by_model_criterion(tables["model_scores"])
    assert ms[("mA", "composite")]["mean_score"] == pytest.approx((2 * 3 + 3 + 4) / 4)


def test_self_exclusion(tmp_path):
    families = {"mA": "anthropic", "mB": "openai", "j2": "openai"}
    cfg = make_cfg(tmp_path, ["mA", "mB"], ["mA", "j2"], families)
    rows = (
        jrows("mA", "mA", 5, 5, 5, judge_family="anthropic", is_self=True)
        + jrows("mA", "j2", 3, 3, 3)
        + jrows("mB", "mA", 4, 4, 4, judge_family="anthropic")
        + jrows("mB", "j2", 2, 2, 2)
    )
    tables = aggregate.build_tables(cfg, rows)
    ms = by_model_criterion(tables["model_scores"])
    assert ms[("mA", "soundness")]["mean_score"] == pytest.approx(4.0)  # primary keeps self
    assert ms[("mA", "soundness")]["mean_score_self_excluded"] == pytest.approx(3.0)
    assert ms[("mB", "soundness")]["mean_score_self_excluded"] == pytest.approx(3.0)


def test_model_judged_only_by_itself_yields_null_self_excluded(tmp_path):
    families = {"mC": "anthropic"}
    cfg = make_cfg(tmp_path, ["mC"], ["mC"], families)
    rows = jrows("mC", "mC", 5, 5, 5, judge_family="anthropic", is_self=True)
    tables = aggregate.build_tables(cfg, rows)
    ms = by_model_criterion(tables["model_scores"])
    assert ms[("mC", "soundness")]["mean_score"] == pytest.approx(5.0)
    assert ms[("mC", "soundness")]["mean_score_self_excluded"] is None
    assert ms[("mC", "soundness")]["rank_self_excluded"] is None


def test_missing_judgment_handling(tmp_path):
    cfg = make_cfg(tmp_path, ["mA", "mB"], ["j1", "j2"], FAMILIES)
    rows = (
        jrows("mA", "j1", 4, 4, 4, judge_family="anthropic")
        + jrows("mA", "j2", None, None, None)  # judge returned no valid scores
        + jrows("mB", "j1", 2, 2, 2, judge_family="anthropic")
        + jrows("mB", "j2", 4, 4, 4)
    )
    tables = aggregate.build_tables(cfg, rows)
    ms = by_model_criterion(tables["model_scores"])
    assert ms[("mA", "soundness")]["mean_score"] == pytest.approx(4.0)
    assert ms[("mA", "soundness")]["n_judgments"] == 1
    rel = {r["criterion"]: r for r in tables["reliability"]}
    assert rel["soundness"]["icc_n_complete"] == 1  # only mB__s0 has both judges


def test_bias_gap(tmp_path):
    cfg = make_cfg(tmp_path, ["mA", "mB"], ["j1", "j2"], FAMILIES)
    rows = (
        jrows("mA", "j1", 4, 4, 4, judge_family="anthropic")  # mA is j1's family
        + jrows("mB", "j1", 3, 3, 3, judge_family="anthropic")
        + jrows("mA", "j2", 3, 3, 3)
        + jrows("mB", "j2", 3, 3, 3)
    )
    tables = aggregate.build_tables(cfg, rows)
    bias = {(r["judge"], r["criterion"]): r for r in tables["bias"]}
    row = bias[("j1", "soundness")]
    assert row["own_family_mean"] == pytest.approx(4.0)
    assert row["other_mean"] == pytest.approx(3.0)
    assert row["gap"] == pytest.approx(1.0)
    assert row["n_own"] == 1 and row["n_other"] == 1
    assert bias[("j2", "soundness")]["gap"] == pytest.approx(0.0)


def test_deterministic_output(tmp_path):
    cfg = make_cfg(tmp_path, ["mA", "mB"], ["j1", "j2"], FAMILIES)
    t1 = aggregate.build_tables(cfg, two_model_rows())
    t2 = aggregate.build_tables(cfg, two_model_rows())
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)


def test_table_columns_match_spec(tmp_path):
    cfg = make_cfg(tmp_path, ["mA", "mB"], ["j1", "j2"], FAMILIES)
    tables = aggregate.build_tables(cfg, two_model_rows())
    assert {"model", "criterion", "mean_score", "rank", "mean_score_self_excluded"} <= set(
        tables["model_scores"][0]
    )
    assert {"criterion", "krippendorff_alpha", "icc2k", "kendall_w"} <= set(
        tables["reliability"][0]
    )
    assert {"judge", "criterion", "own_family_mean", "other_mean", "gap"} <= set(tables["bias"][0])


def test_run_on_synthetic_artifacts_tree(tmp_path):
    cfg = make_cfg(tmp_path, ["mA", "mB"], ["j1", "j2"], FAMILIES)
    store = Store(cfg.artifacts_dir)
    for model, scores_by_judge in {
        "mA": {"j1": (4, 3, 5), "j2": (2, 3, 3)},
        "mB": {"j1": (5, 5, 5), "j2": (1, 1, 1)},
    }.items():
        iid = f"{model}__s0"
        store.write_json(
            store.response_path(iid),
            {
                "instance_id": iid,
                "model": model,
                "sample_idx": 0,
                "snapshot_id": f"{model}-2026-01-01",
                "ts": "2026-06-10T00:00:00Z",
                "temperature": 1.0,
                "seed": 42,
                "stop_reason": "end_turn",
                "raw_text": "{}",
                "parse_mode": "direct",
                "parsed_json": {},
            },
        )
        for judge, (s, p, c) in scores_by_judge.items():
            store.write_json(
                store.judgment_path(iid, judge),
                {
                    "instance_id": iid,
                    "judge": judge,
                    "judge_family": FAMILIES[judge],
                    "is_self": False,
                    "scores": {
                        "soundness": {"score": s, "justification": "x"},
                        "priors": {"score": p, "justification": "x"},
                        "code": {"score": c, "justification": "x"},
                    },
                },
            )
    aggregate.run(cfg)
    model_scores = store.read_json(store.table_path("model_scores"))
    judgments = store.read_json(store.table_path("judgments"))
    assert len(judgments) == 2 * 2 * 3
    ms = by_model_criterion(model_scores)
    assert ms[("mA", "composite")]["rank"] == 1
    assert (store.tables_dir / "reliability.json").exists()
    assert (store.tables_dir / "bias.json").exists()


def test_null_score_rows_kept_in_judgments_table(tmp_path):
    cfg = make_cfg(tmp_path, ["mA"], ["j1"], {"mA": "anthropic", "j1": "anthropic"})
    rows = [jrow("mA", "j1", "soundness", None, judge_family="anthropic")]
    tables = aggregate.build_tables(cfg, rows)
    ms = by_model_criterion(tables["model_scores"])
    assert ("mA", "soundness") not in ms or ms[("mA", "soundness")]["mean_score"] is None
