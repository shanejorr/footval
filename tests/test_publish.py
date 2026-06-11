import csv

import pytest
from conftest import jrows, make_cfg, valid_response

from footbench import publish
from footbench.artifacts import Store


def test_write_scores_csv_long_form(tmp_path):
    rows = (
        jrows("mA", "j1", 4, 3, 5) + jrows("mB", "j1", 5, 5, 5) + jrows("mA", "j2", 2, 3, 3)
        # j2 never judged mB -> empty cells
    )
    path = tmp_path / "scores.csv"
    publish.write_scores_csv(rows, ["mA", "mB"], ["j1", "j2"], path)
    read = list(csv.reader(open(path)))
    assert read[0] == ["judge", "candidate", "soundness", "priors", "code"]
    assert len(read) == 1 + 2 * 2  # one row per judge x candidate
    assert read[1] == ["j1", "mA", "4", "3", "5"]
    assert read[3] == ["j2", "mA", "2", "3", "3"]
    assert read[4] == ["j2", "mB", "", "", ""]


def test_composite_model_order():
    model_scores = [
        {"model": "mB", "criterion": "composite", "rank": 1, "mean_score": 4.0},
        {"model": "mA", "criterion": "composite", "rank": 2, "mean_score": 3.0},
        {"model": "mA", "criterion": "soundness", "rank": 1, "mean_score": 5.0},
    ]
    order = publish.composite_model_order(model_scores, ("mA", "mB", "mC"))
    assert order == ["mB", "mA", "mC"]  # mC unranked -> appended in config order


def test_compute_cell_xrange_clamps_heavy_tail():
    ranges = [(-1.0, 1.0), (-1.2, 0.8), (-0.9, 1.1), (-50.0, 50.0)]
    lo, hi = publish.compute_cell_xrange(ranges)
    assert -6 < lo < -1  # contains the typical mass, not the nu=1.2 monster
    assert 1 < hi < 6


def test_compute_cell_xrange_single_entry():
    lo, hi = publish.compute_cell_xrange([(-1.0, 1.0)])
    assert lo < -1.0 < 1.0 < hi  # just padded


def test_compute_ymax_caps_confident_spike():
    assert publish.compute_ymax([0.4, 0.5, 8.0]) == pytest.approx(1.5 * 1.05)
    assert publish.compute_ymax([0.4, 0.5, 0.6]) == pytest.approx(0.6 * 1.05)
    assert publish.compute_ymax([]) == 1.0


def _store_with_good_and_bad(tmp_path):
    cfg = make_cfg(
        tmp_path,
        ["good", "bad"],
        ["j1"],
        {"good": "openai", "bad": "openai", "j1": "openai"},
    )
    store = Store(cfg.artifacts_dir)
    store.write_json(
        store.response_path("good__s0"),
        {
            "instance_id": "good__s0",
            "model": "good",
            "parse_mode": "direct",
            "parsed_json": valid_response(),
        },
    )
    store.write_json(
        store.response_path("bad__s0"),
        {
            "instance_id": "bad__s0",
            "model": "bad",
            "parse_mode": "failed",
            "parsed_json": None,
            "raw_text": "not json",
        },
    )
    return cfg, store


def test_collect_priors_excludes_invalid_json(tmp_path):
    cfg, store = _store_with_good_and_bad(tmp_path)
    cells, exclusions, model_list = publish.collect_priors(cfg, store)
    assert model_list == ["good", "bad"]
    assert ("bad", "response was not valid JSON") in exclusions
    assert sum(len(v) for v in cells.values()) == 6
    assert all(entries[0]["model"] == "good" for entries in cells.values())


def test_plots_render_smoke(tmp_path):
    rows = (
        jrows("mA", "j1", 4, 3, 5)
        + jrows("mB", "j1", 5, 5, 5)
        + jrows("mA", "j2", None, None, None)
    )
    heat = tmp_path / "heat.png"
    publish.plot_scores_by_judge(rows, ["mA", "mB"], ["j1", "j2"], heat)
    assert heat.exists() and heat.stat().st_size > 0

    cfg, store = _store_with_good_and_bad(tmp_path)
    priors = tmp_path / "priors.png"
    publish.plot_priors_comparison(cfg, store, priors)
    assert priors.exists() and priors.stat().st_size > 0
