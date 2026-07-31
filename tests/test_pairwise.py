import json

import pytest
from conftest import make_cfg, valid_response

from footval import pairwise, providers
from footval.artifacts import Store

FAMILIES = {"m1": "openai", "m2": "openai", "m3": "google", "j1": "anthropic"}
IIDS_12 = [f"m{i:02d}__s0" for i in range(12)]


def pw_cfg(tmp_path, both_orders=False, judges=("j1",)):
    families = dict(FAMILIES)
    families.update({f"m{i:02d}": "openai" for i in range(12)})
    return make_cfg(
        tmp_path,
        candidates=list(families),
        families=families,
        pairwise_judges=tuple(judges),
        pairwise_both_orders=both_orders,
    )


def seeded_store(tmp_path, instance_ids):
    store = Store(tmp_path / "artifacts")
    for iid in instance_ids:
        store.write_json(
            store.response_path(iid),
            {
                "instance_id": iid,
                "model": iid.rsplit("__s", 1)[0],
                "parse_mode": "direct",
                "parsed_json": valid_response(),
            },
        )
    return store


# --- rubric --------------------------------------------------------------


def test_pairwise_system_has_criteria_1_2_only():
    assert pairwise.RUBRIC_SOUNDNESS in pairwise.SYSTEM
    assert pairwise.RUBRIC_PRIORS in pairwise.SYSTEM
    assert "Criterion 3" not in pairwise.SYSTEM
    assert "code" not in pairwise.SYSTEM


def test_pairwise_system_has_no_numeric_scale():
    # The 1-5 scoring scale was removed everywhere; the comparison is a pure
    # forced choice with stronger-vs-weaker prose anchors, no numeric anchors.
    sys = pairwise.SYSTEM
    assert "closer to a 5" not in sys
    for anchor in ("**5**", "**3**", "**1**", "5/3/1"):
        assert anchor not in sys
    assert "stronger" in sys and "weaker" in sys


def test_priors_rubric_demotes_reproduction_to_tiebreaker():
    # Parameter reproduction must be framed as a secondary/tiebreaker signal,
    # and the SYSTEM must warn against tallying correlated failing cells — the
    # two behaviors that sank Sonnet 4.6's priors under the old prompt.
    assert "tiebreaker" in pairwise.RUBRIC_PRIORS.lower()
    assert "substance" in pairwise.RUBRIC_PRIORS.lower()
    assert "count of failing cells" in pairwise.SYSTEM


def test_repro_summary_collapses_common_cause():
    def strat(idx, family, ok):
        return {"idx": idx, "family": family, "params_ok": ok}

    # all pass
    assert "all 3 cells reproduce" in pairwise._repro_summary(
        [strat(i, "normal", True) for i in range(3)]
    )
    # several failures sharing one family read as a single repeated mistake
    same = pairwise._repro_summary(
        [strat(0, "skew_normal", False), strat(1, "skew_normal", False), strat(2, "normal", True)]
    )
    assert "2 of 3" in same and "one repeated conversion mistake" in same
    # mixed-family failures are not collapsed
    mixed = pairwise._repro_summary(
        [strat(0, "skew_normal", False), strat(1, "student_t", False)]
    )
    assert "repeated conversion mistake" not in mixed
    # empty
    assert pairwise._repro_summary([]) == "no strategies to check"


def test_check_summary_includes_repro_rollup_and_per_strategy():
    from footval.checks import check_instance

    report = check_instance(valid_response(), "direct", 0.1)
    summary = json.loads(pairwise.check_summary(report))
    assert "per_strategy" in summary
    assert "params_reproduce_summary" in summary
    assert isinstance(summary["params_reproduce_summary"], str)


# --- enumeration & ids -------------------------------------------------------------


def test_66_pairs_single_order(tmp_path):
    cfg = pw_cfg(tmp_path)
    comps = pairwise.enumerate_comparisons(cfg, IIDS_12)
    assert len(comps) == 66
    pairs = {frozenset((e["instance_a"], e["instance_b"])) for e in comps.values()}
    assert len(pairs) == 66
    assert all(e["order"] == "ab" for e in comps.values())


def test_assignment_deterministic_and_order_independent(tmp_path):
    cfg = pw_cfg(tmp_path)
    a = pairwise.enumerate_comparisons(cfg, IIDS_12)
    b = pairwise.enumerate_comparisons(cfg, list(reversed(IIDS_12)))
    assert a == b
    # the seeded coin must produce a mix, not a constant assignment
    firsts = [e["instance_a"] for e in a.values()]
    lows = sum(1 for e in a.values() if e["instance_a"] < e["instance_b"])
    assert 0 < lows < 66, firsts


def test_both_orders_doubles_with_swapped_twins(tmp_path):
    cfg = pw_cfg(tmp_path, both_orders=True)
    comps = pairwise.enumerate_comparisons(cfg, IIDS_12)
    assert len(comps) == 132
    for idx in range(66):
        ab = comps[pairwise.custom_id_for(idx, "ab")]
        ba = comps[pairwise.custom_id_for(idx, "ba")]
        assert ab["instance_a"] == ba["instance_b"]
        assert ab["instance_b"] == ba["instance_a"]


def test_custom_id_round_trip():
    import re

    for idx in range(66):
        for order in ("ab", "ba"):
            cid = pairwise.custom_id_for(idx, order)
            assert pairwise.parse_custom_id(cid) == (idx, order)
            assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid)
    with pytest.raises(ValueError):
        pairwise.parse_custom_id("gpt-5.5__vs__claude")


# --- prompt construction --------------------------------------------------------------


def test_bundle_byte_stability_across_pairs(tmp_path):
    store = seeded_store(tmp_path, ["mA__s0", "mB__s0", "mC__s0"])
    a1, _ = pairwise.build_bundle(store, "mA__s0", "A")
    a2, _ = pairwise.build_bundle(store, "mA__s0", "A")
    assert a1 == a2
    parts_ab = pairwise.build_parts("TASK", a1, "bundle-b", retry=False)
    parts_ac = pairwise.build_parts("TASK", a1, "bundle-c", retry=False)
    assert parts_ab[0].text == parts_ac[0].text
    assert parts_ab[pairwise.CACHE_PART_IDX].text == parts_ac[pairwise.CACHE_PART_IDX].text
    assert parts_ab[2].text != parts_ac[2].text


def test_bundle_label_and_check_report(tmp_path):
    store = seeded_store(tmp_path, ["mA__s0"])
    text, _n = pairwise.build_bundle(store, "mA__s0", "B")
    assert text.startswith("=== RESPONSE B ===")
    assert "AUTOMATED CHECK REPORT FOR RESPONSE B" in text


def test_retry_part_appended(tmp_path):
    parts = pairwise.build_parts("TASK", "a", "b", retry=True)
    assert len(parts) == 5
    assert "REMINDER" in parts[-1].text


# --- verdict parsing --------------------------------------------------------------------


def _verdict_json(sound="A", priors="B"):
    return json.dumps(
        {
            "soundness": {"winner": sound, "justification": "x wins"},
            "priors": {"winner": priors, "justification": "y wins"},
        }
    )


def test_parse_verdicts_valid_direct_and_fenced():
    assert pairwise.parse_verdicts(_verdict_json())["soundness"]["winner"] == "A"
    fenced = f"```json\n{_verdict_json()}\n```"
    assert pairwise.parse_verdicts(fenced)["priors"]["winner"] == "B"


def test_parse_verdicts_rejects_bad_shapes():
    assert pairwise.parse_verdicts("not json") is None
    assert pairwise.parse_verdicts(_verdict_json(sound="C")) is None
    assert pairwise.parse_verdicts(_verdict_json(sound="a")) is None
    missing = json.dumps({"soundness": {"winner": "A", "justification": "x"}})
    assert pairwise.parse_verdicts(missing) is None
    empty_just = json.dumps(
        {
            "soundness": {"winner": "A", "justification": " "},
            "priors": {"winner": "B", "justification": "y"},
        }
    )
    assert pairwise.parse_verdicts(empty_just) is None
    tie = json.dumps(
        {
            "soundness": {"winner": "tie", "justification": "x"},
            "priors": {"winner": "B", "justification": "y"},
        }
    )
    assert pairwise.parse_verdicts(tie) is None


# --- batch request builders ----------------------------------------------------------------


def _item(cfg, judge_name="j1", cache_idx=pairwise.CACHE_PART_IDX):
    parts = pairwise.build_parts("TASK", "bundle-a", "bundle-b", retry=False)
    req = pairwise.build_request(cfg, cfg.model(judge_name), parts)
    return providers.BatchItem(custom_id="p00-ab", req=req, cache_part_idx=cache_idx)


def test_anthropic_batch_params_cache_injection(tmp_path):
    cfg = pw_cfg(tmp_path)
    params = providers.anthropic_batch_params(_item(cfg))
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    content = params["messages"][0]["content"]
    assert content[pairwise.CACHE_PART_IDX]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in c for i, c in enumerate(content) if i != 1)


def test_openai_batch_line_shapes(tmp_path):
    cfg = pw_cfg(tmp_path, judges=("m1",))  # m1 is family openai in FAMILIES
    item = _item(cfg, judge_name="m1")
    line = providers.openai_batch_line(item, "/v1/responses")
    assert line["url"] == "/v1/responses"
    assert line["custom_id"] == "p00-ab"
    assert line["body"]["instructions"] == pairwise.SYSTEM
    chat = providers.openai_batch_line(item, "/v1/chat/completions")
    assert chat["body"]["messages"][0]["role"] == "system"


def test_gemini_batch_request_shape_and_text_only(tmp_path):
    cfg = pw_cfg(tmp_path, judges=("m3",))  # m3 -> google/gemini provider
    item = _item(cfg, judge_name="m3")
    req = providers.gemini_batch_request(item)
    assert req["metadata"] == {"custom_id": "p00-ab"}
    assert req["config"]["system_instruction"] == pairwise.SYSTEM
    assert req["contents"][0]["parts"][0]["text"] == "=== TASK GIVEN TO BOTH MODELS ===\nTASK"
    bad = providers.BatchItem(
        custom_id="x",
        req=providers.LLMRequest(
            model=cfg.model("m3"),
            system=None,
            parts=(providers.ContentPart(kind="image_png", png_bytes=b"x"),),
            temperature=None,
            seed=None,
            max_output_tokens=10,
        ),
    )
    with pytest.raises(ValueError):
        providers.gemini_batch_request(bad)


# --- manifest & outstanding set ---------------------------------------------------------


def test_manifest_round_trip_and_outstanding(tmp_path):
    cfg = pw_cfg(tmp_path)
    store = seeded_store(tmp_path, ["mA__s0", "mB__s0", "mC__s0"])
    manifest = pairwise.load_or_write_manifest(cfg, store)
    assert len(manifest["comparisons"]) == 3
    outstanding = pairwise.outstanding_items(store, manifest, ["j1"])
    assert len(outstanding) == 3
    judge_name, cid = outstanding[0]
    store.write_json(store.pairwise_verdict_path(judge_name, cid), {"verdicts": None})
    assert len(pairwise.outstanding_items(store, manifest, ["j1"])) == 2
    # idempotent reload
    again = pairwise.load_or_write_manifest(cfg, store)
    assert again["comparisons"] == manifest["comparisons"]


def test_manifest_drift_guard(tmp_path):
    cfg = pw_cfg(tmp_path)
    store = seeded_store(tmp_path, ["mA__s0", "mB__s0"])
    pairwise.load_or_write_manifest(cfg, store)
    # a new instance appearing changes the pair set -> allowed (pure extension)
    store.write_json(
        store.response_path("mZ__s0"),
        {"instance_id": "mZ__s0", "model": "mZ", "parse_mode": "direct", "parsed_json": {}},
    )
    extended = pairwise.load_or_write_manifest(cfg, store)
    assert len(extended["comparisons"]) == 3
    # but a changed seed contradicts stored assignments -> hard fail
    import dataclasses

    cfg2 = dataclasses.replace(cfg, seed=43)
    with pytest.raises(SystemExit):
        pairwise.load_or_write_manifest(cfg2, store)


# --- csv ------------------------------------------------------------------------------------


def test_write_csv_winner_mapping(tmp_path, capsys):
    cfg = pw_cfg(tmp_path, both_orders=True)
    store = seeded_store(tmp_path, ["mA__s0", "mB__s0"])
    manifest = pairwise.load_or_write_manifest(cfg, store)
    assert set(manifest["comparisons"]) == {"p00-ab", "p00-ba"}
    for cid in ("p00-ab", "p00-ba"):
        record = pairwise._verdict_record(
            cfg,
            manifest,
            "j1",
            cid,
            via="batch",
            verdicts={
                "soundness": {"winner": "A", "justification": "x"},
                "priors": {"winner": "B", "justification": "y"},
            },
        )
        store.write_json(store.pairwise_verdict_path("j1", cid), record)

    pairwise.write_csv(cfg)
    import csv as csvmod

    rows = list(csvmod.reader(open(cfg.outputs_data_dir / "pairwise_results.csv")))
    assert rows[0] == ["judge", "model_a", "model_b", "criterion", "winner"]
    assert len(rows) == 1 + 2 * 2  # 2 comparisons x 2 criteria
    by_key = {(r[1], r[2], r[3]): r[4] for r in rows[1:]}
    ab = manifest["comparisons"]["p00-ab"]
    ba = manifest["comparisons"]["p00-ba"]
    # winner "A" maps to whichever model was shown first in THAT ordering
    assert by_key[(ab["model_a"], ab["model_b"], "soundness")] == ab["model_a"]
    assert by_key[(ba["model_a"], ba["model_b"], "soundness")] == ba["model_a"]
    assert by_key[(ab["model_a"], ab["model_b"], "priors")] == ab["model_b"]


def test_write_csv_blank_winner_for_null_verdicts(tmp_path):
    cfg = pw_cfg(tmp_path)
    store = seeded_store(tmp_path, ["mA__s0", "mB__s0"])
    manifest = pairwise.load_or_write_manifest(cfg, store)
    cid = next(iter(manifest["comparisons"]))
    record = pairwise._verdict_record(
        cfg, manifest, "j1", cid, via="batch", verdicts=None, error="malformed"
    )
    store.write_json(store.pairwise_verdict_path("j1", cid), record)
    pairwise.write_csv(cfg)
    import csv as csvmod

    rows = list(csvmod.reader(open(cfg.outputs_data_dir / "pairwise_results.csv")))
    assert len(rows) == 3
    assert all(r[4] == "" for r in rows[1:])
