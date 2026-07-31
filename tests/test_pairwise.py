import json

import pytest
from conftest import JUDGE_PROMPT_TEXT, make_cfg, valid_response

from footval import pairwise, providers
from footval.artifacts import Store

FAMILIES = {"m1": "openai", "m2": "openai", "m3": "google", "j1": "anthropic", "z1": "zai"}
# 8 candidates on the live roster -> C(8,2) = 28 unordered pairs.
N_CANDIDATES = 8
N_PAIRS = 28
IIDS = [f"m{i:02d}__s0" for i in range(N_CANDIDATES)]


def pw_cfg(tmp_path, both_orders=False, judges=("j1",), judge_params=None):
    families = dict(FAMILIES)
    families.update({f"m{i:02d}": "openai" for i in range(N_CANDIDATES)})
    return make_cfg(
        tmp_path,
        candidates=list(families),
        families=families,
        pairwise_judges=tuple(judges),
        pairwise_both_orders=both_orders,
        judge_params=judge_params,
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


def test_pairwise_system_has_soundness_criterion_only():
    assert "**Criterion 1 — Soundness of recommendations**" in JUDGE_PROMPT_TEXT
    assert "Criterion 2" not in JUDGE_PROMPT_TEXT
    assert "Criterion 3" not in JUDGE_PROMPT_TEXT
    # the plotting-script part of the task was removed; nothing about it is judged
    assert "code" not in JUDGE_PROMPT_TEXT.lower()
    assert pairwise.CRITERIA_PAIRWISE == ("soundness",)


def test_pairwise_system_has_no_numeric_scale():
    # The 1-5 scoring scale was removed everywhere; the comparison is a pure
    # forced choice with stronger-vs-weaker prose anchors, no numeric anchors.
    sys = JUDGE_PROMPT_TEXT
    assert "closer to a 5" not in sys
    for anchor in ("**5**", "**3**", "**1**", "5/3/1"):
        assert anchor not in sys
    assert "stronger" in sys.lower() and "weaker" in sys.lower()


def test_soundness_rubric_is_specific_and_ordered():
    # The rubric must name its dimensions and give the judge an explicit
    # tie-break order, not just a stronger/weaker blob.
    sys = JUDGE_PROMPT_TEXT
    for dimension in (
        "Genuinely underutilized",
        "Credibly win-positive",
        "Correct bucket attribution",
        "Grid discipline and distinctness",
        "Quality of the rationale",
    ):
        assert dimension in sys, dimension
    assert "priority order" in sys
    assert "Do not reward" in sys


def test_priors_are_out_of_scope_for_judges():
    sys = JUDGE_PROMPT_TEXT
    assert "out of scope" in sys.lower()
    assert "NOT being judged" in sys


def test_check_summary_shows_grid_facts_but_no_prior_mechanics():
    from footval.checks import check_instance

    report = check_instance(valid_response(), "direct", 0.1)
    summary = json.loads(pairwise.check_summary(report))
    assert summary["json_valid"] is True
    assert summary["grid_complete"] is True
    assert [s["cell"] for s in summary["per_strategy"]]
    # priors mechanics are computed and published, but withheld from judges
    text = json.dumps(summary)
    for leaked in ("params_reproduce", "family", "interval_sane", "intervals_ok"):
        assert leaked not in text, leaked


# --- enumeration & ids -------------------------------------------------------------


def test_pairs_single_order(tmp_path):
    cfg = pw_cfg(tmp_path)
    comps = pairwise.enumerate_comparisons(cfg, IIDS)
    assert len(comps) == N_PAIRS
    pairs = {frozenset((e["instance_a"], e["instance_b"])) for e in comps.values()}
    assert len(pairs) == N_PAIRS
    assert all(e["order"] == "ab" for e in comps.values())


def test_assignment_deterministic_and_order_independent(tmp_path):
    cfg = pw_cfg(tmp_path)
    a = pairwise.enumerate_comparisons(cfg, IIDS)
    b = pairwise.enumerate_comparisons(cfg, list(reversed(IIDS)))
    assert a == b
    # the seeded coin must produce a mix, not a constant assignment
    firsts = [e["instance_a"] for e in a.values()]
    lows = sum(1 for e in a.values() if e["instance_a"] < e["instance_b"])
    assert 0 < lows < N_PAIRS, firsts


def test_both_orders_doubles_with_swapped_twins(tmp_path):
    cfg = pw_cfg(tmp_path, both_orders=True)
    comps = pairwise.enumerate_comparisons(cfg, IIDS)
    assert len(comps) == 2 * N_PAIRS
    for idx in range(N_PAIRS):
        ab = comps[pairwise.custom_id_for(idx, "ab")]
        ba = comps[pairwise.custom_id_for(idx, "ba")]
        assert ab["instance_a"] == ba["instance_b"]
        assert ab["instance_b"] == ba["instance_a"]


def test_custom_id_round_trip():
    import re

    for idx in range(N_PAIRS):
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
    # every cached prefix part is byte-identical when candidate A is unchanged
    for idx in pairwise.CACHE_PART_IDXS:
        assert parts_ab[idx].text == parts_ac[idx].text, idx
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


def _verdict_json(sound="A"):
    return json.dumps({"soundness": {"winner": sound, "justification": "x wins"}})


def test_parse_verdicts_valid_direct_and_fenced():
    assert pairwise.parse_verdicts(_verdict_json())["soundness"]["winner"] == "A"
    fenced = f"```json\n{_verdict_json(sound='B')}\n```"
    assert pairwise.parse_verdicts(fenced)["soundness"]["winner"] == "B"


def test_parse_verdicts_ignores_a_priors_key():
    # Priors are no longer judged; a judge that volunteers one is still valid,
    # and the extra key must not reach the CSV.
    verdict = pairwise.parse_verdicts(
        json.dumps(
            {
                "soundness": {"winner": "A", "justification": "x"},
                "priors": {"winner": "B", "justification": "y"},
            }
        )
    )
    assert verdict == {"soundness": {"winner": "A", "justification": "x"}}


def test_parse_verdicts_rejects_bad_shapes():
    assert pairwise.parse_verdicts("not json") is None
    assert pairwise.parse_verdicts(_verdict_json(sound="C")) is None
    assert pairwise.parse_verdicts(_verdict_json(sound="a")) is None
    assert (
        pairwise.parse_verdicts(json.dumps({"priors": {"winner": "A", "justification": "x"}}))
        is None
    )
    empty_just = json.dumps({"soundness": {"winner": "A", "justification": " "}})
    assert pairwise.parse_verdicts(empty_just) is None
    tie = json.dumps({"soundness": {"winner": "tie", "justification": "x"}})
    assert pairwise.parse_verdicts(tie) is None


# --- batch request builders ----------------------------------------------------------------


def _item(cfg, judge_name="j1"):
    parts = pairwise.build_parts("TASK", "bundle-a", "bundle-b", retry=False)
    req = pairwise.build_request(cfg, cfg.judge_model(judge_name), parts)
    return providers.BatchItem(custom_id="p00-ab", req=req)


def test_anthropic_batch_params_cache_injection(tmp_path):
    cfg = pw_cfg(tmp_path)
    params = providers.anthropic_batch_params(_item(cfg))
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    content = params["messages"][0]["content"]
    for idx in pairwise.CACHE_PART_IDXS:
        assert content[idx]["cache_control"] == {"type": "ephemeral"}, idx
    # nothing after the last breakpoint is marked (bundle B, instruction)
    assert all(
        "cache_control" not in c for i, c in enumerate(content) if i not in pairwise.CACHE_PART_IDXS
    )
    # ... and the system prompt + at most 3 message breakpoints stay under
    # Anthropic's cap of 4 per request
    assert 1 + len(pairwise.CACHE_PART_IDXS) <= 4


def test_sync_anthropic_path_carries_the_same_breakpoints(tmp_path):
    # The breakpoints live on the request, so a resubmitted straggler run through
    # complete() caches identically to the batch body.
    cfg = pw_cfg(tmp_path)
    item = _item(cfg)
    assert providers._anthropic_kwargs(item.req) == providers.anthropic_batch_params(item)


def test_judge_params_override_candidate_effort(tmp_path):
    cfg = pw_cfg(
        tmp_path, judges=("j1",), judge_params={"j1": {"output_config": {"effort": "xhigh"}}}
    )
    assert cfg.model("j1").params == {}
    assert cfg.judge_model("j1").params == {"output_config": {"effort": "xhigh"}}
    # merging must not mutate the stored candidate config
    assert cfg.model("j1").params == {}


def test_openai_batch_line_shapes(tmp_path):
    cfg = pw_cfg(tmp_path, judges=("m1",))  # m1 is family openai in FAMILIES
    item = _item(cfg, judge_name="m1")
    line = providers.openai_batch_line(item, "/v1/responses")
    assert line["url"] == "/v1/responses"
    assert line["custom_id"] == "p00-ab"
    assert line["body"]["instructions"] == JUDGE_PROMPT_TEXT
    # stable cache key so every comparison this judge makes shares a cache
    assert line["body"]["prompt_cache_key"] == "footval-pairwise-m1"
    chat = providers.openai_batch_line(item, "/v1/chat/completions")
    assert chat["body"]["messages"][0]["role"] == "system"
    assert chat["body"]["prompt_cache_key"] == "footval-pairwise-m1"


def test_prompt_cache_key_not_sent_to_non_openai_chat_providers(tmp_path):
    # z.ai's OpenAI-compatible endpoint would reject the unknown field.
    cfg = pw_cfg(tmp_path, judges=("z1",))
    parts = pairwise.build_parts("TASK", "bundle-a", "bundle-b", retry=False)
    req = pairwise.build_request(cfg, cfg.judge_model("z1"), parts)
    assert cfg.model("z1").provider == "zai"
    assert "prompt_cache_key" not in providers._chat_kwargs(req)


def test_gemini_batch_request_shape_and_text_only(tmp_path):
    cfg = pw_cfg(tmp_path, judges=("m3",))  # m3 -> google/gemini provider
    item = _item(cfg, judge_name="m3")
    req = providers.gemini_batch_request(item)
    assert req["metadata"] == {"custom_id": "p00-ab"}
    assert req["config"]["system_instruction"] == JUDGE_PROMPT_TEXT
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
            verdicts={"soundness": {"winner": "A", "justification": "x"}},
        )
        store.write_json(store.pairwise_verdict_path("j1", cid), record)

    pairwise.write_csv(cfg)
    import csv as csvmod

    rows = list(csvmod.reader(open(cfg.outputs_data_dir / "pairwise_results.csv")))
    assert rows[0] == ["judge", "model_a", "model_b", "criterion", "winner"]
    assert len(rows) == 1 + 2  # 2 comparisons x 1 criterion
    assert {r[3] for r in rows[1:]} == {"soundness"}
    by_key = {(r[1], r[2], r[3]): r[4] for r in rows[1:]}
    ab = manifest["comparisons"]["p00-ab"]
    ba = manifest["comparisons"]["p00-ba"]
    # winner "A" maps to whichever model was shown first in THAT ordering
    assert by_key[(ab["model_a"], ab["model_b"], "soundness")] == ab["model_a"]
    assert by_key[(ba["model_a"], ba["model_b"], "soundness")] == ba["model_a"]


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
    assert len(rows) == 2
    assert all(r[4] == "" for r in rows[1:])
