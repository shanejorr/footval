import json

import pytest
from conftest import JUDGE_PROMPTS, make_cfg, valid_response

from footval import pairwise, providers
from footval.artifacts import Store
from footval.config import ModelCfg

FAMILIES = {"m1": "openai", "m2": "openai", "m3": "google", "j1": "anthropic", "z1": "zai"}
# 7 candidates on the live roster -> C(7,2) = 21 unordered pairs.
N_CANDIDATES = 7
N_PAIRS = 21
N_CRITERIA = len(pairwise.CRITERIA_PAIRWISE)
IIDS = [f"m{i:02d}__s0" for i in range(N_CANDIDATES)]

ANL = "analytical_reasoning"
INT = "intuitive_reasoning"


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


def test_two_criteria_each_scoped_to_its_grid_row():
    assert pairwise.CRITERIA_PAIRWISE == (ANL, INT)
    assert pairwise.CRITERION_BUCKET == {ANL: "analytics", INT: "intuition"}
    a, i = JUDGE_PROMPTS[ANL], JUDGE_PROMPTS[INT]
    assert "Criterion — Analytical reasoning" in a
    assert "Criterion — Intuitive reasoning" in i
    # each prompt names its own row and rules the other row out of scope
    assert "reduced to its `analytics` row" in a and "The `intuition` row" in a
    assert "reduced to its `intuition` row" in i and "The `analytics` row" in i
    # each prompt demands the verdict under its own criterion key
    assert f'{{"{ANL}"' in a and f'{{"{INT}"' in i


def test_prompts_judge_the_inference_not_the_conclusion():
    for prompt in JUDGE_PROMPTS.values():
        assert "Judge the inference, not the conclusion" in prompt


def test_under_adoption_is_an_argued_case_not_a_judge_fact_check():
    for prompt in JUDGE_PROMPTS.values():
        assert "The case for under-adoption" in prompt
        assert "not your own belief about current league practice" in prompt
        assert "flagrantly wrong" in prompt


def test_prompts_have_no_numeric_scale():
    for prompt in JUDGE_PROMPTS.values():
        for anchor in ("**5**", "**3**", "**1**", "5/3/1", "closer to a 5"):
            assert anchor not in prompt
        assert "stronger" in prompt.lower() and "weaker" in prompt.lower()


def test_rubrics_are_specific_and_ordered():
    for dim in (
        "Faithful use of evidence",
        "Credibly win-positive mechanism",
        "Situational precision",
        "The case for under-adoption",
        "Engagement with the obvious counterargument",
    ):
        assert dim in JUDGE_PROMPTS[ANL], dim
    for dim in (
        "Genuine intuition",
        "Plausibility of the causal story",
        "Originality without fantasy",
        "The case for under-adoption",
        "Epistemic honesty",
    ):
        assert dim in JUDGE_PROMPTS[INT], dim
    for prompt in JUDGE_PROMPTS.values():
        assert "priority order" in prompt
        assert "Do not reward" in prompt
        assert "forced choice" in prompt.lower()


def test_priors_are_out_of_scope_for_judges():
    for prompt in JUDGE_PROMPTS.values():
        assert "out of scope" in prompt.lower()
        assert "NOT being judged" in prompt


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
    assert len(comps) == N_PAIRS * N_CRITERIA
    pairs = {frozenset((e["instance_a"], e["instance_b"])) for e in comps.values()}
    assert len(pairs) == N_PAIRS
    assert all(e["order"] == "ab" for e in comps.values())
    for crit in pairwise.CRITERIA_PAIRWISE:
        assert sum(1 for e in comps.values() if e["criterion"] == crit) == N_PAIRS


def test_assignment_deterministic_and_order_independent(tmp_path):
    cfg = pw_cfg(tmp_path)
    a = pairwise.enumerate_comparisons(cfg, IIDS)
    b = pairwise.enumerate_comparisons(cfg, list(reversed(IIDS)))
    assert a == b
    # the seeded coin must produce a mix, not a constant assignment
    firsts = [e["instance_a"] for e in a.values()]
    lows = sum(1 for e in a.values() if e["instance_a"] < e["instance_b"])
    assert 0 < lows < len(a), firsts


def test_criteria_share_the_pair_slot_assignment(tmp_path):
    # One coin flip per pair: both criteria show the same response in slot A,
    # so per-criterion results stay comparable pair-by-pair.
    cfg = pw_cfg(tmp_path)
    comps = pairwise.enumerate_comparisons(cfg, IIDS)
    for idx in range(N_PAIRS):
        anl = comps[pairwise.custom_id_for(idx, "ab", ANL)]
        intu = comps[pairwise.custom_id_for(idx, "ab", INT)]
        assert (anl["instance_a"], anl["instance_b"]) == (intu["instance_a"], intu["instance_b"])


def test_both_orders_doubles_with_swapped_twins(tmp_path):
    cfg = pw_cfg(tmp_path, both_orders=True)
    comps = pairwise.enumerate_comparisons(cfg, IIDS)
    assert len(comps) == 2 * N_PAIRS * N_CRITERIA
    for idx in range(N_PAIRS):
        for crit in pairwise.CRITERIA_PAIRWISE:
            ab = comps[pairwise.custom_id_for(idx, "ab", crit)]
            ba = comps[pairwise.custom_id_for(idx, "ba", crit)]
            assert ab["instance_a"] == ba["instance_b"]
            assert ab["instance_b"] == ba["instance_a"]


def test_custom_id_round_trip():
    import re

    for idx in range(N_PAIRS):
        for order in ("ab", "ba"):
            for crit in pairwise.CRITERIA_PAIRWISE:
                cid = pairwise.custom_id_for(idx, order, crit)
                assert pairwise.parse_custom_id(cid) == (idx, order, crit)
                assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid)
    with pytest.raises(ValueError):
        pairwise.parse_custom_id("p00-ab")  # retired single-criterion format
    with pytest.raises(ValueError):
        pairwise.parse_custom_id("gpt-5.5__vs__claude")


# --- prompt construction --------------------------------------------------------------


def test_bundle_byte_stability_across_pairs(tmp_path):
    store = seeded_store(tmp_path, ["mA__s0", "mB__s0", "mC__s0"])
    a1, _ = pairwise.build_bundle(store, "mA__s0", "A", ANL)
    a2, _ = pairwise.build_bundle(store, "mA__s0", "A", ANL)
    assert a1 == a2
    parts_ab = pairwise.build_parts("TASK", a1, "bundle-b", ANL, retry=False)
    parts_ac = pairwise.build_parts("TASK", a1, "bundle-c", ANL, retry=False)
    # every cached prefix part is byte-identical when candidate A is unchanged
    for idx in pairwise.CACHE_PART_IDXS:
        assert parts_ab[idx] == parts_ac[idx], idx
    assert parts_ab[2] != parts_ac[2]


def test_bundle_label_and_check_report(tmp_path):
    store = seeded_store(tmp_path, ["mA__s0"])
    text, _n = pairwise.build_bundle(store, "mA__s0", "B", INT)
    assert text.startswith("=== RESPONSE B ===")
    assert "AUTOMATED CHECK REPORT FOR RESPONSE B" in text


def test_scope_to_row_pure_and_shape_tolerant():
    resp = valid_response()
    anl = pairwise.scope_to_row(resp, "analytics")
    assert len(anl["strategies"]) == 3
    assert all(s["bucket"] == "analytics" for s in anl["strategies"])
    assert len(resp["strategies"]) == 6  # input not mutated
    assert pairwise.scope_to_row(["x"], "analytics") == ["x"]
    assert pairwise.scope_to_row({"strategies": "nope"}, "analytics") == {"strategies": "nope"}
    # malformed strategies belong to neither row
    scoped = pairwise.scope_to_row({"strategies": ["s", {"bucket": "weird"}]}, "analytics")
    assert scoped["strategies"] == []


def test_bundles_are_row_scoped(tmp_path):
    store = seeded_store(tmp_path, ["mA__s0"])
    anl, _ = pairwise.build_bundle(store, "mA__s0", "A", ANL)
    intu, _ = pairwise.build_bundle(store, "mA__s0", "A", INT)
    # the pretty-printed body carries only the criterion's own row (the check
    # report below it may still name all six titles — that is mechanical fact)
    assert '"bucket": "analytics"' in anl and '"bucket": "intuition"' not in anl
    assert '"bucket": "intuition"' in intu and '"bucket": "analytics"' not in intu
    assert "`analytics`-row strategies" in anl
    assert "`intuition`-row strategies" in intu


def test_strip_priors_pure_and_shape_tolerant():
    resp = valid_response()
    stripped = pairwise.strip_priors(resp)
    assert all("prior" not in s for s in stripped["strategies"])
    assert all("prior" in s for s in resp["strategies"])  # input not mutated
    assert stripped["quantity"] == resp["quantity"]
    # malformed shapes pass through untouched
    assert pairwise.strip_priors(["x"]) == ["x"]
    assert pairwise.strip_priors({"strategies": "nope"}) == {"strategies": "nope"}
    assert pairwise.strip_priors({"strategies": ["nope"]}) == {"strategies": ["nope"]}


def test_judge_bundle_has_no_prior_blocks(tmp_path):
    # Priors are out of scope; the bundle withholds them like the prior-mechanics
    # check results, rather than relying on judges to ignore them.
    store = seeded_store(tmp_path, ["mA__s0"])
    for crit in pairwise.CRITERIA_PAIRWISE:
        text, _n = pairwise.build_bundle(store, "mA__s0", "A", crit)
        assert '"prior"' not in text
        assert '"recommendation"' in text and '"rationale"' in text
        assert "prior blocks removed" in text


def test_unparsed_bundle_shown_raw_and_whole(tmp_path):
    store = Store(tmp_path / "artifacts")
    raw = 'not json, but mentions "prior": {"mu": 0.5} and both rows inline'
    store.write_json(
        store.response_path("mR__s0"),
        {
            "instance_id": "mR__s0",
            "model": "mR",
            "parse_mode": "failed",
            "parsed_json": None,
            "raw_text": raw,
        },
    )
    for crit in pairwise.CRITERIA_PAIRWISE:
        text, _n = pairwise.build_bundle(store, "mR__s0", "A", crit)
        assert "did NOT parse" in text
        assert raw in text


def test_retry_part_appended(tmp_path):
    parts = pairwise.build_parts("TASK", "a", "b", INT, retry=True)
    assert len(parts) == 5
    assert "REMINDER" in parts[-1]
    assert INT in parts[-1]  # the retry note names the call's own criterion key


# --- verdict parsing --------------------------------------------------------------------


def _verdict_json(crit=ANL, winner="A"):
    return json.dumps({crit: {"winner": winner, "justification": "x wins"}})


def test_parse_verdicts_valid_direct_and_fenced():
    assert pairwise.parse_verdicts(_verdict_json(), ANL)[ANL]["winner"] == "A"
    fenced = f"```json\n{_verdict_json(INT, winner='B')}\n```"
    assert pairwise.parse_verdicts(fenced, INT)[INT]["winner"] == "B"


def test_parse_verdicts_requires_the_calls_own_criterion():
    # a verdict under the wrong criterion key is not a verdict for this call
    assert pairwise.parse_verdicts(_verdict_json(ANL), INT) is None
    # extra volunteered keys are ignored, not copied through
    both = json.dumps(
        {
            ANL: {"winner": "A", "justification": "x"},
            INT: {"winner": "B", "justification": "y"},
        }
    )
    assert pairwise.parse_verdicts(both, ANL) == {ANL: {"winner": "A", "justification": "x"}}


def test_parse_verdicts_rejects_bad_shapes():
    assert pairwise.parse_verdicts("not json", ANL) is None
    assert pairwise.parse_verdicts(_verdict_json(winner="C"), ANL) is None
    assert pairwise.parse_verdicts(_verdict_json(winner="a"), ANL) is None
    empty_just = json.dumps({ANL: {"winner": "A", "justification": " "}})
    assert pairwise.parse_verdicts(empty_just, ANL) is None
    tie = json.dumps({ANL: {"winner": "tie", "justification": "x"}})
    assert pairwise.parse_verdicts(tie, ANL) is None


# --- batch request builders ----------------------------------------------------------------


def _item(cfg, judge_name="j1", criterion=ANL):
    parts = pairwise.build_parts("TASK", "bundle-a", "bundle-b", criterion, retry=False)
    req = pairwise.build_request(cfg, cfg.judge_model(judge_name), parts, criterion)
    return providers.BatchItem(custom_id=pairwise.custom_id_for(0, "ab", criterion), req=req)


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


def test_per_criterion_system_prompt_and_cache_key(tmp_path):
    cfg = pw_cfg(tmp_path)
    anl_req = _item(cfg, criterion=ANL).req
    int_req = _item(cfg, criterion=INT).req
    assert anl_req.system == JUDGE_PROMPTS[ANL]
    assert int_req.system == JUDGE_PROMPTS[INT]
    # distinct cache keys so the two criteria never share an OpenAI prefix route
    assert anl_req.cache_key != int_req.cache_key
    assert ANL in anl_req.cache_key and INT in int_req.cache_key


def test_first_cache_breakpoint_prefix_clears_anthropic_minimum():
    # Anthropic ignores a cache_control whose prefix is under the model's
    # minimum cacheable length (1024 tokens on large models). The prefix at the
    # first message breakpoint is [system rubric][task prompt]; if either file
    # is ever slimmed to where that sum dips under the minimum, the task
    # breakpoint silently stops caching. ~4 chars/token.
    from pathlib import Path

    task = (Path(__file__).resolve().parent.parent / "footval.prompt.md").read_text()
    for crit, rubric in JUDGE_PROMPTS.items():
        prefix_tokens = (len(rubric) + len(task)) // 4
        assert prefix_tokens >= 1024, (crit, prefix_tokens)


def test_openai_batch_line_shapes(tmp_path):
    cfg = pw_cfg(tmp_path, judges=("m1",))  # m1 is family openai in FAMILIES
    item = _item(cfg, judge_name="m1")
    line = providers.openai_batch_line(item, "/v1/responses")
    assert line["url"] == "/v1/responses"
    assert line["custom_id"] == f"p00-ab-{ANL}"
    assert line["body"]["instructions"] == JUDGE_PROMPTS[ANL]
    # stable cache key so every comparison this judge makes on this criterion
    # shares a cache
    assert line["body"]["prompt_cache_key"] == f"footval-pairwise-{ANL}-m1"
    chat = providers.openai_batch_line(item, "/v1/chat/completions")
    assert chat["body"]["messages"][0]["role"] == "system"
    assert chat["body"]["prompt_cache_key"] == f"footval-pairwise-{ANL}-m1"


def test_prompt_cache_key_not_sent_to_non_openai_chat_providers(tmp_path):
    # z.ai's OpenAI-compatible endpoint would reject the unknown field.
    cfg = pw_cfg(tmp_path, judges=("z1",))
    parts = pairwise.build_parts("TASK", "bundle-a", "bundle-b", ANL, retry=False)
    req = pairwise.build_request(cfg, cfg.judge_model("z1"), parts, ANL)
    assert cfg.model("z1").provider == "zai"
    assert "prompt_cache_key" not in providers._chat_kwargs(req)


def test_zai_vendor_params_ride_extra_body():
    # `thinking` is a z.ai knob, not an OpenAI SDK kwarg: passed top-level the
    # SDK raises TypeError before any HTTP call, so it must move to extra_body.
    mcfg = ModelCfg(
        name="glm",
        provider="zai",
        family="zai",
        params={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    )
    req = providers.LLMRequest(
        model=mcfg,
        system="s",
        parts=("hello",),
        temperature=0.0,
        seed=None,
        max_output_tokens=10,
    )
    kwargs = providers._zai_call_kwargs(req)
    assert "thinking" not in kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    # reasoning_effort IS a real SDK param and stays top-level
    assert kwargs["reasoning_effort"] == "high"


def test_gemini_batch_request_shape(tmp_path):
    cfg = pw_cfg(tmp_path, judges=("m3",))  # m3 -> google/gemini provider
    item = _item(cfg, judge_name="m3", criterion=INT)
    req = providers.gemini_batch_request(item)
    assert req["metadata"] == {"custom_id": f"p00-ab-{INT}"}
    assert req["config"]["system_instruction"] == JUDGE_PROMPTS[INT]
    assert req["contents"][0]["parts"][0]["text"] == "=== TASK GIVEN TO BOTH MODELS ===\nTASK"


# --- manifest & outstanding set ---------------------------------------------------------


def test_manifest_round_trip_and_outstanding(tmp_path):
    cfg = pw_cfg(tmp_path)
    store = seeded_store(tmp_path, ["mA__s0", "mB__s0", "mC__s0"])
    manifest = pairwise.load_or_write_manifest(cfg, store)
    assert manifest["criteria"] == list(pairwise.CRITERIA_PAIRWISE)
    assert len(manifest["comparisons"]) == 3 * N_CRITERIA  # 3 pairs x 2 criteria
    outstanding = pairwise.outstanding_items(store, manifest, ["j1"])
    assert len(outstanding) == 3 * N_CRITERIA
    judge_name, cid = outstanding[0]
    store.write_json(store.pairwise_verdict_path(judge_name, cid), {"verdicts": None})
    assert len(pairwise.outstanding_items(store, manifest, ["j1"])) == 3 * N_CRITERIA - 1
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
    assert len(extended["comparisons"]) == 3 * N_CRITERIA
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
    assert len(manifest["comparisons"]) == 4  # 1 pair x 2 orders x 2 criteria
    for cid, entry in manifest["comparisons"].items():
        record = pairwise._verdict_record(
            cfg,
            manifest,
            "j1",
            cid,
            via="batch",
            verdicts={entry["criterion"]: {"winner": "A", "justification": "x"}},
        )
        store.write_json(store.pairwise_verdict_path("j1", cid), record)

    pairwise.write_csv(cfg)
    import csv as csvmod

    rows = list(csvmod.reader(open(cfg.outputs_data_dir / "pairwise_results.csv")))
    assert rows[0] == ["judge", "model_a", "model_b", "criterion", "winner"]
    assert len(rows) == 1 + 4  # 1 pair x 2 orders x 2 criteria
    assert {r[3] for r in rows[1:]} == set(pairwise.CRITERIA_PAIRWISE)
    # winner "A" maps to whichever model was shown first in THAT ordering
    by_key = {(r[1], r[2], r[3]): r[4] for r in rows[1:]}
    for entry in manifest["comparisons"].values():
        key = (entry["model_a"], entry["model_b"], entry["criterion"])
        assert by_key[key] == entry["model_a"]


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
    assert len(rows) == 2  # header + the one verdict file (the rest are missing)
    assert all(r[4] == "" for r in rows[1:])
