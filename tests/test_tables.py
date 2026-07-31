import dataclasses

from conftest import make_cfg

from footval import tables
from footval.config import ModelCfg


def test_billable_output_includes_separately_reported_thinking():
    # Gemini reports thinking as thoughts_tokens, billed at the output rate
    assert tables.billable_output_tokens({"output_tokens": 2702, "thoughts_tokens": 5452}) == 8154
    # Anthropic/OpenAI/GLM fold thinking into output_tokens already
    assert tables.billable_output_tokens({"output_tokens": 6610}) == 6610
    assert tables.billable_output_tokens({"output_tokens": 100, "thoughts_tokens": None}) == 100
    assert tables.billable_output_tokens({}) is None
    assert tables.billable_output_tokens(None) is None


def test_output_cost_usd():
    assert tables.output_cost_usd({"output_tokens": 1_000_000}, 25.0) == 25.0
    assert tables.output_cost_usd({"output_tokens": 6610}, 50.0) == 6610 * 50.0 / 1e6
    # no rate or no usage -> no cost, never a silent zero
    assert tables.output_cost_usd({"output_tokens": 6610}, None) is None
    assert tables.output_cost_usd(None, 50.0) is None


def test_candidate_costs_aggregates_and_flags_unknowns(tmp_path):
    cfg = make_cfg(tmp_path, candidates=["m1", "m2"], families={"m1": "openai", "m2": "zai"})
    models = dict(cfg.models)
    models["m1"] = dataclasses.replace(models["m1"], output_usd_per_mtok=30.0)
    # m2 keeps rate=None (e.g. retired model still on disk)
    cfg = dataclasses.replace(cfg, models=models)

    responses = [
        {"model": "m1", "usage": {"output_tokens": 1000}},
        {"model": "m1", "usage": {"output_tokens": 2000, "thoughts_tokens": 500}},
        {"model": "m2", "usage": {"output_tokens": 4000}},
    ]
    costs = {row["model"]: row for row in tables.candidate_costs(cfg, responses)}

    assert costs["m1"]["samples"] == 2
    assert costs["m1"]["billable_output_tokens"] == 3500
    assert costs["m1"]["output_cost_usd"] == round(3500 * 30.0 / 1e6, 6)
    assert costs["m2"]["billable_output_tokens"] == 4000
    assert costs["m2"]["output_usd_per_mtok"] is None
    assert costs["m2"]["output_cost_usd"] is None


def test_live_config_prices_every_candidate():
    # Every model on the live roster must carry an output price, or the cost
    # column silently goes blank on the next run.
    from footval.config import load_config

    cfg = load_config()
    for name in cfg.candidate_models:
        assert cfg.models[name].output_usd_per_mtok is not None, name


def test_model_cfg_defaults_to_no_rate():
    assert ModelCfg(name="x", provider="openai", family="openai").output_usd_per_mtok is None
