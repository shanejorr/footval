"""Tests for the pure data-prep of the Bradley-Terry stage. The PyMC fit itself
is not exercised here (slow/stochastic); only the deterministic CSV -> Design
mapping and the self-family indicator are covered."""

from __future__ import annotations

import polars as pl
import pytest

from footval import bradley_terry as bt

# m1/m2 openai, m3/m4 anthropic, judge j1 anthropic.
FAM = {"m1": "openai", "m2": "openai", "m3": "anthropic", "m4": "anthropic", "j1": "anthropic"}


_SCHEMA = {
    "judge": pl.Utf8,
    "model_a": pl.Utf8,
    "model_b": pl.Utf8,
    "criterion": pl.Utf8,
    "winner": pl.Utf8,
}


def _row(a: str, b: str, crit: str, winner: str, judge: str = "j1") -> dict:
    return {"judge": judge, "model_a": a, "model_b": b, "criterion": crit, "winner": winner}


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _row("m1", "m2", "soundness", "m1"),
            _row("m3", "m1", "soundness", "m1"),
            _row("m1", "m3", "soundness", ""),  # blank winner -> dropped
        ],
        schema=_SCHEMA,
    )


# --- self-family signed indicator -----------------------------------------------------


def test_self_sign_four_cases():
    assert bt.self_sign(FAM, "m3", "m1", "j1") == 1.0  # only A shares judge family
    assert bt.self_sign(FAM, "m1", "m3", "j1") == -1.0  # only B shares
    assert bt.self_sign(FAM, "m3", "m4", "j1") == 0.0  # both share -> cancels
    assert bt.self_sign(FAM, "m1", "m2", "j1") == 0.0  # neither shares


# --- outcome encoding & blank dropping ------------------------------------------------


def test_won_a_encoding_and_blank_dropped():
    d = bt.build_design(_df(), "soundness", FAM)
    assert d.n_obs == 2  # the blank-winner row is dropped
    # row order is preserved from the (filtered) frame: m1 beats m2, then m1 beats m3
    pairs = list(zip([d.models[i] for i in d.a_idx], [d.models[i] for i in d.b_idx], strict=True))
    assert pairs == [("m1", "m2"), ("m3", "m1")]
    assert list(d.won_a) == [1, 0]  # slot-A won the first, lost the second


def test_self_sign_array_matches_rows():
    d = bt.build_design(_df(), "soundness", FAM)
    # (m1 vs m2): neither anthropic -> 0; (m3 vs m1): only A anthropic -> +1
    assert list(d.self_sign) == [0.0, 1.0]


# --- tracks ---------------------------------------------------------------------------


def test_soundness_is_the_only_track():
    # Judging is a single criterion, so there is no per-criterion facet and no
    # pooled "overall" track to fit.
    assert bt.TRACKS == ("soundness",)


def test_track_filters_by_criterion():
    df = pl.concat([_df(), pl.DataFrame([_row("m2", "m3", "other", "m3")], schema=_SCHEMA)])
    assert bt.build_design(df, "soundness", FAM).n_obs == 2
    assert bt.build_design(df, "other", FAM).n_obs == 1


# --- index maps cover all CSV entities ------------------------------------------------


def test_index_maps_cover_entities():
    d = bt.build_design(_df(), "soundness", FAM)
    assert d.models == ["m1", "m2", "m3"]  # sorted union of model_a/model_b (m4 absent here)
    assert d.judges == ["j1"]
    assert set(d.a_idx) | set(d.b_idx) <= set(range(len(d.models)))
    assert set(d.j_idx) <= set(range(len(d.judges)))


# --- validation -----------------------------------------------------------------------


def test_winner_must_be_a_or_b():
    bad = pl.DataFrame([_row("m1", "m2", "soundness", "m3")], schema=_SCHEMA)
    with pytest.raises(ValueError):
        bt.build_design(bad, "soundness", FAM)


def test_read_results_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        bt.read_results(tmp_path / "nope.csv")
