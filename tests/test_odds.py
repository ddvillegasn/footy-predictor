import pytest

from footy.betting.odds import (
    fair_odds, implied_prob, model_margin, market_margin,
    decorate_outcome, decorate_group,
)


def test_fair_odds_basic():
    assert fair_odds(0.5) == 2.0
    assert fair_odds(0.8) == 1.25
    assert fair_odds(1.0) == 1.0


def test_fair_odds_zero_is_none():
    assert fair_odds(0.0) is None


def test_implied_prob():
    assert implied_prob(2.0) == 0.5
    assert round(implied_prob(1.25), 4) == 0.8


def test_model_margin_from_raw_probs_is_zero():
    assert abs(model_margin([0.86, 0.09, 0.05])) < 1e-9


def test_market_margin_from_book_odds_positive():
    # Typical vig: implied probs sum above 1.
    assert market_margin([1.45, 4.2, 7.0]) > 0.0


def test_decorate_outcome_no_sim_status():
    d = decorate_outcome(0.0)
    assert d["fair_odds"] is None and d["status"] == "no_sim"


def test_decorate_group_adds_margin():
    g = decorate_group({"home": 0.86, "draw": 0.09, "away": 0.05})
    assert g["home"]["fair_odds"] == round(1 / 0.86, 2)
    assert abs(g["margin"]) < 1e-9
