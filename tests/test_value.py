import pytest

from footy.betting.value import assess_value, assess_market

VCFG = {"threshold": 0.0, "ev_medium": 0.10, "reliability_low": 0.40,
        "reliability_high": 0.70, "kelly_quarter_divisor": 4}


def test_positive_value_detected():
    v = assess_value(model_prob=0.60, book_odds=2.0, reliability=0.8, config=VCFG)
    assert v["ev_per_unit"] == round(0.60 * 2.0 - 1.0, 4)
    assert v["is_value"] is True
    assert v["kelly_fraction_quarter"] == round(v["kelly_fraction_raw"] / 4, 4)


def test_negative_value_is_skip():
    v = assess_value(model_prob=0.40, book_odds=2.0, reliability=0.9, config=VCFG)
    assert v["ev_per_unit"] < 0
    assert v["is_value"] is False
    assert v["stake_recommendation"] == "skip"
    assert v["kelly_fraction_raw"] == 0.0


def test_low_reliability_forces_skip_even_with_edge():
    v = assess_value(model_prob=0.70, book_odds=2.0, reliability=0.3, config=VCFG)
    assert v["ev_per_unit"] > 0
    assert v["stake_recommendation"] == "skip"


def test_high_ev_and_high_reliability_is_medium():
    v = assess_value(model_prob=0.80, book_odds=2.0, reliability=0.9, config=VCFG)
    assert v["stake_recommendation"] == "medium"


def test_invalid_book_odds_raises():
    with pytest.raises(ValueError):
        assess_value(model_prob=0.5, book_odds=1.0, reliability=0.8, config=VCFG)


def test_assess_market_only_priced_outcomes():
    probs = {"home": 0.60, "draw": 0.25, "away": 0.15}
    odds = {"home": 2.0}  # only home priced
    out = assess_market(probs, odds, reliability=0.8, config=VCFG)
    assert set(out.keys()) == {"home"}
    assert out["home"]["is_value"] is True
