import pandas as pd

from footy.predict import Predictor

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 5}
BET_CFG = {"over_under_lines": [0.5, 1.5, 2.5, 3.5, 4.5],
           "handicap_lines": [-1.5, -0.5, 0.5, 1.5], "top_scores": 5,
           "value": {"threshold": 0.0, "ev_medium": 0.10, "reliability_low": 0.40,
                     "reliability_high": 0.70, "kelly_quarter_divisor": 4}}


def _matches():
    rows = []
    for _ in range(12):
        rows.append(("2019-01-01", "Brazil", "Haiti", 3, 0))
        rows.append(("2019-06-01", "Haiti", "Brazil", 0, 2))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    return df


def _predictor():
    return Predictor.from_matches(
        _matches(), model_config=MODEL_CFG, mc_config=MC_CFG, canonical=lambda x: x,
        as_of=pd.Timestamp("2020-01-01"), betting_config=BET_CFG,
        betting_config_version="testhash",
    )


def test_no_flags_matches_sp1_shape():
    out = _predictor().predict("Brazil", "Haiti", neutral=True)
    assert "markets" not in out and "value" not in out
    assert "simulation_meta" not in out


def test_include_markets_adds_markets_and_meta():
    out = _predictor().predict("Brazil", "Haiti", neutral=True, include_markets=True)
    assert out["betting_version"] == "sp2-v1.0.0"
    assert out["simulation_meta"]["betting_config_version"] == "testhash"
    assert out["simulation_meta"]["dc_enabled"] is False
    m = out["markets"]
    assert set(m) == {"1x2", "double_chance", "over_under", "btts", "correct_score", "handicap"}
    assert m["1x2"]["home"]["fair_odds"] is not None
    assert "value" not in out


def test_book_odds_adds_value_only_for_priced():
    book = {"1x2": {"home": 1.05}}
    out = _predictor().predict("Brazil", "Haiti", neutral=True, book_odds=book)
    assert set(out["value"].keys()) == {"1x2"}
    assert set(out["value"]["1x2"].keys()) == {"home"}
    assert "markets" in out  # book_odds implies markets


def test_book_odds_over_under_nested():
    book = {"over_under": {"2.5": {"over": 1.5}}}
    out = _predictor().predict("Brazil", "Haiti", neutral=True, book_odds=book)
    assert "over" in out["value"]["over_under"]["2.5"]
