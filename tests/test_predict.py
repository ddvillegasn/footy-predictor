import pandas as pd
import pytest

from footy.predict import Predictor

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


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
        _matches(),
        model_config=MODEL_CFG,
        mc_config=MC_CFG,
        canonical=lambda x: x,
        as_of=pd.Timestamp("2020-01-01"),
    )


def test_predict_returns_full_dict():
    out = _predictor().predict("Brazil", "Haiti", neutral=True)
    for key in [
        "team_a_win", "draw", "team_b_win", "expected_goals_a", "expected_goals_b",
        "most_likely_score", "score_distribution", "confidence_interval",
        "prediction_reliability", "model_version",
    ]:
        assert key in out
    assert out["model_version"] == "baseline-v1.0.0"
    assert out["team_a_win"] > out["team_b_win"]


def test_neutral_zeroes_home_advantage():
    pred = _predictor()
    home = pred.predict("Brazil", "Haiti", neutral=False)
    neutral = pred.predict("Brazil", "Haiti", neutral=True)
    assert home["expected_goals_a"] >= neutral["expected_goals_a"]


def test_unknown_team_raises_with_suggestion():
    with pytest.raises(ValueError, match="Unknown team"):
        _predictor().predict("Brazil", "Atlantis", neutral=True)
