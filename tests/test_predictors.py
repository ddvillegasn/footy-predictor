import pandas as pd

from footy.eval.predictors import (DixonColesPredictor, EloFavoritePredictor,
                                    NaiveGlobalPredictor, RandomPredictor, build_predictors)
from footy.models.poisson import fit_dixon_coles
from footy.models.montecarlo import simulate

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25, "ridge": 0.01}
MC_CFG = {"n_sims": 40000, "seed": 1, "max_goals": 10, "ci_level": 0.90, "top_scores": 5}
ELO_CFG = {"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
           "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4}}
FALLBACK = {"home": 0.45, "draw": 0.27, "away": 0.28}


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


def test_random_is_uniform():
    p = RandomPredictor().probs("A", "B", True)
    assert p == {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    assert RandomPredictor().goals("A", "B", True) is None


def test_naive_returns_fixed_rates():
    p = NaiveGlobalPredictor(FALLBACK).probs("A", "B", True)
    assert p == FALLBACK


def test_dixon_coles_probs_sum_to_one_and_match_sim():
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    pred = DixonColesPredictor(model, FALLBACK, max_goals=10)
    p = pred.probs("Brazil", "Haiti", neutral=True)
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
    sim = simulate(*model.rates("Brazil", "Haiti", neutral=True), MC_CFG)
    assert abs(p["home"] - sim["team_a_win"] / 100.0) < 0.04   # analytic ≈ MC
    assert pred.goals("Brazil", "Haiti", neutral=True) is not None


def test_dixon_coles_unknown_team_falls_back():
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    pred = DixonColesPredictor(model, FALLBACK, max_goals=10)
    assert pred.probs("Brazil", "Atlantis", neutral=True) == FALLBACK


def test_elo_favorite_gives_more_to_higher_rating():
    pred = EloFavoritePredictor({"Strong": 1800.0, "Weak": 1400.0}, 0.25, 65.0, FALLBACK)
    p = pred.probs("Strong", "Weak", neutral=True)
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
    assert p["home"] > p["away"] and p["draw"] == 0.25


def test_build_predictors_has_standard_set():
    preds = build_predictors(_matches(), MODEL_CFG, ELO_CFG, as_of=pd.Timestamp("2020-01-01"))
    assert set(preds) == {"BASE", "Elo", "naive", "random"}
    for p in preds.values():
        probs = p.probs("Brazil", "Haiti", neutral=True)
        assert abs(sum(probs.values()) - 1.0) < 1e-9
