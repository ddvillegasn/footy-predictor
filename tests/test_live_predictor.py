import pandas as pd

from footy.predict import Predictor
from footy.ui.service import build_live_predictor

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 5}


def _base_matches():
    rows = []
    for _ in range(12):
        rows.append(("2019-01-01", "Brazil", "Haiti", 1, 1))   # historically even-ish
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    return df


def test_live_refit_shifts_lambda_toward_recent_results():
    base = Predictor.from_matches(_base_matches(), model_config=MODEL_CFG, mc_config=MC_CFG,
                                  canonical=lambda x: x, as_of=pd.Timestamp("2020-01-01"))
    base_a, base_b = base.model.rates("Brazil", "Haiti", neutral=True)

    played = [{"team_a": "Brazil", "team_b": "Haiti", "goals_a": 5, "goals_b": 0}] * 6
    live = build_live_predictor(base, played, tournament_date="2026-06-15",
                                model_config=MODEL_CFG, mc_config=MC_CFG)
    live_a, live_b = live.model.rates("Brazil", "Haiti", neutral=True)
    # Brazil thrashed Haiti recently -> Brazil's rate should rise vs Haiti's.
    assert (live_a - live_b) > (base_a - base_b)
    assert isinstance(live, Predictor)
