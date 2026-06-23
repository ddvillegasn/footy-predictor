import pandas as pd

from footy.data.external.elo import final_ratings

CFG = {"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
       "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4}}


def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2019-02-01"]),
        "home_team": ["Brazil", "Brazil"],
        "away_team": ["Haiti", "Haiti"],
        "home_score": [3, 2],
        "away_score": [0, 0],
        "tournament": ["Friendly", "Friendly"],
        "neutral": [False, False],
    })


def test_final_ratings_winner_above_loser():
    r = final_ratings(_matches(), CFG)
    assert r["Brazil"] > 1500.0 > r["Haiti"]
    # zero-sum around the initial total
    assert abs((r["Brazil"] - 1500.0) + (r["Haiti"] - 1500.0)) < 1e-6
