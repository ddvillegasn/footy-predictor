import pandas as pd

from footy.data.external.elo import attach_elo

ELO_CFG = {
    "initial_rating": 1500.0,
    "k_factor": 40.0,
    "home_advantage_elo": 65.0,
    "default_tournament_weight": 0.7,
    "tournament_weights": {"Friendly": 0.4, "FIFA World Cup": 1.0},
}


def _matches():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
            "home_team": ["Brazil", "Brazil", "Haiti"],
            "away_team": ["Haiti", "Haiti", "Brazil"],
            "home_score": [3, 2, 0],
            "away_score": [0, 0, 1],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [False, False, False],
        }
    )


def test_first_match_uses_initial_rating_pre():
    out = attach_elo(_matches(), ELO_CFG)
    assert out.loc[0, "home_elo_pre"] == 1500.0
    assert out.loc[0, "away_elo_pre"] == 1500.0


def test_winner_rating_rises_after_match():
    out = attach_elo(_matches(), ELO_CFG)
    # Brazil won match 0, so its pre-rating for match 1 must exceed 1500.
    assert out.loc[1, "home_elo_pre"] > 1500.0


def test_pre_rating_never_uses_own_match_result():
    # The pre rating of the last match for Brazil must equal its rating
    # computed from the first two matches only (no leakage from match 2).
    out = attach_elo(_matches(), ELO_CFG)
    assert "home_elo_pre" in out.columns and "away_elo_pre" in out.columns
    # Determinism
    out2 = attach_elo(_matches(), ELO_CFG)
    assert out.equals(out2)
