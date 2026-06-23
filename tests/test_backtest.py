import pandas as pd

from footy.eval.backtest import backtest_edition, run_backtest

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25, "ridge": 0.01}
ELO_CFG = {"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
           "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4, "Cup": 1.0}}


def _dataset():
    rows = []
    # History (Friendlies) before the cup: Strong beats Weak repeatedly.
    for i in range(20):
        rows.append(("2012-01-01", "Strong", "Weak", 3, 0, "Friendly"))
        rows.append(("2013-01-01", "Mid", "Weak", 2, 0, "Friendly"))
        rows.append(("2013-06-01", "Strong", "Mid", 2, 1, "Friendly"))
    # The 2014 "Cup" edition (to be evaluated).
    rows.append(("2014-06-01", "Strong", "Weak", 1, 0, "Cup"))
    rows.append(("2014-06-02", "Mid", "Weak", 1, 0, "Cup"))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament"])
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = True
    return df


def test_backtest_edition_is_out_of_sample():
    ds = _dataset()
    res = backtest_edition(ds, "Cup", 2014, MODEL_CFG, ELO_CFG)
    assert res["n"] == 2
    assert res["start"] == "2014-06-01"
    # all four predictors evaluated
    assert set(res["models"]) == {"BASE", "Elo", "naive", "random"}
    # BASE/Elo should beat random on this clearly separable data
    assert res["models"]["BASE"]["accuracy"] >= res["models"]["random"]["accuracy"]


def test_run_backtest_aggregates():
    ds = _dataset()
    out = run_backtest(ds, [("Cup", 2014)], MODEL_CFG, ELO_CFG)
    assert "Cup 2014" in out["editions"]
    assert "BASE" in out["aggregate"]
    assert out["aggregate"]["BASE"]["n"] == 2
