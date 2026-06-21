import pandas as pd

from footy.evaluate import temporal_holdout

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


def _matches():
    # Mixed outcomes so the naive global-frequency baseline is non-degenerate
    # (home/draw/away all present), while the model still has real team signal:
    # Brazil strong, Peru mid, Haiti weak.
    rows = []
    for _ in range(10):
        rows.append(("2018-01-01", "Brazil", "Haiti", 3, 0))   # home
        rows.append(("2018-02-01", "Haiti", "Brazil", 0, 3))   # away
        rows.append(("2018-03-01", "Brazil", "Peru", 2, 1))    # home
        rows.append(("2018-04-01", "Peru", "Brazil", 1, 1))    # draw
        rows.append(("2018-05-01", "Peru", "Haiti", 2, 0))     # home
        rows.append(("2018-05-15", "Haiti", "Peru", 1, 1))     # draw
    # Test period: favourites at home; the team-aware model should beat the baseline.
    for _ in range(4):
        rows.append(("2020-01-01", "Brazil", "Haiti", 3, 0))
        rows.append(("2020-02-01", "Peru", "Haiti", 2, 0))
        rows.append(("2020-03-01", "Brazil", "Peru", 2, 1))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    return df


def test_holdout_reports_model_and_naive():
    report = temporal_holdout(
        _matches(), split_date=pd.Timestamp("2019-06-01"),
        model_config=MODEL_CFG, mc_config=MC_CFG, canonical=lambda x: x,
    )
    for key in ["model", "naive", "beats_baseline", "n_test"]:
        assert key in report
    for metric in ["log_loss", "brier", "accuracy"]:
        assert metric in report["model"]
        assert metric in report["naive"]
    assert report["n_test"] == 12


def test_model_beats_naive_on_separable_data():
    report = temporal_holdout(
        _matches(), split_date=pd.Timestamp("2019-06-01"),
        model_config=MODEL_CFG, mc_config=MC_CFG, canonical=lambda x: x,
    )
    # Non-degenerate naive baseline; the team-aware model should beat it on log loss.
    assert report["model"]["log_loss"] < report["naive"]["log_loss"]
    assert report["beats_baseline"] is True
