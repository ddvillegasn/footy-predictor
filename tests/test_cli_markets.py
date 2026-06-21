import json

import pandas as pd

from footy import cli
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


def test_parse_book_odds_string():
    parsed = cli.parse_book_odds("1x2.home=1.45,1x2.draw=4.2,over_under.2.5.over=1.67")
    assert parsed == {"1x2": {"home": 1.45, "draw": 4.2},
                      "over_under": {"2.5": {"over": 1.67}}}


def test_cli_markets_flag(capsys):
    code = cli.run(["Brazil", "Haiti", "--neutral", "--markets"], predictor=_predictor())
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and "markets" in payload


def test_cli_book_odds_flag(capsys):
    # Brazil's neutral win prob (~0.86) gives fair odds ~1.16, so a 1.45 book
    # odd is a value bet (positive EV); a 1.05 odd would not be.
    code = cli.run(["Brazil", "Haiti", "--neutral", "--book-odds", "1x2.home=1.45"],
                   predictor=_predictor())
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["value"]["1x2"]["home"]["is_value"] is True
