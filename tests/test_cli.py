import json

import pandas as pd

from footy import cli


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


def test_cli_render_outputs_json(capsys, monkeypatch):
    from footy.predict import Predictor
    MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
                 "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
    MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}
    predictor = Predictor.from_matches(
        _matches(), model_config=MODEL_CFG, mc_config=MC_CFG,
        canonical=lambda x: x, as_of=pd.Timestamp("2020-01-01"),
    )
    code = cli.run(["Brazil", "Haiti", "--neutral"], predictor=predictor)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["team_a_win"] > payload["team_b_win"]
