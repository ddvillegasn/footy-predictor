import json

import pandas as pd

from footy.eval.report import run_report

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25, "ridge": 0.01}
ELO_CFG = {"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
           "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4, "Cup": 1.0}}


def _dataset(tmp_path):
    rows = []
    for _ in range(20):
        rows.append(("2012-01-01", "Strong", "Weak", 3, 0, "Friendly"))
        rows.append(("2013-06-01", "Strong", "Mid", 2, 1, "Friendly"))
        rows.append(("2013-01-01", "Mid", "Weak", 2, 0, "Friendly"))
    rows.append(("2014-06-01", "Strong", "Weak", 1, 0, "Cup"))
    rows.append(("2014-06-02", "Mid", "Weak", 1, 0, "Cup"))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament"])
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = True
    p = tmp_path / "results.csv"
    df.to_csv(p, index=False)
    return p


def test_run_report_writes_json(tmp_path):
    ds = _dataset(tmp_path)
    out_path = tmp_path / "backtest_report.json"
    report = run_report(dataset_path=ds, editions=[("Cup", 2014)],
                        model_config=MODEL_CFG, elo_config=ELO_CFG, out_path=out_path)
    assert "editions" in report and "aggregate" in report and "meta" in report
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["aggregate"]["BASE"]["n"] == 2
    assert saved["meta"]["editions"] == ["Cup 2014"]
