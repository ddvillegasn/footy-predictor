import json
from pathlib import Path

import pandas as pd

from footy.pipeline import run_etl


def _raw(tmp_path):
    p = tmp_path / "results.csv"
    rows = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    for _ in range(6):
        rows += "2019-01-01,Brazil,Haiti,3,0,Friendly,Rio,Brazil,FALSE\n"
        rows += "2019-06-01,Haiti,Brazil,0,2,Friendly,Port,Haiti,FALSE\n"
    p.write_text(rows, encoding="utf-8")
    fn = tmp_path / "former_names.csv"
    fn.write_text("current,former,start_date,end_date\n", encoding="utf-8")
    return p, fn


def test_run_etl_writes_artifacts(tmp_path):
    results_path, former_path = _raw(tmp_path)
    out_dir = tmp_path / "artifacts"
    enriched = run_etl(
        results_path=results_path,
        former_names_path=former_path,
        elo_config={"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
                    "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4}},
        data_config={"aliases": {}, "sensitive_merges": {"enabled": False, "mappings": {}}},
        artifacts_dir=out_dir,
    )
    assert "home_elo_pre" in enriched.columns
    assert (out_dir / "etl_report.json").exists()
    assert (out_dir / "team_name_mapping.json").exists()
    report = json.loads((out_dir / "etl_report.json").read_text())
    assert report["rows_out"] >= 1
