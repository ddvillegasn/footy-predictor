from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from footy.eval.backtest import run_backtest

DEFAULT_EDITIONS = [("FIFA World Cup", 2014), ("FIFA World Cup", 2018),
                    ("FIFA World Cup", 2022), ("FIFA World Cup", 2026)]


def run_report(dataset_path, editions, model_config, elo_config,
               out_path="artifacts/backtest_report.json") -> dict:
    """Run the historical backtest over `editions` and write a cached JSON report.

    Each World Cup edition is evaluated out-of-sample (trained on rows before it); the
    most recent edition (e.g. WC2026) is the live comparison.
    """
    dataset = pd.read_csv(dataset_path, parse_dates=["date"])
    for col in ("home_score", "away_score"):
        dataset = dataset[dataset[col].notna()]
    dataset["home_score"] = dataset["home_score"].astype(int)
    dataset["away_score"] = dataset["away_score"].astype(int)
    if "neutral" not in dataset:
        dataset["neutral"] = True

    bt = run_backtest(dataset, editions, model_config, elo_config)
    report = {
        "editions": bt["editions"],
        "aggregate": bt["aggregate"],
        "meta": {"editions": list(bt["editions"].keys()),
                 "generado": datetime.now().strftime("%Y-%m-%d %H:%M")},
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
