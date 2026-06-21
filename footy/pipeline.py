from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from footy.data.loaders import load_results, load_former_names
from footy.data.clean import clean_results
from footy.data.names import NameCanonicalizer
from footy.data.external.elo import attach_elo


def run_etl(results_path, former_names_path, elo_config: dict, data_config: dict,
            artifacts_dir) -> pd.DataFrame:
    """Load -> clean -> canonicalize -> Elo. Writes artifacts; returns enriched frame."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw = load_results(results_path)
    former = load_former_names(former_names_path)
    clean = clean_results(raw)

    canon = NameCanonicalizer(
        former, data_config.get("aliases", {}), data_config.get("sensitive_merges", {})
    )
    df = clean.df.copy()
    df["home_team"] = df["home_team"].map(canon.canonical)
    df["away_team"] = df["away_team"].map(canon.canonical)

    enriched = attach_elo(df, elo_config)
    enriched.to_parquet(artifacts_dir / "enriched_matches.parquet", index=False)

    (artifacts_dir / "etl_report.json").write_text(
        json.dumps(clean.report, indent=2), encoding="utf-8"
    )
    clean.dropped.to_csv(artifacts_dir / "dropped_rows.csv", index=False)
    (artifacts_dir / "team_name_mapping.json").write_text(
        json.dumps(canon.mapping_table(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return enriched
