"""End-to-end smoke test against the real bundled dataset."""
from pathlib import Path

import pandas as pd
import pytest

from footy.data.loaders import load_results, load_former_names
from footy.data.clean import clean_results
from footy.data.names import NameCanonicalizer
from footy.predict import Predictor

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "international_results" / "results.csv"

MODEL_CFG = {"xi": 0.0018, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 30000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


@pytest.mark.skipif(not RESULTS.exists(), reason="real dataset not present")
def test_brazil_beats_haiti_on_real_data():
    raw = load_results(RESULTS)
    former = load_former_names(ROOT / "international_results" / "former_names.csv")
    clean = clean_results(raw)
    canon = NameCanonicalizer(former, {}, {"enabled": False, "mappings": {}})
    as_of = clean.df["date"].max() + pd.Timedelta(days=1)
    predictor = Predictor.from_matches(
        clean.df, model_config=MODEL_CFG, mc_config=MC_CFG,
        canonical=canon.canonical, as_of=as_of,
    )
    out = predictor.predict("Brazil", "Haiti", neutral=True)
    assert out["team_a_win"] > out["team_b_win"]
    assert 99.0 <= out["team_a_win"] + out["draw"] + out["team_b_win"] <= 101.0
