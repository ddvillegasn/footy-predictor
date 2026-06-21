from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from footy.config import load_config
from footy.data.loaders import load_results, load_former_names
from footy.data.clean import clean_results
from footy.data.names import NameCanonicalizer
from footy.predict import Predictor


def _build_default_predictor() -> Predictor:
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    mc_cfg = load_config("montecarlo")

    raw_dir = data_cfg["raw_dir"]
    results = load_results(f"{raw_dir}/{data_cfg['files']['results']}")
    former = load_former_names(f"{raw_dir}/{data_cfg['files']['former_names']}")
    clean = clean_results(results)

    canon = NameCanonicalizer(
        former, data_cfg.get("aliases", {}), data_cfg.get("sensitive_merges", {})
    )
    as_of = clean.df["date"].max() + pd.Timedelta(days=1)
    return Predictor.from_matches(
        clean.df, model_config=model_cfg, mc_config=mc_cfg,
        canonical=canon.canonical, as_of=as_of,
    )


def run(argv: list[str], predictor: Predictor | None = None) -> int:
    parser = argparse.ArgumentParser(prog="predict", description="Predict a national-team match.")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--neutral", action="store_true", help="Neutral venue (home_advantage=0)")
    parser.add_argument("--tournament", default="Friendly")
    args = parser.parse_args(argv)

    if predictor is None:
        predictor = _build_default_predictor()

    try:
        result = predictor.predict(
            args.team_a, args.team_b, neutral=args.neutral, tournament=args.tournament
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))
