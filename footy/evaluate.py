from __future__ import annotations

import pandas as pd

from footy.predict import Predictor
from footy.metrics import log_loss_1x2, brier_1x2, accuracy_1x2, naive_baseline_probs


def _outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "draw"


def temporal_holdout(matches: pd.DataFrame, split_date, model_config: dict,
                     mc_config: dict, canonical) -> dict:
    """Train on rows before split_date, evaluate on rows on/after it.

    Never trains on the future. Compares the model's 1X2 probabilities against
    the naive global-frequency baseline derived from the training period only.
    """
    split_date = pd.Timestamp(split_date)
    train = matches[matches["date"] < split_date]
    test = matches[matches["date"] >= split_date]

    predictor = Predictor.from_matches(
        train, model_config=model_config, mc_config=mc_config,
        canonical=canonical, as_of=split_date,
    )

    train_outcomes = [
        _outcome(int(r.home_score), int(r.away_score))
        for r in train.itertuples(index=False)
    ]
    naive = naive_baseline_probs(train_outcomes)

    model_probs: list[dict] = []
    naive_probs: list[dict] = []
    actuals: list[str] = []
    for row in test.itertuples(index=False):
        try:
            pred = predictor.predict(
                row.home_team, row.away_team, neutral=bool(row.neutral)
            )
        except ValueError:
            # Team unseen in the training period; skip (cannot score fairly).
            continue
        model_probs.append({
            "home": pred["team_a_win"] / 100.0,
            "draw": pred["draw"] / 100.0,
            "away": pred["team_b_win"] / 100.0,
        })
        naive_probs.append(dict(naive))
        actuals.append(_outcome(int(row.home_score), int(row.away_score)))

    model_metrics = {
        "log_loss": log_loss_1x2(model_probs, actuals),
        "brier": brier_1x2(model_probs, actuals),
        "accuracy": accuracy_1x2(model_probs, actuals),
    }
    naive_metrics = {
        "log_loss": log_loss_1x2(naive_probs, actuals),
        "brier": brier_1x2(naive_probs, actuals),
        "accuracy": accuracy_1x2(naive_probs, actuals),
    }
    return {
        "model": model_metrics,
        "naive": naive_metrics,
        "beats_baseline": bool(model_metrics["log_loss"] < naive_metrics["log_loss"]),
        "n_test": len(actuals),
        "split_date": str(split_date.date()),
    }
