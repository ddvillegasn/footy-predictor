from footy.eval.predictors import build_predictors
from footy.eval.evaluate_models import evaluate


def _edition(dataset, tournament, year):
    sub = dataset[(dataset["tournament"] == tournament) & (dataset["date"].dt.year == year)]
    return sub.sort_values("date")


def backtest_edition(dataset, tournament, year, model_config, elo_config) -> dict:
    """Train predictors on rows strictly before the edition; evaluate on the edition."""
    edition = _edition(dataset, tournament, year)
    if len(edition) == 0:
        return {"n": 0, "start": None, "models": {}}
    start = edition["date"].min()
    train = dataset[dataset["date"] < start]
    preds = build_predictors(train, model_config, elo_config, as_of=start)
    matches = [{"team_a": r.home_team, "team_b": r.away_team, "neutral": bool(r.neutral),
                "goals_a": int(r.home_score), "goals_b": int(r.away_score)}
               for r in edition.itertuples(index=False)]
    return {"n": len(matches), "start": str(start.date()),
            "models": {name: evaluate(p, matches) for name, p in preds.items()}}


def _aggregate(editions: dict) -> dict:
    agg: dict = {}
    for ed in editions.values():
        for name, m in ed.get("models", {}).items():
            a = agg.setdefault(name, {"n": 0, "hits": 0, "_ll": 0.0, "_br": 0.0})
            a["n"] += m["n"]
            a["hits"] += m["hits"]
            a["_ll"] += m["log_loss"] * m["n"]
            a["_br"] += m["brier"] * m["n"]
    out = {}
    for name, a in agg.items():
        n = a["n"] or 1
        out[name] = {"n": a["n"], "hits": a["hits"],
                     "accuracy": round(a["hits"] / n, 4),
                     "log_loss": round(a["_ll"] / n, 4),
                     "brier": round(a["_br"] / n, 4)}
    return out


def run_backtest(dataset, editions, model_config, elo_config) -> dict:
    """editions = list of (tournament, year). Returns per-edition + aggregate metrics."""
    per_edition = {}
    for tournament, year in editions:
        per_edition[f"{tournament} {year}"] = backtest_edition(
            dataset, tournament, year, model_config, elo_config)
    return {"editions": per_edition, "aggregate": _aggregate(per_edition)}
