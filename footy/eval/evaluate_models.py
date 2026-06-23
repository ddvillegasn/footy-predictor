from footy.metrics import log_loss_1x2, brier_1x2, accuracy_1x2, calibration_buckets


def _outcome(ga, gb):
    if ga > gb:
        return "home"
    if ga < gb:
        return "away"
    return "draw"


def evaluate(predictor, matches: list) -> dict:
    """Evaluate a 1X2 predictor over matches with known results."""
    probs, actuals = [], []
    goal_err, goal_n = 0.0, 0
    hits = 0
    for m in matches:
        neutral = bool(m.get("neutral", True))
        p = predictor.probs(m["team_a"], m["team_b"], neutral)
        actual = _outcome(m["goals_a"], m["goals_b"])
        probs.append(p)
        actuals.append(actual)
        hits += int(max(p, key=p.get) == actual)
        g = predictor.goals(m["team_a"], m["team_b"], neutral)
        if g is not None:
            goal_err += abs(g[0] - m["goals_a"]) + abs(g[1] - m["goals_b"])
            goal_n += 1
    n = len(matches)
    return {
        "n": n,
        "hits": hits,
        "accuracy": accuracy_1x2(probs, actuals),
        "log_loss": round(log_loss_1x2(probs, actuals), 4),
        "brier": round(brier_1x2(probs, actuals), 4),
        "goal_mae": round(goal_err / (2 * goal_n), 3) if goal_n else None,
        "calibration": calibration_buckets(probs, actuals),
    }
