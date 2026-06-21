from footy.metrics import log_loss_1x2, brier_1x2, accuracy_1x2


def _outcome(ga, gb):
    if ga > gb:
        return "home"
    if ga < gb:
        return "away"
    return "draw"


def scoreboard(predictor, played_matches: list) -> dict:
    """Compare the (pre-tournament) model's predictions to actual played results.

    Out-of-sample: the model is NOT retrained on tournament results (anti-leakage).
    """
    if not played_matches:
        return {"n": 0, "accuracy": None, "log_loss": None, "brier": None,
                "goal_mae": None, "matches": []}

    probs, actuals, details, goal_err = [], [], [], 0.0
    for m in played_matches:
        pred = predictor.predict(m["team_a"], m["team_b"], neutral=True)
        p = {"home": pred["team_a_win"] / 100.0, "draw": pred["draw"] / 100.0,
             "away": pred["team_b_win"] / 100.0}
        actual = _outcome(m["goals_a"], m["goals_b"])
        predicted = max(p, key=p.get)
        probs.append(p)
        actuals.append(actual)
        goal_err += abs(pred["expected_goals_a"] - m["goals_a"]) + abs(pred["expected_goals_b"] - m["goals_b"])
        details.append({
            "match": f"{m['team_a']} vs {m['team_b']}",
            "predicted_score": pred["most_likely_score"],
            "actual_score": f"{m['goals_a']}-{m['goals_b']}",
            "predicted_outcome": predicted,
            "actual_outcome": actual,
            "predicted_prob": round(p[predicted], 4),
            "actual_prob": round(p[actual], 4),
            "hit": predicted == actual,
        })

    n = len(played_matches)
    return {"n": n,
            "accuracy": accuracy_1x2(probs, actuals),
            "log_loss": log_loss_1x2(probs, actuals),
            "brier": brier_1x2(probs, actuals),
            "goal_mae": round(goal_err / (2 * n), 3),
            "matches": details}
