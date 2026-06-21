import math

OUTCOMES = ("home", "draw", "away")


def log_loss_1x2(probs: list[dict], actuals: list[str]) -> float:
    """Mean negative log-likelihood of the realised 1X2 outcome."""
    eps = 1e-15
    total = 0.0
    for p, actual in zip(probs, actuals):
        prob = min(1.0, max(eps, p[actual]))
        total += -math.log(prob)
    return total / len(actuals)


def brier_1x2(probs: list[dict], actuals: list[str]) -> float:
    """Mean multiclass Brier score over the 1X2 vector."""
    total = 0.0
    for p, actual in zip(probs, actuals):
        for outcome in OUTCOMES:
            target = 1.0 if outcome == actual else 0.0
            total += (p[outcome] - target) ** 2
    return total / len(actuals)


def naive_baseline_probs(outcomes: list[str]) -> dict:
    """Global historical 1X2 frequencies — the dumb baseline to beat."""
    n = len(outcomes)
    return {o: outcomes.count(o) / n for o in OUTCOMES}


def accuracy_1x2(probs: list[dict], actuals: list[str]) -> float:
    """Share of matches where the highest-probability outcome was realised."""
    correct = 0
    for p, actual in zip(probs, actuals):
        predicted = max(OUTCOMES, key=lambda o: p[o])
        correct += int(predicted == actual)
    return correct / len(actuals)
