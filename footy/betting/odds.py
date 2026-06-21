def fair_odds(prob: float):
    """Fair decimal odds = 1/prob. prob==0 -> None (no simulated occurrences)."""
    if prob <= 0:
        return None
    return round(1.0 / prob, 2)


def implied_prob(decimal_odds: float) -> float:
    """Implied probability of a decimal odd = 1/odds."""
    return 1.0 / decimal_odds


def model_margin(probs) -> float:
    """Overround of the model's fair odds, from RAW probabilities: sum(probs)-1.

    Uses raw (unrounded) probabilities so rounding never produces a false margin.
    """
    return sum(probs) - 1.0


def market_margin(book_odds) -> float:
    """Overround of real bookmaker odds: sum(1/odd) - 1 (the vig)."""
    return sum(1.0 / o for o in book_odds) - 1.0


def decorate_outcome(prob: float) -> dict:
    """One outcome -> {prob, fair_odds[, status]}."""
    odds = fair_odds(prob)
    out = {"prob": round(prob, 4), "fair_odds": odds}
    if odds is None:
        out["status"] = "no_sim"
    return out


def decorate_group(prob_dict: dict) -> dict:
    """Decorate every outcome in a group and attach the model margin (raw probs)."""
    result = {k: decorate_outcome(v) for k, v in prob_dict.items()}
    result["margin"] = round(model_margin(list(prob_dict.values())), 4)
    return result
