from footy.betting.odds import implied_prob, fair_odds


def _stake_recommendation(ev: float, reliability: float, config: dict) -> str:
    if ev <= 0 or reliability < config["reliability_low"]:
        return "skip"
    if ev > config["ev_medium"] and reliability >= config["reliability_high"]:
        return "medium"
    return "small"


def assess_value(model_prob: float, book_odds: float, reliability: float, config: dict) -> dict:
    """Compare a model probability against a bookmaker decimal odd.

    EV assumes the model probability is correct: it is value relative to the
    model, not guaranteed profit. Reliability gates the stake recommendation.
    """
    if book_odds <= 1.0:
        raise ValueError(f"book_odds must be > 1.0, got {book_odds}")

    book_imp = implied_prob(book_odds)
    ev = model_prob * book_odds - 1.0
    kelly_raw = (model_prob * book_odds - 1.0) / (book_odds - 1.0)
    if kelly_raw < 0:
        kelly_raw = 0.0
    divisor = config.get("kelly_quarter_divisor", 4)

    return {
        "model_prob": round(model_prob, 4),
        "fair_odds": fair_odds(model_prob),
        "book_odds": round(book_odds, 2),
        "book_implied": round(book_imp, 4),
        "edge_pct": round(ev * 100.0, 2),
        "ev_per_unit": round(ev, 4),
        "kelly_fraction_raw": round(kelly_raw, 4),
        "kelly_fraction_quarter": round(kelly_raw / divisor, 4),
        "is_value": bool(ev > config.get("threshold", 0.0)),
        "stake_recommendation": _stake_recommendation(ev, reliability, config),
    }


def assess_market(prob_dict: dict, book_odds_dict: dict, reliability: float, config: dict) -> dict:
    """Assess value only for outcomes that have a supplied bookmaker odd."""
    out = {}
    for outcome, odd in book_odds_dict.items():
        if outcome not in prob_dict:
            continue  # odd for a nonexistent outcome -> ignored
        out[outcome] = assess_value(prob_dict[outcome], odd, reliability, config)
    return out
