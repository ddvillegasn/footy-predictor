def compute_reliability(
    matches_a: int,
    matches_b: int,
    recent_a: int,
    recent_b: int,
    data_age_days: float,
    dispersion: float,
    missing_rankings: int,
    min_matches: int,
) -> float:
    """Honest reliability in [0, 1]. NOT a probability of being correct.

    Combines: sample size of both teams, recent activity, data freshness,
    simulation dispersion, and whether Elo/FIFA signals were available.
    """
    # Sample-size component (saturates around min_matches).
    sample = min(matches_a, matches_b) / float(max(min_matches, 1))
    sample = max(0.0, min(1.0, sample))

    # Recent-activity component (want both teams active).
    recent = min(recent_a, recent_b) / 10.0
    recent = max(0.0, min(1.0, recent))

    # Freshness: 1.0 if recent data, decays to 0 over ~10 years.
    freshness = max(0.0, 1.0 - data_age_days / 3650.0)

    # Stability: lower dispersion -> higher reliability.
    stability = max(0.0, min(1.0, 1.0 / (1.0 + dispersion)))

    # Ranking availability penalty (0, 1 or 2 missing).
    ranking = 1.0 - 0.15 * missing_rankings
    ranking = max(0.0, min(1.0, ranking))

    score = (
        0.35 * sample
        + 0.20 * recent
        + 0.15 * freshness
        + 0.15 * stability
        + 0.15 * ranking
    )
    return round(max(0.0, min(1.0, score)), 3)
