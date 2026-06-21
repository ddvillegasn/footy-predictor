import numpy as np


def simulate_goals(lambda_a: float, lambda_b: float, config: dict):
    """Seeded independent-Poisson goal samples for both teams.

    Returns (goals_a, goals_b, meta). meta records seed/n_sims/lambdas plus
    dc_enabled (False: sampling is plain Poisson; Dixon-Coles tau lives in the
    fit, not the sampler) and clip_max. Raises ValueError if a lambda <= 0.
    """
    if lambda_a <= 0 or lambda_b <= 0:
        raise ValueError(f"lambdas must be > 0, got {lambda_a}, {lambda_b}")
    n = int(config["n_sims"])
    seed = int(config["seed"])
    max_goals = int(config["max_goals"])

    rng = np.random.default_rng(seed)
    goals_a = np.clip(rng.poisson(lambda_a, n), 0, max_goals)
    goals_b = np.clip(rng.poisson(lambda_b, n), 0, max_goals)

    meta = {
        "seed": seed,
        "n_sims": n,
        "lambda_a": round(float(lambda_a), 4),
        "lambda_b": round(float(lambda_b), 4),
        "dc_enabled": False,
        "clip_max": max_goals,
    }
    return goals_a, goals_b, meta


def aggregate_outcomes(goals_a, goals_b, lambda_a: float, lambda_b: float, config: dict) -> dict:
    """1X2, score distribution, expected goals (= lambdas), and goal CI from samples."""
    n = len(goals_a)
    ci_level = float(config["ci_level"])
    top_scores = int(config.get("top_scores", 8))

    a_win = float(np.mean(goals_a > goals_b)) * 100.0
    draw = float(np.mean(goals_a == goals_b)) * 100.0
    b_win = float(np.mean(goals_a < goals_b)) * 100.0

    pair_counts = {}
    for ga, gb in zip(goals_a.tolist(), goals_b.tolist()):
        key = f"{ga}-{gb}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    ordered = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    most_likely = ordered[0][0]
    distribution = {k: round(v / n * 100.0, 2) for k, v in ordered[:top_scores]}

    lo = (1.0 - ci_level) / 2.0 * 100.0
    hi = (1.0 + ci_level) / 2.0 * 100.0
    ci = {
        "goals_a": [int(np.percentile(goals_a, lo)), int(np.percentile(goals_a, hi))],
        "goals_b": [int(np.percentile(goals_b, lo)), int(np.percentile(goals_b, hi))],
        "level": ci_level,
    }

    return {
        "team_a_win": round(a_win, 2),
        "draw": round(draw, 2),
        "team_b_win": round(b_win, 2),
        "expected_goals_a": round(float(lambda_a), 2),
        "expected_goals_b": round(float(lambda_b), 2),
        "most_likely_score": most_likely,
        "score_distribution": distribution,
        "confidence_interval": ci,
        "lambda_dispersion": round(float(np.std(goals_a) + np.std(goals_b)), 4),
    }


def simulate(lambda_a: float, lambda_b: float, config: dict) -> dict:
    """Backward-compatible: same output as before, now via the two helpers."""
    goals_a, goals_b, _ = simulate_goals(lambda_a, lambda_b, config)
    return aggregate_outcomes(goals_a, goals_b, lambda_a, lambda_b, config)
