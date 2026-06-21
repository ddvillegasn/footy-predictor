import numpy as np


def markets_from_samples(goals_a, goals_b, config: dict) -> dict:
    """Compute raw market probabilities by counting events over goal samples.

    Pure counting: no model recomputation. Returns nested dict of probabilities
    (floats rounded to 4 decimals). Handicap covers Asian .5 lines only.
    """
    goals_a = np.asarray(goals_a)
    goals_b = np.asarray(goals_b)
    if len(goals_a) != len(goals_b) or len(goals_a) == 0:
        raise ValueError("goal arrays must be non-empty and equal length")
    n = len(goals_a)
    total = goals_a + goals_b

    def p(mask) -> float:
        return round(float(np.mean(mask)), 4)

    home = p(goals_a > goals_b)
    draw = p(goals_a == goals_b)
    away = p(goals_a < goals_b)

    over_under = {}
    for line in config["over_under_lines"]:
        over_under[str(line)] = {"over": p(total > line), "under": p(total < line)}

    handicap = {}
    for h in config["handicap_lines"]:
        a_eff = goals_a + h
        handicap[str(h)] = {"home": p(a_eff > goals_b), "away": p(a_eff < goals_b)}

    # Correct score: full counts -> top-N + remaining mass.
    pair_counts = {}
    for ga, gb in zip(goals_a.tolist(), goals_b.tolist()):
        key = f"{ga}-{gb}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    ordered = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_n = int(config["top_scores"])
    top = {k: round(v / n, 4) for k, v in ordered[:top_n]}
    all_mass = round(sum(v / n for v in pair_counts.values()), 4)
    other = round(1.0 - sum(top.values()), 4)

    return {
        "1x2": {"home": home, "draw": draw, "away": away},
        "double_chance": {
            "1X": p(goals_a >= goals_b),
            "12": p(goals_a != goals_b),
            "X2": p(goals_a <= goals_b),
        },
        "over_under": over_under,
        "btts": {"yes": p((goals_a > 0) & (goals_b > 0)), "no": p(~((goals_a > 0) & (goals_b > 0)))},
        "correct_score": {"top": top, "other_probability": other, "all_mass_check": all_mass},
        "handicap": handicap,
    }
