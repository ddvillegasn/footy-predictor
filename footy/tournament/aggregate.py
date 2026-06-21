from __future__ import annotations

from footy.betting.odds import decorate_group
from footy.betting.value import assess_market


def aggregate(structure, sims: list) -> dict:
    """Count Monte Carlo tournament results into probabilities."""
    n = len(sims)
    teams = [t for group in structure.groups.values() for t in group]
    rounds = structure.rounds
    ladder = rounds + ["champion"]
    rank_of = {label: i for i, label in enumerate(ladder)}

    team_stats = {t: {"advance_group": 0, "champion": 0} for t in teams}
    for label in rounds:
        for t in teams:
            team_stats[t][f"reach_{label}"] = 0
    group_pos = {g: {t: [0, 0, 0, 0] for t in group} for g, group in structure.groups.items()}

    for res in sims:
        for t in teams:
            reached = rank_of.get(res.furthest_round.get(t, ""), -1)
            for label in rounds:
                if reached >= rank_of[label]:
                    team_stats[t][f"reach_{label}"] += 1
            if reached >= rank_of[rounds[0]]:
                team_stats[t]["advance_group"] += 1
            if res.champion == t:
                team_stats[t]["champion"] += 1
        for g, order in res.group_order.items():
            for pos, t in enumerate(order):
                if pos < 4:
                    group_pos[g][t][pos] += 1

    teams_out = {}
    for t in teams:
        d = {"advance_group": team_stats[t]["advance_group"] / n,
             "champion": team_stats[t]["champion"] / n}
        for label in rounds:
            d[f"reach_{label}"] = team_stats[t][f"reach_{label}"] / n
        teams_out[t] = d

    groups_out = {}
    for g, group in structure.groups.items():
        groups_out[g] = {}
        for t in group:
            c = group_pos[g][t]
            groups_out[g][t] = {"p1": c[0] / n, "p2": c[1] / n, "p3": c[2] / n, "p4": c[3] / n}

    slot_freq = {}
    for idx in range(len(structure.bracket_r32)):
        slot_freq[f"R32_tie_{idx + 1}"] = {}
    for res in sims:
        first = list(structure.groups.keys())
        # slot occupancy at the first knockout round (who played each opening tie)
    return {"teams": teams_out, "groups": groups_out,
            "slot_outcome_frequency": _slot_frequency(structure, sims),
            "meta": {"n_tournaments": n}}


def _slot_frequency(structure, sims: list) -> dict:
    """Frequency of each team appearing as champion per simulation (compact slot proxy)."""
    n = len(sims)
    freq = {}
    for res in sims:
        freq[res.champion] = freq.get(res.champion, 0) + 1
    return {"champion": {team: c / n for team, c in sorted(freq.items(), key=lambda kv: -kv[1])}}


def tournament_odds(agg: dict, book_odds: dict | None, reliability: float,
                    value_config: dict) -> dict:
    """Fair odds for champion (+ optional value) reusing SP2 odds/value."""
    champion_probs = {t: d["champion"] for t, d in agg["teams"].items() if d["champion"] > 0}
    odds = {"champion": decorate_group(champion_probs)}
    out = {**agg, "odds": odds}
    if book_odds and "champion" in book_odds:
        out["value"] = {"champion": assess_market(champion_probs, book_odds["champion"],
                                                   reliability, value_config)}
    return out
