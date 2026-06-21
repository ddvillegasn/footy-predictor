from __future__ import annotations

from itertools import combinations


def group_fixtures(teams: list) -> list:
    """All unordered pairs (round-robin), deterministic order."""
    return [(a, b) for a, b in combinations(teams, 2)]


def group_table(teams: list, results: list, points: dict) -> dict:
    """Build a standings dict from match results.

    results: list of (team_a, team_b, goals_a, goals_b).
    Returns {team: {points, w, d, l, gf, ga, gd}}.
    """
    table = {t: {"points": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0}
             for t in teams}
    for a, b, ga, gb in results:
        table[a]["gf"] += ga; table[a]["ga"] += gb
        table[b]["gf"] += gb; table[b]["ga"] += ga
        if ga > gb:
            table[a]["points"] += points["win"]; table[a]["w"] += 1
            table[b]["points"] += points["loss"]; table[b]["l"] += 1
        elif ga < gb:
            table[b]["points"] += points["win"]; table[b]["w"] += 1
            table[a]["points"] += points["loss"]; table[a]["l"] += 1
        else:
            table[a]["points"] += points["draw"]; table[a]["d"] += 1
            table[b]["points"] += points["draw"]; table[b]["d"] += 1
    for t in teams:
        table[t]["gd"] = table[t]["gf"] - table[t]["ga"]
    return table
