from __future__ import annotations

from itertools import combinations, groupby


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


def _global_key(stats: dict) -> tuple:
    return (stats["points"], stats["gd"], stats["gf"])


def _h2h_table(tied: list, results: list, points: dict) -> dict:
    """Mini-table using only matches among the tied teams."""
    sub = [(a, b, ga, gb) for (a, b, ga, gb) in results if a in tied and b in tied]
    return group_table(tied, sub, points)


def _break_ties(tied: list, results: list, points: dict, rng) -> list:
    h2h = _h2h_table(tied, results, points)
    ordered = sorted(tied, key=lambda t: _global_key(h2h[t]), reverse=True)
    out = []
    for _, grp in groupby(ordered, key=lambda t: _global_key(h2h[t])):
        still = list(grp)
        if len(still) == 1:
            out.extend(still)
        else:
            # fair_play is neutral (no data) -> final fallback: reproducible lots.
            lots = sorted(still, key=lambda t: rng.random())
            out.extend(lots)
    return out


def rank_group(teams: list, results: list, points: dict, tiebreakers: list, rng) -> list:
    """Order a group applying FIFA tie-breakers: points, GD, GF, then head-to-head
    mini-table among equals, then reproducible drawing of lots."""
    table = group_table(teams, results, points)
    ordered = sorted(teams, key=lambda t: _global_key(table[t]), reverse=True)
    out = []
    for _, grp in groupby(ordered, key=lambda t: _global_key(table[t])):
        tied = list(grp)
        if len(tied) == 1:
            out.extend(tied)
        else:
            out.extend(_break_ties(tied, results, points, rng))
    return out


def rank_thirds(thirds: list, thirds_ranking: list, rng) -> list:
    """Rank third-placed entries. thirds: list of (team, stats dict with
    points/gd/gf). Returns teams best-first."""
    def key(entry):
        _, stats = entry
        return (stats["points"], stats["gd"], stats["gf"])

    ordered = sorted(thirds, key=key, reverse=True)
    out = []
    for _, grp in groupby(ordered, key=key):
        tied = list(grp)
        if len(tied) == 1:
            out.append(tied[0][0])
        else:
            for team, _ in sorted(tied, key=lambda e: rng.random()):
                out.append(team)
    return out
