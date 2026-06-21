from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from footy.tournament.groups import group_fixtures, group_table, rank_group, rank_thirds
from footy.tournament.knockout import build_bracket, resolve_match


@dataclass
class TournamentResult:
    champion: str
    furthest_round: dict          # team -> round name or "champion"
    group_order: dict             # group -> ordered team list
    group_position: dict          # team -> 1..N
    group_points: dict            # team -> points
    group_gd: dict                # team -> goal difference
    group_gf: dict                # team -> goals for
    run_id: int = 0


def _played_group_score(pm, a, b):
    """Return (goals_a, goals_b) oriented to (a, b)."""
    if pm.team_a == a:
        return pm.goals_a, pm.goals_b
    return pm.goals_b, pm.goals_a


def _played_knockout_winner(pm, a, b):
    if pm.winner is not None:
        return pm.winner
    ga, gb = _played_group_score(pm, a, b)
    if ga == gb:
        raise ValueError(f"played knockout {pm.match_id} is a draw without a winner field")
    return a if ga > gb else b


def run_tournament(structure, results, sampler, rng) -> TournamentResult:
    run_id = int(rng.integers(0, 2**31 - 1))
    group_order, group_position = {}, {}
    group_points, group_gd, group_gf = {}, {}, {}
    thirds_entries = []

    for g, teams in structure.groups.items():
        match_results = []
        for (a, b) in group_fixtures(teams):
            pm = results.lookup_group(g, a, b)
            if pm is not None:
                ga, gb = _played_group_score(pm, a, b)
            else:
                sa, sb = sampler.scorelines(a, b, structure.neutral_default, 1, rng)
                ga, gb = int(sa[0]), int(sb[0])
            match_results.append((a, b, ga, gb))

        table = group_table(teams, match_results, structure.points)
        order = rank_group(teams, match_results, structure.points, structure.tiebreakers, rng)
        group_order[g] = order
        for pos, t in enumerate(order, start=1):
            group_position[t] = pos
            group_points[t] = table[t]["points"]
            group_gd[t] = table[t]["gd"]
            group_gf[t] = table[t]["gf"]
        if structure.best_thirds > 0 and len(order) >= 3:
            third = order[2]
            thirds_entries.append((third, {"points": table[third]["points"],
                                           "gd": table[third]["gd"], "gf": table[third]["gf"]}))

    thirds_ranked = []
    if structure.best_thirds > 0:
        thirds_ranked = rank_thirds(thirds_entries, structure.thirds_ranking, rng)[:structure.best_thirds]

    group_ranks = {g: order for g, order in group_order.items()}
    ties = build_bracket(group_ranks, thirds_ranked, structure.bracket_r32)

    furthest = {}
    for tie in ties:
        for t in tie:
            furthest[t] = structure.rounds[0]

    for ridx, round_name in enumerate(structure.rounds):
        winners = []
        for (a, b) in ties:
            pm = results.lookup_knockout(round_name, a, b)
            if pm is not None:
                winner = _played_knockout_winner(pm, a, b)
            else:
                sa, sb = sampler.scorelines(a, b, structure.neutral_default, 1, rng)
                winner = resolve_match(a, b, int(sa[0]), int(sb[0]), sampler, rng,
                                       structure.neutral_default)
            winners.append(winner)
            next_label = "champion" if ridx == len(structure.rounds) - 1 else structure.rounds[ridx + 1]
            furthest[winner] = next_label
        ties = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)] if len(winners) > 1 else []
        if len(winners) == 1:
            champion = winners[0]

    return TournamentResult(
        champion=champion, furthest_round=furthest, group_order=group_order,
        group_position=group_position, group_points=group_points,
        group_gd=group_gd, group_gf=group_gf, run_id=run_id,
    )


def simulate_tournaments(structure, results, sampler, n: int, seed: int) -> list:
    master = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        child = np.random.default_rng(master.integers(0, 2**63 - 1))
        out.append(run_tournament(structure, results, sampler, child))
    return out
