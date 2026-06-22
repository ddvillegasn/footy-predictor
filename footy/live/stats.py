def team_stats(structure, results) -> dict:
    """Per-team tournament table from played matches (group + knockout):
    {team: {played, points, gf, ga, gd, wins, draws, losses, form}}.
    form = list of 'W'/'D'/'L', oldest first."""
    points = structure.points
    teams = {t for group in structure.groups.values() for t in group}
    stats = {t: {"played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0,
                 "wins": 0, "draws": 0, "losses": 0, "form": []} for t in teams}
    for pm in results.played:
        for team, gf, ga in ((pm.team_a, pm.goals_a, pm.goals_b),
                             (pm.team_b, pm.goals_b, pm.goals_a)):
            if team not in stats:
                continue
            s = stats[team]
            s["played"] += 1
            s["gf"] += gf
            s["ga"] += ga
            if gf > ga:
                s["points"] += points["win"]; s["wins"] += 1; s["form"].append("W")
            elif gf < ga:
                s["points"] += points["loss"]; s["losses"] += 1; s["form"].append("L")
            else:
                s["points"] += points["draw"]; s["draws"] += 1; s["form"].append("D")
    for t in stats:
        stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]
    return stats
