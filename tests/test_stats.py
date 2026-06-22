from types import SimpleNamespace

from footy.tournament.results import TournamentResults, PlayedMatch
from footy.live.stats import team_stats


def _structure():
    return SimpleNamespace(
        groups={"A": ["Mexico", "South Africa", "South Korea", "Czech Republic"]},
        points={"win": 3, "draw": 1, "loss": 0})


def test_team_stats_accumulates():
    results = TournamentResults([
        PlayedMatch("1", "group", "Mexico", "South Africa", 2, 0, group="A"),
        PlayedMatch("2", "group", "South Korea", "Mexico", 1, 1, group="A"),
    ])
    stats = team_stats(_structure(), results)
    mx = stats["Mexico"]
    assert mx["played"] == 2 and mx["points"] == 4          # win + draw
    assert mx["gf"] == 3 and mx["ga"] == 1 and mx["gd"] == 2
    assert mx["wins"] == 1 and mx["draws"] == 1 and mx["losses"] == 0
    assert mx["form"] == ["W", "D"]
    assert stats["Czech Republic"]["played"] == 0           # no matches yet
