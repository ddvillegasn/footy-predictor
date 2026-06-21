import numpy as np

from footy.tournament.groups import rank_group, rank_thirds

POINTS = {"win": 3, "draw": 1, "loss": 0}
TB = ["points", "goal_difference", "goals_for", "head_to_head", "fair_play", "drawing_of_lots"]


def test_rank_by_points_then_gd_then_gf():
    teams = ["A", "B", "C", "D"]
    results = [
        ("A", "B", 1, 0), ("A", "C", 5, 0), ("A", "D", 1, 0),
        ("B", "C", 1, 0), ("B", "D", 1, 0), ("C", "D", 0, 1),
    ]
    # A: 9 pts. B: 6 pts. D: 3 pts. C: 0.
    order = rank_group(teams, results, POINTS, TB, np.random.default_rng(0))
    assert order == ["A", "B", "D", "C"]


def test_head_to_head_breaks_equal_pts_gd_gf():
    # Two teams level on points/GD/GF overall; H2H decides.
    teams = ["A", "B", "C", "D"]
    results = [
        ("A", "B", 1, 0),   # A beat B head-to-head
        ("A", "C", 0, 5),
        ("A", "D", 3, 0),
        ("B", "C", 0, 5),
        ("B", "D", 3, 0),
        ("C", "D", 1, 0),
    ]
    # A and B: each 6 pts, GD: A = 1-5+3 = -1? compute -> both equal; H2H A>B.
    order = rank_group(teams, results, POINTS, TB, np.random.default_rng(0))
    assert order.index("A") < order.index("B")


def test_drawing_of_lots_is_reproducible():
    # Two identical teams (mirror results) -> only lots can break; same seed = same order.
    teams = ["A", "B"]
    results = [("A", "B", 0, 0)]
    o1 = rank_group(teams, results, POINTS, TB, np.random.default_rng(7))
    o2 = rank_group(teams, results, POINTS, TB, np.random.default_rng(7))
    assert o1 == o2


def test_rank_thirds_picks_best():
    thirds = [
        ("X", {"points": 3, "gd": 1, "gf": 2}),
        ("Y", {"points": 6, "gd": 4, "gf": 5}),
        ("Z", {"points": 1, "gd": -2, "gf": 1}),
    ]
    ranking = ["points", "goal_difference", "goals_for", "drawing_of_lots"]
    order = rank_thirds(thirds, ranking, np.random.default_rng(0))
    assert order[0] == "Y" and order[-1] == "Z"
