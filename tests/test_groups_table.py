from footy.tournament.groups import group_fixtures, group_table

POINTS = {"win": 3, "draw": 1, "loss": 0}


def test_round_robin_fixtures():
    fx = group_fixtures(["A", "B", "C", "D"])
    assert len(fx) == 6
    assert ("A", "B") in fx and ("C", "D") in fx
    # no self matches, no duplicates
    assert all(a != b for a, b in fx)
    assert len(set(map(frozenset, fx))) == 6


def test_table_points_and_gd():
    teams = ["A", "B", "C", "D"]
    # A beats B 2-0, A beats C 1-0, A draws D 1-1
    results = [
        ("A", "B", 2, 0), ("A", "C", 1, 0), ("A", "D", 1, 1),
        ("B", "C", 0, 0), ("B", "D", 1, 2), ("C", "D", 3, 3),
    ]
    table = group_table(teams, results, POINTS)
    assert table["A"]["points"] == 7  # 2W 1D
    assert table["A"]["gf"] == 4 and table["A"]["ga"] == 1
    assert table["A"]["gd"] == 3
    assert table["D"]["points"] == 5  # W vs B, 2 draws
