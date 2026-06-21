from footy.live.scoreboard import scoreboard


class FakePredictor:
    """Returns canned predictions keyed by (team_a, team_b)."""
    def __init__(self, table):
        self.table = table

    def predict(self, team_a, team_b, neutral=False):
        return self.table[(team_a, team_b)]


def test_empty_played_returns_none_metrics():
    board = scoreboard(FakePredictor({}), [])
    assert board["n"] == 0 and board["accuracy"] is None and board["matches"] == []


def test_scoreboard_metrics_and_details():
    table = {
        ("Brazil", "Mexico"): {"team_a_win": 70.0, "draw": 20.0, "team_b_win": 10.0,
                               "expected_goals_a": 2.0, "expected_goals_b": 0.5,
                               "most_likely_score": "2-0"},
        ("USA", "Iran"): {"team_a_win": 30.0, "draw": 30.0, "team_b_win": 40.0,
                          "expected_goals_a": 1.0, "expected_goals_b": 1.2,
                          "most_likely_score": "1-1"},
    }
    played = [
        {"team_a": "Brazil", "team_b": "Mexico", "goals_a": 2, "goals_b": 1},  # home win -> hit
        {"team_a": "USA", "team_b": "Iran", "goals_a": 3, "goals_b": 0},        # home win -> predicted away, miss
    ]
    board = scoreboard(FakePredictor(table), played)
    assert board["n"] == 2
    assert 0.0 <= board["accuracy"] <= 1.0
    assert board["accuracy"] == 0.5            # 1 of 2 correct
    first = board["matches"][0]
    assert first["hit"] is True
    assert first["predicted_outcome"] == "home" and first["actual_outcome"] == "home"
    assert first["predicted_prob"] == 0.7 and first["actual_prob"] == 0.7
    second = board["matches"][1]
    assert second["hit"] is False
    assert second["actual_outcome"] == "home" and second["actual_prob"] == 0.3
    assert board["goal_mae"] is not None and board["log_loss"] > 0
