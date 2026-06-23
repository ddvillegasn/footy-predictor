from footy.eval.evaluate_models import evaluate
from footy.eval.predictors import RandomPredictor, NaiveGlobalPredictor


class PerfectGoalPredictor:
    """Always predicts the realized outcome with goals — for goal_mae coverage."""
    def __init__(self, table):
        self.table = table

    def probs(self, a, b, neutral):
        return self.table[(a, b)]["probs"]

    def goals(self, a, b, neutral):
        return self.table[(a, b)]["goals"]


def _matches():
    return [
        {"team_a": "A", "team_b": "B", "neutral": True, "goals_a": 2, "goals_b": 0},
        {"team_a": "C", "team_b": "D", "neutral": True, "goals_a": 1, "goals_b": 1},
    ]


def test_random_metrics_shape():
    out = evaluate(RandomPredictor(), _matches())
    assert out["n"] == 2
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["log_loss"] > 0 and out["brier"] >= 0
    assert out["goal_mae"] is None            # random predicts no goals
    assert "ece" in out["calibration"]


def test_goal_mae_present_for_goal_predictor():
    table = {("A", "B"): {"probs": {"home": 0.8, "draw": 0.1, "away": 0.1}, "goals": (2.0, 0.0)},
             ("C", "D"): {"probs": {"home": 0.3, "draw": 0.4, "away": 0.3}, "goals": (1.0, 1.0)}}
    out = evaluate(PerfectGoalPredictor(table), _matches())
    assert out["goal_mae"] == 0.0             # exact goals
    assert out["accuracy"] == 1.0             # argmax matches both outcomes
