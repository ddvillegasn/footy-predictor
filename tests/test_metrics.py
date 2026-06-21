import math

from footy.metrics import log_loss_1x2, brier_1x2, naive_baseline_probs


def test_log_loss_perfect_is_near_zero():
    probs = [{"home": 1.0, "draw": 0.0, "away": 0.0}]
    assert log_loss_1x2(probs, ["home"]) < 1e-6


def test_log_loss_penalises_wrong():
    probs = [{"home": 0.01, "draw": 0.01, "away": 0.98}]
    assert log_loss_1x2(probs, ["home"]) > 1.0


def test_brier_range():
    probs = [{"home": 0.5, "draw": 0.3, "away": 0.2}]
    score = brier_1x2(probs, ["home"])
    assert 0.0 <= score <= 2.0


def test_naive_baseline_sums_to_one():
    outcomes = ["home", "home", "draw", "away", "home"]
    probs = naive_baseline_probs(outcomes)
    assert abs(probs["home"] + probs["draw"] + probs["away"] - 1.0) < 1e-9
    assert probs["home"] == 0.6
