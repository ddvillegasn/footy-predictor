from footy.models.montecarlo import simulate

MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


def test_probabilities_sum_to_one():
    res = simulate(2.5, 0.4, MC_CFG)
    total = res["team_a_win"] + res["draw"] + res["team_b_win"]
    assert abs(total - 100.0) < 0.01


def test_seed_is_deterministic():
    a = simulate(1.5, 1.2, MC_CFG)
    b = simulate(1.5, 1.2, MC_CFG)
    assert a == b


def test_favorite_has_higher_win_prob():
    res = simulate(2.8, 0.3, MC_CFG)
    assert res["team_a_win"] > res["team_b_win"]
    assert res["most_likely_score"].count("-") == 1


def test_expected_goals_reported():
    res = simulate(2.0, 0.5, MC_CFG)
    assert res["expected_goals_a"] == 2.0
    assert res["expected_goals_b"] == 0.5
    assert "goals_a" in res["confidence_interval"]
