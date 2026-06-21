import numpy as np
import pytest

from footy.models.montecarlo import simulate_goals, aggregate_outcomes, simulate

CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


def test_simulate_goals_shapes_and_meta():
    ga, gb, meta = simulate_goals(2.0, 0.5, CFG)
    assert len(ga) == 20000 and len(gb) == 20000
    assert meta["seed"] == 42 and meta["n_sims"] == 20000
    assert meta["lambda_a"] == 2.0 and meta["lambda_b"] == 0.5
    assert meta["dc_enabled"] is False
    assert meta["clip_max"] == 10


def test_simulate_goals_is_seed_deterministic():
    a1, b1, _ = simulate_goals(1.5, 1.2, CFG)
    a2, b2, _ = simulate_goals(1.5, 1.2, CFG)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_simulate_goals_respects_clip():
    ga, gb, _ = simulate_goals(5.0, 5.0, {**CFG, "max_goals": 3})
    assert ga.max() <= 3 and gb.max() <= 3


def test_simulate_goals_rejects_nonpositive_lambda():
    with pytest.raises(ValueError):
        simulate_goals(0.0, 1.0, CFG)
    with pytest.raises(ValueError):
        simulate_goals(1.0, -0.5, CFG)


def test_simulate_output_unchanged_after_refactor():
    # simulate() must equal aggregate_outcomes over the same goal arrays.
    ga, gb, _ = simulate_goals(1.7, 0.9, CFG)
    expected = aggregate_outcomes(ga, gb, 1.7, 0.9, CFG)
    assert simulate(1.7, 0.9, CFG) == expected
