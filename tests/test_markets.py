import math

import numpy as np

from footy.models.montecarlo import simulate_goals
from footy.betting.markets import markets_from_samples

CFG = {
    "n_sims": 60000, "seed": 7, "max_goals": 10,
    "over_under_lines": [0.5, 1.5, 2.5, 3.5, 4.5],
    "handicap_lines": [-1.5, -0.5, 0.5, 1.5],
    "top_scores": 5,
}


def _markets(lam_a, lam_b):
    ga, gb, _ = simulate_goals(lam_a, lam_b, CFG)
    return markets_from_samples(ga, gb, CFG)


def test_1x2_and_double_chance_consistency():
    m = _markets(1.8, 1.0)
    o = m["1x2"]
    assert abs(o["home"] + o["draw"] + o["away"] - 1.0) < 1e-9
    dc = m["double_chance"]
    assert abs(dc["1X"] - (o["home"] + o["draw"])) < 1e-9
    assert abs(dc["X2"] - (o["draw"] + o["away"])) < 1e-9
    assert abs(dc["12"] - (o["home"] + o["away"])) < 1e-9


def test_over_under_sums_to_one_per_line():
    m = _markets(1.4, 1.1)
    for line, ou in m["over_under"].items():
        assert abs(ou["over"] + ou["under"] - 1.0) < 1e-9


def test_correct_score_top_and_mass():
    m = _markets(1.6, 0.8)
    cs = m["correct_score"]
    assert len(cs["top"]) == 5
    assert abs(cs["all_mass_check"] - 1.0) < 1e-9
    assert abs(cs["other_probability"] - (1.0 - sum(cs["top"].values()))) < 1e-6


def test_handicap_complementary():
    m = _markets(2.0, 0.7)
    for line, h in m["handicap"].items():
        assert abs(h["home"] + h["away"] - 1.0) < 1e-9


def test_cross_check_btts_and_over05_closed_form():
    # Valid ONLY for dc_enabled=False, independent Poisson sampling.
    lam_a, lam_b = 1.7, 1.1
    m = _markets(lam_a, lam_b)
    btts_closed = (1 - math.exp(-lam_a)) * (1 - math.exp(-lam_b))
    over05_closed = 1 - math.exp(-(lam_a + lam_b))
    assert abs(m["btts"]["yes"] - btts_closed) < 0.02
    assert abs(m["over_under"]["0.5"]["over"] - over05_closed) < 0.02
