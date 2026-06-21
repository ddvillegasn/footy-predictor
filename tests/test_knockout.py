import numpy as np

from footy.tournament.knockout import build_bracket, resolve_match


class FakeSampler:
    """Deterministic sampler for tests: fixed extra-time goals, known lambdas."""
    def __init__(self, et_goals=(0, 0), lambdas=(2.0, 1.0)):
        self.et_goals = et_goals
        self._lambdas = lambdas

    def lambdas(self, a, b, neutral):
        return self._lambdas

    def sample_goals(self, lam_a, lam_b, n, rng):
        return (np.array([self.et_goals[0]] * n), np.array([self.et_goals[1]] * n))


def test_build_bracket_resolves_slots():
    group_ranks = {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}
    cfg = [["winner_A", "runner_B"], ["winner_B", "runner_A"]]
    ties = build_bracket(group_ranks, [], cfg)
    assert ties == [("A1", "B2"), ("B1", "A2")]


def test_build_bracket_with_thirds():
    group_ranks = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    cfg = [["winner_A", "third_slot_1"], ["winner_B", "third_slot_2"]]
    ties = build_bracket(group_ranks, ["C3", "D3"], cfg)
    assert ties == [("A1", "C3"), ("B1", "D3")]


def test_resolve_clear_winner():
    s = FakeSampler()
    w = resolve_match("A", "B", 2, 0, s, np.random.default_rng(0), neutral=True)
    assert w == "A"


def test_resolve_draw_goes_to_extra_time():
    # Regulation 1-1, extra time 1-0 for A -> A wins.
    s = FakeSampler(et_goals=(1, 0))
    w = resolve_match("A", "B", 1, 1, s, np.random.default_rng(0), neutral=True)
    assert w == "A"


def test_resolve_penalties_favour_stronger_over_many_seeds():
    # Regulation draw, extra time 0-0 -> penalties weighted by lambdas (A stronger).
    s = FakeSampler(et_goals=(0, 0), lambdas=(3.0, 1.0))
    wins_a = sum(
        resolve_match("A", "B", 0, 0, s, np.random.default_rng(seed), neutral=True) == "A"
        for seed in range(400)
    )
    assert wins_a > 240  # ~0.75 share, clearly above 50%


def test_penalty_clipping_extremes():
    # Even a huge lambda gap is clipped to [0.05, 0.95]: underdog still wins sometimes.
    s = FakeSampler(et_goals=(0, 0), lambdas=(50.0, 0.1))
    wins_b = sum(
        resolve_match("A", "B", 0, 0, s, np.random.default_rng(seed), neutral=True) == "B"
        for seed in range(400)
    )
    assert wins_b > 0  # clipping guarantees a floor
