import numpy as np

from footy.tournament.structure import TournamentConfig
from footy.tournament.results import TournamentResults, PlayedMatch
from footy.tournament.simulator import run_tournament, simulate_tournaments


class FakeSampler:
    """Strong teams (lower index in each group) score more, deterministically-ish."""
    def __init__(self):
        self.strength = {"A1": 3.0, "A2": 2.0, "A3": 1.0, "A4": 0.5,
                         "B1": 3.0, "B2": 2.0, "B3": 1.0, "B4": 0.5}

    def lambdas(self, a, b, neutral):
        return self.strength[a], self.strength[b]

    def sample_goals(self, lam_a, lam_b, n, rng):
        return np.clip(rng.poisson(lam_a, n), 0, 10), np.clip(rng.poisson(lam_b, n), 0, 10)

    def scorelines(self, a, b, neutral, n, rng):
        la, lb = self.lambdas(a, b, neutral)
        return self.sample_goals(la, lb, n, rng)


def _struct():
    return TournamentConfig(
        name="Mini", neutral_default=True, points={"win": 3, "draw": 1, "loss": 0},
        groups={"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]},
        group_schedule="round_robin", per_group_advance=2, best_thirds=0,
        tiebreakers=["points", "goal_difference", "goals_for", "head_to_head", "fair_play", "drawing_of_lots"],
        thirds_ranking=["points", "goal_difference", "goals_for", "drawing_of_lots"],
        rounds=["SF", "F"],
        bracket_r32=[["winner_A", "runner_B"], ["winner_B", "runner_A"]],
        thirds_assignment="ranked_order",
    )


def test_run_tournament_produces_champion_and_rounds():
    res = run_tournament(_struct(), TournamentResults([]), FakeSampler(), np.random.default_rng(1))
    assert res.champion in {"A1", "A2", "B1", "B2"}  # only qualifiers can win
    assert res.furthest_round[res.champion] == "champion"
    # group stage debug fields present
    assert "A1" in res.group_points and res.group_points["A1"] >= 0


def test_played_group_result_is_fixed():
    # Force A4 to thrash everyone via played results -> A4 must top group A every run.
    played = TournamentResults([
        PlayedMatch("g1", "group", "A4", "A1", 9, 0, group="A"),
        PlayedMatch("g2", "group", "A4", "A2", 9, 0, group="A"),
        PlayedMatch("g3", "group", "A4", "A3", 9, 0, group="A"),
    ])
    for seed in range(5):
        res = run_tournament(_struct(), played, FakeSampler(), np.random.default_rng(seed))
        assert res.group_order["A"][0] == "A4"  # fixed wins put A4 first


def test_simulate_tournaments_runs_many():
    results = simulate_tournaments(_struct(), TournamentResults([]), FakeSampler(), n=50, seed=3)
    assert len(results) == 50
    assert all(r.champion is not None for r in results)
