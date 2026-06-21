import numpy as np

from footy.tournament.structure import TournamentConfig
from footy.tournament.results import TournamentResults
from footy.tournament.simulator import simulate_tournaments
from footy.tournament.aggregate import aggregate, tournament_odds


class FakeSampler:
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


def _sims():
    return simulate_tournaments(_struct(), TournamentResults([]), FakeSampler(), n=300, seed=5)


def test_champion_probs_sum_to_one():
    agg = aggregate(_struct(), _sims())
    total = sum(t["champion"] for t in agg["teams"].values())
    assert abs(total - 1.0) < 1e-9


def test_group_positions_sum_to_one_per_team():
    agg = aggregate(_struct(), _sims())
    a1 = agg["groups"]["A"]["A1"]
    assert abs(a1["p1"] + a1["p2"] + a1["p3"] + a1["p4"] - 1.0) < 1e-9


def test_round_probs_monotone():
    agg = aggregate(_struct(), _sims())
    for t in agg["teams"].values():
        assert t["reach_F"] >= t["champion"] - 1e-9


def test_stronger_team_more_likely_champion():
    agg = aggregate(_struct(), _sims())
    assert agg["teams"]["A1"]["champion"] > agg["teams"]["A4"]["champion"]


def test_tournament_odds_fair_and_value():
    agg = aggregate(_struct(), _sims())
    vcfg = {"threshold": 0.0, "ev_medium": 0.10, "reliability_low": 0.40,
            "reliability_high": 0.70, "kelly_quarter_divisor": 4}
    out = tournament_odds(agg, book_odds={"champion": {"A1": 50.0}},
                          reliability=0.6, value_config=vcfg)
    assert out["odds"]["champion"]["A1"]["fair_odds"] is not None
    assert "A1" in out["value"]["champion"]
