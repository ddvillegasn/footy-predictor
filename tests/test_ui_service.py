from types import SimpleNamespace

import numpy as np

from footy.ui.service import team_list, match_prediction, tournament_probs, live_scoreboard
from footy.tournament.structure import TournamentConfig
from footy.tournament.results import TournamentResults


class FakePredictor:
    def __init__(self):
        self.model = SimpleNamespace(attack={"Brazil": 0.1, "Argentina": 0.2, "Haiti": -0.3})
        self.last_kwargs = None

    def predict(self, team_a, team_b, neutral=False, include_markets=False, book_odds=None):
        self.last_kwargs = {"neutral": neutral, "include_markets": include_markets,
                            "book_odds": book_odds}
        return {"team_a_win": 50.0, "draw": 30.0, "team_b_win": 20.0,
                "expected_goals_a": 1.5, "expected_goals_b": 1.0, "most_likely_score": "1-1"}


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


def test_team_list_sorted():
    assert team_list(FakePredictor()) == ["Argentina", "Brazil", "Haiti"]


def test_match_prediction_always_includes_markets():
    fp = FakePredictor()
    match_prediction(fp, "Brazil", "Haiti", neutral=True)
    assert fp.last_kwargs["include_markets"] is True


def test_match_prediction_with_book_odds_forces_markets():
    fp = FakePredictor()
    match_prediction(fp, "Brazil", "Haiti", book_odds={"1x2": {"home": 1.5}})
    assert fp.last_kwargs["include_markets"] is True
    assert fp.last_kwargs["book_odds"] == {"1x2": {"home": 1.5}}


def test_tournament_probs_returns_aggregate():
    agg = tournament_probs(_struct(), TournamentResults([]), FakeSampler(), n=30, seed=1)
    assert "teams" in agg and "groups" in agg


def test_live_scoreboard_empty_is_none():
    assert live_scoreboard(FakePredictor(), [])["n"] == 0
