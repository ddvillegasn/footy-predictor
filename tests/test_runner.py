import numpy as np

from footy.tournament.structure import TournamentConfig
from footy.live.provider import ProviderMatch
from footy.live.runner import TournamentRunner, watch


class FakeProvider:
    def __init__(self, matches):
        self._matches = matches

    def fetch_finished(self):
        return self._matches


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


class FakePredictor:
    def predict(self, team_a, team_b, neutral=False):
        return {"team_a_win": 60.0, "draw": 25.0, "team_b_win": 15.0,
                "expected_goals_a": 1.5, "expected_goals_b": 0.8, "most_likely_score": "1-0"}


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


def test_cycle_returns_aggregate_scoreboard_meta(tmp_path):
    out = tmp_path / "r.yaml"
    runner = TournamentRunner(_struct(), {}, {"GROUP_STAGE": "group"}, out,
                              FakeSampler(), FakePredictor(), n=30, seed=1)
    provider = FakeProvider([ProviderMatch("1", "A1", "A2", 2, 0, "GROUP_STAGE", "GROUP_A", "FINISHED")])
    result = runner.cycle(provider)
    assert result["played"] == 1
    assert "teams" in result["aggregate"]
    assert result["scoreboard"]["n"] == 1
    assert result["meta"]["n_tournaments"] == 30


def test_watch_stops_on_keyboard_interrupt():
    class BoomRunner:
        def __init__(self):
            self.calls = 0

        def cycle(self, provider):
            self.calls += 1
            raise KeyboardInterrupt

    runner = BoomRunner()
    emitted = []
    watch(runner, provider=None, interval_minutes=0, emit=emitted.append)
    assert runner.calls == 1 and emitted == []   # raised before emit, loop exits cleanly
