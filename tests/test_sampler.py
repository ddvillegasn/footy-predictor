import numpy as np

from footy.tournament.sampler import MatchSampler


class StubModel:
    """Minimal model exposing rates() like DixonColesModel."""
    def __init__(self):
        self.calls = 0

    def rates(self, team_a, team_b, neutral=False):
        self.calls += 1
        base = {"Strong": 2.2, "Weak": 0.6}
        la = base.get(team_a, 1.0)
        lb = base.get(team_b, 1.0)
        if not neutral:
            la += 0.2
        return la, lb


CFG = {"max_goals": 10}


def test_scorelines_shapes_and_determinism():
    s = MatchSampler(StubModel(), CFG, "m1", "h1")
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a1, b1 = s.scorelines("Strong", "Weak", True, 5000, rng1)
    a2, b2 = s.scorelines("Strong", "Weak", True, 5000, rng2)
    assert len(a1) == 5000
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert a1.mean() > b1.mean()  # strong scores more


def test_lambda_cache_avoids_remodel():
    model = StubModel()
    s = MatchSampler(model, CFG, "m1", "h1")
    s.lambdas("Strong", "Weak", True)
    s.lambdas("Strong", "Weak", True)
    assert model.calls == 1  # cached on second call


def test_sample_goals_respects_clip():
    s = MatchSampler(StubModel(), {"max_goals": 2}, "m1", "h1")
    rng = np.random.default_rng(1)
    ga, gb = s.sample_goals(8.0, 8.0, 1000, rng)
    assert ga.max() <= 2 and gb.max() <= 2
