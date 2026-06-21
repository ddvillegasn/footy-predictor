from __future__ import annotations

import numpy as np


class MatchSampler:
    """Sole source of randomness for goals. Caches lambdas (not scorelines)."""

    def __init__(self, model, mc_config: dict, model_version: str, config_hash: str):
        self.model = model
        self.max_goals = int(mc_config["max_goals"])
        self.model_version = model_version
        self.config_hash = config_hash
        self._lambda_cache: dict = {}

    def lambdas(self, team_a: str, team_b: str, neutral: bool) -> tuple[float, float]:
        key = (team_a, team_b, neutral, self.model_version, self.config_hash)
        if key not in self._lambda_cache:
            self._lambda_cache[key] = self.model.rates(team_a, team_b, neutral=neutral)
        return self._lambda_cache[key]

    def sample_goals(self, lam_a: float, lam_b: float, n: int, rng):
        ga = np.clip(rng.poisson(lam_a, n), 0, self.max_goals)
        gb = np.clip(rng.poisson(lam_b, n), 0, self.max_goals)
        return ga, gb

    def scorelines(self, team_a: str, team_b: str, neutral: bool, n: int, rng):
        lam_a, lam_b = self.lambdas(team_a, team_b, neutral)
        return self.sample_goals(lam_a, lam_b, n, rng)
