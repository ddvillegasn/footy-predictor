from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from footy.models.poisson import fit_dixon_coles
from footy.data.external.elo import final_ratings


def _global_rates(matches) -> dict:
    h = d = a = 0
    for row in matches.itertuples(index=False):
        if row.home_score > row.away_score:
            h += 1
        elif row.home_score < row.away_score:
            a += 1
        else:
            d += 1
    n = h + d + a or 1
    return {"home": h / n, "draw": d / n, "away": a / n}


class DixonColesPredictor:
    name = "BASE"

    def __init__(self, model, fallback: dict, max_goals: int = 10):
        self.model = model
        self.fallback = fallback
        self.max_goals = max_goals

    def probs(self, team_a, team_b, neutral) -> dict:
        try:
            la, lb = self.model.rates(team_a, team_b, neutral=neutral)
        except KeyError:
            return dict(self.fallback)
        xs = np.arange(0, self.max_goals + 1)
        grid = np.outer(poisson.pmf(xs, la), poisson.pmf(xs, lb))
        rho = self.model.rho
        grid[0, 0] *= max(1e-9, 1.0 - la * lb * rho)
        grid[0, 1] *= max(1e-9, 1.0 + la * rho)
        grid[1, 0] *= max(1e-9, 1.0 + lb * rho)
        grid[1, 1] *= max(1e-9, 1.0 - rho)
        grid /= grid.sum()
        x = np.arange(grid.shape[0])[:, None]
        y = np.arange(grid.shape[1])[None, :]
        home = float(grid[x > y].sum())
        draw = float(np.trace(grid))
        away = float(grid[x < y].sum())
        s = home + draw + away
        return {"home": home / s, "draw": draw / s, "away": away / s}

    def goals(self, team_a, team_b, neutral):
        try:
            return self.model.rates(team_a, team_b, neutral=neutral)
        except KeyError:
            return None


class EloFavoritePredictor:
    name = "Elo"

    def __init__(self, ratings: dict, draw_rate: float, home_adv_elo: float, fallback: dict):
        self.ratings = ratings
        self.draw_rate = draw_rate
        self.home_adv_elo = home_adv_elo
        self.fallback = fallback

    def probs(self, team_a, team_b, neutral) -> dict:
        ra = self.ratings.get(team_a)
        rb = self.ratings.get(team_b)
        if ra is None or rb is None:
            return dict(self.fallback)
        adv = 0.0 if neutral else self.home_adv_elo
        p_home = 1.0 / (1.0 + 10.0 ** ((rb - (ra + adv)) / 400.0))
        rest = 1.0 - self.draw_rate
        return {"home": rest * p_home, "draw": self.draw_rate, "away": rest * (1.0 - p_home)}

    def goals(self, team_a, team_b, neutral):
        return None


class NaiveGlobalPredictor:
    name = "naive"

    def __init__(self, rates: dict):
        self.rates = rates

    def probs(self, team_a, team_b, neutral) -> dict:
        return dict(self.rates)

    def goals(self, team_a, team_b, neutral):
        return None


class RandomPredictor:
    name = "random"

    def probs(self, team_a, team_b, neutral) -> dict:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}

    def goals(self, team_a, team_b, neutral):
        return None


def build_predictors(train, model_config: dict, elo_config: dict, as_of) -> dict:
    """Construct the standard predictor set from a training DataFrame."""
    rates = _global_rates(train)
    model = fit_dixon_coles(train, model_config, as_of=as_of)
    ratings = final_ratings(train, elo_config)
    return {
        "BASE": DixonColesPredictor(model, fallback=rates, max_goals=int(model_config["max_goals"])),
        "Elo": EloFavoritePredictor(ratings, draw_rate=rates["draw"],
                                    home_adv_elo=float(elo_config["home_advantage_elo"]),
                                    fallback=rates),
        "naive": NaiveGlobalPredictor(rates),
        "random": RandomPredictor(),
    }
