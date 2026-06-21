from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles low-score dependency correction (vectorised)."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


@dataclass
class DixonColesModel:
    attack: dict
    defense: dict
    home_adv: float
    intercept: float
    rho: float

    def rates(self, team_a: str, team_b: str, neutral: bool = False) -> tuple[float, float]:
        if team_a not in self.attack:
            raise KeyError(team_a)
        if team_b not in self.attack:
            raise KeyError(team_b)
        adv = 0.0 if neutral else self.home_adv
        lam_a = np.exp(self.intercept + self.attack[team_a] - self.defense[team_b] + adv)
        lam_b = np.exp(self.intercept + self.attack[team_b] - self.defense[team_a])
        return float(lam_a), float(lam_b)


def fit_dixon_coles(matches: pd.DataFrame, config: dict, as_of) -> DixonColesModel:
    """Fit a time-decayed Dixon-Coles model by weighted maximum likelihood."""
    xi = float(config["xi"])
    ridge = float(config.get("ridge", 0.0))
    home_init = float(config.get("home_advantage_init", 0.25))
    as_of = pd.Timestamp(as_of)

    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_i = df["home_team"].map(idx).to_numpy()
    away_i = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy(dtype=int)
    ag = df["away_score"].to_numpy(dtype=int)
    neutral = df["neutral"].to_numpy(dtype=bool) if "neutral" in df else np.zeros(len(df), bool)

    age_days = (as_of - df["date"]).dt.days.to_numpy(dtype=float)
    weights = np.exp(-xi * np.clip(age_days, 0, None))

    # Parameter vector: [intercept, home_adv, attack(n), defense(n), rho]
    def unpack(p):
        intercept = p[0]
        home_adv = p[1]
        attack = p[2:2 + n]
        defense = p[2 + n:2 + 2 * n]
        rho = p[-1]
        return intercept, home_adv, attack, defense, rho

    def neg_log_lik(p):
        intercept, home_adv, attack, defense, rho = unpack(p)
        adv = np.where(neutral, 0.0, home_adv)
        log_lam = intercept + attack[home_i] - defense[away_i] + adv
        log_mu = intercept + attack[away_i] - defense[home_i]
        lam = np.exp(np.clip(log_lam, -10, 10))
        mu = np.exp(np.clip(log_mu, -10, 10))
        tau = _tau(hg, ag, lam, mu, rho)
        tau = np.clip(tau, 1e-9, None)
        ll = (
            -lam + hg * np.log(lam)
            - mu + ag * np.log(mu)
            + np.log(tau)
        )
        penalty = ridge * (np.sum(attack ** 2) + np.sum(defense ** 2))
        return -np.sum(weights * ll) + penalty

    x0 = np.zeros(2 + 2 * n + 1)
    x0[0] = np.log(max(df[["home_score", "away_score"]].to_numpy().mean(), 0.1))
    x0[1] = home_init
    bounds = (
        [(-3, 3), (-1, 1)]
        + [(-3, 3)] * (2 * n)
        + [(-0.2, 0.2)]
    )
    res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
    intercept, home_adv, attack, defense, rho = unpack(res.x)

    return DixonColesModel(
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        home_adv=float(home_adv),
        intercept=float(intercept),
        rho=float(rho),
    )
