from __future__ import annotations

import difflib

import pandas as pd

from footy.models.poisson import fit_dixon_coles, DixonColesModel
from footy.models.montecarlo import simulate
from footy.features.strength import recent_form
from footy.reliability import compute_reliability


class Predictor:
    """Bundles a fitted model + match history to answer predict() queries."""

    def __init__(self, model: DixonColesModel, matches: pd.DataFrame,
                 model_config: dict, mc_config: dict, canonical):
        self.model = model
        self.matches = matches
        self.model_config = model_config
        self.mc_config = mc_config
        self.canonical = canonical
        self._teams = set(model.attack.keys())

    @classmethod
    def from_matches(cls, matches: pd.DataFrame, model_config: dict, mc_config: dict,
                     canonical, as_of) -> "Predictor":
        canon = matches.copy()
        canon["home_team"] = canon["home_team"].map(canonical)
        canon["away_team"] = canon["away_team"].map(canonical)
        model = fit_dixon_coles(canon, model_config, as_of=as_of)
        return cls(model, canon, model_config, mc_config, canonical)

    def _resolve(self, team: str) -> str:
        name = self.canonical(team)
        if name not in self._teams:
            suggestions = difflib.get_close_matches(name, sorted(self._teams), n=3)
            raise ValueError(f"Unknown team: {team!r}. Did you mean {suggestions}?")
        return name

    def _match_count(self, team: str) -> int:
        m = self.matches
        return int(((m["home_team"] == team) | (m["away_team"] == team)).sum())

    def predict(self, team_a: str, team_b: str, neutral: bool = False,
                tournament: str = "Friendly") -> dict:
        a = self._resolve(team_a)
        b = self._resolve(team_b)

        lam_a, lam_b = self.model.rates(a, b, neutral=neutral)
        sim = simulate(lam_a, lam_b, self.mc_config)

        as_of = self.matches["date"].max() + pd.Timedelta(days=1)
        form_a = recent_form(self.matches, a, as_of, window=10)
        form_b = recent_form(self.matches, b, as_of, window=10)

        reliability = compute_reliability(
            matches_a=self._match_count(a),
            matches_b=self._match_count(b),
            recent_a=form_a["matches"],
            recent_b=form_b["matches"],
            data_age_days=0.0,
            dispersion=sim["lambda_dispersion"],
            missing_rankings=0,
            min_matches=int(self.model_config["min_matches_reliable"]),
        )

        return {
            "team_a": a,
            "team_b": b,
            "team_a_win": sim["team_a_win"],
            "draw": sim["draw"],
            "team_b_win": sim["team_b_win"],
            "expected_goals_a": sim["expected_goals_a"],
            "expected_goals_b": sim["expected_goals_b"],
            "most_likely_score": sim["most_likely_score"],
            "score_distribution": sim["score_distribution"],
            "confidence_interval": sim["confidence_interval"],
            "prediction_reliability": reliability,
            "model_version": self.model_config["model_version"],
        }
