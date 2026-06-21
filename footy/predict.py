from __future__ import annotations

import difflib

import pandas as pd

from footy.models.poisson import fit_dixon_coles, DixonColesModel
from footy.models.montecarlo import simulate_goals, aggregate_outcomes
from footy.features.strength import recent_form
from footy.reliability import compute_reliability
from footy.betting import BETTING_VERSION
from footy.betting.markets import markets_from_samples
from footy.betting.odds import decorate_outcome, decorate_group
from footy.betting.value import assess_market


def _decorate_markets(raw: dict) -> dict:
    """Walk raw market probabilities and attach fair odds (+ group margins)."""
    out = {
        "1x2": decorate_group(raw["1x2"]),
        "double_chance": decorate_group(raw["double_chance"]),
        "over_under": {line: decorate_group(g) for line, g in raw["over_under"].items()},
        "btts": decorate_group(raw["btts"]),
        "handicap": {line: decorate_group(g) for line, g in raw["handicap"].items()},
        "correct_score": {
            "top": {s: decorate_outcome(p) for s, p in raw["correct_score"]["top"].items()},
            "other_probability": raw["correct_score"]["other_probability"],
            "all_mass_check": raw["correct_score"]["all_mass_check"],
        },
    }
    return out


def _value_tree(raw: dict, book_odds: dict, reliability: float, value_config: dict) -> dict:
    """Assess value for each market/outcome the user priced. Flat and per-line markets."""
    out = {}
    for market, priced in book_odds.items():
        if market not in raw:
            continue
        if market in ("1x2", "double_chance", "btts"):
            out[market] = assess_market(raw[market], priced, reliability, value_config)
        elif market in ("over_under", "handicap"):
            per_line = {}
            for line, odds_group in priced.items():
                if line in raw[market]:
                    per_line[line] = assess_market(raw[market][line], odds_group, reliability, value_config)
            out[market] = per_line
        elif market == "correct_score":
            out[market] = assess_market(raw[market]["top"], priced, reliability, value_config)
    return out


class Predictor:
    """Bundles a fitted model + match history to answer predict() queries."""

    def __init__(self, model: DixonColesModel, matches: pd.DataFrame,
                 model_config: dict, mc_config: dict, canonical,
                 betting_config: dict | None = None, betting_config_version: str | None = None):
        self.model = model
        self.matches = matches
        self.model_config = model_config
        self.mc_config = mc_config
        self.canonical = canonical
        self.betting_config = betting_config
        self.betting_config_version = betting_config_version
        self._teams = set(model.attack.keys())

    @classmethod
    def from_matches(cls, matches: pd.DataFrame, model_config: dict, mc_config: dict,
                     canonical, as_of, betting_config: dict | None = None,
                     betting_config_version: str | None = None) -> "Predictor":
        canon = matches.copy()
        canon["home_team"] = canon["home_team"].map(canonical)
        canon["away_team"] = canon["away_team"].map(canonical)
        model = fit_dixon_coles(canon, model_config, as_of=as_of)
        return cls(model, canon, model_config, mc_config, canonical,
                   betting_config, betting_config_version)

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
                tournament: str = "Friendly", include_markets: bool = False,
                book_odds: dict | None = None) -> dict:
        a = self._resolve(team_a)
        b = self._resolve(team_b)

        lam_a, lam_b = self.model.rates(a, b, neutral=neutral)
        goals_a, goals_b, meta = simulate_goals(lam_a, lam_b, self.mc_config)
        sim = aggregate_outcomes(goals_a, goals_b, lam_a, lam_b, self.mc_config)

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

        result = {
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

        if not include_markets and book_odds is None:
            return result

        if self.betting_config is None:
            raise ValueError("betting_config is required for markets/value output")

        raw = markets_from_samples(goals_a, goals_b, self.betting_config)
        meta["betting_config_version"] = self.betting_config_version
        result["betting_version"] = BETTING_VERSION
        result["simulation_meta"] = meta
        result["markets"] = _decorate_markets(raw)

        if book_odds:
            result["value"] = _value_tree(
                raw, book_odds, reliability, self.betting_config["value"]
            )
        return result
