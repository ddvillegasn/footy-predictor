import pandas as pd

from footy.predict import Predictor
from footy.tournament.simulator import simulate_tournaments
from footy.tournament.aggregate import aggregate
from footy.live.scoreboard import scoreboard


def team_list(predictor) -> list:
    """Sorted team names for the dropdowns (the fitted model's keys)."""
    return sorted(predictor.model.attack.keys())


def match_prediction(predictor, team_a, team_b, neutral=False, book_odds=None) -> dict:
    """Wrap predict(); markets are always included for the UI. Value/EV needs book_odds."""
    return predictor.predict(team_a, team_b, neutral=neutral,
                             include_markets=True, book_odds=book_odds)


def tournament_probs(structure, results, sampler, n, seed) -> dict:
    """Run N tournaments conditioned on `results`, return aggregated probabilities."""
    sims = simulate_tournaments(structure, results, sampler, n, seed)
    return aggregate(structure, sims)


def live_scoreboard(predictor, played_matches) -> dict:
    """Predicted-vs-actual scoreboard over the played matches."""
    return scoreboard(predictor, played_matches)


def build_live_predictor(base_predictor, played_matches, tournament_date,
                         model_config, mc_config):
    """LIVE model: base dataset + played WC matches (recent date, neutral) refit.
    Reacts to tournament form (lightly, by design). Used for predictions, NOT scoreboard."""
    base = base_predictor.matches.copy()
    rows = [{"date": pd.Timestamp(tournament_date),
             "home_team": pm["team_a"], "away_team": pm["team_b"],
             "home_score": pm["goals_a"], "away_score": pm["goals_b"],
             "tournament": "FIFA World Cup", "neutral": True} for pm in played_matches]
    live_matches = pd.concat([base, pd.DataFrame(rows)], ignore_index=True) if rows else base
    as_of = live_matches["date"].max() + pd.Timedelta(days=1)
    return Predictor.from_matches(live_matches, model_config=model_config, mc_config=mc_config,
                                  canonical=base_predictor.canonical, as_of=as_of)
