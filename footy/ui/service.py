from footy.tournament.simulator import simulate_tournaments
from footy.tournament.aggregate import aggregate
from footy.live.scoreboard import scoreboard


def team_list(predictor) -> list:
    """Sorted team names for the dropdowns (the fitted model's keys)."""
    return sorted(predictor.model.attack.keys())


def match_prediction(predictor, team_a, team_b, neutral=False, book_odds=None) -> dict:
    """Wrap predict(). Force include_markets=True whenever book_odds is provided."""
    include_markets = book_odds is not None
    return predictor.predict(team_a, team_b, neutral=neutral,
                             include_markets=include_markets, book_odds=book_odds)


def tournament_probs(structure, results, sampler, n, seed) -> dict:
    """Run N tournaments conditioned on `results`, return aggregated probabilities."""
    sims = simulate_tournaments(structure, results, sampler, n, seed)
    return aggregate(structure, sims)


def live_scoreboard(predictor, played_matches) -> dict:
    """Predicted-vs-actual scoreboard over the played matches."""
    return scoreboard(predictor, played_matches)
