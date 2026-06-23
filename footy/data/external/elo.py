import pandas as pd


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _outcome(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def attach_elo(matches: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add home_elo_pre / away_elo_pre columns (rating BEFORE each match).

    Ratings are updated chronologically; the value written to a row is the
    pre-match rating, so no row ever sees its own result (anti-leakage).
    """
    init = float(config["initial_rating"])
    k = float(config["k_factor"])
    home_adv = float(config["home_advantage_elo"])
    weights = config.get("tournament_weights", {})
    default_w = float(config.get("default_tournament_weight", 1.0))

    df = matches.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    home_pre = []
    away_pre = []

    for row in df.itertuples(index=False):
        ra = ratings.get(row.home_team, init)
        rb = ratings.get(row.away_team, init)
        home_pre.append(ra)
        away_pre.append(rb)

        adv = 0.0 if bool(row.neutral) else home_adv
        exp_home = _expected(ra + adv, rb)
        score_home = _outcome(int(row.home_score), int(row.away_score))
        weight = float(weights.get(row.tournament, default_w))
        delta = k * weight * (score_home - exp_home)
        ratings[row.home_team] = ra + delta
        ratings[row.away_team] = rb - delta

    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre
    return df


def final_ratings(matches, config: dict) -> dict:
    """Elo rating of every team AFTER processing all matches chronologically."""
    init = float(config["initial_rating"])
    k = float(config["k_factor"])
    home_adv = float(config["home_advantage_elo"])
    weights = config.get("tournament_weights", {})
    default_w = float(config.get("default_tournament_weight", 1.0))

    df = matches.sort_values("date")
    ratings: dict = {}
    for row in df.itertuples(index=False):
        ra = ratings.get(row.home_team, init)
        rb = ratings.get(row.away_team, init)
        adv = 0.0 if bool(row.neutral) else home_adv
        exp_home = 1.0 / (1.0 + 10.0 ** ((rb - (ra + adv)) / 400.0))
        if row.home_score > row.away_score:
            score = 1.0
        elif row.home_score < row.away_score:
            score = 0.0
        else:
            score = 0.5
        weight = float(weights.get(row.tournament, default_w))
        delta = k * weight * (score - exp_home)
        ratings[row.home_team] = ra + delta
        ratings[row.away_team] = rb - delta
    return ratings
