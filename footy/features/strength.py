import pandas as pd

from footy.features.leakage import matches_before


def _team_matches(history: pd.DataFrame, team: str) -> pd.DataFrame:
    return history[(history["home_team"] == team) | (history["away_team"] == team)]


def recent_form(history: pd.DataFrame, team: str, date, window: int) -> dict:
    """Goals for/against, clean sheets over the team's last `window` matches
    strictly before `date`."""
    past = matches_before(history, date)
    team_rows = _team_matches(past, team).sort_values("date").tail(window)

    goals_for = goals_against = clean_sheets = 0
    for row in team_rows.itertuples(index=False):
        if row.home_team == team:
            gf, ga = int(row.home_score), int(row.away_score)
        else:
            gf, ga = int(row.away_score), int(row.home_score)
        goals_for += gf
        goals_against += ga
        if ga == 0:
            clean_sheets += 1

    return {
        "matches": int(len(team_rows)),
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
        "clean_sheets": int(clean_sheets),
    }


def head_to_head(history: pd.DataFrame, team_a: str, team_b: str, date) -> dict:
    """Directional H2H record of team_a vs team_b strictly before `date`."""
    past = matches_before(history, date)
    pair = past[
        ((past["home_team"] == team_a) & (past["away_team"] == team_b))
        | ((past["home_team"] == team_b) & (past["away_team"] == team_a))
    ]

    wins = draws = losses = goals_for = goals_against = 0
    for row in pair.itertuples(index=False):
        if row.home_team == team_a:
            ga_for, ga_against = int(row.home_score), int(row.away_score)
        else:
            ga_for, ga_against = int(row.away_score), int(row.home_score)
        goals_for += ga_for
        goals_against += ga_against
        if ga_for > ga_against:
            wins += 1
        elif ga_for < ga_against:
            losses += 1
        else:
            draws += 1

    return {
        "matches": int(len(pair)),
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
    }
