import pandas as pd

# Minimal confederation map; extend via data file in a later sub-project.
_CONTINENT = {
    "Brazil": "South America", "Argentina": "South America", "Peru": "South America",
    "Uruguay": "South America", "Colombia": "South America", "Chile": "South America",
    "Haiti": "North America", "United States": "North America", "Mexico": "North America",
    "Canada": "North America", "Jamaica": "North America",
    "France": "Europe", "Germany": "Europe", "Spain": "Europe", "England": "Europe",
    "Italy": "Europe", "Portugal": "Europe", "Netherlands": "Europe", "Belgium": "Europe",
    "Nigeria": "Africa", "Ghana": "Africa", "Egypt": "Africa", "Senegal": "Africa",
    "Cameroon": "Africa", "Morocco": "Africa",
    "Japan": "Asia", "South Korea": "Asia", "Iran": "Asia", "Saudi Arabia": "Asia",
    "Qatar": "Asia", "Australia": "Asia",
}


def tournament_weight(tournament: str, weights: dict, default: float) -> float:
    return float(weights.get(tournament, default))


def continent_of(team: str) -> str:
    return _CONTINENT.get(team, "Unknown")


def last_ranking_before(ranking: pd.DataFrame | None, team: str, date) -> int | None:
    """Most recent rank strictly before `date` (merge_asof backward semantics).
    Best-effort: missing table or no prior record -> None. Never future data.
    """
    if ranking is None or len(ranking) == 0:
        return None
    date = pd.Timestamp(date)
    sub = ranking[(ranking["team"] == team) & (ranking["date"] < date)]
    if len(sub) == 0:
        return None
    latest = sub.sort_values("date").iloc[-1]
    return int(latest["rank"])
