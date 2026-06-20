from pathlib import Path

import pandas as pd

RESULTS_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]
GOALSCORERS_COLUMNS = [
    "date", "home_team", "away_team", "team", "scorer", "minute",
    "own_goal", "penalty",
]
SHOOTOUTS_COLUMNS = ["date", "home_team", "away_team", "winner", "first_shooter"]
FORMER_NAMES_COLUMNS = ["current", "former", "start_date", "end_date"]


def _require_columns(df: pd.DataFrame, columns: list[str], source) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "TRUE": True, "FALSE": False, "True": True, "False": False,
        True: True, False: False,
    }
    return series.map(mapping).astype(bool)


def load_results(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, RESULTS_COLUMNS, path.name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    df["neutral"] = _to_bool(df["neutral"])
    # Keep only rows whose scores are present, then downcast to plain int.
    valid = df["home_score"].notna() & df["away_score"].notna()
    df.loc[valid, "home_score"] = df.loc[valid, "home_score"]
    df["home_score"] = df["home_score"].astype("Int64")
    return df.reset_index(drop=True)


def load_goalscorers(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, GOALSCORERS_COLUMNS, path.name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.reset_index(drop=True)


def load_shootouts(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, SHOOTOUTS_COLUMNS, path.name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.reset_index(drop=True)


def load_former_names(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, FORMER_NAMES_COLUMNS, path.name)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df.reset_index(drop=True)
