import pandas as pd


def matches_before(history: pd.DataFrame, date) -> pd.DataFrame:
    """Return only rows strictly before `date` (anti-leakage window)."""
    date = pd.Timestamp(date)
    return history[history["date"] < date]


def assert_no_leakage(feature_fn, history: pd.DataFrame) -> None:
    """Golden guard: a feature must produce identical output whether it sees
    the full history or only rows strictly before the match date.

    If appending future rows changes the feature value, the function used
    information from the present/future and leaks. Raises AssertionError.
    """
    history = history.sort_values("date").reset_index(drop=True)
    for row in history.itertuples(index=False):
        past = matches_before(history, row.date)
        value_past = feature_fn(past, row.home_team, row.date)
        value_full = feature_fn(history, row.home_team, row.date)
        assert value_past == value_full, (
            f"feature leak detected for {row.home_team} at {row.date}: "
            f"past={value_past} full={value_full}"
        )
