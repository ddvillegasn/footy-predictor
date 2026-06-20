from pathlib import Path

import pandas as pd

REQUIRED = {"date", "team", "rank"}


def load_fifa_ranking(path) -> pd.DataFrame | None:
    """Load cached FIFA ranking CSV. Best-effort: missing/invalid -> None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if not REQUIRED.issubset(df.columns):
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None
