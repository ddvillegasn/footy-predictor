from pathlib import Path

import pandas as pd

REQUIRED = {"team", "market_value"}


def load_market_values(path) -> pd.DataFrame | None:
    """Load cached Transfermarkt values. Best-effort: any problem -> None.

    Never raises; the pipeline must continue without this signal.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if not REQUIRED.issubset(df.columns):
            return None
        return df.reset_index(drop=True)
    except Exception:
        return None
