from dataclasses import dataclass

import pandas as pd

KEY = ["date", "home_team", "away_team"]


@dataclass
class CleanResult:
    df: pd.DataFrame
    report: dict
    dropped: pd.DataFrame


def clean_results(df: pd.DataFrame) -> CleanResult:
    """Remove invalid/duplicate rows, logging every drop with a reason."""
    work = df.copy()
    dropped_frames = []

    # 1. Invalid dates.
    bad_date = work["date"].isna()
    if bad_date.any():
        rec = work[bad_date].copy()
        rec["drop_reason"] = "invalid_date"
        dropped_frames.append(rec)
        work = work[~bad_date]

    # 2. Null scores (cannot train on them).
    null_score = work["home_score"].isna() | work["away_score"].isna()
    if null_score.any():
        rec = work[null_score].copy()
        rec["drop_reason"] = "null_score"
        dropped_frames.append(rec)
        work = work[~null_score]

    # 3. Exact duplicates on (date, home, away): keep first, log the rest.
    dup_mask = work.duplicated(subset=KEY, keep="first")
    if dup_mask.any():
        rec = work[dup_mask].copy()
        rec["drop_reason"] = "duplicate"
        dropped_frames.append(rec)
        work = work[~dup_mask]

    work = work.sort_values("date").reset_index(drop=True)
    work["home_score"] = work["home_score"].astype(int)
    work["away_score"] = work["away_score"].astype(int)

    dropped = (
        pd.concat(dropped_frames, ignore_index=True)
        if dropped_frames
        else pd.DataFrame(columns=list(df.columns) + ["drop_reason"])
    )
    report = {
        "rows_in": int(len(df)),
        "rows_out": int(len(work)),
        "invalid_dates_removed": int((dropped.get("drop_reason") == "invalid_date").sum())
        if len(dropped) else 0,
        "null_scores_removed": int((dropped.get("drop_reason") == "null_score").sum())
        if len(dropped) else 0,
        "duplicates_removed": int((dropped.get("drop_reason") == "duplicate").sum())
        if len(dropped) else 0,
    }
    return CleanResult(df=work, report=report, dropped=dropped)
