from pathlib import Path

from footy.data.loaders import load_results
from footy.data.clean import clean_results

FIX = Path(__file__).parent / "fixtures"


def test_clean_dedups_and_reports():
    raw = load_results(FIX / "results.csv")  # contains 1 exact duplicate
    result = clean_results(raw)
    # 6 raw rows, one duplicated 2021-06-01 Brazil-Haiti pair -> 5 kept
    assert len(result.df) == 5
    assert result.report["duplicates_removed"] == 1
    assert len(result.dropped) == 1
    assert result.dropped.iloc[0]["drop_reason"] == "duplicate"


def test_clean_drops_null_scores_with_reason():
    raw = load_results(FIX / "results.csv").copy()
    raw.loc[0, "home_score"] = None
    result = clean_results(raw)
    reasons = set(result.dropped["drop_reason"])
    assert "null_score" in reasons
    assert result.report["null_scores_removed"] == 1


def test_clean_is_deterministic():
    raw = load_results(FIX / "results.csv")
    a = clean_results(raw)
    b = clean_results(raw)
    assert a.df.reset_index(drop=True).equals(b.df.reset_index(drop=True))
