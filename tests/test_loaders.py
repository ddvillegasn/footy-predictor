from pathlib import Path

import pandas as pd
import pytest

from footy.data.loaders import load_results, load_former_names

FIX = Path(__file__).parent / "fixtures"


def test_load_results_types():
    df = load_results(FIX / "results.csv")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["home_score"].dtype.kind == "i"
    assert df["neutral"].dtype == bool
    assert df.loc[0, "neutral"] is True or bool(df.loc[0, "neutral"]) is True
    assert len(df) == 6


def test_load_results_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("date,home_team\n2019-01-01,Brazil\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_results(bad)


def test_load_former_names():
    df = load_former_names(FIX / "former_names.csv")
    assert list(df.columns) == ["current", "former", "start_date", "end_date"]
    assert df.loc[0, "current"] == "Haiti"
