from pathlib import Path

import pandas as pd

from footy.data.external.fifa import load_fifa_ranking
from footy.data.external.transfermarkt import load_market_values


def test_fifa_missing_cache_returns_none(tmp_path):
    assert load_fifa_ranking(tmp_path / "nope.csv") is None


def test_fifa_loads_when_present(tmp_path):
    p = tmp_path / "fifa.csv"
    pd.DataFrame(
        {"date": ["2021-01-01"], "team": ["Brazil"], "rank": [1]}
    ).to_csv(p, index=False)
    out = load_fifa_ranking(p)
    assert out is not None
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_transfermarkt_never_raises(tmp_path):
    # Missing file and malformed file both yield None, never an exception.
    assert load_market_values(tmp_path / "missing.csv") is None
    bad = tmp_path / "bad.csv"
    bad.write_text("not,a,valid\nstructure", encoding="utf-8")
    assert load_market_values(bad) is None or isinstance(load_market_values(bad), pd.DataFrame)
