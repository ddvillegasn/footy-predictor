import pandas as pd
import pytest

from footy.features.leakage import matches_before, assert_no_leakage


def _matches():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
            "home_team": ["Brazil", "Brazil", "Haiti"],
            "away_team": ["Haiti", "Peru", "Brazil"],
        }
    )


def test_matches_before_is_strictly_past():
    m = _matches()
    past = matches_before(m, pd.Timestamp("2019-02-01"))
    assert len(past) == 1
    assert past["date"].max() < pd.Timestamp("2019-02-01")


def test_assert_no_leakage_passes_for_past_only_feature():
    m = _matches()

    def good_feature(history, team, date):
        sub = matches_before(history, date)
        return len(sub)

    # Should not raise.
    assert_no_leakage(good_feature, m)


def test_assert_no_leakage_detects_future_use():
    m = _matches()

    def leaky_feature(history, team, date):
        # Deliberately uses the row at `date` and later -> leakage.
        sub = history[history["date"] >= date]
        return len(sub)

    with pytest.raises(AssertionError, match="leak"):
        assert_no_leakage(leaky_feature, m)
