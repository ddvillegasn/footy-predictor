import pandas as pd

from footy.features.strength import recent_form, head_to_head
from footy.features.leakage import assert_no_leakage


def _history():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2019-01-01", "2019-02-01", "2019-03-01", "2019-04-01"]
            ),
            "home_team": ["Brazil", "Haiti", "Brazil", "Peru"],
            "away_team": ["Haiti", "Brazil", "Haiti", "Brazil"],
            "home_score": [3, 1, 2, 0],
            "away_score": [0, 1, 0, 4],
        }
    )


def test_recent_form_counts_only_past():
    h = _history()
    form = recent_form(h, "Brazil", pd.Timestamp("2019-03-01"), window=5)
    # Brazil before 2019-03-01: won 3-0, drew 1-1 (as away). 2 matches.
    assert form["matches"] == 2
    assert form["goals_for"] == 4      # 3 + 1
    assert form["goals_against"] == 1  # 0 + 1
    assert form["clean_sheets"] == 1


def test_recent_form_no_leakage():
    h = _history()
    assert_no_leakage(
        lambda hist, team, date: recent_form(hist, team, date, window=5)["matches"],
        h,
    )


def test_head_to_head_directional_counts():
    h = _history()
    h2h = head_to_head(h, "Brazil", "Haiti", pd.Timestamp("2019-04-01"))
    # Brazil vs Haiti before 2019-04-01: 3-0 (BRA win), 1-1 (draw), 2-0 (BRA win)
    assert h2h["wins"] == 2
    assert h2h["draws"] == 1
    assert h2h["losses"] == 0
    assert h2h["goals_for"] == 6
    assert h2h["goals_against"] == 1
