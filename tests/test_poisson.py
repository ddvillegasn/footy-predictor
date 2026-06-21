import numpy as np
import pandas as pd

from footy.models.poisson import fit_dixon_coles

MODEL_CFG = {
    "xi": 0.0,            # disable decay for a deterministic small-data test
    "max_goals": 10,
    "home_advantage_init": 0.25,
    "ridge": 0.01,
}


def _matches():
    # Brazil clearly stronger than Haiti across repeated matches.
    rows = []
    for i in range(12):
        rows.append(("2019-01-01", "Brazil", "Haiti", 3, 0))
        rows.append(("2019-06-01", "Haiti", "Brazil", 0, 2))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    return df


def test_fit_returns_rates_with_favorite_higher():
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    lam_a, lam_b = model.rates("Brazil", "Haiti", neutral=True)
    assert lam_a > lam_b
    assert lam_a > 0 and lam_b > 0


def test_fit_is_deterministic():
    m = _matches()
    a = fit_dixon_coles(m, MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    b = fit_dixon_coles(m, MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    assert a.rates("Brazil", "Haiti", neutral=True) == b.rates("Brazil", "Haiti", neutral=True)


def test_home_advantage_increases_home_rate():
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    lam_home, _ = model.rates("Brazil", "Haiti", neutral=False)
    lam_neutral, _ = model.rates("Brazil", "Haiti", neutral=True)
    assert lam_home >= lam_neutral


def test_unknown_team_raises():
    import pytest
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    with pytest.raises(KeyError):
        model.rates("Brazil", "Atlantis", neutral=True)
