import pandas as pd

from footy.features.context import tournament_weight, continent_of, last_ranking_before


def test_tournament_weight_uses_config_and_default():
    weights = {"FIFA World Cup": 1.0, "Friendly": 0.4}
    assert tournament_weight("FIFA World Cup", weights, default=0.7) == 1.0
    assert tournament_weight("Unknown Cup", weights, default=0.7) == 0.7


def test_continent_lookup_defaults_unknown():
    assert continent_of("Brazil") == "South America"
    assert continent_of("Atlantis") == "Unknown"


def test_last_ranking_before_is_backward_only():
    ranking = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            "team": ["Brazil", "Brazil", "Brazil"],
            "rank": [3, 2, 1],
        }
    )
    # On 2021-06-01 the most recent *prior* rank is the 2021-01-01 one (2).
    assert last_ranking_before(ranking, "Brazil", pd.Timestamp("2021-06-01")) == 2
    # Before any record -> None.
    assert last_ranking_before(ranking, "Brazil", pd.Timestamp("2019-06-01")) is None
    # Missing ranking table -> None.
    assert last_ranking_before(None, "Brazil", pd.Timestamp("2021-06-01")) is None
