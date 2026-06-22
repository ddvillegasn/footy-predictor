from footy.ui.components import (reliability_label, header_html, prob_cards_html,
                                 hbars_html, value_badge_html, reliability_badge_html,
                                 odds_table_html)


def test_reliability_label_thresholds():
    assert reliability_label(0.80) == ("Fiabilidad alta", "alta")
    assert reliability_label(0.55) == ("Fiabilidad media", "media")
    assert reliability_label(0.20) == ("Fiabilidad baja", "baja")


def test_prob_cards_highlight_favorite_and_labels():
    html = prob_cards_html("Brasil", 85.0, 9.0, "Haití", 6.0)
    assert "Gana Brasil" in html and "Gana Haití" in html and "Empate" in html
    assert "85%" in html
    # favorite card (Brasil) gets the 'fav' class exactly once here
    assert html.count("fty-card fav") == 1


def test_hbars_width_proportional():
    html = hbars_html([("Brasil", 85.0), ("Empate", 9.0)])
    assert "width:85%" in html and "width:9%" in html


def test_value_badge_three_states():
    assert "Hay valor" in value_badge_html(True)
    assert "No hay valor" in value_badge_html(False)
    assert "neutro" in value_badge_html(None)


def test_reliability_badge_uses_class():
    assert "fty-badge alta" in reliability_badge_html(0.9)


def test_header_html_shows_status():
    html = header_html("hoy 14:00", 40, "BASE")
    assert "Predicción Mundial 2026" in html
    assert "40" in html and "BASE" in html


def test_odds_table_uses_team_names_and_markets():
    markets = {
        "1x2": {"home": {"prob": 0.7, "fair_odds": 1.43},
                "draw": {"prob": 0.2, "fair_odds": 5.0},
                "away": {"prob": 0.1, "fair_odds": 10.0}},
        "over_under": {"2.5": {"over": {"prob": 0.6, "fair_odds": 1.67},
                               "under": {"prob": 0.4, "fair_odds": 2.5}}},
        "btts": {"yes": {"prob": 0.5, "fair_odds": 2.0}},
    }
    html = odds_table_html(markets, "Brasil", "Haití")
    assert "Gana Brasil" in html and "Gana Haití" in html
    assert "Más de 2.5 goles" in html and "Ambos marcan" in html
    assert "1.43" in html
