def reliability_label(score: float):
    """0..1 -> (texto humano, clase css)."""
    if score >= 0.70:
        return ("Fiabilidad alta", "alta")
    if score >= 0.45:
        return ("Fiabilidad media", "media")
    return ("Fiabilidad baja", "baja")


def header_html(updated: str, played, model: str) -> str:
    return (
        '<div>'
        '<p class="fty-title">Predicción Mundial 2026</p>'
        '<p class="fty-sub">Probabilidades, goles esperados, simulación del torneo y '
        'valor en cuotas.</p>'
        '<div class="fty-chips">'
        f'<span class="fty-chip">Última actualización: {updated}</span>'
        f'<span class="fty-chip">Partidos jugados: {played}</span>'
        f'<span class="fty-chip">Modelo: {model}</span>'
        '</div></div>'
    )


def prob_cards_html(team_a: str, pa: float, draw: float, team_b: str, pb: float) -> str:
    fav = max((pa, "a"), (draw, "x"), (pb, "b"))[1]

    def card(label, pct, key):
        cls = "fty-card fav" if key == fav else "fty-card"
        return (f'<div class="{cls}"><div class="lbl">{label}</div>'
                f'<div class="pct">{pct:.0f}%</div></div>')

    return ('<div class="fty-cards">'
            + card(f"Gana {team_a}", pa, "a")
            + card("Empate", draw, "x")
            + card(f"Gana {team_b}", pb, "b")
            + '</div>')


def hbars_html(rows) -> str:
    out = ['<div class="fty-bars">']
    for label, pct in rows:
        out.append(
            '<div class="fty-bar-row">'
            f'<div class="fty-bar-lbl">{label}</div>'
            f'<div class="fty-bar-track"><div class="fty-bar-fill" style="width:{pct:.0f}%"></div></div>'
            f'<div class="fty-bar-val">{pct:.0f}%</div>'
            '</div>')
    out.append('</div>')
    return "".join(out)


def value_badge_html(is_value) -> str:
    if is_value is None:
        return '<span class="fty-badge neutro">—</span>'
    if is_value:
        return '<span class="fty-badge valor">Hay valor</span>'
    return '<span class="fty-badge novalor">No hay valor</span>'


def reliability_badge_html(score: float) -> str:
    text, cls = reliability_label(score)
    return f'<span class="fty-badge {cls}">{text}</span>'


def odds_table_html(markets: dict, team_a: str, team_b: str) -> str:
    o = markets["1x2"]
    rows = [
        (f"Gana {team_a}", o["home"]["prob"], o["home"]["fair_odds"]),
        ("Empate", o["draw"]["prob"], o["draw"]["fair_odds"]),
        (f"Gana {team_b}", o["away"]["prob"], o["away"]["fair_odds"]),
    ]
    ou = markets.get("over_under", {}).get("2.5")
    if ou:
        rows.append(("Más de 2.5 goles", ou["over"]["prob"], ou["over"]["fair_odds"]))
        rows.append(("Menos de 2.5 goles", ou["under"]["prob"], ou["under"]["fair_odds"]))
    btts = markets.get("btts", {}).get("yes")
    if btts:
        rows.append(("Ambos marcan", btts["prob"], btts["fair_odds"]))

    html = ['<div class="fty-odds">']
    for name, prob, odds in rows:
        html.append('<div class="fty-odds-row">'
                    f'<span>{name}</span>'
                    f'<span>{prob * 100:.0f}%</span>'
                    f'<span class="fty-odds-cuota">{odds}</span></div>')
    html.append('</div>')
    return "".join(html)
