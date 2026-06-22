# Dashboard Redesign Implementation Plan (SP7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Streamlit dashboard into a clean, dark, sports-broadcast UI in natural Spanish, with a match tab that answers instantly, a tournament tab, a betting tab (shared selection), and a plain-language scoreboard — UI only, model untouched.

**Architecture:** Testable HTML/CSS builders in `footy/ui/styles.py` and `footy/ui/components.py` (pure functions returning strings); a dark base theme in `.streamlit/config.toml`; `app/streamlit_app.py` rewritten as 4 tabs that inject the CSS and render the components, sharing the selected match via `st.session_state`. No model/service changes.

**Tech Stack:** Python 3.10.6, Streamlit, pandas, pytest.

**Conventions:** branch `feature/baseline-v1`. TDD: failing test first. Commit per task, `git add` ONLY named files (never `__pycache__`/`.pyc`). Commit trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## File Structure

| File | Responsibility |
|---|---|
| `.streamlit/config.toml` | dark base theme |
| `footy/ui/styles.py` | `CSS` string (design classes) |
| `footy/ui/components.py` | HTML builders + `reliability_label` |
| `app/streamlit_app.py` | 4-tab render using components + session_state |
| `tests/test_ui_components.py` | builder tests |

---

## Task 1: dark theme + styles.py

**Files:**
- Create: `.streamlit/config.toml`, `footy/ui/styles.py`
- Test: `tests/test_ui_styles.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ui_styles.py`:
```python
from footy.ui.styles import CSS


def test_css_defines_core_classes():
    for cls in [".fty-card", ".fty-card.fav", ".fty-bar-fill", ".fty-chip",
                ".fty-badge.alta", ".fty-badge.valor", ".fty-badge.novalor",
                ".fty-odds-row"]:
        assert cls in CSS
    assert CSS.strip().startswith("<style>") and CSS.strip().endswith("</style>")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui_styles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.ui.styles'`

- [ ] **Step 3: Write implementation**

`.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#27c281"
backgroundColor = "#0e1420"
secondaryBackgroundColor = "#161d2e"
textColor = "#e8edf4"
font = "sans serif"
```

`footy/ui/styles.py`:
```python
CSS = """
<style>
.fty-title {font-size:1.9rem; font-weight:800; color:#e8edf4; margin:0;}
.fty-sub {color:#9fb0c3; font-size:0.95rem; margin:2px 0 10px;}
.fty-chips {display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;}
.fty-chip {background:#161d2e; border:1px solid #233047; color:#cdd9e6;
           border-radius:999px; padding:4px 12px; font-size:0.82rem;}
.fty-cards {display:flex; gap:14px; margin:12px 0;}
.fty-card {flex:1; background:#161d2e; border:1px solid #233047; border-radius:14px;
           padding:18px 14px; text-align:center;}
.fty-card.fav {border-color:#27c281; box-shadow:0 0 0 1px #27c281 inset;}
.fty-card .lbl {color:#9fb0c3; font-size:0.85rem; text-transform:uppercase; letter-spacing:.04em;}
.fty-card .pct {color:#e8edf4; font-size:2.1rem; font-weight:800; margin-top:6px;}
.fty-bars {margin:10px 0;}
.fty-bar-row {display:flex; align-items:center; gap:10px; margin:6px 0;}
.fty-bar-lbl {width:130px; color:#cdd9e6; font-size:0.9rem;}
.fty-bar-track {flex:1; background:#0e1420; border:1px solid #233047; border-radius:8px;
                height:18px; overflow:hidden;}
.fty-bar-fill {height:100%; background:linear-gradient(90deg,#27c281,#1f9e6b);}
.fty-bar-val {width:46px; text-align:right; color:#e8edf4; font-size:0.9rem;}
.fty-line {color:#cdd9e6; font-size:1.0rem; margin:8px 0;}
.fty-badge {display:inline-block; border-radius:999px; padding:4px 12px;
            font-weight:700; font-size:0.85rem;}
.fty-badge.alta {background:#10311f; color:#27c281; border:1px solid #27c281;}
.fty-badge.media {background:#33270e; color:#e0b341; border:1px solid #e0b341;}
.fty-badge.baja {background:#3a1620; color:#ff5a7a; border:1px solid #ff5a7a;}
.fty-badge.valor {background:#10311f; color:#27c281; border:1px solid #27c281;}
.fty-badge.novalor {background:#3a1620; color:#ff5a7a; border:1px solid #ff5a7a;}
.fty-badge.neutro {background:#1b2333; color:#9fb0c3; border:1px solid #2a3550;}
.fty-odds {background:#161d2e; border:1px solid #233047; border-radius:12px; padding:6px 12px;}
.fty-odds-row {display:flex; justify-content:space-between; gap:12px;
               padding:8px 2px; border-bottom:1px solid #1d2740; color:#cdd9e6;}
.fty-odds-row:last-child {border-bottom:none;}
.fty-odds-cuota {color:#27c281; font-weight:700;}
</style>
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui_styles.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add .streamlit/config.toml footy/ui/styles.py tests/test_ui_styles.py
git commit -m "feat: dark sports theme + dashboard CSS

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: components.py (HTML builders)

**Files:**
- Create: `footy/ui/components.py`
- Test: `tests/test_ui_components.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ui_components.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui_components.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.ui.components'`

- [ ] **Step 3: Write implementation**

`footy/ui/components.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui_components.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/ui/components.py tests/test_ui_components.py
git commit -m "feat: dashboard HTML components (cards, bars, badges, odds)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: rewrite the app (4 tabs, session_state, components)

**Files:**
- Modify: `app/streamlit_app.py`
- Test: `tests/test_streamlit_app_imports.py` (must keep passing)

- [ ] **Step 1: Confirm the smoke test passes today**

Run: `python -m pytest tests/test_streamlit_app_imports.py -v`
Expected: PASS (1 passed)

- [ ] **Step 2: Rewrite `app/streamlit_app.py`**

```python
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from footy.cli import _build_default_predictor
from footy.config import load_config, config_fingerprint
from footy.persist import load_or_build, files_fingerprint
from footy.tournament.structure import load_structure
from footy.tournament.results import load_results, TournamentResults
from footy.tournament.sampler import MatchSampler
from footy.live.provider import FootballDataProvider
from footy.live.name_map import load_name_map
from footy.live.ingest import ingest
from footy.live.structure_sync import sync_structure
from footy.live.stats import team_stats
from footy.ui.service import (team_list, match_prediction, tournament_probs,
                              live_scoreboard, build_live_predictor)
from footy.ui.styles import CSS
from footy.ui import components as C

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "configs" / "tournaments" / "wc2026_results.yaml"
STRUCTURE_PATH = ROOT / "configs" / "tournaments" / "wc2026.yaml"


@st.cache_resource
def build_engine():
    predictor = _build_default_predictor()
    structure = load_structure(STRUCTURE_PATH)
    canon = predictor.canonical
    for g, teams in structure.groups.items():
        structure.groups[g] = [canon(t) for t in teams]
    sampler = MatchSampler(predictor.model, load_config("montecarlo"),
                           model_version=load_config("model")["model_version"],
                           config_hash=config_fingerprint("montecarlo"))
    return predictor, structure, sampler


@st.cache_resource
def build_live(_base_predictor, results_token):
    if not RESULTS_PATH.exists():
        return _base_predictor
    structure = _current_structure(_base_predictor)
    try:
        results = load_results(RESULTS_PATH, structure.groups)
    except ValueError:
        return _base_predictor
    played = [{"team_a": pm.team_a, "team_b": pm.team_b,
               "goals_a": pm.goals_a, "goals_b": pm.goals_b} for pm in results.played]
    if not played:
        return _base_predictor
    fingerprint = files_fingerprint([RESULTS_PATH, "configs/model.yaml"]) + "-live-v2"
    return load_or_build(
        "artifacts/live_predictor.pkl", fingerprint,
        lambda: build_live_predictor(_base_predictor, played, tournament_date="2026-06-15",
                                     model_config=load_config("model"),
                                     mc_config=load_config("montecarlo")))


def _results_token() -> str:
    return str(RESULTS_PATH.stat().st_mtime) if RESULTS_PATH.exists() else "none"


def _current_structure(base_predictor):
    structure = load_structure(STRUCTURE_PATH)
    canon = base_predictor.canonical
    for g, teams in structure.groups.items():
        structure.groups[g] = [canon(t) for t in teams]
    return structure


def _played_dicts(base_predictor):
    if not RESULTS_PATH.exists():
        return []
    structure = _current_structure(base_predictor)
    try:
        results = load_results(RESULTS_PATH, structure.groups)
    except ValueError:
        return []
    return [{"team_a": pm.team_a, "team_b": pm.team_b,
             "goals_a": pm.goals_a, "goals_b": pm.goals_b} for pm in results.played]


def _data_status(base_predictor):
    played = len(_played_dicts(base_predictor))
    if RESULTS_PATH.exists():
        updated = datetime.fromtimestamp(RESULTS_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    else:
        updated = "—"
    return updated, played


def _refresh_from_api(base_predictor):
    try:
        provider = FootballDataProvider.from_config(load_config("live"))
        name_map = load_name_map("configs/name_map.yaml")
        known = set(base_predictor.model.attack.keys())
        sync_structure(provider, name_map, known, STRUCTURE_PATH)
        structure = _current_structure(base_predictor)
        n = ingest(provider, structure, name_map, load_config("live")["stage_map"], RESULTS_PATH)
        st.success(f"Actualizado desde la API: {n} partidos jugados.")
        st.cache_resource.clear()
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo actualizar desde la API: {exc}")


def _get_predictor(base_predictor, use_live):
    if use_live:
        with st.spinner("Ajustando modelo LIVE (solo la primera vez)…"):
            return build_live(base_predictor, _results_token())
    return base_predictor


def _render_match_tab(base_predictor):
    teams = team_list(base_predictor)
    ss = st.session_state
    ss.setdefault("team_a", "Brazil" if "Brazil" in teams else teams[0])
    ss.setdefault("team_b", "Argentina" if "Argentina" in teams else teams[1])
    ss.setdefault("neutral", True)
    ss.setdefault("use_live", False)

    c1, c2 = st.columns(2)
    ss.team_a = c1.selectbox("Equipo A", teams, index=teams.index(ss.team_a))
    ss.team_b = c2.selectbox("Equipo B", teams, index=teams.index(ss.team_b))
    ss.neutral = st.checkbox("Cancha neutral", value=ss.neutral)
    ss.use_live = st.toggle("Histórico + Mundial actual", value=ss.use_live,
                            help="El modelo LIVE ajusta ligeramente la fuerza según los "
                                 "partidos ya jugados del Mundial.")
    st.caption("Modelo: **Histórico + Mundial actual (LIVE)**" if ss.use_live
               else "Modelo: **Histórico reciente (BASE)**")

    predictor = _get_predictor(base_predictor, ss.use_live)
    try:
        out = match_prediction(predictor, ss.team_a, ss.team_b, neutral=ss.neutral)
    except ValueError as exc:
        st.error(str(exc))
        return
    ss.last_prediction = out

    st.markdown(C.prob_cards_html(ss.team_a, out["team_a_win"], out["draw"],
                                  ss.team_b, out["team_b_win"]), unsafe_allow_html=True)
    st.markdown(C.hbars_html([(f"Gana {ss.team_a}", out["team_a_win"]),
                              ("Empate", out["draw"]),
                              (f"Gana {ss.team_b}", out["team_b_win"])]),
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="fty-line">Goles esperados: <b>{ss.team_a} {out["expected_goals_a"]}</b> '
        f'– <b>{out["expected_goals_b"]} {ss.team_b}</b></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="fty-line">Marcador probable: <b>{out["most_likely_score"]}</b> &nbsp; '
        + C.reliability_badge_html(out["prediction_reliability"]) + '</div>',
        unsafe_allow_html=True)


def _render_betting_tab(base_predictor):
    ss = st.session_state
    if "team_a" not in ss:
        st.info("Elige primero un partido en la pestaña **Partido**.")
        return
    st.caption(f"Partido: **{ss.team_a} vs {ss.team_b}** "
               f"({'neutral' if ss.get('neutral', True) else 'con localía'})")
    predictor = _get_predictor(base_predictor, ss.get("use_live", False))
    o1, o2, o3 = st.columns(3)
    odd_home = o1.number_input(f"Cuota {ss.team_a}", min_value=0.0, value=0.0, step=0.05)
    odd_draw = o2.number_input("Cuota empate", min_value=0.0, value=0.0, step=0.05)
    odd_away = o3.number_input(f"Cuota {ss.team_b}", min_value=0.0, value=0.0, step=0.05)
    one = {}
    if odd_home > 1.0:
        one["home"] = odd_home
    if odd_draw > 1.0:
        one["draw"] = odd_draw
    if odd_away > 1.0:
        one["away"] = odd_away
    book = {"1x2": one} if one else None
    try:
        out = match_prediction(predictor, ss.team_a, ss.team_b,
                               neutral=ss.get("neutral", True), book_odds=book)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.subheader("Cuotas justas del modelo")
    st.markdown(C.odds_table_html(out["markets"], ss.team_a, ss.team_b), unsafe_allow_html=True)
    if "value" in out and out["value"].get("1x2"):
        st.subheader("Valor vs tus cuotas")
        names = {"home": f"Gana {ss.team_a}", "draw": "Empate", "away": f"Gana {ss.team_b}"}
        for k, v in out["value"]["1x2"].items():
            st.markdown(
                f'<div class="fty-line">{names[k]} · cuota {v["book_odds"]} · '
                f'EV {v["edge_pct"]}% &nbsp; ' + C.value_badge_html(v["is_value"]) + '</div>',
                unsafe_allow_html=True)


def _render_tournament_tab(base_predictor):
    structure = _current_structure(base_predictor)
    try:
        results = (load_results(RESULTS_PATH, structure.groups)
                   if RESULTS_PATH.exists() else TournamentResults([]))
    except ValueError:
        results = TournamentResults([])

    mode = st.radio("Simulación", ["Histórico (BASE)", "Histórico + Mundial (LIVE)"],
                    horizontal=True)
    n = st.slider("Número de torneos a simular", 500, 5000, 1000, 500)
    if st.button("Simular Mundial", type="primary"):
        if mode.startswith("Histórico +"):
            live = _get_predictor(base_predictor, True)
            model = live.model
        else:
            model = base_predictor.model
        sampler = MatchSampler(model, load_config("montecarlo"),
                               model_version=load_config("model")["model_version"],
                               config_hash=config_fingerprint("montecarlo"))
        with st.spinner(f"Simulando {n} torneos…"):
            agg = tournament_probs(structure, results, sampler, n, seed=42)
        st.session_state.agg = agg

    agg = st.session_state.get("agg")
    if not agg:
        st.info("Aprieta **Simular Mundial** para ver probabilidades.")
        return

    stats = team_stats(structure, results) if results.played else {}
    group_opts = ["Todos"] + sorted(structure.groups.keys())
    group = st.selectbox("Filtrar por grupo", group_opts)
    if group == "Todos":
        teams = [t for g in structure.groups.values() for t in g]
    else:
        teams = structure.groups[group]

    rows = []
    for t in teams:
        d = agg["teams"].get(t, {})
        s = stats.get(t, {})
        rows.append({
            "Equipo": t,
            "Clasificar %": round(d.get("advance_group", 0) * 100, 1),
            "Campeón %": round(d.get("champion", 0) * 100, 1),
            "Puntos": s.get("points", 0),
            "Forma": "".join(s.get("form", [])[-5:]),
        })
    df = pd.DataFrame(rows).sort_values("Campeón %", ascending=False)
    st.dataframe(df, width="stretch", hide_index=True)


def _render_scoreboard_tab(base_predictor):
    st.caption("Mide qué tan bien predijo el modelo en partidos ya jugados del Mundial "
               "(modelo histórico, sin hacer trampa con lo ya visto).")
    played = _played_dicts(base_predictor)
    if not played:
        st.info("Aún no hay resultados cargados. Usa **Actualizar desde API**.")
        return
    board = live_scoreboard(base_predictor, played)
    c1, c2, c3 = st.columns(3)
    c1.metric("Aciertos", f"{round((board['accuracy'] or 0) * 100)}%")
    c2.metric("Error de goles (medio)", board["goal_mae"])
    c3.metric("Partidos evaluados", board["n"])
    st.subheader("Predicho vs real (últimos)")
    rows = [{"Partido": m["match"], "Predicho": m["predicted_score"],
             "Real": m["actual_score"], "Acierto": "Sí" if m["hit"] else "No"}
            for m in board["matches"][-12:]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with st.expander("Métricas avanzadas"):
        st.write(f"Log loss: {board['log_loss']} · Brier: {board['brier']}")


def main():
    st.set_page_config(page_title="Predicción Mundial 2026", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.spinner("Cargando modelo (solo la primera vez)…"):
        base_predictor, _structure, _sampler = build_engine()

    updated, played = _data_status(base_predictor)
    model_label = "LIVE" if st.session_state.get("use_live") else "BASE"
    st.markdown(C.header_html(updated, played, model_label), unsafe_allow_html=True)

    with st.sidebar:
        st.header("Datos en vivo")
        if st.button("🔄 Actualizar desde API"):
            _refresh_from_api(base_predictor)
        st.caption("El modelo reacciona poco a un resultado suelto — es correcto. "
                   "Las cuotas/valor dependen del modelo; no son garantía.")

    t1, t2, t3, t4 = st.tabs(["Partido", "Mundial", "Apuestas", "Scoreboard"])
    with t1:
        _render_match_tab(base_predictor)
    with t2:
        _render_tournament_tab(base_predictor)
    with t3:
        _render_betting_tab(base_predictor)
    with t4:
        _render_scoreboard_tab(base_predictor)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the smoke test**

Run: `python -m pytest tests/test_streamlit_app_imports.py -v`
Expected: PASS (1 passed — module imports, `main`/`build_engine` present)

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: all PASS (smoke ~2 min; model unchanged).

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: redesigned 4-tab dashboard (dark, cards, shared selection)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: visual verification

- [ ] **Step 1: Render a static preview of the components**

Create a throwaway script that writes `artifacts/ui_preview.html` = `styles.CSS` plus a sample
`header_html` + `prob_cards_html` + `hbars_html` + `reliability_badge_html` + `odds_table_html`
on a `#0e1420` page, to eyeball the real styled output:

```python
from footy.ui.styles import CSS
from footy.ui import components as C
markets = {"1x2": {"home": {"prob": 0.85, "fair_odds": 1.18},
                   "draw": {"prob": 0.09, "fair_odds": 11.1},
                   "away": {"prob": 0.06, "fair_odds": 16.7}},
           "over_under": {"2.5": {"over": {"prob": 0.71, "fair_odds": 1.41},
                                  "under": {"prob": 0.29, "fair_odds": 3.45}}},
           "btts": {"yes": {"prob": 0.42, "fair_odds": 2.38}}}
body = (C.header_html("2026-06-22 14:00", 40, "BASE")
        + C.prob_cards_html("Brasil", 85, 9, "Haití", 6)
        + C.hbars_html([("Gana Brasil", 85), ("Empate", 9), ("Gana Haití", 6)])
        + '<div class="fty-line">Goles esperados: <b>Brasil 3.0</b> – <b>0.7 Haití</b></div>'
        + '<div class="fty-line">Marcador probable: <b>3-0</b> &nbsp; '
        + C.reliability_badge_html(0.9) + '</div>'
        + '<h3 style="color:#e8edf4">Apuestas</h3>'
        + C.odds_table_html(markets, "Brasil", "Haití"))
html = f'<html><body style="background:#0e1420;font-family:sans-serif;padding:24px">{CSS}{body}</body></html>'
import pathlib
pathlib.Path("artifacts").mkdir(exist_ok=True)
pathlib.Path("artifacts/ui_preview.html").write_text(html, encoding="utf-8")
print("wrote artifacts/ui_preview.html")
```

Run it with `PYTHONPATH=. python <script>`; open `artifacts/ui_preview.html` to confirm the
cards/bars/badges/odds look right (dark, high-contrast, clean).

- [ ] **Step 2: Boot the app headless**

Run `PYTHONPATH=. streamlit run app/streamlit_app.py --server.headless true --server.port 8530 &`,
wait ~10s, confirm the log shows "You can now view your Streamlit app" and the port is open,
then stop the process.

- [ ] **Step 3: No commit** (verification only). `artifacts/` is gitignored.

---

## Self-Review

**Spec coverage:**
- §2 architecture (styles/components/app, testable) → Tasks 1–3. ✓
- §3 dark theme (config.toml + CSS) → Task 1. ✓
- §4 tabs: header chips, Partido (cards/bars/xG/score/reliability/model chip), Mundial (BASE/LIVE radio with real sampler + group filter + compact table), Apuestas (shared session_state selection + fair odds + value badges), Scoreboard (plain language, advanced metrics hidden) → Tasks 2, 3. ✓
- §5 Spanish labels → components + app. ✓
- §6 error handling (st.error/info, LIVE fallback, spinners) → Task 3. ✓
- §7 testing (components, smoke) → Tasks 1–3. ✓
- §8 visual verification (preview HTML + boot) → Task 4. ✓
- Adjustments: Mundial LIVE builds a real sampler from `live.model` (Task 3); claridad>cantidad (compact df + filter); log loss/Brier in expander; first screen answers instantly (cards first); model untouched (no model files modified). ✓

**Placeholder scan:** all code complete; app rewrite is a full file; no TODO/TBD.

**Type consistency:** `reliability_label`/`prob_cards_html`/`hbars_html`/`value_badge_html`/`reliability_badge_html`/`odds_table_html`/`header_html` signatures in Task 2 match their calls in Task 3; `match_prediction`/`tournament_probs`/`team_stats`/`live_scoreboard`/`build_live_predictor` reused unchanged from SP5/SP6; `MatchSampler` built from `base.model` or `live.model` (same interface). App keeps `main`/`build_engine` for the smoke test. ✓
