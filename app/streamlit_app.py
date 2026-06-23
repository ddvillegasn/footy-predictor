from datetime import datetime
from pathlib import Path
import json as _json

import pandas as pd
import streamlit as st

from footy.cli import _build_default_predictor
from footy.config import load_config, config_fingerprint
from footy.eval.report import run_report, DEFAULT_EDITIONS
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
REPORT_PATH = ROOT / "artifacts" / "backtest_report.json"


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


def _render_eval_tab(base_predictor):
    st.caption("Compara el modelo contra baselines (Elo, naive, azar) y hace backtest: "
               "entrena antes de cada Mundial y evalúa en él. Así sabes si 60% es bueno.")
    if st.button("Recalcular backtest (~6 min)"):
        with st.spinner("Entrenando y evaluando por edición…"):
            data_cfg = load_config("data")
            raw = data_cfg["raw_dir"]
            run_report(dataset_path=f"{raw}/{data_cfg['files']['results']}",
                       editions=DEFAULT_EDITIONS, model_config=load_config("model"),
                       elo_config=load_config("elo"), out_path=REPORT_PATH)
        st.success("Backtest actualizado.")
    if not REPORT_PATH.exists():
        st.info("Aún no hay reporte. Aprieta **Recalcular backtest**.")
        return
    report = _json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    st.subheader("Resumen (todas las ediciones)")
    agg = report["aggregate"]
    st.dataframe(pd.DataFrame([
        {"Modelo": name, "Aciertos %": round(d["accuracy"] * 100, 1),
         "Log loss": d["log_loss"], "Brier": d["brier"], "Partidos": d["n"]}
        for name, d in agg.items()], ).sort_values("Aciertos %", ascending=False),
        width="stretch", hide_index=True)
    edition = st.selectbox("Ver edición", list(report["editions"].keys()))
    ed = report["editions"][edition]
    if ed["n"] == 0:
        st.info("Sin partidos para esa edición en el dataset.")
        return
    st.dataframe(pd.DataFrame([
        {"Modelo": name, "Aciertos %": round(m["accuracy"] * 100, 1),
         "Log loss": m["log_loss"], "Brier": m["brier"],
         "Error goles": m["goal_mae"], "Partidos": m["n"]}
        for name, m in ed["models"].items()],
    ).sort_values("Aciertos %", ascending=False), width="stretch", hide_index=True)


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

    t1, t2, t3, t4, t5 = st.tabs(["Partido", "Mundial", "Apuestas", "Scoreboard", "Evaluación"])
    with t1:
        _render_match_tab(base_predictor)
    with t2:
        _render_tournament_tab(base_predictor)
    with t3:
        _render_betting_tab(base_predictor)
    with t4:
        _render_scoreboard_tab(base_predictor)
    with t5:
        _render_eval_tab(base_predictor)


if __name__ == "__main__":
    main()
