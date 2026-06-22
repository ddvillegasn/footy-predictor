from pathlib import Path

import pandas as pd
import streamlit as st

from footy.cli import _build_default_predictor
from footy.config import load_config, config_fingerprint
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

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "configs" / "tournaments" / "wc2026_results.yaml"
STRUCTURE_PATH = ROOT / "configs" / "tournaments" / "wc2026.yaml"


@st.cache_resource
def build_engine():
    """BASE model (pre-tournament) + sampler + structure. Cached per server session."""
    predictor = _build_default_predictor()
    structure = load_structure(STRUCTURE_PATH)
    canon = predictor.canonical
    for g, teams in structure.groups.items():
        structure.groups[g] = [canon(t) for t in teams]
    mc_cfg = load_config("montecarlo")
    model_cfg = load_config("model")
    sampler = MatchSampler(predictor.model, mc_cfg,
                           model_version=model_cfg["model_version"],
                           config_hash=config_fingerprint("montecarlo"))
    return predictor, structure, sampler


@st.cache_resource
def build_live(_base_predictor, results_token):
    """LIVE model refit with played matches. results_token busts the cache when results change."""
    if not RESULTS_PATH.exists():
        return _base_predictor
    structure = load_structure(STRUCTURE_PATH)
    canon = _base_predictor.canonical
    for g, teams in structure.groups.items():
        structure.groups[g] = [canon(t) for t in teams]
    results = load_results(RESULTS_PATH, structure.groups)
    played = [{"team_a": pm.team_a, "team_b": pm.team_b,
               "goals_a": pm.goals_a, "goals_b": pm.goals_b} for pm in results.played]
    if not played:
        return _base_predictor
    return build_live_predictor(_base_predictor, played, tournament_date="2026-06-15",
                                model_config=load_config("model"), mc_config=load_config("montecarlo"))


def _results_token() -> str:
    return str(RESULTS_PATH.stat().st_mtime) if RESULTS_PATH.exists() else "none"


def _current_structure(base_predictor):
    """Load the on-disk wc2026.yaml fresh and canonicalize, so it always matches the
    on-disk results (avoids crashes when the cached engine structure is stale)."""
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
        # results/structure briefly out of sync (e.g. right after a refresh) -> treat as empty
        return []
    return [{"team_a": pm.team_a, "team_b": pm.team_b,
             "goals_a": pm.goals_a, "goals_b": pm.goals_b} for pm in results.played]


def _refresh_from_api(base_predictor):
    try:
        provider = FootballDataProvider.from_config(load_config("live"))
        name_map = load_name_map("configs/name_map.yaml")
        # Known teams = the FULL dataset (the fitted model's teams), not the current
        # placeholder structure — otherwise every real team that is not a placeholder
        # is wrongly reported as "not in dataset".
        known = set(base_predictor.model.attack.keys())
        sync_structure(provider, name_map, known, STRUCTURE_PATH)
        # Reload the freshly-written real structure (canonicalized) so ingest validates
        # played results against the real teams, not the old placeholders.
        structure = load_structure(STRUCTURE_PATH)
        canon = base_predictor.canonical
        for g, teams in structure.groups.items():
            structure.groups[g] = [canon(t) for t in teams]
        n = ingest(provider, structure, name_map, load_config("live")["stage_map"], RESULTS_PATH)
        st.success(f"Actualizado desde la API: {n} partidos jugados.")
        st.cache_resource.clear()
    except Exception as exc:  # noqa: BLE001 - surface any API/mapping problem to the user
        st.error(f"No se pudo actualizar desde la API: {exc}")


def _render_match_tab(base_predictor, structure):
    use_live = st.toggle("Usar modelo LIVE (con resultados del Mundial)", value=False)
    predictor = build_live(base_predictor, _results_token()) if use_live else base_predictor
    st.caption(f"Modelo en uso: {'LIVE (re-fit con jugados)' if use_live else 'BASE (histórico)'}")

    teams = team_list(base_predictor)
    c1, c2 = st.columns(2)
    team_a = c1.selectbox("Equipo A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    team_b = c2.selectbox("Equipo B", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
    neutral = st.checkbox("Cancha neutral", value=True)
    with st.expander("Cuotas de tu casa (opcional, para detectar valor)"):
        o1, o2, o3 = st.columns(3)
        odd_home = o1.number_input("Cuota A", min_value=0.0, value=0.0, step=0.05)
        odd_draw = o2.number_input("Cuota X", min_value=0.0, value=0.0, step=0.05)
        odd_away = o3.number_input("Cuota B", min_value=0.0, value=0.0, step=0.05)

    if st.button("Predecir", type="primary"):
        one = {}
        if odd_home > 1.0:
            one["home"] = odd_home
        if odd_draw > 1.0:
            one["draw"] = odd_draw
        if odd_away > 1.0:
            one["away"] = odd_away
        book = {"1x2": one} if one else None
        try:
            out = match_prediction(predictor, team_a, team_b, neutral=neutral, book_odds=book)
        except ValueError as exc:
            st.error(str(exc))
            return
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a}", f"{out['team_a_win']}%")
        m2.metric("Empate", f"{out['draw']}%")
        m3.metric(f"{team_b}", f"{out['team_b_win']}%")
        st.caption(f"xG {out['expected_goals_a']} - {out['expected_goals_b']} · "
                   f"marcador {out['most_likely_score']} · fiabilidad {out['prediction_reliability']}")
        st.bar_chart(pd.DataFrame({"prob %": [out["team_a_win"], out["draw"], out["team_b_win"]]},
                                  index=[team_a, "Empate", team_b]))
        mk = out["markets"]
        rows = [{"mercado": "1X2", "resultado": k, "prob": v["prob"], "cuota": v["fair_odds"]}
                for k, v in mk["1x2"].items() if isinstance(v, dict)]
        for line, ou in mk["over_under"].items():
            rows.append({"mercado": f"O/U {line}", "resultado": "over",
                         "prob": ou["over"]["prob"], "cuota": ou["over"]["fair_odds"]})
        rows.append({"mercado": "BTTS", "resultado": "sí",
                     "prob": mk["btts"]["yes"]["prob"], "cuota": mk["btts"]["yes"]["fair_odds"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        if "value" in out and out["value"].get("1x2"):
            vrows = [{"resultado": k, "edge %": v["edge_pct"], "EV": v["ev_per_unit"],
                      "stake": v["stake_recommendation"], "value": v["is_value"]}
                     for k, v in out["value"]["1x2"].items()]
            st.subheader("Valor vs tus cuotas")
            st.dataframe(pd.DataFrame(vrows), use_container_width=True)


def _render_tournament_tab(base_predictor, sampler):
    structure = _current_structure(base_predictor)
    try:
        results = (load_results(RESULTS_PATH, structure.groups)
                   if RESULTS_PATH.exists() else TournamentResults([]))
    except ValueError:
        results = TournamentResults([])
    if results.played:
        stats = team_stats(structure, results)
        st.subheader("Stats por selección (en el torneo)")
        sdf = pd.DataFrame([
            {"equipo": t, "PJ": s["played"], "Pts": s["points"], "GF": s["gf"],
             "GC": s["ga"], "DG": s["gd"], "forma": "".join(s["form"][-5:])}
            for t, s in sorted(stats.items(), key=lambda kv: -kv[1]["points"]) if s["played"] > 0])
        st.dataframe(sdf, use_container_width=True)
    else:
        st.info("Aún no hay partidos jugados. Usa 'Actualizar desde API' (barra lateral).")

    n = st.slider("Número de torneos a simular", 500, 10000, 1000, 500)
    if st.button("Simular Mundial", type="primary"):
        with st.spinner(f"Simulando {n} torneos…"):
            agg = tournament_probs(structure, results, sampler, n, seed=42)
        champ = sorted(agg["teams"].items(), key=lambda kv: -kv[1]["champion"])[:16]
        df = pd.DataFrame(
            {"campeón %": [round(d["champion"] * 100, 1) for _, d in champ],
             "avanza %": [round(d["advance_group"] * 100, 1) for _, d in champ]},
            index=[t for t, _ in champ])
        st.bar_chart(df[["campeón %"]])
        st.dataframe(df, use_container_width=True)


def _render_scoreboard_tab(base_predictor):
    played = _played_dicts(base_predictor)
    if not played:
        st.info("Aún no hay resultados cargados. Usa 'Actualizar desde API'.")
        return
    board = live_scoreboard(base_predictor, played)   # BASE model, out-of-sample
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", board["accuracy"])
    c2.metric("Log loss", board["log_loss"])
    c3.metric("Brier", board["brier"])
    st.caption(f"Partidos: {board['n']} · goal MAE {board['goal_mae']} · modelo BASE (out-of-sample)")
    st.dataframe(pd.DataFrame(board["matches"]), use_container_width=True)


def main():
    st.set_page_config(page_title="Footy — Mundial 2026", layout="wide")
    st.title("⚽ Footy — Mundial 2026 en vivo")
    st.caption("El modelo reacciona poco a resultados sueltos (es correcto). "
               "Cuotas/EV dependen del modelo; no son garantía.")
    with st.spinner("Cargando modelo (solo la primera vez)…"):
        base_predictor, structure, sampler = build_engine()
    with st.sidebar:
        st.header("Datos en vivo")
        if st.button("🔄 Actualizar desde API"):
            _refresh_from_api(base_predictor)
    tab1, tab2, tab3 = st.tabs(["Predecir partido", "Mundial / Grupos", "Scoreboard"])
    with tab1:
        _render_match_tab(base_predictor, structure)
    with tab2:
        _render_tournament_tab(base_predictor, sampler)
    with tab3:
        _render_scoreboard_tab(base_predictor)


if __name__ == "__main__":
    main()
