from pathlib import Path

import pandas as pd
import streamlit as st

from footy.cli import _build_default_predictor
from footy.config import load_config, config_fingerprint
from footy.tournament.structure import load_structure
from footy.tournament.results import load_results, TournamentResults
from footy.tournament.sampler import MatchSampler
from footy.ui.service import team_list, match_prediction, tournament_probs, live_scoreboard

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "configs" / "tournaments" / "wc2026_results.yaml"
STRUCTURE_PATH = ROOT / "configs" / "tournaments" / "wc2026.yaml"


@st.cache_resource
def build_engine():
    """Fit the real model once and build the tournament sampler (cached per server)."""
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


def _render_match_tab(predictor):
    teams = team_list(predictor)
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Equipo A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    team_b = col2.selectbox("Equipo B", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
    neutral = st.checkbox("Cancha neutral", value=True)

    with st.expander("Cuotas de tu casa (opcional, para detectar valor)"):
        oc1, oc2, oc3 = st.columns(3)
        odd_home = oc1.number_input("Cuota A (1)", min_value=0.0, value=0.0, step=0.05)
        odd_draw = oc2.number_input("Cuota empate (X)", min_value=0.0, value=0.0, step=0.05)
        odd_away = oc3.number_input("Cuota B (2)", min_value=0.0, value=0.0, step=0.05)

    if st.button("Predecir", type="primary"):
        book = {}
        one_x_two = {}
        if odd_home > 1.0:
            one_x_two["home"] = odd_home
        if odd_draw > 1.0:
            one_x_two["draw"] = odd_draw
        if odd_away > 1.0:
            one_x_two["away"] = odd_away
        if one_x_two:
            book["1x2"] = one_x_two
        try:
            out = match_prediction(predictor, team_a, team_b, neutral=neutral,
                                   book_odds=book or None)
        except ValueError as exc:
            st.error(str(exc))
            return

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a} gana", f"{out['team_a_win']}%")
        m2.metric("Empate", f"{out['draw']}%")
        m3.metric(f"{team_b} gana", f"{out['team_b_win']}%")
        st.caption(f"Goles esperados {out['expected_goals_a']} - {out['expected_goals_b']} · "
                   f"marcador más probable {out['most_likely_score']} · "
                   f"fiabilidad {out['prediction_reliability']}")
        st.bar_chart(pd.DataFrame(
            {"prob %": [out["team_a_win"], out["draw"], out["team_b_win"]]},
            index=[team_a, "Empate", team_b]))

        if "markets" in out:
            mk = out["markets"]
            rows = [{"mercado": "1X2", "resultado": k, "prob": v["prob"], "cuota justa": v["fair_odds"]}
                    for k, v in mk["1x2"].items() if isinstance(v, dict)]
            for line, ou in mk["over_under"].items():
                rows.append({"mercado": f"O/U {line}", "resultado": "over",
                             "prob": ou["over"]["prob"], "cuota justa": ou["over"]["fair_odds"]})
            rows.append({"mercado": "BTTS", "resultado": "sí",
                         "prob": mk["btts"]["yes"]["prob"], "cuota justa": mk["btts"]["yes"]["fair_odds"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if "value" in out and out["value"].get("1x2"):
            vrows = [{"resultado": k, "edge %": v["edge_pct"], "EV": v["ev_per_unit"],
                      "stake": v["stake_recommendation"], "value": v["is_value"]}
                     for k, v in out["value"]["1x2"].items()]
            st.subheader("Valor vs tus cuotas")
            st.dataframe(pd.DataFrame(vrows), use_container_width=True)


def _render_tournament_tab(structure, sampler):
    n = st.slider("Número de torneos a simular", 500, 10000, 1000, 500)
    if st.button("Simular Mundial", type="primary"):
        results = (load_results(RESULTS_PATH, structure.groups)
                   if RESULTS_PATH.exists() else TournamentResults([]))
        with st.spinner(f"Simulando {n} torneos…"):
            agg = tournament_probs(structure, results, sampler, n, seed=42)
        champ = sorted(agg["teams"].items(), key=lambda kv: -kv[1]["champion"])[:16]
        df = pd.DataFrame(
            {"campeón %": [round(d["champion"] * 100, 1) for _, d in champ],
             "avanza grupo %": [round(d["advance_group"] * 100, 1) for _, d in champ]},
            index=[t for t, _ in champ])
        st.bar_chart(df[["campeón %"]])
        st.dataframe(df, use_container_width=True)
        group = st.selectbox("Ver grupo", sorted(structure.groups.keys()))
        gp = agg["groups"][group]
        gdf = pd.DataFrame(
            {"1º %": [round(gp[t]["p1"] * 100, 1) for t in gp],
             "2º %": [round(gp[t]["p2"] * 100, 1) for t in gp]},
            index=list(gp.keys()))
        st.dataframe(gdf, use_container_width=True)


def _render_scoreboard_tab(predictor, structure):
    if not RESULTS_PATH.exists():
        st.info("Aún no hay resultados cargados. Agrega partidos jugados (manual o con "
                "`update-and-simulate`) para ver el desempeño del modelo.")
        return
    results = load_results(RESULTS_PATH, structure.groups)
    played = [{"team_a": pm.team_a, "team_b": pm.team_b,
               "goals_a": pm.goals_a, "goals_b": pm.goals_b} for pm in results.played]
    if not played:
        st.info("Aún no hay resultados cargados.")
        return
    board = live_scoreboard(predictor, played)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", board["accuracy"])
    c2.metric("Log loss", board["log_loss"])
    c3.metric("Brier", board["brier"])
    st.caption(f"Partidos evaluados: {board['n']} · goal MAE {board['goal_mae']}")
    st.dataframe(pd.DataFrame(board["matches"]), use_container_width=True)


def main():
    st.set_page_config(page_title="Footy predictor", layout="wide")
    st.title("⚽ Footy — predictor de selecciones")
    st.caption("Cuotas/EV dependen del modelo; no son garantía.")
    with st.spinner("Cargando modelo (solo la primera vez)…"):
        predictor, structure, sampler = build_engine()
    tab1, tab2, tab3 = st.tabs(["Predecir partido", "Simulador Mundial", "Scoreboard en vivo"])
    with tab1:
        _render_match_tab(predictor)
    with tab2:
        _render_tournament_tab(structure, sampler)
    with tab3:
        _render_scoreboard_tab(predictor, structure)


if __name__ == "__main__":
    main()
