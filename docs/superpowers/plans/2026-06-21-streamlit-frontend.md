# Streamlit Frontend Implementation Plan (SP5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 3-tab Streamlit app (match predictor, tournament simulator, live scoreboard) on top of SP1–SP4, with the heavy model fit cached once.

**Architecture:** Testable logic in `footy/ui/service.py` (thin wrappers over `predict`, `simulate_tournaments`/`aggregate`, `scoreboard`); render-only `app/streamlit_app.py` calls the service and caches the engine via `@st.cache_resource`. Streamlit is an optional `[ui]` dependency.

**Tech Stack:** Python 3.10.6, Streamlit, pandas, pytest. Run from repo root: `python -m pytest`.

**Conventions:** branch `feature/baseline-v1`. TDD: failing test first. Commit per task, `git add` ONLY named files (never `__pycache__`/`.pyc`). Commit trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## File Structure

| File | Responsibility |
|---|---|
| `footy/ui/__init__.py` | package marker |
| `footy/ui/service.py` | `team_list`, `match_prediction`, `tournament_probs`, `live_scoreboard` |
| `app/streamlit_app.py` | render: 3 tabs, cached engine, `main()` |
| `pyproject.toml` | add `[project.optional-dependencies] ui = ["streamlit>=1.30"]` |
| `tests/test_ui_service.py`, `tests/test_streamlit_app_imports.py` | tests |

---

## Task 1: ui/service.py

**Files:**
- Create: `footy/ui/__init__.py`, `footy/ui/service.py`
- Test: `tests/test_ui_service.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ui_service.py`:
```python
from types import SimpleNamespace

import numpy as np

from footy.ui.service import team_list, match_prediction, tournament_probs, live_scoreboard
from footy.tournament.structure import TournamentConfig
from footy.tournament.results import TournamentResults


class FakePredictor:
    def __init__(self):
        self.model = SimpleNamespace(attack={"Brazil": 0.1, "Argentina": 0.2, "Haiti": -0.3})
        self.last_kwargs = None

    def predict(self, team_a, team_b, neutral=False, include_markets=False, book_odds=None):
        self.last_kwargs = {"neutral": neutral, "include_markets": include_markets,
                            "book_odds": book_odds}
        return {"team_a_win": 50.0, "draw": 30.0, "team_b_win": 20.0,
                "expected_goals_a": 1.5, "expected_goals_b": 1.0, "most_likely_score": "1-1"}


class FakeSampler:
    def __init__(self):
        self.strength = {"A1": 3.0, "A2": 2.0, "A3": 1.0, "A4": 0.5,
                         "B1": 3.0, "B2": 2.0, "B3": 1.0, "B4": 0.5}

    def lambdas(self, a, b, neutral):
        return self.strength[a], self.strength[b]

    def sample_goals(self, lam_a, lam_b, n, rng):
        return np.clip(rng.poisson(lam_a, n), 0, 10), np.clip(rng.poisson(lam_b, n), 0, 10)

    def scorelines(self, a, b, neutral, n, rng):
        la, lb = self.lambdas(a, b, neutral)
        return self.sample_goals(la, lb, n, rng)


def _struct():
    return TournamentConfig(
        name="Mini", neutral_default=True, points={"win": 3, "draw": 1, "loss": 0},
        groups={"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]},
        group_schedule="round_robin", per_group_advance=2, best_thirds=0,
        tiebreakers=["points", "goal_difference", "goals_for", "head_to_head", "fair_play", "drawing_of_lots"],
        thirds_ranking=["points", "goal_difference", "goals_for", "drawing_of_lots"],
        rounds=["SF", "F"],
        bracket_r32=[["winner_A", "runner_B"], ["winner_B", "runner_A"]],
        thirds_assignment="ranked_order",
    )


def test_team_list_sorted():
    assert team_list(FakePredictor()) == ["Argentina", "Brazil", "Haiti"]


def test_match_prediction_without_book_odds_keeps_markets_off():
    fp = FakePredictor()
    match_prediction(fp, "Brazil", "Haiti", neutral=True)
    assert fp.last_kwargs["include_markets"] is False


def test_match_prediction_with_book_odds_forces_markets():
    fp = FakePredictor()
    match_prediction(fp, "Brazil", "Haiti", book_odds={"1x2": {"home": 1.5}})
    assert fp.last_kwargs["include_markets"] is True
    assert fp.last_kwargs["book_odds"] == {"1x2": {"home": 1.5}}


def test_tournament_probs_returns_aggregate():
    agg = tournament_probs(_struct(), TournamentResults([]), FakeSampler(), n=30, seed=1)
    assert "teams" in agg and "groups" in agg


def test_live_scoreboard_empty_is_none():
    assert live_scoreboard(FakePredictor(), [])["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.ui'`

- [ ] **Step 3: Write implementation**

`footy/ui/__init__.py`:
```python
```
(empty — a single newline)

`footy/ui/service.py`:
```python
from footy.tournament.simulator import simulate_tournaments
from footy.tournament.aggregate import aggregate
from footy.live.scoreboard import scoreboard


def team_list(predictor) -> list:
    """Sorted team names for the dropdowns (the fitted model's keys)."""
    return sorted(predictor.model.attack.keys())


def match_prediction(predictor, team_a, team_b, neutral=False, book_odds=None) -> dict:
    """Wrap predict(). Force include_markets=True whenever book_odds is provided."""
    include_markets = book_odds is not None
    return predictor.predict(team_a, team_b, neutral=neutral,
                             include_markets=include_markets, book_odds=book_odds)


def tournament_probs(structure, results, sampler, n, seed) -> dict:
    """Run N tournaments conditioned on `results`, return aggregated probabilities."""
    sims = simulate_tournaments(structure, results, sampler, n, seed)
    return aggregate(structure, sims)


def live_scoreboard(predictor, played_matches) -> dict:
    """Predicted-vs-actual scoreboard over the played matches."""
    return scoreboard(predictor, played_matches)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui_service.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/ui/__init__.py footy/ui/service.py tests/test_ui_service.py
git commit -m "feat: UI service layer wrapping predict/simulate/scoreboard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Streamlit app + dependency + smoke test

**Files:**
- Create: `app/streamlit_app.py`
- Modify: `pyproject.toml`
- Test: `tests/test_streamlit_app_imports.py`

- [ ] **Step 1: Install Streamlit and write the failing test**

Install (needed to import the app): `pip install "streamlit>=1.30"`

`tests/test_streamlit_app_imports.py`:
```python
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_module_imports_without_running_main():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # runs top-level; main() is guarded by __main__
    assert hasattr(mod, "main") and hasattr(mod, "build_engine")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_streamlit_app_imports.py -v`
Expected: FAIL (the app file does not exist yet → `spec` is None / FileNotFoundError)

- [ ] **Step 3: Write the app + add the optional dependency**

`app/streamlit_app.py`:
```python
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
```

In `pyproject.toml`, under `[project.optional-dependencies]` (which already has `dev`), add:
```toml
ui = ["streamlit>=1.30"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_streamlit_app_imports.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full suite + commit**

Run: `python -m pytest -q`
Expected: all tests PASS (smoke test may take ~2 min).

```bash
git add app/streamlit_app.py pyproject.toml tests/test_streamlit_app_imports.py
git commit -m "feat: Streamlit 3-tab frontend (match, tournament, scoreboard)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §1 architecture (service vs render, [ui] dep) → Tasks 1, 2. ✓
- §2 service functions (team_list, match_prediction force-markets, tournament_probs, live_scoreboard) → Task 1. ✓
- §3 three tabs + cached engine + disclaimer + N default 1000 + scoreboard friendly-missing → Task 2 (`_render_*`, `build_engine`, `st.caption` disclaimer, slider default 1000, `st.info` on missing results). ✓
- §4 error handling (unknown team → st.error; missing results → st.info; spinner) → Task 2. ✓
- §5 testing (service with fakes; app import smoke; streamlit optional) → Tasks 1, 2. ✓

**Placeholder scan:** all code complete; no TODO/TBD. The `pyproject.toml` edit is additive with an explicit anchor.

**Type consistency:** `match_prediction(..., book_odds=None)` (T1) called by `_render_match_tab` with `book or None` (T2); `tournament_probs(structure, results, sampler, n, seed)` (T1) called in `_render_tournament_tab` (T2); `team_list(predictor)` / `live_scoreboard(predictor, played)` (T1) used in T2; `build_engine` returns `(predictor, structure, sampler)` consumed by `main` (T2). Output keys read in render (`team_a_win`, `markets`, `value`, `aggregate["teams"]/["groups"]`, `scoreboard` keys) match SP1/SP3/SP4 contracts. ✓
