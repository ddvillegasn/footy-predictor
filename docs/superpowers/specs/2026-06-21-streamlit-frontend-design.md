# Spec — Frontend Streamlit · Sub-proyecto 5

- **Fecha:** 2026-06-21
- **Estado:** Aprobado (Gate PASS)
- **Enfoque:** Streamlit, lógica testeable en `footy/ui/service.py`, render delgado en `app/streamlit_app.py`. 3 pestañas: partido, torneo, scoreboard. Reusa SP1–SP4.
- **Depende de:** SP1 (`Predictor`/`predict`, `cli._build_default_predictor`), SP3 (`load_structure`, `MatchSampler`, `simulate_tournaments`, `aggregate`, `load_results`), SP4 (`scoreboard`). 138 tests verdes.

---

## 1. Arquitectura

```
footy/ui/
  __init__.py
  service.py            # funciones puras testeables (wrappean SP1-SP4 -> dicts)
app/
  streamlit_app.py      # render delgado: 3 tabs; @st.cache_resource para el motor
configs (reuse)
tests/test_ui_service.py
pyproject.toml          # dependencia opcional [ui] = streamlit
```

Separación: toda la lógica vive en `service.py` (testeable con fakes); `streamlit_app.py`
solo renderiza (st.*). Charts nativos de Streamlit (`st.bar_chart`, `st.dataframe`), sin plotly.

---

## 2. `footy/ui/service.py`

```python
def team_list(predictor) -> list:
    """Equipos ordenados para los desplegables (claves del modelo)."""

def match_prediction(predictor, team_a, team_b, neutral=False, book_odds=None) -> dict:
    """Wrap de predict(). Fuerza include_markets=True cuando hay book_odds (micro-ajuste 1)."""

def tournament_probs(structure, sampler, n, seed) -> dict:
    """simulate_tournaments + aggregate -> dict de probabilidades (campeón/avance/grupos)."""

def live_scoreboard(predictor, played_matches) -> dict:
    """Wrap de scoreboard(predictor, played_matches)."""
```

Reglas:
- `match_prediction`: si `book_odds` no es None → `include_markets=True` (aunque el flag venga False).
- Funciones puras respecto al motor: reciben `predictor`/`sampler`/`structure` ya construidos
  (el build pesado lo hace el app cacheado, o el test inyecta fakes).

---

## 3. `app/streamlit_app.py` (render, 3 pestañas)

`@st.cache_resource def build_engine()` construye **una vez** el predictor real
(`cli._build_default_predictor`, ~2 min), el `MatchSampler` y la estructura WC2026
canonicalizada; cacheado por sesión de servidor. El cuerpo de render vive en `def main()`
(el módulo es importable sin ejecutar el server; `main()` se llama al final).

**Disclaimer global (micro-ajuste 2):** banner corto siempre visible:
> "Cuotas/EV dependen del modelo; no son garantía."

**Tab 1 — Predecir partido:** 2 `st.selectbox` (de `team_list`), checkbox `neutral`, inputs
opcionales de cuota bookie (1X2). Botón → `match_prediction` → `st.metric` 1X2, `st.bar_chart`
de probabilidades, `st.dataframe` de mercados (O/U, BTTS, hándicap, cuotas justas), y bloque
de valor/EV si se pegaron cuotas.

**Tab 2 — Simulador Mundial:** `st.slider` N torneos con **default 1000** (micro-ajuste 3),
rango p. ej. 500–10000; botón "Simular" → `tournament_probs` → tabla/`bar_chart` de prob
campeón (top-N) y `st.selectbox` de grupo → tabla de posiciones.

**Tab 3 — Scoreboard en vivo:** carga `configs/tournaments/wc2026_results.yaml`; si **no existe
o está vacío** → mensaje amigable (`st.info`, "aún no hay resultados cargados…") en vez de
romper (micro-ajuste 4). Si hay → `live_scoreboard` → `st.metric` accuracy/log-loss/Brier +
`st.dataframe` acierto/fallo por partido.

---

## 4. Error handling

| Caso | Comportamiento |
|---|---|
| Equipo desconocido en predict | `predict` lanza ValueError → la UI lo muestra con `st.error` (no crashea) |
| `wc2026_results.yaml` ausente/vacío | mensaje amigable `st.info`, no excepción |
| Cuota bookie ≤ 1.0 | `value` lanza ValueError → `st.error` claro |
| Modelo aún fiteando | spinner `st.spinner` durante `build_engine` |

---

## 5. Testing

`tests/test_ui_service.py` con `FakePredictor`/`FakeSampler` (sin fit real):
- `team_list` ordena las claves del modelo.
- `match_prediction` sin book_odds → no fuerza markets; con book_odds → `include_markets=True`
  (verificado vía un FakePredictor que registra los kwargs recibidos).
- `tournament_probs` con estructura mini + FakeSampler → dict con `teams`/`groups`.
- `live_scoreboard` con FakePredictor + played → métricas; lista vacía → None.

El render Streamlit no se testea como server; sí un smoke test de que `app/streamlit_app.py`
**importa** sin ejecutar `main()`. La lógica de fondo ya está cubierta (138 tests SP1-SP4).

Dependencia: **streamlit** en `[project.optional-dependencies] ui`. Correr:
`pip install -e .[ui]` → `streamlit run app/streamlit_app.py`.

---

## 6. Decisiones / micro-ajustes

1. `match_prediction` fuerza `include_markets=True` cuando hay `book_odds`.
2. Disclaimer permanente "Cuotas/EV dependen del modelo; no son garantía."
3. Simulador Mundial: N default **1000** (subible por slider).
4. Scoreboard sin `wc2026_results.yaml` → mensaje amigable, no crash.
5. Lógica testeable en `service.py`; render delgado; motor cacheado con `@st.cache_resource`.
6. Charts nativos Streamlit (sin plotly); streamlit como dependencia opcional `[ui]`.
