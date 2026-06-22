# Spec — Rediseño del dashboard (UI/UX) · Sub-proyecto 7

- **Fecha:** 2026-06-22
- **Estado:** Aprobado (Gate PASS)
- **Alcance:** SOLO capa de presentación (Streamlit + CSS + componentes + tests). **El modelo, servicios y simulador NO cambian.**
- **Tema:** Dark deportivo (broadcast). Español natural.
- **Depende de:** SP1–SP6 (motor, simulador, live, persistencia). 152 tests verdes.

---

## 1. Objetivo

Dashboard limpio, moderno y entendible para una persona normal. La **primera pantalla
responde de inmediato**: quién es favorito, por cuánto, marcador probable, qué tan confiable,
y si se usa modelo histórico (BASE) o con Mundial (LIVE). Claridad antes que cantidad de datos.

---

## 2. Arquitectura

Lógica visual testeable separada del render (patrón SP5):
```
.streamlit/config.toml      # tema base dark (fondo, fuente, primaryColor)
footy/ui/styles.py          # CSS() -> string con las clases del diseño
footy/ui/components.py      # builders HTML puros y testeables (devuelven str)
app/streamlit_app.py        # render: 4 pestañas; usa components; modelo intacto
tests/test_ui_components.py # tests de los builders
```

`components.py` (funciones puras → testeables):
- `reliability_label(score) -> (texto, clase)`: ≥0.70 "Alta", 0.45–0.70 "Media", <0.45 "Baja".
- `prob_cards_html(team_a, pa, draw, team_b, pb) -> str`: 3 cards grandes ("Gana X" / "Empate").
- `hbars_html(rows) -> str`: barras horizontales (label, %, color).
- `value_badge_html(is_value) -> str`: "Hay valor" (verde) / "No hay valor" (rojo); None → "—" gris.
- `header_html(updated, played, model) -> str`: título + subtítulo + chips de estado.
- `odds_table_html(markets) -> str`: mercados en formato entendible (no tabla cruda técnica).

`styles.py::CSS` define: `.fty-card`, `.fty-prob`, `.fty-bar`, `.fty-badge-*`, `.fty-chip`, etc.

---

## 3. Tema visual (config.toml + CSS)

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
CSS: cards con fondo `#161d2e`, borde `1px solid #233047`, radio 14px, sombra suave; texto
claro alto contraste (sin grises ilegibles); barras horizontales con gradiente; badges de
color (verde valor, rojo riesgo, gris neutro); menos espacio vacío; estética broadcast.

---

## 4. Pestañas

**Encabezado (siempre):** "Predicción Mundial 2026" + subtítulo
"Probabilidades, goles esperados, simulación del torneo y valor en cuotas." + chips:
`Última actualización · Partidos jugados: N · Modelo: BASE/LIVE`.

**Estado compartido:** la selección de equipos (A, B, neutral, modelo) vive en
`st.session_state` → la pestaña **Apuestas comparte el mismo partido** (sin selector duplicado).

### Tab 1 — Partido (primera vista, responde de inmediato)
- Selector A vs B + checkbox neutral (guardan en session_state).
- Toggle: **"Histórico reciente"** / **"Histórico + Mundial actual"** + nota:
  *"El modelo LIVE ajusta ligeramente la fuerza según partidos ya jugados del Mundial."*
- **3 cards grandes**: "Gana Brasil 85%" · "Empate 9%" · "Gana Haití 6%". El favorito resaltado.
- **Barras horizontales** limpias (mismas probabilidades).
- Línea: **"Goles esperados: Brasil 3.0 – 0.7 Haití"**.
- Card **"Marcador probable: 3–0"**.
- Badge **"Fiabilidad alta/media/baja"** (color).
- Chip del modelo en uso (BASE/LIVE) visible arriba.

### Tab 2 — Mundial
- Selector de simulación **BASE histórico** (default) / **LIVE con Mundial**.
  - **LIVE bien implementado:** construye un `MatchSampler` desde `live_predictor.model`
    (no solo cambia texto). Usa el live model cacheado; si aún no existe, lo ajusta (spinner,
    ~2-3 min la primera vez). Si falla, cae a BASE con aviso.
- Tabla ordenada y compacta: **Equipo · Clasificar % · Campeón % · Puntos · Forma**.
- **Filtro por grupo** (selectbox A–L; "Todos").
- No es la primera vista; evita tabla gigante de golpe (top-N + filtro).

### Tab 3 — Apuestas
- Usa el **partido seleccionado** en Partido (session_state); si no hay, pide elegirlo allí.
- Cuotas justas entendibles (mercado, resultado, probabilidad, cuota).
- Inputs de cuota de casa (1X2) → por resultado: **"Hay valor"** (verde) / **"No hay valor"**
  (rojo) / neutro (gris), con edge% y EV en lenguaje simple.

### Tab 4 — Scoreboard
- Texto: *"Mide qué tan bien predijo el modelo en partidos ya jugados del Mundial."*
- Cards: **Aciertos %** · **Error promedio de goles** · **Partidos evaluados**.
- Tabla **predicho vs real** (últimos, compacta): partido, predicho, real, acierto sí/no.
- **Log loss / Brier ocultos por defecto** → dentro de un `expander` "Métricas avanzadas".

---

## 5. Mapeos / lenguaje (español natural)

- 1X2 → "Gana {A}" / "Empate" / "Gana {B}".
- `prediction_reliability` → "Fiabilidad alta/media/baja".
- xG → "Goles esperados".
- `most_likely_score` → "Marcador probable".
- value → "Hay valor" / "No hay valor".
- BASE → "Histórico reciente"; LIVE → "Histórico + Mundial actual".

---

## 6. Error handling

| Caso | Comportamiento |
|---|---|
| Equipo desconocido en predict | `st.error` con mensaje claro (no crash) |
| Sin resultados del Mundial | chips/scoreboard muestran "—" + aviso amigable |
| LIVE no disponible / falla | cae a BASE con aviso, no rompe |
| Modelo ajustando | spinner "Cargando modelo…" / "Ajustando modelo LIVE…" |

---

## 7. Testing

| Test | Verifica |
|---|---|
| `test_ui_components` | `reliability_label` umbrales (Alta/Media/Baja); `prob_cards_html` contiene "Gana {A}" y los %; `value_badge_html` verde/rojo/gris; `hbars_html` ancho ∝ %; `header_html` chips con N partidos/modelo |
| `test_streamlit_app_imports` | la app importa (render no se testea como server) |
| Suite existente (152) | sin cambios al modelo → siguen verdes |

`components.py` se testea por strings (busca clases/labels/valores). Sin red.

---

## 8. Verificación visual (cierre)

Tras implementar: levantar la app headless (confirmar boot sin errores) **y** renderizar
`styles.CSS + components` a un `artifacts/ui_preview.html` estático para ver el diseño real
(cards, barras, badges) y confirmar que se ve bien antes de cerrar.

---

## 9. Decisiones registradas

1. Solo UI/UX/CSS/componentes/tests. Modelo, servicios y simulador intactos.
2. Tema dark deportivo (config.toml + CSS); español natural.
3. Lógica visual en `components.py` (testeable) + `styles.py`; render delgado.
4. 4 pestañas: Partido (primera vista directa), Mundial, Apuestas (comparte selección vía session_state), Scoreboard.
5. Mundial LIVE = sampler real desde `live_predictor.model`; BASE default; fallback a BASE.
6. Claridad > cantidad: sin tablas gigantes de entrada; top-N + filtros.
7. Log loss/Brier ocultos en `expander` avanzado.
8. Fiabilidad y valor con etiquetas humanas + color.
9. Cierre con boot headless + preview HTML estático.
