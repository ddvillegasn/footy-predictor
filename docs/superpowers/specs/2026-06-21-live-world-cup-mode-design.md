# Spec — Modo Mundial en vivo · Sub-proyecto 6

- **Fecha:** 2026-06-21
- **Estado:** Aprobado (Gate PASS)
- **Confirmado:** football-data.org (token gratis del usuario) entrega **Mundial 2026 EN VIVO**:
  104 partidos 2026, 40 FINISHED / 1 IN_PLAY / 63 TIMED; equipos y grupos reales.
- **Depende de:** SP1 (`Predictor`, `predict`, `metrics`), SP3 (estructura/results/simulador/aggregate),
  SP4 (provider football-data, ingest, scoreboard), SP5 (UI service + app Streamlit). 144 tests verdes.

---

## 1. Objetivo

Convertir la app en un **modo Mundial 2026 en vivo real**: equipos y grupos reales desde la API,
resultados jugados automáticos, **stats por selección del torneo** (PJ, Pts, GF, GC, DG, forma),
tablas de grupo, predicción de partidos con contexto del torneo, simulador condicionado y
scoreboard honesto — todo movido por datos reales.

### Dos modelos (decisión central, anti-leakage)
- **Modelo BASE** = fit histórico pre-torneo. Lo usa el **scoreboard** (out-of-sample, sin leakage).
- **Modelo LIVE** = BASE + partidos jugados del Mundial anexados (fecha torneo → time-decay) y
  re-fiteado. Lo usa la **predicción de partidos** y las apuestas. Marcado claramente como `live_model`.
- Nunca se mezclan: evaluación honesta (BASE) ≠ predicción viva (LIVE).

**Nota:** yo (agente) no tengo el token; tests usan datos mock/FakeProvider. El fetch real corre
en la máquina del usuario. El sync de nombres **junta todos los no-mapeados y los lista de una**.

---

## 2. Arquitectura / módulos

```
footy/
├── live/
│   ├── provider.py         # AÑADIR fetch_structure() (+ from_config: env o archivo local)
│   ├── structure_sync.py   # NUEVO: raw grupos -> canónico -> escribe wc2026.yaml (lista todos los faltantes)
│   ├── stats.py            # NUEVO: team_stats(structure, results) -> PJ/Pts/GF/GC/DG/forma por equipo
│   ├── ingest.py           # (SP4) sin cambios
│   └── scoreboard.py       # (SP4) sin cambios
├── ui/
│   └── service.py          # AÑADIR build_live_predictor(); match_prediction markets-always
app/streamlit_app.py        # botón "Actualizar desde API", tab Stats/Grupos, toggle live, key de archivo
configs/
├── name_map.yaml           # ampliado con equipos reales WC2026 (los que difieren)
├── live.yaml               # (SP4) knockout template para regenerar wc2026.yaml
└── secrets.local.yaml      # GITIGNORED: football_data_api_key (fallback al env)
tests/test_*                # mock/fake, sin red real
```

---

## 3. `provider.fetch_structure()`

Lee TODOS los partidos (sin filtro de status) → de los `GROUP_STAGE` agrupa por `group`,
recolecta los 4 equipos distintos por grupo (nombres crudos de la API).

```python
def fetch_structure(self) -> dict:
    """{'GROUP_A': ['Mexico', 'South Africa', ...], ...} desde los partidos de grupo."""
```
Reusa el GET de matches (mismo endpoint/headers/timeout). Devuelve dict ordenado.

`from_config`: clave de `FOOTBALL_DATA_API_KEY` (env); si no, de `configs/secrets.local.yaml`
(`football_data_api_key`); si ninguno → ValueError.

---

## 4. `structure_sync.py`

```python
def map_groups(raw_groups, name_map, known_teams) -> dict:
    """{'GROUP_A': [...crudos]} -> {'A': [...canónicos]}. Recolecta TODOS los nombres no
    mapeados/ausentes del dataset y lanza UN ValueError con la lista completa (no uno por uno)."""

def write_structure_yaml(groups, knockout_template, out_path) -> None:
    """Escribe wc2026.yaml: groups reales + plantilla fija (qualification, tiebreakers,
    thirds_ranking, knockout.bracket_r32 con winner_/runner_/third_slot)."""

def sync_structure(provider, name_map, known_teams, knockout_template, out_path) -> int:
    """fetch_structure -> map_groups -> write. Devuelve nº de equipos."""
```
- `GROUP_A`→`A`. `known_teams` = equipos canónicos del dataset (claves del modelo).
- Plantilla knockout = constante validada (12 grupos, bracket de 16 cruces, third_slot_1..8).
- Error duro con **lista completa** de faltantes → el usuario llena `name_map.yaml` de una.

---

## 5. `stats.py` — stats vivas por selección

```python
def team_stats(structure, results) -> dict:
    """Por equipo (de los partidos jugados, grupo+knockout):
    {team: {played, points, gf, ga, gd, wins, draws, losses, form}}.
    form = lista de 'W'/'D'/'L' de los últimos partidos (más reciente al final)."""
```
Recorre `results.played`, acumula por equipo (orienta marcador a cada lado). Puntos con `structure.points`.

---

## 6. Modelo LIVE — `service.build_live_predictor`

```python
def build_live_predictor(base_predictor, played_matches, tournament_date,
                         model_config, mc_config) -> Predictor:
    """Re-fit: base_predictor.matches + filas de los jugados (fecha=tournament_date,
    neutral=True, tournament='FIFA World Cup') -> Predictor.from_matches(...). Marcado live."""
```
- Reusa `base_predictor.matches` (las que el BASE usó). Añade los jugados como filas reales.
- `tournament_date` reciente → time-decay los pesa. Caveat honesto: mueve poco con pocos partidos.
- El re-fit tarda ~2 min (como el base). En la app se cachea aparte.

---

## 7. UI (app/streamlit_app.py)

- **Sidebar** "🔄 Actualizar desde API": `sync_structure` + `ingest` resultados → `st.cache_resource.clear()` → rerun. Si la API falla → `st.error`, sigue con lo de disco.
- **Tab Predecir partido:** mercados **siempre** (O/U, BTTS, hándicap, cuotas justas); valor/EV si pegas cuota. Toggle **"usar modelo live (resultados del Mundial)"** → live vs base; etiqueta visible de cuál se usó.
- **Tab Mundial / Grupos:** tablas de grupo con lo jugado + **stats por selección** (PJ, Pts, GF, GC, DG, forma); prob de avanzar/campeón (simulador condicionado).
- **Tab Scoreboard:** SIEMPRE modelo BASE (out-of-sample).
- Banner: "El modelo reacciona poco a resultados sueltos — es correcto. Cuotas/EV no son garantía."

---

## 8. Error handling

| Caso | Comportamiento |
|---|---|
| Equipo WC no mapeado/ausente del dataset | `sync_structure` lanza ValueError con **lista completa** de faltantes |
| API caída / 403 / timeout | `st.error`, la app sigue con `wc2026.yaml`/`wc2026_results.yaml` de disco |
| Sin token (env ni archivo) | ValueError claro al construir provider |
| Sin partidos jugados aún | stats/scoreboard muestran mensaje amigable |

---

## 9. Testing (mock/fake, sin red real)

| Test | Verifica |
|---|---|
| `test_fetch_structure` | `requests.get` mockeado → grupos crudos {A:[4 teams]}; from_config lee env y archivo |
| `test_structure_sync` | `map_groups` mapea; **junta todos los faltantes en un solo error**; `write_structure_yaml` produce wc2026.yaml válido (lo carga `load_structure`) |
| `test_stats` | `team_stats` PJ/Pts/GF/GC/DG/forma correctos sobre played mini |
| `test_live_predictor` | `build_live_predictor` re-fitea (dataset mini + played) y cambia λ en la dirección correcta; sigue siendo Predictor |
| `test_ui_service` (ampliado) | `match_prediction` markets-always |
| `test_streamlit_app_imports` | la app importa (render no se testea como server) |

---

## 10. Decisiones registradas

1. Dos modelos: BASE (scoreboard, out-of-sample) y LIVE (predicción/apuestas, re-fit con jugados). No se mezclan.
2. `fetch_structure` arma los grupos reales desde los partidos de la API; `sync_structure` reescribe `wc2026.yaml`.
3. Mapeo de nombres exacto y duro; el sync **lista todos los faltantes juntos** para llenar `name_map` de una.
4. Plantilla knockout fija (thirds ranked, v1); solo los grupos cambian con el sorteo real.
5. `stats.team_stats`: PJ, Pts, GF, GC, DG, W/D/L, forma — de los jugados.
6. Modelo LIVE = `base.matches` + jugados (fecha torneo, neutral) re-fit; caveat: mueve poco (correcto).
7. Mercados siempre en la pestaña de partido; valor/EV solo con cuota de bookie.
8. Botón "Actualizar desde API" limpia el cache; fallback de key por archivo gitignored.
9. Tests con mock/FakeProvider (el agente no tiene token); fetch real en la máquina del usuario.
