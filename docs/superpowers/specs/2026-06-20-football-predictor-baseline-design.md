# Spec — Predictor de fútbol internacional · Sub-proyecto 1 (Baseline jugable)

- **Fecha:** 2026-06-20
- **Estado:** En revisión
- **Enfoque aprobado:** A — Poisson/Dixon-Coles con time-decay + Elo/FIFA como covariables
- **Autor:** brainstorming colaborativo (superpowers flow)

---

## 1. Contexto y descomposición

El objetivo final del usuario es un sistema profesional de predicción de partidos de
selecciones (11 fases: ETL, features, modelos estadísticos, ML, deep learning, Monte
Carlo, ensemble, backtesting, API, dashboard). Ese alcance abarca múltiples subsistemas
independientes y **no cabe en un solo spec**. Se descompone en sub-proyectos, cada uno
con su propio ciclo spec → plan → build:

1. **ETL + baseline jugable** ← *este spec*
2. Features avanzadas + Elo/ranking ampliado
3. ML (XGBoost/LightGBM/CatBoost) + backtesting/calibración profunda
4. Deep learning (solo si ML no basta — YAGNI)
5. Ensemble (weighted/stacking/blending)
6. Dashboard Streamlit + API REST
7. MLOps / despliegue

Este documento cubre **únicamente el sub-proyecto 1**.

### Reglas heredadas descartadas
Las "reglas no negociables" del prompt original (`clinica_id` multi-tenant/anti-IDOR,
funciones DB best-effort, golden test, deploy `/health`) provienen de otro proyecto
(SaaS clínica) y **no aplican** a un predictor de fútbol. Decisión del usuario:
**descartar todas salvo TDD**. Reinterpretaciones sanas que sí sobreviven:
- "funciones DB best-effort" → loaders de fuentes externas tolerantes a fallos.
- "golden test verde siempre" → **test anti-leakage** verde siempre.

---

## 2. Objetivo del sub-proyecto 1

Entregar un baseline **end-to-end funcional**: el usuario ejecuta
`predict_match("Brazil", "Haiti")` (o el CLI `predict Brazil Haiti`) y obtiene
probabilidades reales, distribución de marcadores, goles esperados (λ), intervalos y
fiabilidad. Sin UI, sin ML/DL todavía.

### Entregable
- Función Python `predict_match(...)` que devuelve un dict completo.
- Comando CLI `predict <team_a> <team_b>`.

### Fuera de alcance (sub-proyectos posteriores)
- ML, deep learning, ensemble.
- Dashboard Streamlit, API REST.
- Backtesting profundo (por torneo/selección, confusion matrix). Aquí solo el
  esqueleto de evaluación temporal + métricas básicas.
- xG basado en tiros (no hay datos de remates para selecciones).

---

## 3. Fuentes de datos

### Locales (repo `international_results/`)
- `results.csv` — ~49.477 partidos 1872–2024 (fecha, equipos, marcador, torneo, ciudad, país, neutral).
- `goalscorers.csv` — goles (jugador, minuto, autogol, penal).
- `shootouts.csv` — tandas de penales.
- `former_names.csv` — nombres antiguos de selecciones.

### Externas (decisión del usuario: incluir; acotadas a las que cubren selecciones)
- **Elo in-house** (`elo.py`) — calculado desde `results.csv`. Sin scraping, determinista. Fuente primaria de Elo.
- **FIFA ranking histórico** (`fifa.py`) — CSV público cacheado en `external/cache/`. Best-effort.
- **Elo externo (eloratings.net)** — opcional, solo cross-check. Redundante con in-house.
- **Transfermarkt** (`transfermarkt.py`) — best-effort con caché; señal marginal. Si falla → None, no bloquea.

> Nota: Understat, Football-Data.co.uk solo cubren clubes → **no se usan** (no aplican a selecciones).

---

## 4. Arquitectura y estructura

Nombre destino de la carpeta raíz: **`analisis_mundial/`** (sin espacios). El rename
físico se ejecuta **una sola vez al cierre** (opción A), porque no se puede renombrar el
directorio de trabajo en uso. El espacio actual no afecta imports (paquete `footy`) ni
CLI (entry-point en `pyproject.toml`).

```
analisis_mundial/
├── footy/
│   ├── __init__.py
│   ├── data/
│   │   ├── loaders.py        # CSV repo -> DataFrames tipados
│   │   ├── clean.py          # nulos, duplicados, formato fechas
│   │   ├── names.py          # homologar selecciones
│   │   └── external/
│   │       ├── elo.py        # Elo in-house desde results
│   │       ├── fifa.py       # ranking FIFA histórico (CSV cacheado)
│   │       ├── transfermarkt.py  # best-effort, cache, nunca bloquea
│   │       └── cache/        # archivos cacheados
│   ├── features/
│   │   ├── strength.py       # ataque/defensa, forma 5/10/20, H2H
│   │   └── context.py        # neutral, torneo, continente, elo, fifa
│   ├── models/
│   │   ├── poisson.py        # ajuste Dixon-Coles + time-decay
│   │   └── montecarlo.py     # 100k simulaciones -> distribución
│   ├── predict.py            # SOLO orquesta
│   └── cli.py                # `predict Brazil Haiti`
├── configs/
│   ├── data.yaml             # rutas CSV, ventana fechas, fusiones sensibles
│   ├── model.yaml            # time-decay xi, home_advantage, Dixon-Coles rho
│   ├── elo.yaml              # K-factor, rating inicial, peso por torneo
│   └── montecarlo.yaml       # n_sims=100000, seed, max_goals
├── artifacts/                # gitignored (salvo .gitkeep)
│   ├── models/               # modelo Poisson ajustado
│   ├── metrics/              # log_loss, brier, accuracy, baseline_naive_comparison
│   ├── plots/                # calibración, distribución goles
│   ├── backtest/             # predicciones vs real
│   ├── prob_matrices/        # matriz marcadores por partido
│   ├── etl_report.json
│   ├── dropped_rows.csv
│   ├── team_name_mapping.json
│   └── enriched_matches.parquet
├── tests/
│   ├── fixtures/             # CSV mini deterministas
│   └── test_*.py             # uno por módulo
├── international_results/     # datos crudos (ya existe)
├── pyproject.toml
├── README.md                 # manual de uso
└── CLAUDE.md                 # guía repo para sesiones futuras
```

**Principio rector:** cada módulo una responsabilidad, interfaz clara, testeable de
forma aislada. `predict.py` orquesta; **no contiene lógica de modelo**. Config-driven:
nada hardcodeado, todo parámetro relevante en `configs/*.yaml`.

---

## 5. Flujo ETL

```
CSV crudos ──loaders──> DataFrames tipados ──clean──> limpio ──names──> homologado
                                                                            │
                          ┌─────────────────────────────────────────────────┤
                          ▼                                                   ▼
                    external (elo/fifa/tm)                             features
                          │                                                   │
                          └──────────────► matches enriquecidos ◄─────────────┘
                                                    │
                                  (cacheado en artifacts/enriched_matches.parquet)
```

### 5.1 loaders.py
Lee los 4 CSV. Tipa columnas (fechas→datetime, scores→int, neutral→bool). Devuelve
DataFrames. **Falla fuerte** si falta archivo o columna (datos crudos = contrato de entrada).

### 5.2 clean.py
- **Duplicados** (misma fecha+team_a+team_b): conserva uno, **registra el eliminado y
  por qué** en `dropped_rows.csv` + `etl_report.json`. No drop silencioso.
- **Nulos**: filas con score nulo → inválidas para entrenar (marcadas, no imputadas con
  basura). Equipos siguen contando para nombres/H2H.
- **Formato**: fechas inválidas → drop con conteo registrado.
- Devuelve DataFrame limpio + reporte (n filas eliminadas por causa).

### 5.3 names.py
Homologa con `former_names.csv` + diccionario de alias. Función `canonical(name) -> str`.
**Fusiones históricas sensibles** (p. ej. `West Germany → Germany`) **no automáticas**:
definidas y conmutables en `data.yaml` (trazabilidad). Mapeo final →
`artifacts/team_name_mapping.json`. Test dedicado.

### 5.4 external/
- `elo.py`: recorre partidos cronológicamente, actualiza Elo (K, rating inicial, peso
  por torneo en `elo.yaml`). **El rating expuesto como feature es el Elo PRE-partido**
  (anti-leakage). Determinista.
- `fifa.py`: carga ranking FIFA desde CSV cacheado. Si no existe → ranking=None,
  pipeline sigue (best-effort).
- `transfermarkt.py`: best-effort total. Caché; si falla → None. Nunca lanza.

### 5.5 Merge / enriquecido
- Elo y FIFA se unen por equipo+fecha con **merge_asof backward**: si no hay ranking
  exacto para la fecha, usa el **último anterior**. **Nunca datos futuros.**
- Resultado cacheado en `artifacts/enriched_matches.parquet`.

### 5.6 Anti-leakage (transversal)
- `assert_no_leakage()`: verifica que toda feature de un partido use solo info con
  fecha `< match_date`. Corre en tests (golden) y opcionalmente al construir features.

---

## 6. Features

Calculadas **solo con partidos previos a la fecha del partido** (anti-leakage).

### strength.py
- Ataque/defensa por equipo = parámetros del ajuste Dixon-Coles (no medias crudas).
- Forma reciente: media goles a favor/contra en últimos 5/10/20 partidos (ventana móvil, solo pasado).
- Porterías en cero.
- H2H: V/E/D y goles en enfrentamientos directos previos.

> **Eliminada** "% conversión de oportunidades": requiere tiros/remates reales que no
> existen para selecciones. No se inventa la feature.

### context.py
- `neutral` (de `results.csv`); con `neutral=True` → `home_advantage = 0`.
- Tipo torneo (Mundial/clasificatoria/amistoso) → peso.
- Continente (mapeo país→confederación).
- Elo pre-partido, ranking FIFA pre-partido (merge_asof backward).

---

## 7. Modelo — poisson.py (Dixon-Coles + time-decay)

Goles de cada equipo ~ Poisson:
```
log(λ_a) = μ + ataque[team_a] - defensa[team_b] + home_advantage
log(λ_b) = μ + ataque[team_b] - defensa[team_a]
```
- Si `neutral=True` → `home_advantage = 0`. Nomenclatura `team_a/team_b` (no local/visitante).
- **Time-decay**: cada partido pesa `exp(-xi · Δt)` (`xi` en `model.yaml`). Reciente
  pesa más; usa toda la historia sin cortar.
- **Corrección Dixon-Coles (τ)**: forma parte de la **función de verosimilitud durante
  el entrenamiento** (no post-hoc). Ajusta dependencia en marcadores bajos (0-0, 1-0, 0-1, 1-1).
- Ajuste por máxima verosimilitud ponderada (`scipy.optimize`).
- Modelo ajustado → `artifacts/models/` (parámetros + fecha de entrenamiento + versión).
- **Determinista**: misma data + config → mismos parámetros.

---

## 8. Monte Carlo — montecarlo.py

- Entrada: λ_a, λ_b del modelo.
- Simula N marcadores (Poisson seedeado). `n_sims=100000`, `seed`, `max_goals` de `montecarlo.yaml`.
- Agrega:
  - P(victoria A), P(empate), P(victoria B).
  - Distribución completa de marcadores; marcador más probable.
  - Intervalos de confianza (percentiles de la distribución de goles).
  - Goles esperados = λ (documentado como λ, **no** xG de tiros).
- Matriz de probabilidad de marcadores → `artifacts/prob_matrices/`.
- Seed fijo → resultados reproducibles.

---

## 9. predict.py — orquestación

```python
predict_match(team_a, team_b, neutral=False, tournament="Friendly") -> dict
```
Pasos: valida equipos vía `canonical()` → carga modelo de `artifacts/` (o entrena si no
existe) → arma features del enfrentamiento → λ del modelo → Monte Carlo → dict.

### Salida (dict)
```json
{
  "team_a_win": 87.2,
  "draw": 8.6,
  "team_b_win": 4.2,
  "expected_goals_a": 2.7,
  "expected_goals_b": 0.4,
  "most_likely_score": "3-0",
  "score_distribution": { "3-0": 12.1, "2-0": 11.4, "1-0": 9.8, "...": "top-N marcadores con su probabilidad %" },
  "confidence_interval": { "goals_a": [1, 4], "goals_b": [0, 1], "level": 0.90 },
  "prediction_reliability": 0.71,
  "model_version": "2026-06-20"
}
```

### `prediction_reliability` (NO es probabilidad de acierto)
Renombrado desde `confidence` para evitar malinterpretación. Medida honesta basada en
factores **documentados explícitamente**:
- Cantidad de partidos recientes de ambos equipos.
- Antigüedad de los datos disponibles.
- Estabilidad de λ entre simulaciones (dispersión).
- Si faltan Elo/FIFA para algún equipo.
- Si el equipo tiene pocos partidos internacionales en total.

---

## 10. Error handling (capas)

| Capa | Comportamiento |
|---|---|
| loaders (datos crudos) | Falla fuerte: falta archivo/columna → excepción clara |
| clean | No lanza por fila mala: registra en `dropped_rows.csv` + `etl_report.json`, sigue |
| external (fifa/tm) | Best-effort: fallo → None, pipeline continúa, nunca bloquea |
| predict_match | Equipo desconocido → error explícito + fuzzy match de nombres cercanos. Equipo con pocos partidos → predice con `prediction_reliability` baja + warning |
| anti-leakage | `assert_no_leakage()` en tests y opcional en build de features |

---

## 11. Testing — TDD estricto

Regla: **test rojo → código → verde** en cada task del plan. Fixtures = CSV mini
deterministas en `tests/fixtures/` (no el dataset real).

| Test | Verifica |
|---|---|
| `test_loaders` | tipos correctos; falla si falta columna |
| `test_clean` | duplicado conservado **y registrado**; fila nula dropeada y reportada |
| `test_names` | `canonical()` homologa; fusiones sensibles respetan `data.yaml` |
| `test_elo` | Elo determinista; feature usa rating **pre-partido** |
| `test_leakage` | **golden anti-leakage**: ninguna feature usa fecha futura. Verde siempre |
| `test_poisson` | ajuste converge; τ Dixon-Coles dentro de la verosimilitud; determinista |
| `test_montecarlo` | seed fijo → mismas probabilidades; suma P = 1.0; respeta n_sims |
| `test_predict` | dict completo con todas las claves; `home_advantage=0` si neutral; equipo desconocido → error |

Suite completa corre en el cierre (Gate 4 — finishing-a-development-branch).

---

## 12. Métricas

Evaluación con **hold-out temporal** (entrenar pasado, evaluar futuro — nunca aleatorio).
- **Log Loss** y **Brier Score** (calibración probabilística — lo que importa en apuestas).
- **Accuracy** 1X2 (referencia, no objetivo único).
- **`baseline_naive_comparison`**: comparar contra baseline tonto = probabilidades
  históricas globales 1X2. **El modelo solo se considera útil si supera ese baseline.**
- Guardadas en `artifacts/metrics/`. Curva de calibración → `artifacts/plots/`.
- Backtesting profundo = sub-proyecto 4. Aquí solo el esqueleto temporal + métricas básicas.

---

## 13. Dependencias

Mínimas para el baseline: `pandas, numpy, scipy, pyyaml, pyarrow`. **Sin** XGBoost / TF /
PyTorch / Optuna / SHAP todavía — llegan en sub-proyectos ML/DL (YAGNI).

---

## 14. Entregables docs (Gate 4)

- **README.md** — manual: instalar, configurar YAMLs, entrenar, `predict Brazil Haiti`, leer output.
- **CLAUDE.md** — guía de repo para sesiones futuras: estructura, comandos, convenciones, anti-leakage.
- **ALCANCE** — qué cubre el sub-proyecto 1 y qué queda para 2–7 (roadmap).
- Spec (este archivo) + plan en `docs/superpowers/`.

---

## 14b. Repositorio y control de versiones

- **Remote:** `origin` = `https://github.com/ddvillegasn/international_results.git`
  (repo vacío del usuario; lienzo limpio).
- **Identidad:** `ddvillegasn` / `cesar.villegas@utp.edu.co`.
- **Un solo repo** en la raíz del proyecto (`git init -b main`). El `.git` anidado del
  dataset (clon de `martj42`) se eliminó; los CSV quedan como datos del proyecto,
  re-clonables si se quiere refrescar el dataset.
- **`.gitignore`:** `artifacts/`, `__pycache__/`, `external/cache/`, `*.pyc`, venvs.
- **Política de commits:** TDD → commit por task (Gate 4). **Push solo cuando el
  usuario lo pida.** Sin `--no-verify`.
- La carpeta local sigue llamándose con espacio hasta el rename de cierre (opción A);
  el repo remoto se llama `international_results`. Nombre de carpeta ≠ nombre de remote,
  sin impacto en imports (`footy`) ni CLI.

---

## 15. Decisiones registradas

1. Reglas de clínica descartadas; solo TDD sobrevive.
2. Construcción por sub-proyectos; este = sub-proyecto 1 (ETL + baseline jugable).
3. Externas acotadas a las que cubren selecciones (Elo in-house primario, FIFA, Elo
   externo opcional, Transfermarkt best-effort).
4. Entregable = función `predict_match` + CLI. Sin UI.
5. Modelo = Poisson/Dixon-Coles con time-decay (enfoque A). Bivariate Poisson queda como
   extensión futura.
6. Rename de raíz a `analisis_mundial/` se ejecuta al cierre (opción A).
7. Anti-leakage como invariante central (Elo pre-partido, merge_asof backward, golden test).
8. "Goles esperados" = λ del modelo, explícitamente NO xG de tiros.
9. `confidence` → `prediction_reliability` con factores documentados.
10. `baseline_naive_comparison` obligatorio como umbral de utilidad.
11. Repo único en la raíz, `origin` = fork vacío `ddvillegasn/international_results`;
    `.git` anidado del dataset eliminado; push solo a petición.
