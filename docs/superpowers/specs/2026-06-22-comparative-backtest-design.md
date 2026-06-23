# Spec — Evaluación comparativa + backtest histórico · Sub-proyecto 8

- **Fecha:** 2026-06-22
- **Estado:** Aprobado (Gate PASS)
- **Objetivo:** saber si la accuracy (~60%) es buena o mala: comparar el modelo contra
  baselines y hacer **backtest histórico** (entrenar antes de un Mundial, evaluar en él).
- **Depende de:** SP1 (`metrics.py`, `evaluate.py`, Dixon-Coles), SP3 (Elo, datos),
  SP6/SP7 (app). 164 tests verdes.

---

## 1. Contexto y motivación

El scoreboard actual mide solo los 40 partidos jugados del WC2026 (accuracy 60%) sin
referencia. SP8 responde "¿60% es bueno?" de dos formas:
1. **Comparativo**: el modelo vs baselines (Elo, naive, random, etc.) sobre un mismo set.
2. **Backtest histórico**: entrenar estrictamente antes de WC2014/2018/2022 y evaluar cada
   uno → así sabemos el rango típico del modelo (55/60/65%) y si supera a los baselines.

**Dato confirmado:** `international_results/results.csv` tiene WC2014/2018/2022 (64 partidos
c/u), Copa América, Euro, eliminatorias, con columna `tournament` y fechas → backtest viable.
**Nota:** el dataset ya incluye partidos de WC2026; por eso el backtest **siempre** filtra por
fecha (`< inicio de la edición`) para ser out-of-sample honesto.

**Baselines con datos faltantes (decisión del usuario: construir los disponibles ahora):**
- ✅ BASE, LIVE, Elo-favorito, naive-histórico, random.
- ⚠️ FIFA-ranking favorito y Bookmaker favorito → **slots "N/A"** (sin ranking FIFA histórico
  ni cuotas históricas en el repo); se activan cuando se agregue un CSV de ranking o cuotas.

---

## 2. Arquitectura

```
footy/eval/
  __init__.py
  predictors.py       # Predictor1X2 (interfaz) + implementaciones
  evaluate_models.py  # evaluate(predictor, matches) -> métricas
  backtest.py         # backtest por edición (entrena antes, evalúa en la edición)
  report.py           # orquesta -> artifacts/backtest_report.json (cacheado)
footy/metrics.py      # AMPLIAR: calibration_buckets()
footy/cli.py          # nuevo comando: `backtest`
app/streamlit_app.py  # nueva pestaña "Evaluación" (lee el JSON; botón recalcular)
tests/test_*          # predictors, evaluate_models, backtest, calibration
```

Principio: predictores con interfaz común → el `evaluate` y el `backtest` son agnósticos del
predictor. Reusa `footy/metrics.py` (log loss, Brier, accuracy, naive base rates) y la lógica
de fit Dixon-Coles existente. Sin duplicar el motor.

---

## 3. Predictores 1X2 (`predictors.py`)

Interfaz:
```python
class Predictor1X2(Protocol):
    name: str
    def probs(self, team_a, team_b, neutral) -> dict   # {"home","draw","away"} suma 1
    def goals(self, team_a, team_b, neutral) -> tuple | None  # (λ_a, λ_b) o None
```

| Predictor | Construcción | `probs` | `goals` |
|---|---|---|---|
| `DixonColesPredictor` (BASE/LIVE) | `fit_dixon_coles(train, cfg, as_of)` | **analítico**: suma malla Poisson×Poisson×τ (0..max_goals) → P(home/draw/away) | (λ_a, λ_b) |
| `EloFavoritePredictor` | Elo final desde `attach_elo(train)` | P(local)=logística(ΔElo+ventaja); empate = tasa empírica del train; reparte resto | None |
| `NaiveGlobalPredictor` | frecuencias 1X2 globales del train | mismas para todo partido | None |
| `RandomPredictor` | — | `{1/3,1/3,1/3}` | None |
| FIFA / Bookmaker | (sin datos) | no se instancian; el reporte marca "N/A" | None |

- **Analítico Dixon-Coles:** P(x,y) = Poisson(x;λ_a)·Poisson(y;λ_b)·τ(x,y;ρ), normalizado sobre
  la malla; home = Σ_{x>y}, draw = Σ_{x=y}, away = Σ_{x<y}. Determinista, rápido (sin Monte Carlo).
  Test: ≈ resultado de `simulate()` dentro de tolerancia.
- Equipo desconocido (no visto en train) → `probs` cae a las frecuencias globales (no rompe).

---

## 4. Evaluación (`evaluate_models.py`)

```python
def evaluate(predictor, matches) -> dict
```
`matches` = lista de partidos con resultado real (team_a, team_b, neutral, goals_a, goals_b).
Devuelve:
- `accuracy` (1X2), `hits`, `n`.
- `log_loss`, `brier` (reusa `footy/metrics.py`).
- `goal_mae`: solo si el predictor da `goals` (MAE por equipo-partido); si no → `None`.
- `calibration`: buckets de `footy/metrics.calibration_buckets`.

`calibration_buckets(probs, actuals, bins=10) -> dict` (en `footy/metrics.py`): por cada bin de
probabilidad-del-resultado-predicho, cuenta `(prob_media, frecuencia_observada, n)`; agrega
**ECE** (Expected Calibration Error). Test: predicciones perfectas → ECE≈0.

---

## 5. Backtest histórico (`backtest.py`)

```python
def backtest_edition(dataset, tournament, year, predictors_spec, configs) -> dict
def run_backtest(dataset, editions, configs) -> dict
```
- Por edición (tournament+year): `start = ` fecha del primer partido de esa edición en el dataset.
  `train = dataset[date < start]`, `eval = dataset[(tournament==t) & (year==year)]`.
  Construye los predictores **trainables** con `train` (`as_of = start`), evalúa en `eval`.
- **LIVE = BASE en el pasado** (no hay torneo en curso histórico; LIVE solo aplica a WC2026 vivo).
- `run_backtest` corre las ediciones [WC2014, WC2018, WC2022] (config) → métricas por edición +
  agregado por modelo. Extensible a Copa América/Euro/eliminatorias por config.
- **Caché:** los modelos por edición son caros (~2 min el fit); se persisten con
  `footy/persist.load_or_build` (clave = edición + fingerprint del dataset).

Anti-leakage: el train se filtra por fecha estricta `< start`; nunca usa partidos de la edición.

---

## 6. Reporte y CLI (`report.py`, `cli.py`)

```python
run_report(out_path="artifacts/backtest_report.json") -> dict
```
Corre `run_backtest` (histórico) **y** el comparativo en vivo (predictores sobre los partidos
jugados de WC2026 desde `wc2026_results.yaml`, si existen). Escribe JSON:
```json
{
  "historico": {"WC2014": {"BASE": {...}, "Elo": {...}, ...}, "WC2018": {...}, "WC2022": {...},
                "agregado": {"BASE": {"accuracy": 0.58, ...}, ...}},
  "en_vivo_wc2026": {"BASE": {...}, "Elo": {...}, "naive": {...}, "random": {...}},
  "meta": {"editions": [...], "n_partidos": {...}, "generado": "<fecha>"}
}
```
CLI: `backtest [--json] [--editions WC2014,WC2018,WC2022]` → corre el reporte, imprime tabla
comparativa legible, guarda el JSON.

---

## 7. UI — pestaña "Evaluación" (Streamlit)

- **Lee** `artifacts/backtest_report.json` (instantáneo). Si no existe → aviso "Corre el
  backtest" + botón **"Recalcular"** (corre `run_report` con spinner, ~6 min).
- Muestra:
  - Tabla comparativa **histórica agregada**: modelo × (accuracy, log loss, Brier, goal MAE).
    Resalta el mejor por métrica. Random/naive como piso.
  - Selector de edición (WC2014/2018/2022) → tabla de esa edición.
  - **Comparativo en vivo WC2026** (mismos predictores sobre los 40 jugados).
  - **Curva de calibración** (prob predicha vs frecuencia real) + ECE, para BASE.
- Lenguaje claro (español); métricas técnicas con tooltip/expander como en SP7.

---

## 8. Error handling

| Caso | Comportamiento |
|---|---|
| Edición sin partidos en el dataset | se omite con aviso en el reporte (no rompe) |
| Equipo no visto en el train | predictor cae a frecuencias globales |
| FIFA/bookmaker sin datos | slot "N/A" en el reporte |
| Reporte ausente en la UI | aviso + botón recalcular |
| Fit largo | caché por edición (persist); reporte cacheado en JSON |

---

## 9. Testing (sin red; determinismo por seed)

| Test | Verifica |
|---|---|
| `test_predictors` | cada `probs` suma 1; `DixonColesPredictor.probs` analítico ≈ `simulate()` (tol); Elo favorito da más prob al de mayor Elo; random = 1/3; equipo desconocido → globales |
| `test_calibration` | `calibration_buckets`: predicciones perfectas → ECE≈0; mal calibrado → ECE alto; bins suman n |
| `test_evaluate_models` | `evaluate` accuracy/log_loss/brier correctos; goal_mae None para baselines, número para Dixon-Coles |
| `test_backtest` | dataset mini sintético con 2 "ediciones": entrena `< start`, evalúa en la edición; **no usa partidos de la edición** (anti-leakage); agrega por modelo |
| `test_report` | `run_report` escribe JSON con `historico`/`en_vivo`/`meta`; estructura correcta |
| smoke app | la pestaña Evaluación importa |

---

## 10. Decisiones registradas

1. Construir baselines disponibles (BASE, LIVE, Elo, naive, random); FIFA/bookmaker = slots N/A.
2. Backtest = script → `artifacts/backtest_report.json` cacheado; pestaña Evaluación lo lee + botón recalcular.
3. Predictores con interfaz común `probs`/`goals`; evaluator y backtest agnósticos.
4. Dixon-Coles 1X2 **analítico** (malla Poisson×τ), no Monte Carlo, para velocidad/determinismo.
5. Backtest WC2014/2018/2022; train por fecha `< inicio` (anti-leakage); extensible por config.
6. LIVE = BASE en el pasado (LIVE solo aplica a WC2026 vivo).
7. Calibración por buckets + ECE en `footy/metrics.py`.
8. goal MAE solo para modelos que predicen goles.
9. Reutiliza `metrics.py`/`evaluate.py`/fit Dixon-Coles; sin duplicar el motor. Modelo intacto.
