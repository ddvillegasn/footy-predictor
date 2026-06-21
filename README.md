# footy — predictor de partidos de selecciones + capa de apuestas

Motor que predice partidos internacionales de selecciones con un modelo
**Dixon-Coles** (Poisson con time-decay) y **Monte Carlo**, y una capa de
**apuestas por partido** (mercados, cuotas justas, valor/EV opcional).

Pensado para colocar **cualquier partido** (ej. `Brazil` vs `Haiti`) y obtener
probabilidades reales, distribución de marcadores, goles esperados y mercados.

> **Honestidad estadística**
> - `expected_goals_*` = la **λ del modelo**, NO xG basado en tiros (no hay datos de
>   remates para selecciones).
> - `prediction_reliability` **no** es la probabilidad de acertar; mide cuánta confianza
>   merece la predicción (tamaño de muestra, frescura de datos, dispersión, rankings).
> - El **valor/EV** asume que la probabilidad del modelo es correcta: es valor *relativo
>   al modelo*, no ganancia garantizada. Gestión de riesgo es responsabilidad tuya.

## Instalación

Requiere Python 3.10+.

```bash
pip install -e .[dev]
```

Dependencias núcleo: pandas, numpy, scipy, pyyaml, pyarrow. Tests: pytest.

## Datos

Los CSV crudos viven en `international_results/` (results, goalscorers, shootouts,
former_names; ~49k partidos 1872–2024). Fuente: dataset martj42/international_results.

## Configuración (`configs/*.yaml`)

| Archivo | Qué controla |
|---|---|
| `data.yaml` | rutas CSV, alias de nombres, fusiones históricas (off por defecto) |
| `model.yaml` | `xi` (time-decay), `home_advantage_init`, `ridge`, `min_matches_reliable`, `model_version` |
| `elo.yaml` | K-factor, rating inicial, peso por torneo |
| `montecarlo.yaml` | `n_sims` (100000), `seed`, `max_goals`, `ci_level`, `top_scores` |
| `betting.yaml` | líneas O/U y hándicap, `top_scores`, umbrales de valor (EV + fiabilidad) |

Nada está hardcodeado: todo parámetro relevante sale de estos YAML.

## Uso — CLI

```bash
# Predicción simple (1X2, marcador, fiabilidad)
predict Brazil Haiti --neutral

# Con mercados de apuestas (cuotas justas)
predict Germany "Ivory Coast" --neutral --markets

# Con detección de valor: pega tus cuotas de bookie
predict Germany "Ivory Coast" --neutral --book-odds 1x2.home=1.55,over_under.2.5.over=2.10
```

`--book-odds` acepta rutas con puntos: `mercado.outcome=cuota` o
`mercado.linea.lado=cuota` (ej. `over_under.2.5.over=1.67`, `handicap.-1.5.home=1.9`).

## Uso — Python

```python
from footy.predict import Predictor
from footy.config import load_config, config_fingerprint
import pandas as pd

# (ver footy/cli.py::_build_default_predictor para el armado completo desde CSV)
out = predictor.predict("Brazil", "Haiti", neutral=True,
                        include_markets=True,
                        book_odds={"1x2": {"home": 1.45}})
```

### Salida (claves principales)

| Clave | Significado |
|---|---|
| `team_a_win` / `draw` / `team_b_win` | probabilidades 1X2 (%) |
| `expected_goals_a/b` | λ del modelo (no xG de tiros) |
| `most_likely_score`, `score_distribution` | marcador más probable + top-N |
| `confidence_interval` | percentiles de goles por equipo |
| `prediction_reliability` | fiabilidad [0,1] (no prob. de acierto) |
| `model_version` / `betting_version` | versiones del motor y de la capa de apuestas |
| `simulation_meta` | seed, n_sims, λ, `dc_enabled`, `clip_max`, `betting_config_version` |
| `markets` | 1X2, doble oportunidad, O/U, BTTS, marcador exacto, hándicap (con cuotas justas) |
| `value` | edge%, EV, Kelly (raw + ¼), `stake_recommendation` (solo donde pegaste cuota) |

`markets`/`value` solo aparecen con `--markets`/`--book-odds`; sin ellos la salida es la
del baseline (compatibilidad total).

## Pipeline ETL (artifacts)

`footy/pipeline.py::run_etl` carga → limpia → homologa nombres → Elo pre-partido, y
escribe en `artifacts/`: `enriched_matches.parquet`, `etl_report.json`,
`dropped_rows.csv`, `team_name_mapping.json`. `artifacts/` está gitignored.

## Evaluación

`footy/evaluate.py::temporal_holdout` entrena hasta una fecha de corte, evalúa después
(nunca con datos futuros) y compara contra el baseline ingenuo de frecuencias globales:
log loss, Brier, accuracy y `beats_baseline`.

## Tests

```bash
python -m pytest          # suite completa
```

TDD estricto. Invariante central **anti-leakage** (features solo con info previa a la
fecha del partido; Elo pre-partido; ranking backward-only) con test golden
(`tests/test_leakage.py`). El cross-check analítico de mercados
(`tests/test_markets.py`) es válido solo con `dc_enabled=False` (sampleo Poisson
independiente).

## Roadmap

Ver `docs/ALCANCE.md`.
