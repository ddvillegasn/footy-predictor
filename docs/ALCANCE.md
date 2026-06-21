# ALCANCE — sistema de predicción de partidos de selecciones

Construcción por sub-proyectos, cada uno con su ciclo spec → plan → build.

## Entregado

### Sub-proyecto 1 — Baseline jugable (motor de partido)
- ETL: loaders tipados, limpieza con reporte, homologación de nombres, Elo in-house
  pre-partido.
- Modelo **Dixon-Coles** (Poisson + time-decay, τ dentro de la verosimilitud).
- **Monte Carlo** (100k sims): 1X2, distribución de marcadores, intervalos.
- `predict_match()` + CLI `predict`.
- Features: forma reciente, porterías en cero, H2H (peso bajo), contexto (torneo,
  continente, Elo/FIFA pre-partido).
- `prediction_reliability` honesto.
- Métricas (log loss, Brier, accuracy) + **baseline ingenuo** + driver de
  **hold-out temporal** (`evaluate.py`).
- Invariante **anti-leakage** con test golden.

### Sub-proyecto 2 — Capa de apuestas por partido
- `simulate_goals()` (refactor de Monte Carlo; una sola fuente de verdad).
- Mercados desde muestras: 1X2, doble oportunidad, over/under, BTTS, marcador exacto
  (top-N + masa restante), hándicap asiático .5.
- Cuotas justas, probabilidad implícita, márgenes.
- **Valor/EV opcional** con cuota de bookie del usuario: edge%, EV, Kelly (raw + ¼),
  `stake_recommendation` (basado en EV **y** fiabilidad).
- `predict(include_markets=, book_odds=)` + flags CLI `--markets` / `--book-odds`.
- Versionado: `betting_version`, `betting_config_version` (hash de config).

## Fuera de alcance (sub-proyectos futuros)

| SP | Contenido |
|---|---|
| 3 | **Simulador de torneo**: colocar un fixture/grupo/bracket; posiciones finales, prob de clasificar/avanzar/campeón (Monte Carlo del torneo) |
| 4 | ML (XGBoost/LightGBM/CatBoost) + backtesting profundo (por torneo/selección, calibración, confusion matrix) + métricas de apuestas **CLV/ROI/Yield** |
| 5 | Deep learning (solo si ML no basta — YAGNI) |
| 6 | Ensemble (weighted / stacking / blending) |
| 7 | Dashboard Streamlit + API REST |
| 8 | MLOps / despliegue |

### No incluido por falta de datos / decisión
- **xG basado en tiros**: no hay datos de remates para selecciones; se usa λ del modelo.
- **τ en el sampleo Monte Carlo**: hoy el sampleo es Poisson independiente
  (`dc_enabled=False`); τ vive en el ajuste. Meterlo al sampleo = v2.
- **Scraping de cuotas de casas**: el usuario pega sus cuotas; no se scrapea.
- Hándicap de líneas enteras (−1, 0, +1) con push: v2.
- Factor **must-win / importancia del partido**: research, no baseline.

## Pendiente operativo
- Rename de la carpeta raíz `analisis mundial` → `analisis_mundial` (hacer al cierre;
  no se puede renombrar el cwd en uso). No afecta imports (`footy`) ni el CLI.
