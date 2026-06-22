# CLAUDE.md — Football Predictor · repo guide

National-team match predictor + betting + live World Cup 2026 simulator + Streamlit
dashboard. Package `footy`. Built via the **superpowers** flow (brainstorm→spec→plan→
subagent-exec→finish), per-sub-project.

## ⏱️ STATE (read first)

- **Branch:** `feature/baseline-v1`, pushed to `origin` = `ddvillegasn/international_results`.
  **PR #1** open (feature/baseline-v1 → main).
- **Full suite green: `python -m pytest` → 164 tests pass** (smoke + live-data ~1.5 min).
- **Done — SP1..SP7 (all complete & pushed):**
  - **SP1** baseline: ETL, Dixon-Coles (`footy/models/poisson.py`), Monte Carlo, `predict_match`, CLI, metrics, temporal hold-out (`footy/evaluate.py`).
  - **SP2** betting per match: `footy/betting/` (markets, odds, value/EV), `predict(include_markets=, book_odds=)`, CLI `--markets/--book-odds`.
  - **SP3** tournament simulator: `footy/tournament/` (structure, results, sampler, groups+FIFA tiebreakers, knockout ET/penalties, simulator, aggregate). Conditions on played results.
  - **SP4** live auto-fetch: `footy/live/` (football-data.org provider, name/stage map, idempotent ingest, scoreboard). CLI `update-and-simulate [--watch] [--json]`.
  - **SP5** Streamlit app + `footy/ui/service.py`.
  - **SP6** live WC2026 mode: `fetch_structure`, `structure_sync`, `stats`, LIVE model refit, refresh button. **Confirmed: football-data free tier returns WC2026 live (104 matches).**
  - **SP7** dashboard redesign: dark theme (`.streamlit/config.toml`), `footy/ui/styles.py` + `footy/ui/components.py`, 4 tabs (Partido/Mundial/Apuestas/Scoreboard), shared selection via `session_state`.
  - **Model persistence:** `footy/persist.py` — BASE/LIVE models pickled to `artifacts/`
    (first fit ~2 min, then loads in ~0.05 s). `artifacts/` is gitignored.

- **NEXT (not started) — SP8: comparative evaluation + historical backtest.** See section below.
  Start with `superpowers:brainstorming` (it's new creative work).

- Specs: `docs/superpowers/specs/` · Plans: `docs/superpowers/plans/` (one pair per SP, dated).

## 🎯 SP8 — what to build next (user's exact ask)

Goal: **know if 60% accuracy is good or bad** by comparing the model against baselines AND
backtesting on past tournaments (not just the 40 live WC2026 matches).

**Comparative scoreboard — evaluate these "models" side by side:**
1. Model BASE (current historical fit)
2. Model LIVE (refit with played)
3. Elo favorite (pick higher pre-match Elo)
4. FIFA-ranking favorite (if ranking data exists — currently FIFA loader is best-effort/None)
5. Bookmaker favorite (if odds provided)
6. Naive: pick team with higher historical win frequency
7. Random 1X2 (theoretical reference)

**Metrics per model:** accuracy 1X2 · hits/matches · log loss · Brier · goal MAE ·
**calibration by probability buckets** (reliability curve / ECE — was SP4 backlog).

**Historical backtest (the important part):** train strictly BEFORE a tournament, evaluate ON it:
- Train < WC2014 → eval WC2014 · Train < WC2018 → eval WC2018 · Train < WC2022 → eval WC2022.
- If data supports: also Copa América, Euro, qualifiers.
- Reuse `footy/evaluate.py::temporal_holdout` + `footy/metrics.py`; the dataset
  (`international_results/results.csv`) has `tournament` column to filter editions and dates
  to split. Anti-leakage: features/fit only from rows with date < tournament start.

Likely shape: `footy/backtest.py` (run a tournament backtest for each model) + a new
Streamlit "Evaluación" tab or section showing the comparative table + calibration. TDD,
mock/fixtures (no network). This realizes the roadmap "SP4 deep backtest (CLV/ROI later)".

## How to run

```bash
python -m pytest                 # full suite
pip install -e .[ui]             # once (installs streamlit)
streamlit run app/streamlit_app.py   # dashboard (1st start ~2 min: fits+caches model)
predict Brazil Haiti --neutral --markets        # CLI match
update-and-simulate --watch 15                  # CLI live fetch+sim (needs token)
```

## Live data (football-data.org)

- Token via env `FOOTBALL_DATA_API_KEY` **or** `configs/secrets.local.yaml`
  (`football_data_api_key:` — gitignored). Free tier covers competition `WC` = WC2026 live.
- App sidebar "🔄 Actualizar desde API" → `sync_structure` (real groups → `wc2026.yaml`) +
  `ingest` (FINISHED → `wc2026_results.yaml`). Name mismatches → hard error listing all;
  add to `configs/name_map.yaml` (e.g. Czechia→Czech Republic, Cape Verde Islands→Cape Verde,
  Congo DR→DR Congo already mapped).
- **Working tree note:** `configs/tournaments/wc2026.yaml` + `wc2026_results.yaml` hold the
  user's real live data locally (modified, intentionally NOT committed — repo keeps clean
  defaults so tests pass on fresh checkout). Don't commit them unless asked.

## Two models (don't mix)

- **BASE** = historical fit (to 2024). Used by the **scoreboard** (out-of-sample, no leakage).
- **LIVE** = BASE dataset + played WC matches refit. Used by **predictions/sim** when toggled.
  Built via `footy/ui/service.build_live_predictor` (carries `betting_config`!).

## Environment quirks (IMPORTANT)

- **Python is 3.10.6, NOT 3.12.** `pyproject.toml requires-python = ">=3.10"`. Code is 3.10-safe.
- **pytest-asyncio in the env is broken** → disabled in `pyproject.toml`
  `[tool.pytest.ini_options] addopts = "-p no:asyncio"`. Do not remove.
- Windows + Git Bash. CRLF warnings on commit are harmless. `gh` CLI NOT installed (PRs via web).
- `streamlit run` boot is fast; the ~2 min cost is the one-time model fit (then cached to disk).

## Execution method (cost control — user's rules)

`superpowers:subagent-driven-development`. **Fresh subagent per task**, reads ONLY its
`## Task N` from the plan, strict TDD (red→green→commit). **Haiku for mechanical, Sonnet for
judgment/integration.** **Main thread reviews each diff itself** (no reviewer subagents): after
each task `git show --stat HEAD`, ensure no `*.pyc` tracked, code matches plan. Trivial/doc
tasks done inline. Commit per task; `git add` ONLY named files. Trailer:
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. **Push only when user asks.**

## Conventions / invariants (non-negotiable)

- **TDD**: failing test first, every task. Never weaken a test to make it pass.
- **Anti-leakage** (first-class): features only from rows with date `< match_date`; Elo
  pre-match; ranking merge backward-only; scoreboard uses BASE (out-of-sample). Golden:
  `footy/features/leakage.py`, `tests/test_leakage.py`.
- **Config-driven**: params in `configs/*.yaml`. No hardcoding.
- **Best-effort external** (FIFA/Transfermarkt): missing → None, never raise.
- **Honesty**: `expected_goals_*` = model λ (NOT shot xG). `prediction_reliability` ≠ P(correct).
  EV = value relative to model, not guaranteed. Model reacts little to one result (correct).
- Tests: **no real network** (FakeProvider / HTTP mock). Discarded clinic rules (clinica_id/
  /health/etc.) — only TDD applies.

## Repo map

```
footy/
  config.py persist.py predict.py reliability.py metrics.py cli.py evaluate.py pipeline.py
  data/        loaders clean names · external/(elo fifa transfermarkt)
  features/    leakage strength context
  models/      poisson(Dixon-Coles) montecarlo(+simulate_goals)
  betting/     markets odds value
  tournament/  structure results sampler groups knockout simulator aggregate
  live/        provider name_map ingest scoreboard structure_sync stats runner
  ui/          service styles components
app/streamlit_app.py        configs/*.yaml (+ tournaments/wc2026*.yaml, name_map, live, secrets.local)
tests/  test_*.py           artifacts/ (gitignored: base_predictor.pkl, live_predictor.pkl, ui_preview.html)
docs/superpowers/  specs/ plans/   docs/ALCANCE.md   README.md
international_results/  raw dataset CSVs (results 49k rows, goalscorers, shootouts, former_names)
```

## Roadmap after SP8

CLV/ROI/Yield betting metrics · ML (XGB/LGBM/CatBoost) ensemble vs Dixon-Coles · API REST ·
MLOps. Pending chore: rename root folder `analisis mundial` → `analisis_mundial` (user doesn't
care for now; do at the very end — can't rename cwd in use).
