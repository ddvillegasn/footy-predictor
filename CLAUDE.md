# CLAUDE.md — Football Predictor · repo guide

National-team match predictor. Package `footy`. Dixon-Coles Poisson + time-decay +
Monte Carlo → `predict_match` + `predict` CLI, plus a per-match **betting layer**
(markets, fair odds, value/EV).

## ⏱️ STATE (read first)

- **Branch:** `feature/baseline-v1`. Pushed to `origin` (ddvillegasn/international_results).
- **SP1 (baseline) — COMPLETE:** Tasks 0–18. ETL, Dixon-Coles, Monte Carlo, features,
  predict, CLI, metrics, ETL pipeline, real-data smoke test, **temporal hold-out**
  (`evaluate.py`), docs.
- **SP2 (betting layer) — COMPLETE:** `simulate_goals` refactor, `footy/betting/`
  (markets, odds, value), `predict(include_markets=, book_odds=)`, CLI `--markets` /
  `--book-odds`.
- **Full suite green: `python -m pytest` → 83 tests pass** (smoke test ~2 min).
- **NEXT (not started):** SP3 — tournament/fixture simulator (groups, bracket,
  qualification probabilities). Needs its own spec→plan→build (brainstorm first).
- Specs: `docs/superpowers/specs/2026-06-20-football-predictor-baseline-design.md`,
  `…/2026-06-20-betting-markets-layer-design.md`
- Plans: `docs/superpowers/plans/2026-06-20-football-predictor-baseline.md`,
  `…/2026-06-20-betting-markets-layer.md`
- Scope/roadmap: `docs/ALCANCE.md`. User manual: `README.md`.

## How to resume execution

Method = **superpowers:subagent-driven-development** (user chose it). Per user cost rules:
- **Fresh subagent per task.** Subagent reads ONLY its `## Task N` section from the
  plan file and implements EXACTLY (TDD: write failing test → confirm fail →
  implement plan code verbatim → confirm pass → commit).
- **Model:** Haiku for mechanical tasks; **Sonnet for judgment** (Task 9 Dixon-Coles,
  Task 12 predict, Task 16 smoke, Task 18 evaluate).
- **Reviewer = the main thread reviews each diff itself** (user said NO separate
  reviewer subagents). After each task: `git show --stat HEAD`, check no `*.pyc`
  tracked, spot-check code matches plan.
- Trivial tasks may be done inline instead of dispatching.

Dispatch prompt template that worked (adapt Task number/files):
> Implement ONE task, strict TDD, don't improvise. cwd `c:\Users\pc\Desktop\analisis mundial`,
> branch feature/baseline-v1, Python 3.10.6, run `python -m pytest <path> -v`.
> Read ONLY `## Task N` from the plan file. Create the failing test, confirm fail,
> create the impl code verbatim, confirm pass, commit. `git add` ONLY the specific
> files (never directories/__pycache__/.pyc). Commit msg from plan Step 5 with trailer
> `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. If a test fails with code
> written exactly as specified, report BLOCKED with output — do NOT weaken tests.

## Remaining tasks (model · one-liner)

- **Task 9** · Sonnet · `footy/models/poisson.py` — Dixon-Coles MLE (τ inside
  likelihood, time-decay, ridge for identifiability). Risk: optimizer convergence on
  fixture. If a test fails, first verify exact transcription of the NLL/_tau/bounds.
- **Task 10** · Haiku · `footy/models/montecarlo.py` — seeded MC, 1X2, score dist, CI.
- **Task 11** · Haiku · `footy/reliability.py` — `compute_reliability` (NOT win prob).
- **Task 12** · Sonnet · `footy/predict.py` — `Predictor` + `predict()` output dict.
- **Task 13** · Haiku · `footy/metrics.py` — log loss, brier, accuracy, naive baseline.
- **Task 14** · Haiku · `footy/cli.py` — `predict <a> <b> [--neutral] [--tournament]`.
- **Task 15** · Haiku · `footy/pipeline.py` — ETL writes report/mapping/enriched parquet.
- **Task 16** · Sonnet · `tests/test_smoke_real.py` — end-to-end on real dataset.
- **Task 17** · Sonnet · docs: `README.md`, this `CLAUDE.md` (refine), `docs/ALCANCE.md`.
- **Task 18** · Sonnet · `footy/evaluate.py` — temporal hold-out vs naive baseline.

After Task 18 → **Gate 4 (superpowers:finishing-a-development-branch):** run full suite,
finalize docs, confirm commits per task, then ask user before any push.

## Commands

```bash
python -m pytest            # full suite (pytest-asyncio auto-disabled via pyproject)
python -m pytest tests/test_X.py -v
predict Brazil Haiti --neutral   # after Task 14 (needs `pip install -e .` OR run module)
```

## Environment quirks (IMPORTANT)

- **Python is 3.10.6, NOT 3.12.** `pyproject.toml` `requires-python` lowered to
  `>=3.10`. Code is 3.10-compatible (PEP 604 `X | None` unions OK).
- **pytest-asyncio in the env is broken** (crashes on package collection:
  `'Package' object has no attribute 'obj'`). Disabled in `pyproject.toml`
  `[tool.pytest.ini_options] addopts = "-p no:asyncio"`. Do not remove.
- Windows + Git Bash. CRLF warnings on commit are harmless.

## Conventions / invariants (non-negotiable)

- **TDD:** failing test first, every task. Never weaken a test to make it pass.
- **Anti-leakage** (first-class): features use only rows with date `< match_date`;
  Elo exposed is **pre-match**; ranking merge is backward-only (`last_ranking_before`).
  Golden guard: `footy/features/leakage.py::assert_no_leakage`, test `tests/test_leakage.py`.
- **Config-driven:** params live in `configs/*.yaml` (data/model/elo/montecarlo). No
  hardcoding.
- **Best-effort external** (FIFA/Transfermarkt): missing/invalid → `None`, never raise.
- **Naming:** `team_a`/`team_b` (not home/away); `neutral=True` ⇒ `home_advantage=0`.
- **Honesty:** `expected_goals_*` = model λ (NOT shot-based xG, no shot data exists).
  `prediction_reliability` ≠ probability of being correct.
- **Versioning:** `model_version = baseline-v1.0.0` (SemVer w/ family prefix).
- Git: commit per task, specific files only, `Co-Authored-By: Claude Opus 4.8
  <noreply@anthropic.com>` trailer. Origin = `ddvillegasn/international_results` (empty
  fork). **Push only when user explicitly asks.**

## Discarded clinic rules

Prompt's "non-negotiable" rules (`clinica_id`/anti-IDOR, DB best-effort, `/health`
deploy, clinic golden test) came from another project. **Discarded; only TDD applies.**
Sane reinterpretations kept: external loaders tolerant to failure; golden = anti-leakage.

## Repo map

```
footy/
  config.py            data/loaders.py clean.py names.py
  data/external/       elo.py fifa.py transfermarkt.py  (+ cache/)
  features/            leakage.py strength.py context.py
  models/              poisson.py(T9) montecarlo.py(T10)
  predict.py(T12) reliability.py(T11) metrics.py(T13) cli.py(T14)
  pipeline.py(T15) evaluate.py(T18)
configs/  data.yaml model.yaml elo.yaml montecarlo.yaml
tests/    test_*.py  fixtures/
international_results/  raw dataset CSVs (results, goalscorers, shootouts, former_names)
artifacts/  gitignored training outputs
docs/superpowers/  specs/ plans/   docs/ALCANCE.md (T17)
```

## Roadmap (later sub-projects)

2 features+Elo ampliado · 3 ML (XGB/LGBM/CatBoost)+backtesting · 4 deep backtest
(per-tournament, calibration, **CLV/ROI/Yield** for betting) · 5 ensemble · 6 Streamlit+API
· 7 MLOps. Also pending: rename root folder `analisis mundial` → `analisis_mundial`
(do at close, can't rename cwd in use).
