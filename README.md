# footy — international football match predictor

[![tests](https://github.com/ddvillegasn/footy-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/ddvillegasn/footy-predictor/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

### ▶ [Try it live](https://footy-predictor.streamlit.app/)

Runs in the browser, nothing to install.

A prediction engine for **national-team** football matches, built on a **Dixon-Coles**
model (time-decayed Poisson) with **Monte Carlo** simulation, plus a per-match
**betting-markets layer** that prices fair odds and flags value against a bookmaker's
prices.

Point it at any fixture — `Brazil` vs `Haiti` — and get outcome probabilities, a full
scoreline distribution, expected goals and priced markets.

> **Project context.** Built for the 2026 World Cup, which finished in July 2026. The
> **tournament** and **scoreboard** views cover that closed event. The prediction engine
> itself is not tied to it — it fits any national-team fixture from the historical record.

*[Léeme en español](README.es.md)*

---

## Why national teams

Most public football models target club leagues, where event-level data is abundant.
International football has the opposite problem: **no shot-level data**, irregular
fixtures, and squads that turn over between tournament cycles. That constrains what can
honestly be built, and this project is designed around those constraints rather than
pretending they do not exist.

The dataset is ~49,000 matches from 1872 onward — results, competition and venue. Nothing
else.

---

## Statistical honesty

Read this before reading any number the tool prints.

- **`expected_goals_*` is the model's λ, not shot-based xG.** It comes from estimated
  attack/defence strength and home advantage, never from shot events. It is not
  comparable to the xG published on statistics sites.
  [Full explanation →](https://github.com/ddvillegasn/footy-predictor/discussions/6)

- **`prediction_reliability` is not the probability of being correct.** It measures how
  much the model's own inputs deserve trust for this fixture: sample size, data
  freshness, dispersion of simulated outcomes, ranking context.
  [Full explanation →](https://github.com/ddvillegasn/footy-predictor/discussions/8)

- **A value flag is not a profitable bet.** Expected value is computed *relative to the
  model*. It assumes the model's probability is correct, which it cannot verify. Risk
  management is yours.
  [Full explanation →](https://github.com/ddvillegasn/footy-predictor/discussions/9)

A model that oversells its own certainty is worse than no model.

---

## Quick start

Requires Python 3.10+.

```bash
pip install -e ".[dev]"

predict Brazil Haiti --neutral
```

Core dependencies: pandas, numpy, scipy, pyyaml, pyarrow. Tests: pytest.

---

## CLI

```bash
# Outcome probabilities, scoreline, reliability
predict Brazil Haiti --neutral

# Add priced markets (fair odds)
predict Germany "Ivory Coast" --neutral --markets

# Add value detection: paste your bookmaker's prices
predict Germany "Ivory Coast" --neutral \
  --book-odds 1x2.home=1.55,over_under.2.5.over=2.10
```

`--book-odds` takes dotted paths — `market.outcome=odds` or `market.line.side=odds`
(e.g. `over_under.2.5.over=1.67`, `handicap.-1.5.home=1.9`).

## Python

```python
from footy.predict import Predictor
from footy.config import load_config, config_fingerprint
import pandas as pd

# See footy/cli.py::_build_default_predictor for the full wiring from CSV
out = predictor.predict(
    "Brazil", "Haiti",
    neutral=True,
    include_markets=True,
    book_odds={"1x2": {"home": 1.45}},
)
```

### Output reference

| Key | Meaning |
|---|---|
| `team_a_win` / `draw` / `team_b_win` | 1X2 probabilities (%) |
| `expected_goals_a/b` | model λ — **not** shot-based xG |
| `most_likely_score`, `score_distribution` | most probable scoreline plus top-N |
| `confidence_interval` | goal percentiles per team |
| `prediction_reliability` | reliability in [0,1] — **not** P(correct) |
| `model_version` / `betting_version` | engine and betting-layer versions |
| `simulation_meta` | seed, `n_sims`, λ, `dc_enabled`, `clip_max`, config version |
| `markets` | 1X2, double chance, O/U, BTTS, correct score, handicap — with fair odds |
| `value` | edge %, EV, Kelly (raw and quarter), stake recommendation |

`markets` and `value` appear only with `--markets` / `--book-odds`. Without them the
output is the baseline shape, fully backward compatible.

---

## Configuration

Every parameter that matters lives in `configs/*.yaml`. Nothing is hardcoded.

| File | Controls |
|---|---|
| `data.yaml` | CSV paths, name aliases, historical team merges (off by default) |
| `model.yaml` | `xi` (time decay), `home_advantage_init`, `ridge`, `min_matches_reliable`, `model_version` |
| `elo.yaml` | K-factor, initial rating, per-tournament weight |
| `montecarlo.yaml` | `n_sims` (100,000), `seed`, `max_goals`, `ci_level`, `top_scores` |
| `betting.yaml` | O/U and handicap lines, `top_scores`, value thresholds (EV + reliability) |

The reasoning behind keeping these in config rather than in code is written up
[here](https://github.com/ddvillegasn/footy-predictor/discussions/10).

**Why Dixon-Coles rather than two independent Poisson distributions?**
Independence understates draws and low scorelines, which is where most markets are
decided. [Explanation →](https://github.com/ddvillegasn/footy-predictor/discussions/7)

---

## Data

Raw CSVs live in `international_results/` — results, goalscorers, shootouts and former
names, roughly 49,000 matches spanning 1872–2024. Source: the
[martj42/international_results](https://github.com/martj42/international_results)
dataset.

## ETL pipeline

`footy/pipeline.py::run_etl` loads → cleans → normalises team names → computes
pre-match Elo, writing to `artifacts/`: `enriched_matches.parquet`, `etl_report.json`,
`dropped_rows.csv`, `team_name_mapping.json`. `artifacts/` is gitignored.

## Evaluation

`footy/evaluate.py::temporal_holdout` trains up to a cutoff date and evaluates strictly
after it — never on future data — comparing against a naive global-frequency baseline
across log loss, Brier score, accuracy and `beats_baseline`.

## Tests

```bash
pytest
```

Developed test-first. The central invariant is **anti-leakage**: features may only use
information available before the match date, Elo is pre-match, and rankings are
backward-only. It is pinned by a golden test in `tests/test_leakage.py`.

The analytical cross-check of market prices (`tests/test_markets.py`) is valid only with
`dc_enabled=False`, since it assumes independent Poisson sampling.

---

## Roadmap

See [`docs/ALCANCE.md`](docs/ALCANCE.md).

## Questions

Technical questions and answers about the modelling choices live in
[Discussions](https://github.com/ddvillegasn/footy-predictor/discussions).

## License

MIT — see [LICENSE](LICENSE).
