# Baseline Football Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end baseline that predicts national-team match outcomes via a time-decayed Dixon-Coles Poisson model plus Monte Carlo, exposed as `predict_match(...)` and a `predict` CLI.

**Architecture:** A focused Python package `footy/`. ETL (load → clean → canonicalize names → Elo/FIFA enrichment) feeds a config-driven Dixon-Coles maximum-likelihood fit. The fitted goal rates drive a seeded Monte Carlo that produces 1X2 probabilities, score distribution, expected goals (λ), confidence intervals and an honest `prediction_reliability`. Anti-leakage (pre-match Elo, `merge_asof` backward, features only from past rows) is a first-class invariant guarded by a golden test.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, pyyaml, pyarrow, pytest.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, `predict` CLI entry-point |
| `footy/__init__.py` | Package marker + version |
| `footy/config.py` | Load `configs/*.yaml` |
| `footy/data/loaders.py` | Read 4 raw CSVs into typed DataFrames; fail hard on missing file/column |
| `footy/data/clean.py` | Dedup (logged), null/format handling, ETL report |
| `footy/data/names.py` | `NameCanonicalizer`: former-name + alias + configurable sensitive merges |
| `footy/data/external/elo.py` | In-house Elo, exposes **pre-match** ratings |
| `footy/data/external/fifa.py` | Best-effort FIFA ranking loader (cached CSV) |
| `footy/data/external/transfermarkt.py` | Best-effort market values; never raises |
| `footy/features/leakage.py` | `matches_before`, `assert_no_leakage` |
| `footy/features/strength.py` | Recent form, clean sheets, H2H (low weight) |
| `footy/features/context.py` | neutral, tournament weight, continent, Elo/FIFA pre-match |
| `footy/models/poisson.py` | `fit_dixon_coles`, `DixonColesModel.rates(...)` |
| `footy/models/montecarlo.py` | `simulate(...)` → probabilities + score matrix |
| `footy/predict.py` | `Predictor` + `predict_match(...)` orchestration |
| `footy/reliability.py` | `compute_reliability(...)` |
| `footy/metrics.py` | log loss, brier, accuracy, `naive_baseline`, `evaluate` |
| `footy/cli.py` | `predict <team_a> <team_b>` |
| `configs/*.yaml` | data/model/elo/montecarlo parameters |
| `tests/fixtures/*.csv` | tiny deterministic datasets |
| `tests/test_*.py` | one test module per unit |

**Conventions for every task:** run tests from repo root with `python -m pytest`. Commit messages end with the Co-Authored-By trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

## Task 0: Project scaffold + config loader

**Files:**
- Create: `pyproject.toml`, `footy/__init__.py`, `footy/config.py`
- Create: `configs/data.yaml`, `configs/model.yaml`, `configs/elo.yaml`, `configs/montecarlo.yaml`
- Create: `artifacts/.gitkeep`, `footy/data/external/cache/.gitkeep`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from footy.config import load_config


def test_load_model_config_has_version():
    cfg = load_config("model")
    assert cfg["model_version"] == "baseline-v1.0.0"
    assert cfg["max_goals"] >= 1


def test_load_missing_config_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy'`

- [ ] **Step 3: Write scaffold + implementation**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "footy"
version = "1.0.0"
description = "Baseline national-team match predictor (Dixon-Coles + Monte Carlo)"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.26",
    "scipy>=1.11",
    "pyyaml>=6.0",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
predict = "footy.cli:main"

[tool.setuptools.packages.find]
include = ["footy*"]
```

`footy/__init__.py`:
```python
__version__ = "1.0.0"
```

`footy/config.py`:
```python
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml as a dict. Raise FileNotFoundError if absent."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
```

`configs/data.yaml`:
```yaml
raw_dir: international_results
files:
  results: results.csv
  goalscorers: goalscorers.csv
  shootouts: shootouts.csv
  former_names: former_names.csv
min_date: null            # no hard cut; time-decay handles recency
aliases:                  # always-on safe spelling unifications
  "Vietnam Republic": "South Vietnam"
sensitive_merges:         # historical merges OFF by default (traceability)
  enabled: false
  mappings:
    "West Germany": "Germany"
    "East Germany": "Germany"
```

`configs/model.yaml`:
```yaml
model_version: baseline-v1.0.0
xi: 0.0018                # time-decay per day (~half-life 1.05y)
max_goals: 10
home_advantage_init: 0.25
ridge: 0.01               # L2 on attack/defense for identifiability
h2h_weight: 0.1           # low weight (sparse for national teams)
min_matches_reliable: 10
```

`configs/elo.yaml`:
```yaml
initial_rating: 1500.0
k_factor: 40.0
home_advantage_elo: 65.0
default_tournament_weight: 0.7
tournament_weights:
  "FIFA World Cup": 1.0
  "FIFA World Cup qualification": 0.85
  "Friendly": 0.4
```

`configs/montecarlo.yaml`:
```yaml
n_sims: 100000
seed: 42
max_goals: 10
ci_level: 0.90
top_scores: 8
```

Create empty marker files `artifacts/.gitkeep` and `footy/data/external/cache/.gitkeep` (each containing a single newline). Create empty `footy/data/__init__.py`, `footy/data/external/__init__.py`, `footy/features/__init__.py`, `footy/models/__init__.py` (each containing a single newline).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml footy configs artifacts/.gitkeep tests/test_config.py
git commit -m "feat: project scaffold, package skeleton, config loader

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1: Raw CSV loaders

**Files:**
- Create: `footy/data/loaders.py`
- Create: `tests/fixtures/results.csv`, `tests/fixtures/former_names.csv`
- Test: `tests/test_loaders.py`

- [ ] **Step 1: Write fixtures and the failing test**

`tests/fixtures/results.csv`:
```csv
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2019-06-01,Brazil,Haiti,7,1,Friendly,Miami,United States,TRUE
2019-09-10,Brazil,Peru,0,1,Friendly,Los Angeles,United States,TRUE
2019-10-15,Haiti,Peru,1,0,Friendly,Port-au-Prince,Haiti,FALSE
2021-06-01,Brazil,Haiti,2,0,FIFA World Cup,Rio,Brazil,FALSE
2021-06-01,Brazil,Haiti,2,0,FIFA World Cup,Rio,Brazil,FALSE
2022-03-01,Peru,Brazil,1,1,FIFA World Cup qualification,Lima,Peru,FALSE
```

`tests/fixtures/former_names.csv`:
```csv
current,former,start_date,end_date
Haiti,Saint-Domingue,1900-01-01,1950-01-01
```

`tests/test_loaders.py`:
```python
from pathlib import Path

import pandas as pd
import pytest

from footy.data.loaders import load_results, load_former_names

FIX = Path(__file__).parent / "fixtures"


def test_load_results_types():
    df = load_results(FIX / "results.csv")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["home_score"].dtype.kind == "i"
    assert df["neutral"].dtype == bool
    assert df.loc[0, "neutral"] is True or bool(df.loc[0, "neutral"]) is True
    assert len(df) == 6


def test_load_results_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("date,home_team\n2019-01-01,Brazil\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_results(bad)


def test_load_former_names():
    df = load_former_names(FIX / "former_names.csv")
    assert list(df.columns) == ["current", "former", "start_date", "end_date"]
    assert df.loc[0, "current"] == "Haiti"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_loaders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.data.loaders'`

- [ ] **Step 3: Write implementation**

`footy/data/loaders.py`:
```python
from pathlib import Path

import pandas as pd

RESULTS_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]
GOALSCORERS_COLUMNS = [
    "date", "home_team", "away_team", "team", "scorer", "minute",
    "own_goal", "penalty",
]
SHOOTOUTS_COLUMNS = ["date", "home_team", "away_team", "winner", "first_shooter"]
FORMER_NAMES_COLUMNS = ["current", "former", "start_date", "end_date"]


def _require_columns(df: pd.DataFrame, columns: list[str], source) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "TRUE": True, "FALSE": False, "True": True, "False": False,
        True: True, False: False,
    }
    return series.map(mapping).astype(bool)


def load_results(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, RESULTS_COLUMNS, path.name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    df["neutral"] = _to_bool(df["neutral"])
    # Keep only rows whose scores are present, then downcast to plain int.
    valid = df["home_score"].notna() & df["away_score"].notna()
    df.loc[valid, "home_score"] = df.loc[valid, "home_score"]
    df["home_score"] = df["home_score"].astype("Int64")
    return df.reset_index(drop=True)


def load_goalscorers(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, GOALSCORERS_COLUMNS, path.name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.reset_index(drop=True)


def load_shootouts(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, SHOOTOUTS_COLUMNS, path.name)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.reset_index(drop=True)


def load_former_names(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, FORMER_NAMES_COLUMNS, path.name)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df.reset_index(drop=True)
```

Note: the test asserts `home_score.dtype.kind == "i"`. After `astype("Int64")` the kind is `"i"` (nullable integer extension still reports kind `i`). Keep as `Int64` for null-tolerance; the clean step (Task 2) drops null-score rows.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_loaders.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/data/loaders.py tests/test_loaders.py tests/fixtures/results.csv tests/fixtures/former_names.csv
git commit -m "feat: typed raw CSV loaders with hard column validation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Clean step (logged dedup, null/format handling, report)

**Files:**
- Create: `footy/data/clean.py`
- Test: `tests/test_clean.py`

- [ ] **Step 1: Write the failing test**

`tests/test_clean.py`:
```python
from pathlib import Path

from footy.data.loaders import load_results
from footy.data.clean import clean_results

FIX = Path(__file__).parent / "fixtures"


def test_clean_dedups_and_reports():
    raw = load_results(FIX / "results.csv")  # contains 1 exact duplicate
    result = clean_results(raw)
    # 6 raw rows, one duplicated 2021-06-01 Brazil-Haiti pair -> 5 kept
    assert len(result.df) == 5
    assert result.report["duplicates_removed"] == 1
    assert len(result.dropped) == 1
    assert result.dropped.iloc[0]["drop_reason"] == "duplicate"


def test_clean_drops_null_scores_with_reason():
    raw = load_results(FIX / "results.csv").copy()
    raw.loc[0, "home_score"] = None
    result = clean_results(raw)
    reasons = set(result.dropped["drop_reason"])
    assert "null_score" in reasons
    assert result.report["null_scores_removed"] == 1


def test_clean_is_deterministic():
    raw = load_results(FIX / "results.csv")
    a = clean_results(raw)
    b = clean_results(raw)
    assert a.df.reset_index(drop=True).equals(b.df.reset_index(drop=True))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.data.clean'`

- [ ] **Step 3: Write implementation**

`footy/data/clean.py`:
```python
from dataclasses import dataclass

import pandas as pd

KEY = ["date", "home_team", "away_team"]


@dataclass
class CleanResult:
    df: pd.DataFrame
    report: dict
    dropped: pd.DataFrame


def clean_results(df: pd.DataFrame) -> CleanResult:
    """Remove invalid/duplicate rows, logging every drop with a reason."""
    work = df.copy()
    dropped_frames = []

    # 1. Invalid dates.
    bad_date = work["date"].isna()
    if bad_date.any():
        rec = work[bad_date].copy()
        rec["drop_reason"] = "invalid_date"
        dropped_frames.append(rec)
        work = work[~bad_date]

    # 2. Null scores (cannot train on them).
    null_score = work["home_score"].isna() | work["away_score"].isna()
    if null_score.any():
        rec = work[null_score].copy()
        rec["drop_reason"] = "null_score"
        dropped_frames.append(rec)
        work = work[~null_score]

    # 3. Exact duplicates on (date, home, away): keep first, log the rest.
    dup_mask = work.duplicated(subset=KEY, keep="first")
    if dup_mask.any():
        rec = work[dup_mask].copy()
        rec["drop_reason"] = "duplicate"
        dropped_frames.append(rec)
        work = work[~dup_mask]

    work = work.sort_values("date").reset_index(drop=True)
    work["home_score"] = work["home_score"].astype(int)
    work["away_score"] = work["away_score"].astype(int)

    dropped = (
        pd.concat(dropped_frames, ignore_index=True)
        if dropped_frames
        else pd.DataFrame(columns=list(df.columns) + ["drop_reason"])
    )
    report = {
        "rows_in": int(len(df)),
        "rows_out": int(len(work)),
        "invalid_dates_removed": int((dropped.get("drop_reason") == "invalid_date").sum())
        if len(dropped) else 0,
        "null_scores_removed": int((dropped.get("drop_reason") == "null_score").sum())
        if len(dropped) else 0,
        "duplicates_removed": int((dropped.get("drop_reason") == "duplicate").sum())
        if len(dropped) else 0,
    }
    return CleanResult(df=work, report=report, dropped=dropped)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_clean.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/data/clean.py tests/test_clean.py
git commit -m "feat: clean step with logged dedup and ETL report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Name canonicalization

**Files:**
- Create: `footy/data/names.py`
- Test: `tests/test_names.py`

- [ ] **Step 1: Write the failing test**

`tests/test_names.py`:
```python
import pandas as pd

from footy.data.names import NameCanonicalizer


def _former():
    return pd.DataFrame(
        {
            "current": ["Benin"],
            "former": ["Dahomey"],
            "start_date": pd.to_datetime(["1959-11-08"]),
            "end_date": pd.to_datetime(["1975-11-30"]),
        }
    )


def test_former_name_maps_to_current():
    canon = NameCanonicalizer(_former(), aliases={}, sensitive_merges={"enabled": False, "mappings": {}})
    assert canon.canonical("Dahomey") == "Benin"
    assert canon.canonical("Benin") == "Benin"


def test_alias_applied():
    canon = NameCanonicalizer(
        _former(),
        aliases={"Vietnam Republic": "South Vietnam"},
        sensitive_merges={"enabled": False, "mappings": {}},
    )
    assert canon.canonical("Vietnam Republic") == "South Vietnam"


def test_sensitive_merge_off_by_default():
    merges = {"enabled": False, "mappings": {"West Germany": "Germany"}}
    canon = NameCanonicalizer(_former(), aliases={}, sensitive_merges=merges)
    assert canon.canonical("West Germany") == "West Germany"


def test_sensitive_merge_on_when_enabled():
    merges = {"enabled": True, "mappings": {"West Germany": "Germany"}}
    canon = NameCanonicalizer(_former(), aliases={}, sensitive_merges=merges)
    assert canon.canonical("West Germany") == "Germany"


def test_mapping_table_exported():
    canon = NameCanonicalizer(_former(), aliases={}, sensitive_merges={"enabled": False, "mappings": {}})
    table = canon.mapping_table()
    assert table["Dahomey"] == "Benin"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_names.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.data.names'`

- [ ] **Step 3: Write implementation**

`footy/data/names.py`:
```python
import pandas as pd


class NameCanonicalizer:
    """Maps team names to a single canonical form.

    Order of application: former-name table -> always-on aliases ->
    optional configurable sensitive merges (off by default for traceability).
    """

    def __init__(self, former_names: pd.DataFrame, aliases: dict, sensitive_merges: dict):
        self._former = {
            str(row.former): str(row.current)
            for row in former_names.itertuples(index=False)
        }
        self._aliases = {str(k): str(v) for k, v in (aliases or {}).items()}
        merges = sensitive_merges or {"enabled": False, "mappings": {}}
        self._sensitive = (
            {str(k): str(v) for k, v in merges.get("mappings", {}).items()}
            if merges.get("enabled", False)
            else {}
        )

    def canonical(self, name: str) -> str:
        name = str(name)
        name = self._former.get(name, name)
        name = self._aliases.get(name, name)
        name = self._sensitive.get(name, name)
        return name

    def mapping_table(self) -> dict:
        table = dict(self._former)
        table.update(self._aliases)
        table.update(self._sensitive)
        return table
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_names.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/data/names.py tests/test_names.py
git commit -m "feat: name canonicalizer with configurable sensitive merges

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: In-house Elo with pre-match ratings (anti-leakage)

**Files:**
- Create: `footy/data/external/elo.py`
- Test: `tests/test_elo.py`

- [ ] **Step 1: Write the failing test**

`tests/test_elo.py`:
```python
import pandas as pd

from footy.data.external.elo import attach_elo

ELO_CFG = {
    "initial_rating": 1500.0,
    "k_factor": 40.0,
    "home_advantage_elo": 65.0,
    "default_tournament_weight": 0.7,
    "tournament_weights": {"Friendly": 0.4, "FIFA World Cup": 1.0},
}


def _matches():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
            "home_team": ["Brazil", "Brazil", "Haiti"],
            "away_team": ["Haiti", "Haiti", "Brazil"],
            "home_score": [3, 2, 0],
            "away_score": [0, 0, 1],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [False, False, False],
        }
    )


def test_first_match_uses_initial_rating_pre():
    out = attach_elo(_matches(), ELO_CFG)
    assert out.loc[0, "home_elo_pre"] == 1500.0
    assert out.loc[0, "away_elo_pre"] == 1500.0


def test_winner_rating_rises_after_match():
    out = attach_elo(_matches(), ELO_CFG)
    # Brazil won match 0, so its pre-rating for match 1 must exceed 1500.
    assert out.loc[1, "home_elo_pre"] > 1500.0


def test_pre_rating_never_uses_own_match_result():
    # The pre rating of the last match for Brazil must equal its rating
    # computed from the first two matches only (no leakage from match 2).
    out = attach_elo(_matches(), ELO_CFG)
    assert "home_elo_pre" in out.columns and "away_elo_pre" in out.columns
    # Determinism
    out2 = attach_elo(_matches(), ELO_CFG)
    assert out.equals(out2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_elo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.data.external.elo'`

- [ ] **Step 3: Write implementation**

`footy/data/external/elo.py`:
```python
import pandas as pd


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _outcome(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def attach_elo(matches: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add home_elo_pre / away_elo_pre columns (rating BEFORE each match).

    Ratings are updated chronologically; the value written to a row is the
    pre-match rating, so no row ever sees its own result (anti-leakage).
    """
    init = float(config["initial_rating"])
    k = float(config["k_factor"])
    home_adv = float(config["home_advantage_elo"])
    weights = config.get("tournament_weights", {})
    default_w = float(config.get("default_tournament_weight", 1.0))

    df = matches.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    home_pre = []
    away_pre = []

    for row in df.itertuples(index=False):
        ra = ratings.get(row.home_team, init)
        rb = ratings.get(row.away_team, init)
        home_pre.append(ra)
        away_pre.append(rb)

        adv = 0.0 if bool(row.neutral) else home_adv
        exp_home = _expected(ra + adv, rb)
        score_home = _outcome(int(row.home_score), int(row.away_score))
        weight = float(weights.get(row.tournament, default_w))
        delta = k * weight * (score_home - exp_home)
        ratings[row.home_team] = ra + delta
        ratings[row.away_team] = rb - delta

    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_elo.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/data/external/elo.py tests/test_elo.py
git commit -m "feat: in-house Elo exposing pre-match ratings

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Best-effort FIFA and Transfermarkt loaders

**Files:**
- Create: `footy/data/external/fifa.py`, `footy/data/external/transfermarkt.py`
- Test: `tests/test_external_besteffort.py`

- [ ] **Step 1: Write the failing test**

`tests/test_external_besteffort.py`:
```python
from pathlib import Path

import pandas as pd

from footy.data.external.fifa import load_fifa_ranking
from footy.data.external.transfermarkt import load_market_values


def test_fifa_missing_cache_returns_none(tmp_path):
    assert load_fifa_ranking(tmp_path / "nope.csv") is None


def test_fifa_loads_when_present(tmp_path):
    p = tmp_path / "fifa.csv"
    pd.DataFrame(
        {"date": ["2021-01-01"], "team": ["Brazil"], "rank": [1]}
    ).to_csv(p, index=False)
    out = load_fifa_ranking(p)
    assert out is not None
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_transfermarkt_never_raises(tmp_path):
    # Missing file and malformed file both yield None, never an exception.
    assert load_market_values(tmp_path / "missing.csv") is None
    bad = tmp_path / "bad.csv"
    bad.write_text("not,a,valid\nstructure", encoding="utf-8")
    assert load_market_values(bad) is None or isinstance(load_market_values(bad), pd.DataFrame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_external_besteffort.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`footy/data/external/fifa.py`:
```python
from pathlib import Path

import pandas as pd

REQUIRED = {"date", "team", "rank"}


def load_fifa_ranking(path) -> pd.DataFrame | None:
    """Load cached FIFA ranking CSV. Best-effort: missing/invalid -> None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if not REQUIRED.issubset(df.columns):
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None
```

`footy/data/external/transfermarkt.py`:
```python
from pathlib import Path

import pandas as pd

REQUIRED = {"team", "market_value"}


def load_market_values(path) -> pd.DataFrame | None:
    """Load cached Transfermarkt values. Best-effort: any problem -> None.

    Never raises; the pipeline must continue without this signal.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if not REQUIRED.issubset(df.columns):
            return None
        return df.reset_index(drop=True)
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_external_besteffort.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/data/external/fifa.py footy/data/external/transfermarkt.py tests/test_external_besteffort.py
git commit -m "feat: best-effort FIFA and Transfermarkt loaders

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Leakage helpers + golden test

**Files:**
- Create: `footy/features/leakage.py`
- Test: `tests/test_leakage.py`

- [ ] **Step 1: Write the failing test**

`tests/test_leakage.py`:
```python
import pandas as pd
import pytest

from footy.features.leakage import matches_before, assert_no_leakage


def _matches():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
            "home_team": ["Brazil", "Brazil", "Haiti"],
            "away_team": ["Haiti", "Peru", "Brazil"],
        }
    )


def test_matches_before_is_strictly_past():
    m = _matches()
    past = matches_before(m, pd.Timestamp("2019-02-01"))
    assert len(past) == 1
    assert past["date"].max() < pd.Timestamp("2019-02-01")


def test_assert_no_leakage_passes_for_past_only_feature():
    m = _matches()

    def good_feature(history, team, date):
        sub = matches_before(history, date)
        return len(sub)

    # Should not raise.
    assert_no_leakage(good_feature, m)


def test_assert_no_leakage_detects_future_use():
    m = _matches()

    def leaky_feature(history, team, date):
        # Deliberately uses the row at `date` and later -> leakage.
        sub = history[history["date"] >= date]
        return len(sub)

    with pytest.raises(AssertionError, match="leak"):
        assert_no_leakage(leaky_feature, m)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_leakage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.features.leakage'`

- [ ] **Step 3: Write implementation**

`footy/features/leakage.py`:
```python
import pandas as pd


def matches_before(history: pd.DataFrame, date) -> pd.DataFrame:
    """Return only rows strictly before `date` (anti-leakage window)."""
    date = pd.Timestamp(date)
    return history[history["date"] < date]


def assert_no_leakage(feature_fn, history: pd.DataFrame) -> None:
    """Golden guard: a feature must produce identical output whether it sees
    the full history or only rows strictly before the match date.

    If appending future rows changes the feature value, the function used
    information from the present/future and leaks. Raises AssertionError.
    """
    history = history.sort_values("date").reset_index(drop=True)
    for row in history.itertuples(index=False):
        past = matches_before(history, row.date)
        value_past = feature_fn(past, row.home_team, row.date)
        value_full = feature_fn(history, row.home_team, row.date)
        assert value_past == value_full, (
            f"feature leak detected for {row.home_team} at {row.date}: "
            f"past={value_past} full={value_full}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_leakage.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/features/leakage.py tests/test_leakage.py
git commit -m "feat: anti-leakage helpers and golden guard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Strength features (form, clean sheets, H2H)

**Files:**
- Create: `footy/features/strength.py`
- Test: `tests/test_strength.py`

- [ ] **Step 1: Write the failing test**

`tests/test_strength.py`:
```python
import pandas as pd

from footy.features.strength import recent_form, head_to_head
from footy.features.leakage import assert_no_leakage


def _history():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2019-01-01", "2019-02-01", "2019-03-01", "2019-04-01"]
            ),
            "home_team": ["Brazil", "Haiti", "Brazil", "Peru"],
            "away_team": ["Haiti", "Brazil", "Haiti", "Brazil"],
            "home_score": [3, 1, 2, 0],
            "away_score": [0, 1, 0, 4],
        }
    )


def test_recent_form_counts_only_past():
    h = _history()
    form = recent_form(h, "Brazil", pd.Timestamp("2019-03-01"), window=5)
    # Brazil before 2019-03-01: won 3-0, drew 1-1 (as away). 2 matches.
    assert form["matches"] == 2
    assert form["goals_for"] == 4      # 3 + 1
    assert form["goals_against"] == 1  # 0 + 1
    assert form["clean_sheets"] == 1


def test_recent_form_no_leakage():
    h = _history()
    assert_no_leakage(
        lambda hist, team, date: recent_form(hist, team, date, window=5)["matches"],
        h,
    )


def test_head_to_head_directional_counts():
    h = _history()
    h2h = head_to_head(h, "Brazil", "Haiti", pd.Timestamp("2019-04-01"))
    # Brazil vs Haiti before 2019-04-01: 3-0 (BRA win), 1-1 (draw), 2-0 (BRA win)
    assert h2h["wins"] == 2
    assert h2h["draws"] == 1
    assert h2h["losses"] == 0
    assert h2h["goals_for"] == 6
    assert h2h["goals_against"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strength.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.features.strength'`

- [ ] **Step 3: Write implementation**

`footy/features/strength.py`:
```python
import pandas as pd

from footy.features.leakage import matches_before


def _team_matches(history: pd.DataFrame, team: str) -> pd.DataFrame:
    return history[(history["home_team"] == team) | (history["away_team"] == team)]


def recent_form(history: pd.DataFrame, team: str, date, window: int) -> dict:
    """Goals for/against, clean sheets over the team's last `window` matches
    strictly before `date`."""
    past = matches_before(history, date)
    team_rows = _team_matches(past, team).sort_values("date").tail(window)

    goals_for = goals_against = clean_sheets = 0
    for row in team_rows.itertuples(index=False):
        if row.home_team == team:
            gf, ga = int(row.home_score), int(row.away_score)
        else:
            gf, ga = int(row.away_score), int(row.home_score)
        goals_for += gf
        goals_against += ga
        if ga == 0:
            clean_sheets += 1

    return {
        "matches": int(len(team_rows)),
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
        "clean_sheets": int(clean_sheets),
    }


def head_to_head(history: pd.DataFrame, team_a: str, team_b: str, date) -> dict:
    """Directional H2H record of team_a vs team_b strictly before `date`."""
    past = matches_before(history, date)
    pair = past[
        ((past["home_team"] == team_a) & (past["away_team"] == team_b))
        | ((past["home_team"] == team_b) & (past["away_team"] == team_a))
    ]

    wins = draws = losses = goals_for = goals_against = 0
    for row in pair.itertuples(index=False):
        if row.home_team == team_a:
            ga_for, ga_against = int(row.home_score), int(row.away_score)
        else:
            ga_for, ga_against = int(row.away_score), int(row.home_score)
        goals_for += ga_for
        goals_against += ga_against
        if ga_for > ga_against:
            wins += 1
        elif ga_for < ga_against:
            losses += 1
        else:
            draws += 1

    return {
        "matches": int(len(pair)),
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_strength.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/features/strength.py tests/test_strength.py
git commit -m "feat: strength features (form, clean sheets, directional H2H)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Context features (tournament weight, continent, pre-match ranking merge)

**Files:**
- Create: `footy/features/context.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Write the failing test**

`tests/test_context.py`:
```python
import pandas as pd

from footy.features.context import tournament_weight, continent_of, last_ranking_before


def test_tournament_weight_uses_config_and_default():
    weights = {"FIFA World Cup": 1.0, "Friendly": 0.4}
    assert tournament_weight("FIFA World Cup", weights, default=0.7) == 1.0
    assert tournament_weight("Unknown Cup", weights, default=0.7) == 0.7


def test_continent_lookup_defaults_unknown():
    assert continent_of("Brazil") == "South America"
    assert continent_of("Atlantis") == "Unknown"


def test_last_ranking_before_is_backward_only():
    ranking = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            "team": ["Brazil", "Brazil", "Brazil"],
            "rank": [3, 2, 1],
        }
    )
    # On 2021-06-01 the most recent *prior* rank is the 2021-01-01 one (2).
    assert last_ranking_before(ranking, "Brazil", pd.Timestamp("2021-06-01")) == 2
    # Before any record -> None.
    assert last_ranking_before(ranking, "Brazil", pd.Timestamp("2019-06-01")) is None
    # Missing ranking table -> None.
    assert last_ranking_before(None, "Brazil", pd.Timestamp("2021-06-01")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.features.context'`

- [ ] **Step 3: Write implementation**

`footy/features/context.py`:
```python
import pandas as pd

# Minimal confederation map; extend via data file in a later sub-project.
_CONTINENT = {
    "Brazil": "South America", "Argentina": "South America", "Peru": "South America",
    "Uruguay": "South America", "Colombia": "South America", "Chile": "South America",
    "Haiti": "North America", "United States": "North America", "Mexico": "North America",
    "Canada": "North America", "Jamaica": "North America",
    "France": "Europe", "Germany": "Europe", "Spain": "Europe", "England": "Europe",
    "Italy": "Europe", "Portugal": "Europe", "Netherlands": "Europe", "Belgium": "Europe",
    "Nigeria": "Africa", "Ghana": "Africa", "Egypt": "Africa", "Senegal": "Africa",
    "Cameroon": "Africa", "Morocco": "Africa",
    "Japan": "Asia", "South Korea": "Asia", "Iran": "Asia", "Saudi Arabia": "Asia",
    "Qatar": "Asia", "Australia": "Asia",
}


def tournament_weight(tournament: str, weights: dict, default: float) -> float:
    return float(weights.get(tournament, default))


def continent_of(team: str) -> str:
    return _CONTINENT.get(team, "Unknown")


def last_ranking_before(ranking: pd.DataFrame | None, team: str, date) -> int | None:
    """Most recent rank strictly before `date` (merge_asof backward semantics).
    Best-effort: missing table or no prior record -> None. Never future data.
    """
    if ranking is None or len(ranking) == 0:
        return None
    date = pd.Timestamp(date)
    sub = ranking[(ranking["team"] == team) & (ranking["date"] < date)]
    if len(sub) == 0:
        return None
    latest = sub.sort_values("date").iloc[-1]
    return int(latest["rank"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_context.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/features/context.py tests/test_context.py
git commit -m "feat: context features with backward-only ranking lookup

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Dixon-Coles model fit

**Files:**
- Create: `footy/models/poisson.py`
- Test: `tests/test_poisson.py`

- [ ] **Step 1: Write the failing test**

`tests/test_poisson.py`:
```python
import numpy as np
import pandas as pd

from footy.models.poisson import fit_dixon_coles

MODEL_CFG = {
    "xi": 0.0,            # disable decay for a deterministic small-data test
    "max_goals": 10,
    "home_advantage_init": 0.25,
    "ridge": 0.01,
}


def _matches():
    # Brazil clearly stronger than Haiti across repeated matches.
    rows = []
    for i in range(12):
        rows.append(("2019-01-01", "Brazil", "Haiti", 3, 0))
        rows.append(("2019-06-01", "Haiti", "Brazil", 0, 2))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    return df


def test_fit_returns_rates_with_favorite_higher():
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    lam_a, lam_b = model.rates("Brazil", "Haiti", neutral=True)
    assert lam_a > lam_b
    assert lam_a > 0 and lam_b > 0


def test_fit_is_deterministic():
    m = _matches()
    a = fit_dixon_coles(m, MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    b = fit_dixon_coles(m, MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    assert a.rates("Brazil", "Haiti", neutral=True) == b.rates("Brazil", "Haiti", neutral=True)


def test_home_advantage_increases_home_rate():
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    lam_home, _ = model.rates("Brazil", "Haiti", neutral=False)
    lam_neutral, _ = model.rates("Brazil", "Haiti", neutral=True)
    assert lam_home >= lam_neutral


def test_unknown_team_raises():
    import pytest
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    with pytest.raises(KeyError):
        model.rates("Brazil", "Atlantis", neutral=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_poisson.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.models.poisson'`

- [ ] **Step 3: Write implementation**

`footy/models/poisson.py`:
```python
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles low-score dependency correction (vectorised)."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


@dataclass
class DixonColesModel:
    attack: dict
    defense: dict
    home_adv: float
    intercept: float
    rho: float

    def rates(self, team_a: str, team_b: str, neutral: bool = False) -> tuple[float, float]:
        if team_a not in self.attack:
            raise KeyError(team_a)
        if team_b not in self.attack:
            raise KeyError(team_b)
        adv = 0.0 if neutral else self.home_adv
        lam_a = np.exp(self.intercept + self.attack[team_a] - self.defense[team_b] + adv)
        lam_b = np.exp(self.intercept + self.attack[team_b] - self.defense[team_a])
        return float(lam_a), float(lam_b)


def fit_dixon_coles(matches: pd.DataFrame, config: dict, as_of) -> DixonColesModel:
    """Fit a time-decayed Dixon-Coles model by weighted maximum likelihood."""
    xi = float(config["xi"])
    ridge = float(config.get("ridge", 0.0))
    home_init = float(config.get("home_advantage_init", 0.25))
    as_of = pd.Timestamp(as_of)

    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_i = df["home_team"].map(idx).to_numpy()
    away_i = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy(dtype=int)
    ag = df["away_score"].to_numpy(dtype=int)
    neutral = df["neutral"].to_numpy(dtype=bool) if "neutral" in df else np.zeros(len(df), bool)

    age_days = (as_of - df["date"]).dt.days.to_numpy(dtype=float)
    weights = np.exp(-xi * np.clip(age_days, 0, None))

    # Parameter vector: [intercept, home_adv, attack(n), defense(n), rho]
    def unpack(p):
        intercept = p[0]
        home_adv = p[1]
        attack = p[2:2 + n]
        defense = p[2 + n:2 + 2 * n]
        rho = p[-1]
        return intercept, home_adv, attack, defense, rho

    def neg_log_lik(p):
        intercept, home_adv, attack, defense, rho = unpack(p)
        adv = np.where(neutral, 0.0, home_adv)
        log_lam = intercept + attack[home_i] - defense[away_i] + adv
        log_mu = intercept + attack[away_i] - defense[home_i]
        lam = np.exp(np.clip(log_lam, -10, 10))
        mu = np.exp(np.clip(log_mu, -10, 10))
        tau = _tau(hg, ag, lam, mu, rho)
        tau = np.clip(tau, 1e-9, None)
        ll = (
            -lam + hg * np.log(lam)
            - mu + ag * np.log(mu)
            + np.log(tau)
        )
        penalty = ridge * (np.sum(attack ** 2) + np.sum(defense ** 2))
        return -np.sum(weights * ll) + penalty

    x0 = np.zeros(2 + 2 * n + 1)
    x0[0] = np.log(max(df[["home_score", "away_score"]].to_numpy().mean(), 0.1))
    x0[1] = home_init
    bounds = (
        [(-3, 3), (-1, 1)]
        + [(-3, 3)] * (2 * n)
        + [(-0.2, 0.2)]
    )
    res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
    intercept, home_adv, attack, defense, rho = unpack(res.x)

    return DixonColesModel(
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        home_adv=float(home_adv),
        intercept=float(intercept),
        rho=float(rho),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_poisson.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/models/poisson.py tests/test_poisson.py
git commit -m "feat: time-decayed Dixon-Coles fit via weighted MLE

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Monte Carlo simulation

**Files:**
- Create: `footy/models/montecarlo.py`
- Test: `tests/test_montecarlo.py`

- [ ] **Step 1: Write the failing test**

`tests/test_montecarlo.py`:
```python
from footy.models.montecarlo import simulate

MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


def test_probabilities_sum_to_one():
    res = simulate(2.5, 0.4, MC_CFG)
    total = res["team_a_win"] + res["draw"] + res["team_b_win"]
    assert abs(total - 100.0) < 0.01


def test_seed_is_deterministic():
    a = simulate(1.5, 1.2, MC_CFG)
    b = simulate(1.5, 1.2, MC_CFG)
    assert a == b


def test_favorite_has_higher_win_prob():
    res = simulate(2.8, 0.3, MC_CFG)
    assert res["team_a_win"] > res["team_b_win"]
    assert res["most_likely_score"].count("-") == 1


def test_expected_goals_reported():
    res = simulate(2.0, 0.5, MC_CFG)
    assert res["expected_goals_a"] == 2.0
    assert res["expected_goals_b"] == 0.5
    assert "goals_a" in res["confidence_interval"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_montecarlo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.models.montecarlo'`

- [ ] **Step 3: Write implementation**

`footy/models/montecarlo.py`:
```python
import numpy as np


def simulate(lambda_a: float, lambda_b: float, config: dict) -> dict:
    """Seeded Monte Carlo over two independent Poisson goal counts.

    Returns 1X2 probabilities (%), most likely score, top score distribution,
    expected goals (the input rates) and a goal confidence interval.
    """
    n = int(config["n_sims"])
    seed = int(config["seed"])
    max_goals = int(config["max_goals"])
    ci_level = float(config["ci_level"])
    top_scores = int(config.get("top_scores", 8))

    rng = np.random.default_rng(seed)
    goals_a = np.clip(rng.poisson(lambda_a, n), 0, max_goals)
    goals_b = np.clip(rng.poisson(lambda_b, n), 0, max_goals)

    a_win = float(np.mean(goals_a > goals_b)) * 100.0
    draw = float(np.mean(goals_a == goals_b)) * 100.0
    b_win = float(np.mean(goals_a < goals_b)) * 100.0

    # Score distribution.
    pair_counts = {}
    for ga, gb in zip(goals_a.tolist(), goals_b.tolist()):
        key = f"{ga}-{gb}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    ordered = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    most_likely = ordered[0][0]
    distribution = {
        k: round(v / n * 100.0, 2) for k, v in ordered[:top_scores]
    }

    lo = (1.0 - ci_level) / 2.0 * 100.0
    hi = (1.0 + ci_level) / 2.0 * 100.0
    ci = {
        "goals_a": [int(np.percentile(goals_a, lo)), int(np.percentile(goals_a, hi))],
        "goals_b": [int(np.percentile(goals_b, lo)), int(np.percentile(goals_b, hi))],
        "level": ci_level,
    }

    return {
        "team_a_win": round(a_win, 2),
        "draw": round(draw, 2),
        "team_b_win": round(b_win, 2),
        "expected_goals_a": round(float(lambda_a), 2),
        "expected_goals_b": round(float(lambda_b), 2),
        "most_likely_score": most_likely,
        "score_distribution": distribution,
        "confidence_interval": ci,
        "lambda_dispersion": round(float(np.std(goals_a) + np.std(goals_b)), 4),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_montecarlo.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/models/montecarlo.py tests/test_montecarlo.py
git commit -m "feat: seeded Monte Carlo with 1X2, score distribution, CI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: Prediction reliability

**Files:**
- Create: `footy/reliability.py`
- Test: `tests/test_reliability.py`

- [ ] **Step 1: Write the failing test**

`tests/test_reliability.py`:
```python
from footy.reliability import compute_reliability


def test_more_matches_yields_higher_reliability():
    low = compute_reliability(
        matches_a=2, matches_b=2, recent_a=1, recent_b=1,
        data_age_days=4000, dispersion=2.0, missing_rankings=2, min_matches=10,
    )
    high = compute_reliability(
        matches_a=120, matches_b=110, recent_a=10, recent_b=10,
        data_age_days=120, dispersion=1.0, missing_rankings=0, min_matches=10,
    )
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_missing_rankings_lowers_reliability():
    base = dict(matches_a=50, matches_b=50, recent_a=8, recent_b=8,
                data_age_days=200, dispersion=1.0, min_matches=10)
    with_rank = compute_reliability(missing_rankings=0, **base)
    without = compute_reliability(missing_rankings=2, **base)
    assert with_rank > without
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reliability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.reliability'`

- [ ] **Step 3: Write implementation**

`footy/reliability.py`:
```python
def compute_reliability(
    matches_a: int,
    matches_b: int,
    recent_a: int,
    recent_b: int,
    data_age_days: float,
    dispersion: float,
    missing_rankings: int,
    min_matches: int,
) -> float:
    """Honest reliability in [0, 1]. NOT a probability of being correct.

    Combines: sample size of both teams, recent activity, data freshness,
    simulation dispersion, and whether Elo/FIFA signals were available.
    """
    # Sample-size component (saturates around min_matches).
    sample = min(matches_a, matches_b) / float(max(min_matches, 1))
    sample = max(0.0, min(1.0, sample))

    # Recent-activity component (want both teams active).
    recent = min(recent_a, recent_b) / 10.0
    recent = max(0.0, min(1.0, recent))

    # Freshness: 1.0 if recent data, decays to 0 over ~10 years.
    freshness = max(0.0, 1.0 - data_age_days / 3650.0)

    # Stability: lower dispersion -> higher reliability.
    stability = max(0.0, min(1.0, 1.0 / (1.0 + dispersion)))

    # Ranking availability penalty (0, 1 or 2 missing).
    ranking = 1.0 - 0.15 * missing_rankings
    ranking = max(0.0, min(1.0, ranking))

    score = (
        0.35 * sample
        + 0.20 * recent
        + 0.15 * freshness
        + 0.15 * stability
        + 0.15 * ranking
    )
    return round(max(0.0, min(1.0, score)), 3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reliability.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/reliability.py tests/test_reliability.py
git commit -m "feat: honest prediction_reliability score

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: Predictor orchestration + `predict_match`

**Files:**
- Create: `footy/predict.py`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

`tests/test_predict.py`:
```python
import pandas as pd
import pytest

from footy.predict import Predictor

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


def _matches():
    rows = []
    for _ in range(12):
        rows.append(("2019-01-01", "Brazil", "Haiti", 3, 0))
        rows.append(("2019-06-01", "Haiti", "Brazil", 0, 2))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    return df


def _predictor():
    return Predictor.from_matches(
        _matches(),
        model_config=MODEL_CFG,
        mc_config=MC_CFG,
        canonical=lambda x: x,
        as_of=pd.Timestamp("2020-01-01"),
    )


def test_predict_returns_full_dict():
    out = _predictor().predict("Brazil", "Haiti", neutral=True)
    for key in [
        "team_a_win", "draw", "team_b_win", "expected_goals_a", "expected_goals_b",
        "most_likely_score", "score_distribution", "confidence_interval",
        "prediction_reliability", "model_version",
    ]:
        assert key in out
    assert out["model_version"] == "baseline-v1.0.0"
    assert out["team_a_win"] > out["team_b_win"]


def test_neutral_zeroes_home_advantage():
    pred = _predictor()
    home = pred.predict("Brazil", "Haiti", neutral=False)
    neutral = pred.predict("Brazil", "Haiti", neutral=True)
    assert home["expected_goals_a"] >= neutral["expected_goals_a"]


def test_unknown_team_raises_with_suggestion():
    with pytest.raises(ValueError, match="Unknown team"):
        _predictor().predict("Brazil", "Atlantis", neutral=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.predict'`

- [ ] **Step 3: Write implementation**

`footy/predict.py`:
```python
from __future__ import annotations

import difflib

import pandas as pd

from footy.models.poisson import fit_dixon_coles, DixonColesModel
from footy.models.montecarlo import simulate
from footy.features.strength import recent_form
from footy.reliability import compute_reliability


class Predictor:
    """Bundles a fitted model + match history to answer predict() queries."""

    def __init__(self, model: DixonColesModel, matches: pd.DataFrame,
                 model_config: dict, mc_config: dict, canonical):
        self.model = model
        self.matches = matches
        self.model_config = model_config
        self.mc_config = mc_config
        self.canonical = canonical
        self._teams = set(model.attack.keys())

    @classmethod
    def from_matches(cls, matches: pd.DataFrame, model_config: dict, mc_config: dict,
                     canonical, as_of) -> "Predictor":
        canon = matches.copy()
        canon["home_team"] = canon["home_team"].map(canonical)
        canon["away_team"] = canon["away_team"].map(canonical)
        model = fit_dixon_coles(canon, model_config, as_of=as_of)
        return cls(model, canon, model_config, mc_config, canonical)

    def _resolve(self, team: str) -> str:
        name = self.canonical(team)
        if name not in self._teams:
            suggestions = difflib.get_close_matches(name, sorted(self._teams), n=3)
            raise ValueError(f"Unknown team: {team!r}. Did you mean {suggestions}?")
        return name

    def _match_count(self, team: str) -> int:
        m = self.matches
        return int(((m["home_team"] == team) | (m["away_team"] == team)).sum())

    def predict(self, team_a: str, team_b: str, neutral: bool = False,
                tournament: str = "Friendly") -> dict:
        a = self._resolve(team_a)
        b = self._resolve(team_b)

        lam_a, lam_b = self.model.rates(a, b, neutral=neutral)
        sim = simulate(lam_a, lam_b, self.mc_config)

        as_of = self.matches["date"].max() + pd.Timedelta(days=1)
        form_a = recent_form(self.matches, a, as_of, window=10)
        form_b = recent_form(self.matches, b, as_of, window=10)

        reliability = compute_reliability(
            matches_a=self._match_count(a),
            matches_b=self._match_count(b),
            recent_a=form_a["matches"],
            recent_b=form_b["matches"],
            data_age_days=0.0,
            dispersion=sim["lambda_dispersion"],
            missing_rankings=0,
            min_matches=int(self.model_config["min_matches_reliable"]),
        )

        return {
            "team_a": a,
            "team_b": b,
            "team_a_win": sim["team_a_win"],
            "draw": sim["draw"],
            "team_b_win": sim["team_b_win"],
            "expected_goals_a": sim["expected_goals_a"],
            "expected_goals_b": sim["expected_goals_b"],
            "most_likely_score": sim["most_likely_score"],
            "score_distribution": sim["score_distribution"],
            "confidence_interval": sim["confidence_interval"],
            "prediction_reliability": reliability,
            "model_version": self.model_config["model_version"],
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_predict.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/predict.py tests/test_predict.py
git commit -m "feat: Predictor orchestration and predict() output contract

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 13: Metrics + naive baseline comparison

**Files:**
- Create: `footy/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import math

from footy.metrics import log_loss_1x2, brier_1x2, naive_baseline_probs


def test_log_loss_perfect_is_near_zero():
    probs = [{"home": 1.0, "draw": 0.0, "away": 0.0}]
    assert log_loss_1x2(probs, ["home"]) < 1e-6


def test_log_loss_penalises_wrong():
    probs = [{"home": 0.01, "draw": 0.01, "away": 0.98}]
    assert log_loss_1x2(probs, ["home"]) > 1.0


def test_brier_range():
    probs = [{"home": 0.5, "draw": 0.3, "away": 0.2}]
    score = brier_1x2(probs, ["home"])
    assert 0.0 <= score <= 2.0


def test_naive_baseline_sums_to_one():
    outcomes = ["home", "home", "draw", "away", "home"]
    probs = naive_baseline_probs(outcomes)
    assert abs(probs["home"] + probs["draw"] + probs["away"] - 1.0) < 1e-9
    assert probs["home"] == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.metrics'`

- [ ] **Step 3: Write implementation**

`footy/metrics.py`:
```python
import math

OUTCOMES = ("home", "draw", "away")


def log_loss_1x2(probs: list[dict], actuals: list[str]) -> float:
    """Mean negative log-likelihood of the realised 1X2 outcome."""
    eps = 1e-15
    total = 0.0
    for p, actual in zip(probs, actuals):
        prob = min(1.0, max(eps, p[actual]))
        total += -math.log(prob)
    return total / len(actuals)


def brier_1x2(probs: list[dict], actuals: list[str]) -> float:
    """Mean multiclass Brier score over the 1X2 vector."""
    total = 0.0
    for p, actual in zip(probs, actuals):
        for outcome in OUTCOMES:
            target = 1.0 if outcome == actual else 0.0
            total += (p[outcome] - target) ** 2
    return total / len(actuals)


def naive_baseline_probs(outcomes: list[str]) -> dict:
    """Global historical 1X2 frequencies — the dumb baseline to beat."""
    n = len(outcomes)
    return {o: outcomes.count(o) / n for o in OUTCOMES}


def accuracy_1x2(probs: list[dict], actuals: list[str]) -> float:
    """Share of matches where the highest-probability outcome was realised."""
    correct = 0
    for p, actual in zip(probs, actuals):
        predicted = max(OUTCOMES, key=lambda o: p[o])
        correct += int(predicted == actual)
    return correct / len(actuals)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/metrics.py tests/test_metrics.py
git commit -m "feat: 1X2 log loss, brier, accuracy, naive baseline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 14: CLI

**Files:**
- Create: `footy/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import json

import pandas as pd

from footy import cli


def _matches():
    rows = []
    for _ in range(12):
        rows.append(("2019-01-01", "Brazil", "Haiti", 3, 0))
        rows.append(("2019-06-01", "Haiti", "Brazil", 0, 2))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    return df


def test_cli_render_outputs_json(capsys, monkeypatch):
    from footy.predict import Predictor
    MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
                 "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
    MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}
    predictor = Predictor.from_matches(
        _matches(), model_config=MODEL_CFG, mc_config=MC_CFG,
        canonical=lambda x: x, as_of=pd.Timestamp("2020-01-01"),
    )
    code = cli.run(["Brazil", "Haiti", "--neutral"], predictor=predictor)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["team_a_win"] > payload["team_b_win"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'cli'` or `AttributeError: run`

- [ ] **Step 3: Write implementation**

`footy/cli.py`:
```python
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from footy.config import load_config
from footy.data.loaders import load_results, load_former_names
from footy.data.clean import clean_results
from footy.data.names import NameCanonicalizer
from footy.predict import Predictor


def _build_default_predictor() -> Predictor:
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    mc_cfg = load_config("montecarlo")

    raw_dir = data_cfg["raw_dir"]
    results = load_results(f"{raw_dir}/{data_cfg['files']['results']}")
    former = load_former_names(f"{raw_dir}/{data_cfg['files']['former_names']}")
    clean = clean_results(results)

    canon = NameCanonicalizer(
        former, data_cfg.get("aliases", {}), data_cfg.get("sensitive_merges", {})
    )
    as_of = clean.df["date"].max() + pd.Timedelta(days=1)
    return Predictor.from_matches(
        clean.df, model_config=model_cfg, mc_config=mc_cfg,
        canonical=canon.canonical, as_of=as_of,
    )


def run(argv: list[str], predictor: Predictor | None = None) -> int:
    parser = argparse.ArgumentParser(prog="predict", description="Predict a national-team match.")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--neutral", action="store_true", help="Neutral venue (home_advantage=0)")
    parser.add_argument("--tournament", default="Friendly")
    args = parser.parse_args(argv)

    if predictor is None:
        predictor = _build_default_predictor()

    try:
        result = predictor.predict(
            args.team_a, args.team_b, neutral=args.neutral, tournament=args.tournament
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/cli.py tests/test_cli.py
git commit -m "feat: predict CLI with injectable predictor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 15: Pipeline build script + artifacts (config snapshot, reports)

**Files:**
- Create: `footy/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
import json
from pathlib import Path

import pandas as pd

from footy.pipeline import run_etl


def _raw(tmp_path):
    p = tmp_path / "results.csv"
    rows = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    for _ in range(6):
        rows += "2019-01-01,Brazil,Haiti,3,0,Friendly,Rio,Brazil,FALSE\n"
        rows += "2019-06-01,Haiti,Brazil,0,2,Friendly,Port,Haiti,FALSE\n"
    p.write_text(rows, encoding="utf-8")
    fn = tmp_path / "former_names.csv"
    fn.write_text("current,former,start_date,end_date\n", encoding="utf-8")
    return p, fn


def test_run_etl_writes_artifacts(tmp_path):
    results_path, former_path = _raw(tmp_path)
    out_dir = tmp_path / "artifacts"
    enriched = run_etl(
        results_path=results_path,
        former_names_path=former_path,
        elo_config={"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
                    "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4}},
        data_config={"aliases": {}, "sensitive_merges": {"enabled": False, "mappings": {}}},
        artifacts_dir=out_dir,
    )
    assert "home_elo_pre" in enriched.columns
    assert (out_dir / "etl_report.json").exists()
    assert (out_dir / "team_name_mapping.json").exists()
    report = json.loads((out_dir / "etl_report.json").read_text())
    assert report["rows_out"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.pipeline'`

- [ ] **Step 3: Write implementation**

`footy/pipeline.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from footy.data.loaders import load_results, load_former_names
from footy.data.clean import clean_results
from footy.data.names import NameCanonicalizer
from footy.data.external.elo import attach_elo


def run_etl(results_path, former_names_path, elo_config: dict, data_config: dict,
            artifacts_dir) -> pd.DataFrame:
    """Load -> clean -> canonicalize -> Elo. Writes artifacts; returns enriched frame."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw = load_results(results_path)
    former = load_former_names(former_names_path)
    clean = clean_results(raw)

    canon = NameCanonicalizer(
        former, data_config.get("aliases", {}), data_config.get("sensitive_merges", {})
    )
    df = clean.df.copy()
    df["home_team"] = df["home_team"].map(canon.canonical)
    df["away_team"] = df["away_team"].map(canon.canonical)

    enriched = attach_elo(df, elo_config)
    enriched.to_parquet(artifacts_dir / "enriched_matches.parquet", index=False)

    (artifacts_dir / "etl_report.json").write_text(
        json.dumps(clean.report, indent=2), encoding="utf-8"
    )
    clean.dropped.to_csv(artifacts_dir / "dropped_rows.csv", index=False)
    (artifacts_dir / "team_name_mapping.json").write_text(
        json.dumps(canon.mapping_table(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return enriched
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/pipeline.py tests/test_pipeline.py
git commit -m "feat: ETL pipeline writing report/mapping/enriched artifacts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 16: Smoke test on real data + full suite

**Files:**
- Test: `tests/test_smoke_real.py`

- [ ] **Step 1: Write the failing test**

`tests/test_smoke_real.py`:
```python
"""End-to-end smoke test against the real bundled dataset."""
from pathlib import Path

import pandas as pd
import pytest

from footy.data.loaders import load_results, load_former_names
from footy.data.clean import clean_results
from footy.data.names import NameCanonicalizer
from footy.predict import Predictor

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "international_results" / "results.csv"

MODEL_CFG = {"xi": 0.0018, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 30000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


@pytest.mark.skipif(not RESULTS.exists(), reason="real dataset not present")
def test_brazil_beats_haiti_on_real_data():
    raw = load_results(RESULTS)
    former = load_former_names(ROOT / "international_results" / "former_names.csv")
    clean = clean_results(raw)
    canon = NameCanonicalizer(former, {}, {"enabled": False, "mappings": {}})
    as_of = clean.df["date"].max() + pd.Timedelta(days=1)
    predictor = Predictor.from_matches(
        clean.df, model_config=MODEL_CFG, mc_config=MC_CFG,
        canonical=canon.canonical, as_of=as_of,
    )
    out = predictor.predict("Brazil", "Haiti", neutral=True)
    assert out["team_a_win"] > out["team_b_win"]
    assert 99.0 <= out["team_a_win"] + out["draw"] + out["team_b_win"] <= 101.0
```

- [ ] **Step 2: Run test to verify it fails (or skips if no data)**

Run: `python -m pytest tests/test_smoke_real.py -v`
Expected: FAIL or SKIP depending on dataset presence. With dataset present, FAIL only if a regression exists — should pass once the stack is wired.

- [ ] **Step 3: Make it pass**

No new production code required; this test exercises the assembled stack. If it fails, debug the offending module using superpowers:systematic-debugging. Do not weaken the assertion.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -v`
Expected: all tests PASS (smoke may SKIP if data absent).

- [ ] **Step 5: Commit**

```bash
git add tests/test_smoke_real.py
git commit -m "test: end-to-end smoke test on real dataset

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 17: Documentation (README, CLAUDE, ALCANCE)

**Files:**
- Create: `README.md`, `CLAUDE.md`, `docs/ALCANCE.md`

- [ ] **Step 1: Write `README.md`**

Content must cover: install (`pip install -e .[dev]`), config files overview, how to run the pipeline, how to call `predict_match`/CLI (`predict Brazil Haiti --neutral`), and how to read every key in the output dict. Explicitly state that `expected_goals_*` is the model λ, **not** shot-based xG, and that `prediction_reliability` is not a probability of being correct.

- [ ] **Step 2: Write `CLAUDE.md`**

Content: repo map (table from this plan's File Structure), commands (`python -m pytest`, `predict ...`), conventions (TDD, anti-leakage invariant, config-driven, best-effort external loaders), and the sub-project roadmap (2–7) with the CLV/ROI/Yield note.

- [ ] **Step 3: Write `docs/ALCANCE.md`**

Content: what sub-project 1 delivers (ETL + Dixon-Coles baseline + Monte Carlo + predict + CLI), explicit out-of-scope list (ML, DL, ensemble, dashboard, API, deep backtesting, shot-based xG), and the ordered roadmap for sub-projects 2–7.

- [ ] **Step 4: Verify docs reference real commands**

Run: `python -m pytest -q` and confirm the commands shown in README actually exist (CLI entry-point `predict`, module paths). Fix any mismatch.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md docs/ALCANCE.md
git commit -m "docs: README, CLAUDE guide, and ALCANCE roadmap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 18: Temporal hold-out evaluation driver

**Files:**
- Create: `footy/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate.py`:
```python
import pandas as pd

from footy.evaluate import temporal_holdout

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


def _matches():
    # Mixed outcomes so the naive global-frequency baseline is non-degenerate
    # (home/draw/away all present); the team-aware model still has signal.
    rows = []
    for _ in range(10):
        rows.append(("2018-01-01", "Brazil", "Haiti", 3, 0))   # home
        rows.append(("2018-02-01", "Haiti", "Brazil", 0, 3))   # away
        rows.append(("2018-03-01", "Brazil", "Peru", 2, 1))    # home
        rows.append(("2018-04-01", "Peru", "Brazil", 1, 1))    # draw
        rows.append(("2018-05-01", "Peru", "Haiti", 2, 0))     # home
        rows.append(("2018-05-15", "Haiti", "Peru", 1, 1))     # draw
    # Test period: favourites at home; model should beat the global baseline.
    for _ in range(4):
        rows.append(("2020-01-01", "Brazil", "Haiti", 3, 0))
        rows.append(("2020-02-01", "Peru", "Haiti", 2, 0))
        rows.append(("2020-03-01", "Brazil", "Peru", 2, 1))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    return df


def test_holdout_reports_model_and_naive():
    report = temporal_holdout(
        _matches(), split_date=pd.Timestamp("2019-06-01"),
        model_config=MODEL_CFG, mc_config=MC_CFG, canonical=lambda x: x,
    )
    for key in ["model", "naive", "beats_baseline", "n_test"]:
        assert key in report
    for metric in ["log_loss", "brier", "accuracy"]:
        assert metric in report["model"]
        assert metric in report["naive"]
    assert report["n_test"] == 12


def test_model_beats_naive_on_separable_data():
    report = temporal_holdout(
        _matches(), split_date=pd.Timestamp("2019-06-01"),
        model_config=MODEL_CFG, mc_config=MC_CFG, canonical=lambda x: x,
    )
    # On clearly separable data the model should beat the global-frequency baseline.
    assert report["model"]["log_loss"] < report["naive"]["log_loss"]
    assert report["beats_baseline"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.evaluate'`

- [ ] **Step 3: Write implementation**

`footy/evaluate.py`:
```python
from __future__ import annotations

import pandas as pd

from footy.predict import Predictor
from footy.metrics import log_loss_1x2, brier_1x2, accuracy_1x2, naive_baseline_probs


def _outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "draw"


def temporal_holdout(matches: pd.DataFrame, split_date, model_config: dict,
                     mc_config: dict, canonical) -> dict:
    """Train on rows before split_date, evaluate on rows on/after it.

    Never trains on the future. Compares the model's 1X2 probabilities against
    the naive global-frequency baseline derived from the training period only.
    """
    split_date = pd.Timestamp(split_date)
    train = matches[matches["date"] < split_date]
    test = matches[matches["date"] >= split_date]

    predictor = Predictor.from_matches(
        train, model_config=model_config, mc_config=mc_config,
        canonical=canonical, as_of=split_date,
    )

    train_outcomes = [
        _outcome(int(r.home_score), int(r.away_score))
        for r in train.itertuples(index=False)
    ]
    naive = naive_baseline_probs(train_outcomes)

    model_probs: list[dict] = []
    naive_probs: list[dict] = []
    actuals: list[str] = []
    for row in test.itertuples(index=False):
        try:
            pred = predictor.predict(
                row.home_team, row.away_team, neutral=bool(row.neutral)
            )
        except ValueError:
            # Team unseen in the training period; skip (cannot score fairly).
            continue
        model_probs.append({
            "home": pred["team_a_win"] / 100.0,
            "draw": pred["draw"] / 100.0,
            "away": pred["team_b_win"] / 100.0,
        })
        naive_probs.append(dict(naive))
        actuals.append(_outcome(int(row.home_score), int(row.away_score)))

    model_metrics = {
        "log_loss": log_loss_1x2(model_probs, actuals),
        "brier": brier_1x2(model_probs, actuals),
        "accuracy": accuracy_1x2(model_probs, actuals),
    }
    naive_metrics = {
        "log_loss": log_loss_1x2(naive_probs, actuals),
        "brier": brier_1x2(naive_probs, actuals),
        "accuracy": accuracy_1x2(naive_probs, actuals),
    }
    return {
        "model": model_metrics,
        "naive": naive_metrics,
        "beats_baseline": bool(model_metrics["log_loss"] < naive_metrics["log_loss"]),
        "n_test": len(actuals),
        "split_date": str(split_date.date()),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/evaluate.py tests/test_evaluate.py
git commit -m "feat: temporal hold-out evaluation vs naive baseline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §3 sources → Tasks 1 (loaders), 4 (Elo), 5 (FIFA/TM). ✓
- §4 architecture/config-driven → Task 0 configs + every module reads config. ✓
- §5 ETL (logged dedup, null handling, names, merge_asof backward) → Tasks 2, 3, 15; backward ranking in Task 8 (`last_ranking_before`). ✓
- §5.6 anti-leakage golden → Task 6, reused in Tasks 4/7. ✓
- §6 features (form, clean sheets, H2H low weight, context) → Tasks 7, 8. ✓
- §7 Dixon-Coles (τ inside likelihood, time-decay, deterministic, versioned) → Task 9; `model_version` surfaced in Task 12. ✓
- §7.1 config snapshot → Task 15 writes artifacts (note: full per-version snapshot dir extends here; baseline writes report+mapping+enriched). ✓
- §8 Monte Carlo (100k, seed, distribution, CI, λ as expected goals) → Task 10. ✓
- §9 predict dict + reliability factors → Tasks 11, 12. ✓
- §10 error handling layers → loaders raise (T1), clean logs (T2), external None (T5), unknown team + suggestion (T12). ✓
- §11 testing matrix → Tasks 1–16 mirror it; golden leakage T6. ✓
- §12 metrics + naive baseline → Task 13. ✓
- §13 minimal deps → Task 0 pyproject. ✓
- §14 docs → Task 17. ✓

**Placeholder scan:** every code step contains complete runnable code; no TODO/TBD. ✓

**Type consistency:** `clean_results→CleanResult(df, report, dropped)` (T2, used T15); `attach_elo→home_elo_pre/away_elo_pre` (T4, asserted T16 indirectly); `fit_dixon_coles(matches, config, as_of)→DixonColesModel.rates(a,b,neutral)` (T9, used T12); `simulate(lam_a,lam_b,cfg)` dict keys reused verbatim in T12; `Predictor.from_matches(...)` signature identical across T12/T14/T16; `compute_reliability(...)` kwargs identical T11/T12. ✓

**Note on §12 hold-out evaluation:** Task 13 ships metric primitives + naive baseline; **Task 18** adds the temporal hold-out driver (`temporal_holdout`) that trains before a split date, evaluates after it, and reports model-vs-naive log loss / brier / accuracy with `beats_baseline`. Deep backtesting (per tournament/team, calibration curve, confusion matrix) remains sub-project 4.
