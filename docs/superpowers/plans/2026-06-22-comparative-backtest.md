# Comparative Evaluation + Historical Backtest Implementation Plan (SP8)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tell whether ~60% accuracy is good by comparing the Dixon-Coles model against baselines (Elo favorite, naive, random) and backtesting by training strictly before WC2014/2018/2022/2026 and evaluating each edition out-of-sample.

**Architecture:** New `footy/eval/` package: a common 1X2 predictor interface with analytic Dixon-Coles / Elo / naive / random implementations; a model-agnostic `evaluate`; a `backtest` that trains the trainable predictors on rows before each edition; a `report` that writes a cached JSON. `footy/metrics.py` gains calibration buckets/ECE. A Streamlit "Evaluación" tab reads the cached JSON.

**Tech Stack:** Python 3.10.6, numpy, scipy, pandas, streamlit, pytest. Run from repo root: `python -m pytest`.

**Conventions:** branch `feature/baseline-v1`. TDD: failing test first. Commit per task, `git add` ONLY named files (never `__pycache__`/`.pyc`). No real network in tests. Commit trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## File Structure

| File | Responsibility |
|---|---|
| `footy/metrics.py` | ADD `calibration_buckets` |
| `footy/data/external/elo.py` | ADD `final_ratings` |
| `footy/eval/__init__.py` | package marker |
| `footy/eval/predictors.py` | `DixonColesPredictor`, `EloFavoritePredictor`, `NaiveGlobalPredictor`, `RandomPredictor`, `build_predictors` |
| `footy/eval/evaluate_models.py` | `evaluate(predictor, matches)` |
| `footy/eval/backtest.py` | `backtest_edition`, `run_backtest` |
| `footy/eval/report.py` | `run_report` (writes JSON) |
| `footy/cli.py` | ADD `backtest` command (`run_backtest_cli`, `main_backtest`) |
| `app/streamlit_app.py` | ADD "Evaluación" tab |
| `tests/test_*` | per unit |

---

## Task 1: calibration buckets + ECE

**Files:**
- Modify: `footy/metrics.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

`tests/test_calibration.py`:
```python
from footy.metrics import calibration_buckets


def test_perfect_confidence_correct_is_zero_ece():
    probs = [{"home": 1.0, "draw": 0.0, "away": 0.0}] * 5
    actuals = ["home"] * 5
    c = calibration_buckets(probs, actuals, bins=10)
    assert c["ece"] == 0.0
    assert sum(b["n"] for b in c["bins"]) == 5


def test_overconfident_wrong_has_high_ece():
    probs = [{"home": 1.0, "draw": 0.0, "away": 0.0}] * 4
    actuals = ["home", "home", "away", "away"]   # conf 1.0 but only 50% correct
    c = calibration_buckets(probs, actuals, bins=10)
    assert c["ece"] >= 0.49


def test_bins_record_prob_and_frequency():
    probs = [{"home": 0.6, "draw": 0.2, "away": 0.2},
             {"home": 0.6, "draw": 0.2, "away": 0.2}]
    actuals = ["home", "away"]   # one hit, one miss; conf 0.6 bucket
    c = calibration_buckets(probs, actuals, bins=10)
    bucket = next(b for b in c["bins"] if b["n"] == 2)
    assert bucket["prob_media"] == 0.6 and bucket["frecuencia"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibration.py -v`
Expected: FAIL with `ImportError: cannot import name 'calibration_buckets'`

- [ ] **Step 3: Append to `footy/metrics.py`**

```python
def calibration_buckets(probs: list, actuals: list, bins: int = 10) -> dict:
    """Reliability buckets for the predicted (argmax) outcome's confidence vs how often
    it was right, plus the Expected Calibration Error (ECE)."""
    buckets = [{"lo": i / bins, "hi": (i + 1) / bins, "n": 0, "sum_p": 0.0, "hits": 0}
               for i in range(bins)]
    for p, actual in zip(probs, actuals):
        pred = max(p, key=p.get)
        conf = p[pred]
        idx = min(bins - 1, int(conf * bins))
        b = buckets[idx]
        b["n"] += 1
        b["sum_p"] += conf
        b["hits"] += int(pred == actual)

    total = sum(b["n"] for b in buckets) or 1
    ece = 0.0
    out = []
    for b in buckets:
        if b["n"] == 0:
            out.append({"lo": round(b["lo"], 2), "hi": round(b["hi"], 2),
                        "n": 0, "prob_media": None, "frecuencia": None})
            continue
        prob_media = b["sum_p"] / b["n"]
        freq = b["hits"] / b["n"]
        ece += (b["n"] / total) * abs(prob_media - freq)
        out.append({"lo": round(b["lo"], 2), "hi": round(b["hi"], 2), "n": b["n"],
                    "prob_media": round(prob_media, 4), "frecuencia": round(freq, 4)})
    return {"bins": out, "ece": round(ece, 4)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_calibration.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/metrics.py tests/test_calibration.py
git commit -m "feat: calibration buckets + ECE metric

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Elo final ratings

**Files:**
- Modify: `footy/data/external/elo.py`
- Test: `tests/test_elo_final.py`

- [ ] **Step 1: Write the failing test**

`tests/test_elo_final.py`:
```python
import pandas as pd

from footy.data.external.elo import final_ratings

CFG = {"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
       "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4}}


def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2019-02-01"]),
        "home_team": ["Brazil", "Brazil"],
        "away_team": ["Haiti", "Haiti"],
        "home_score": [3, 2],
        "away_score": [0, 0],
        "tournament": ["Friendly", "Friendly"],
        "neutral": [False, False],
    })


def test_final_ratings_winner_above_loser():
    r = final_ratings(_matches(), CFG)
    assert r["Brazil"] > 1500.0 > r["Haiti"]
    # zero-sum around the initial total
    assert abs((r["Brazil"] - 1500.0) + (r["Haiti"] - 1500.0)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_elo_final.py -v`
Expected: FAIL with `ImportError: cannot import name 'final_ratings'`

- [ ] **Step 3: Append to `footy/data/external/elo.py`**

```python
def final_ratings(matches, config: dict) -> dict:
    """Elo rating of every team AFTER processing all matches chronologically."""
    init = float(config["initial_rating"])
    k = float(config["k_factor"])
    home_adv = float(config["home_advantage_elo"])
    weights = config.get("tournament_weights", {})
    default_w = float(config.get("default_tournament_weight", 1.0))

    df = matches.sort_values("date")
    ratings: dict = {}
    for row in df.itertuples(index=False):
        ra = ratings.get(row.home_team, init)
        rb = ratings.get(row.away_team, init)
        adv = 0.0 if bool(row.neutral) else home_adv
        exp_home = 1.0 / (1.0 + 10.0 ** ((rb - (ra + adv)) / 400.0))
        if row.home_score > row.away_score:
            score = 1.0
        elif row.home_score < row.away_score:
            score = 0.0
        else:
            score = 0.5
        weight = float(weights.get(row.tournament, default_w))
        delta = k * weight * (score - exp_home)
        ratings[row.home_team] = ra + delta
        ratings[row.away_team] = rb - delta
    return ratings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_elo_final.py tests/test_elo.py -v`
Expected: all PASS (new + existing elo tests stay green)

- [ ] **Step 5: Commit**

```bash
git add footy/data/external/elo.py tests/test_elo_final.py
git commit -m "feat: Elo final_ratings (post-match ratings per team)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: predictors

**Files:**
- Create: `footy/eval/__init__.py`, `footy/eval/predictors.py`
- Test: `tests/test_predictors.py`

- [ ] **Step 1: Write the failing test**

`tests/test_predictors.py`:
```python
import pandas as pd

from footy.eval.predictors import (DixonColesPredictor, EloFavoritePredictor,
                                    NaiveGlobalPredictor, RandomPredictor, build_predictors)
from footy.models.poisson import fit_dixon_coles
from footy.models.montecarlo import simulate

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25, "ridge": 0.01}
MC_CFG = {"n_sims": 40000, "seed": 1, "max_goals": 10, "ci_level": 0.90, "top_scores": 5}
ELO_CFG = {"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
           "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4}}
FALLBACK = {"home": 0.45, "draw": 0.27, "away": 0.28}


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


def test_random_is_uniform():
    p = RandomPredictor().probs("A", "B", True)
    assert p == {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    assert RandomPredictor().goals("A", "B", True) is None


def test_naive_returns_fixed_rates():
    p = NaiveGlobalPredictor(FALLBACK).probs("A", "B", True)
    assert p == FALLBACK


def test_dixon_coles_probs_sum_to_one_and_match_sim():
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    pred = DixonColesPredictor(model, FALLBACK, max_goals=10)
    p = pred.probs("Brazil", "Haiti", neutral=True)
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
    sim = simulate(*model.rates("Brazil", "Haiti", neutral=True), MC_CFG)
    assert abs(p["home"] - sim["team_a_win"] / 100.0) < 0.04   # analytic ≈ MC
    assert pred.goals("Brazil", "Haiti", neutral=True) is not None


def test_dixon_coles_unknown_team_falls_back():
    model = fit_dixon_coles(_matches(), MODEL_CFG, as_of=pd.Timestamp("2020-01-01"))
    pred = DixonColesPredictor(model, FALLBACK, max_goals=10)
    assert pred.probs("Brazil", "Atlantis", neutral=True) == FALLBACK


def test_elo_favorite_gives_more_to_higher_rating():
    pred = EloFavoritePredictor({"Strong": 1800.0, "Weak": 1400.0}, 0.25, 65.0, FALLBACK)
    p = pred.probs("Strong", "Weak", neutral=True)
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
    assert p["home"] > p["away"] and p["draw"] == 0.25


def test_build_predictors_has_standard_set():
    preds = build_predictors(_matches(), MODEL_CFG, ELO_CFG, as_of=pd.Timestamp("2020-01-01"))
    assert set(preds) == {"BASE", "Elo", "naive", "random"}
    for p in preds.values():
        probs = p.probs("Brazil", "Haiti", neutral=True)
        assert abs(sum(probs.values()) - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predictors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.eval'`

- [ ] **Step 3: Write implementation**

`footy/eval/__init__.py`:
```python
```
(empty — a single newline)

`footy/eval/predictors.py`:
```python
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from footy.models.poisson import fit_dixon_coles
from footy.data.external.elo import final_ratings


def _global_rates(matches) -> dict:
    h = d = a = 0
    for row in matches.itertuples(index=False):
        if row.home_score > row.away_score:
            h += 1
        elif row.home_score < row.away_score:
            a += 1
        else:
            d += 1
    n = h + d + a or 1
    return {"home": h / n, "draw": d / n, "away": a / n}


class DixonColesPredictor:
    name = "BASE"

    def __init__(self, model, fallback: dict, max_goals: int = 10):
        self.model = model
        self.fallback = fallback
        self.max_goals = max_goals

    def probs(self, team_a, team_b, neutral) -> dict:
        try:
            la, lb = self.model.rates(team_a, team_b, neutral=neutral)
        except KeyError:
            return dict(self.fallback)
        xs = np.arange(0, self.max_goals + 1)
        grid = np.outer(poisson.pmf(xs, la), poisson.pmf(xs, lb))
        rho = self.model.rho
        grid[0, 0] *= max(1e-9, 1.0 - la * lb * rho)
        grid[0, 1] *= max(1e-9, 1.0 + la * rho)
        grid[1, 0] *= max(1e-9, 1.0 + lb * rho)
        grid[1, 1] *= max(1e-9, 1.0 - rho)
        grid /= grid.sum()
        x = np.arange(grid.shape[0])[:, None]
        y = np.arange(grid.shape[1])[None, :]
        home = float(grid[x > y].sum())
        draw = float(np.trace(grid))
        away = float(grid[x < y].sum())
        s = home + draw + away
        return {"home": home / s, "draw": draw / s, "away": away / s}

    def goals(self, team_a, team_b, neutral):
        try:
            return self.model.rates(team_a, team_b, neutral=neutral)
        except KeyError:
            return None


class EloFavoritePredictor:
    name = "Elo"

    def __init__(self, ratings: dict, draw_rate: float, home_adv_elo: float, fallback: dict):
        self.ratings = ratings
        self.draw_rate = draw_rate
        self.home_adv_elo = home_adv_elo
        self.fallback = fallback

    def probs(self, team_a, team_b, neutral) -> dict:
        ra = self.ratings.get(team_a)
        rb = self.ratings.get(team_b)
        if ra is None or rb is None:
            return dict(self.fallback)
        adv = 0.0 if neutral else self.home_adv_elo
        p_home = 1.0 / (1.0 + 10.0 ** ((rb - (ra + adv)) / 400.0))
        rest = 1.0 - self.draw_rate
        return {"home": rest * p_home, "draw": self.draw_rate, "away": rest * (1.0 - p_home)}

    def goals(self, team_a, team_b, neutral):
        return None


class NaiveGlobalPredictor:
    name = "naive"

    def __init__(self, rates: dict):
        self.rates = rates

    def probs(self, team_a, team_b, neutral) -> dict:
        return dict(self.rates)

    def goals(self, team_a, team_b, neutral):
        return None


class RandomPredictor:
    name = "random"

    def probs(self, team_a, team_b, neutral) -> dict:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}

    def goals(self, team_a, team_b, neutral):
        return None


def build_predictors(train, model_config: dict, elo_config: dict, as_of) -> dict:
    """Construct the standard predictor set from a training DataFrame."""
    rates = _global_rates(train)
    model = fit_dixon_coles(train, model_config, as_of=as_of)
    ratings = final_ratings(train, elo_config)
    return {
        "BASE": DixonColesPredictor(model, fallback=rates, max_goals=int(model_config["max_goals"])),
        "Elo": EloFavoritePredictor(ratings, draw_rate=rates["draw"],
                                    home_adv_elo=float(elo_config["home_advantage_elo"]),
                                    fallback=rates),
        "naive": NaiveGlobalPredictor(rates),
        "random": RandomPredictor(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_predictors.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/eval/__init__.py footy/eval/predictors.py tests/test_predictors.py
git commit -m "feat: 1X2 predictors (Dixon-Coles analytic, Elo, naive, random)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: evaluate_models

**Files:**
- Create: `footy/eval/evaluate_models.py`
- Test: `tests/test_evaluate_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate_models.py`:
```python
from footy.eval.evaluate_models import evaluate
from footy.eval.predictors import RandomPredictor, NaiveGlobalPredictor


class PerfectGoalPredictor:
    """Always predicts the realized outcome with goals — for goal_mae coverage."""
    def __init__(self, table):
        self.table = table

    def probs(self, a, b, neutral):
        return self.table[(a, b)]["probs"]

    def goals(self, a, b, neutral):
        return self.table[(a, b)]["goals"]


def _matches():
    return [
        {"team_a": "A", "team_b": "B", "neutral": True, "goals_a": 2, "goals_b": 0},
        {"team_a": "C", "team_b": "D", "neutral": True, "goals_a": 1, "goals_b": 1},
    ]


def test_random_metrics_shape():
    out = evaluate(RandomPredictor(), _matches())
    assert out["n"] == 2
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["log_loss"] > 0 and out["brier"] >= 0
    assert out["goal_mae"] is None            # random predicts no goals
    assert "ece" in out["calibration"]


def test_goal_mae_present_for_goal_predictor():
    table = {("A", "B"): {"probs": {"home": 0.8, "draw": 0.1, "away": 0.1}, "goals": (2.0, 0.0)},
             ("C", "D"): {"probs": {"home": 0.3, "draw": 0.4, "away": 0.3}, "goals": (1.0, 1.0)}}
    out = evaluate(PerfectGoalPredictor(table), _matches())
    assert out["goal_mae"] == 0.0             # exact goals
    assert out["accuracy"] == 1.0             # argmax matches both outcomes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.eval.evaluate_models'`

- [ ] **Step 3: Write implementation**

`footy/eval/evaluate_models.py`:
```python
from footy.metrics import log_loss_1x2, brier_1x2, accuracy_1x2, calibration_buckets


def _outcome(ga, gb):
    if ga > gb:
        return "home"
    if ga < gb:
        return "away"
    return "draw"


def evaluate(predictor, matches: list) -> dict:
    """Evaluate a 1X2 predictor over matches with known results."""
    probs, actuals = [], []
    goal_err, goal_n = 0.0, 0
    hits = 0
    for m in matches:
        neutral = bool(m.get("neutral", True))
        p = predictor.probs(m["team_a"], m["team_b"], neutral)
        actual = _outcome(m["goals_a"], m["goals_b"])
        probs.append(p)
        actuals.append(actual)
        hits += int(max(p, key=p.get) == actual)
        g = predictor.goals(m["team_a"], m["team_b"], neutral)
        if g is not None:
            goal_err += abs(g[0] - m["goals_a"]) + abs(g[1] - m["goals_b"])
            goal_n += 1
    n = len(matches)
    return {
        "n": n,
        "hits": hits,
        "accuracy": accuracy_1x2(probs, actuals),
        "log_loss": round(log_loss_1x2(probs, actuals), 4),
        "brier": round(brier_1x2(probs, actuals), 4),
        "goal_mae": round(goal_err / (2 * goal_n), 3) if goal_n else None,
        "calibration": calibration_buckets(probs, actuals),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate_models.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/eval/evaluate_models.py tests/test_evaluate_models.py
git commit -m "feat: model-agnostic 1X2 evaluation (accuracy/logloss/brier/mae/calibration)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: backtest driver

**Files:**
- Create: `footy/eval/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test**

`tests/test_backtest.py`:
```python
import pandas as pd

from footy.eval.backtest import backtest_edition, run_backtest

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25, "ridge": 0.01}
ELO_CFG = {"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
           "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4, "Cup": 1.0}}


def _dataset():
    rows = []
    # History (Friendlies) before the cup: Strong beats Weak repeatedly.
    for i in range(20):
        rows.append(("2012-01-01", "Strong", "Weak", 3, 0, "Friendly"))
        rows.append(("2013-01-01", "Mid", "Weak", 2, 0, "Friendly"))
        rows.append(("2013-06-01", "Strong", "Mid", 2, 1, "Friendly"))
    # The 2014 "Cup" edition (to be evaluated).
    rows.append(("2014-06-01", "Strong", "Weak", 1, 0, "Cup"))
    rows.append(("2014-06-02", "Mid", "Weak", 1, 0, "Cup"))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament"])
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = True
    return df


def test_backtest_edition_is_out_of_sample():
    ds = _dataset()
    res = backtest_edition(ds, "Cup", 2014, MODEL_CFG, ELO_CFG)
    assert res["n"] == 2
    assert res["start"] == "2014-06-01"
    # all four predictors evaluated
    assert set(res["models"]) == {"BASE", "Elo", "naive", "random"}
    # BASE/Elo should beat random on this clearly separable data
    assert res["models"]["BASE"]["accuracy"] >= res["models"]["random"]["accuracy"]


def test_run_backtest_aggregates():
    ds = _dataset()
    out = run_backtest(ds, [("Cup", 2014)], MODEL_CFG, ELO_CFG)
    assert "Cup 2014" in out["editions"]
    assert "BASE" in out["aggregate"]
    assert out["aggregate"]["BASE"]["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.eval.backtest'`

- [ ] **Step 3: Write implementation**

`footy/eval/backtest.py`:
```python
from footy.eval.predictors import build_predictors
from footy.eval.evaluate_models import evaluate


def _edition(dataset, tournament, year):
    sub = dataset[(dataset["tournament"] == tournament) & (dataset["date"].dt.year == year)]
    return sub.sort_values("date")


def backtest_edition(dataset, tournament, year, model_config, elo_config) -> dict:
    """Train predictors on rows strictly before the edition; evaluate on the edition."""
    edition = _edition(dataset, tournament, year)
    if len(edition) == 0:
        return {"n": 0, "start": None, "models": {}}
    start = edition["date"].min()
    train = dataset[dataset["date"] < start]
    preds = build_predictors(train, model_config, elo_config, as_of=start)
    matches = [{"team_a": r.home_team, "team_b": r.away_team, "neutral": bool(r.neutral),
                "goals_a": int(r.home_score), "goals_b": int(r.away_score)}
               for r in edition.itertuples(index=False)]
    return {"n": len(matches), "start": str(start.date()),
            "models": {name: evaluate(p, matches) for name, p in preds.items()}}


def _aggregate(editions: dict) -> dict:
    agg: dict = {}
    for ed in editions.values():
        for name, m in ed.get("models", {}).items():
            a = agg.setdefault(name, {"n": 0, "hits": 0, "_ll": 0.0, "_br": 0.0})
            a["n"] += m["n"]
            a["hits"] += m["hits"]
            a["_ll"] += m["log_loss"] * m["n"]
            a["_br"] += m["brier"] * m["n"]
    out = {}
    for name, a in agg.items():
        n = a["n"] or 1
        out[name] = {"n": a["n"], "hits": a["hits"],
                     "accuracy": round(a["hits"] / n, 4),
                     "log_loss": round(a["_ll"] / n, 4),
                     "brier": round(a["_br"] / n, 4)}
    return out


def run_backtest(dataset, editions, model_config, elo_config) -> dict:
    """editions = list of (tournament, year). Returns per-edition + aggregate metrics."""
    per_edition = {}
    for tournament, year in editions:
        per_edition[f"{tournament} {year}"] = backtest_edition(
            dataset, tournament, year, model_config, elo_config)
    return {"editions": per_edition, "aggregate": _aggregate(per_edition)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/eval/backtest.py tests/test_backtest.py
git commit -m "feat: train-before-edition backtest driver (out-of-sample)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: report + CLI

**Files:**
- Create: `footy/eval/report.py`
- Modify: `footy/cli.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
import json

import pandas as pd

from footy.eval.report import run_report

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25, "ridge": 0.01}
ELO_CFG = {"initial_rating": 1500.0, "k_factor": 40.0, "home_advantage_elo": 65.0,
           "default_tournament_weight": 0.7, "tournament_weights": {"Friendly": 0.4, "Cup": 1.0}}


def _dataset(tmp_path):
    rows = []
    for _ in range(20):
        rows.append(("2012-01-01", "Strong", "Weak", 3, 0, "Friendly"))
        rows.append(("2013-06-01", "Strong", "Mid", 2, 1, "Friendly"))
        rows.append(("2013-01-01", "Mid", "Weak", 2, 0, "Friendly"))
    rows.append(("2014-06-01", "Strong", "Weak", 1, 0, "Cup"))
    rows.append(("2014-06-02", "Mid", "Weak", 1, 0, "Cup"))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament"])
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = True
    p = tmp_path / "results.csv"
    df.to_csv(p, index=False)
    return p


def test_run_report_writes_json(tmp_path):
    ds = _dataset(tmp_path)
    out_path = tmp_path / "backtest_report.json"
    report = run_report(dataset_path=ds, editions=[("Cup", 2014)],
                        model_config=MODEL_CFG, elo_config=ELO_CFG, out_path=out_path)
    assert "editions" in report and "aggregate" in report and "meta" in report
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["aggregate"]["BASE"]["n"] == 2
    assert saved["meta"]["editions"] == ["Cup 2014"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.eval.report'`

- [ ] **Step 3: Write implementation**

`footy/eval/report.py`:
```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from footy.eval.backtest import run_backtest

DEFAULT_EDITIONS = [("FIFA World Cup", 2014), ("FIFA World Cup", 2018),
                    ("FIFA World Cup", 2022), ("FIFA World Cup", 2026)]


def run_report(dataset_path, editions, model_config, elo_config,
               out_path="artifacts/backtest_report.json") -> dict:
    """Run the historical backtest over `editions` and write a cached JSON report.

    Each World Cup edition is evaluated out-of-sample (trained on rows before it); the
    most recent edition (e.g. WC2026) is the live comparison.
    """
    dataset = pd.read_csv(dataset_path, parse_dates=["date"])
    for col in ("home_score", "away_score"):
        dataset = dataset[dataset[col].notna()]
    dataset["home_score"] = dataset["home_score"].astype(int)
    dataset["away_score"] = dataset["away_score"].astype(int)
    if "neutral" not in dataset:
        dataset["neutral"] = True

    bt = run_backtest(dataset, editions, model_config, elo_config)
    report = {
        "editions": bt["editions"],
        "aggregate": bt["aggregate"],
        "meta": {"editions": list(bt["editions"].keys()),
                 "generado": datetime.now().strftime("%Y-%m-%d %H:%M")},
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
```

Append to `footy/cli.py` (keep existing imports/functions; add the import and functions):
```python
from footy.eval.report import run_report, DEFAULT_EDITIONS


def run_backtest_cli(argv) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="backtest",
                                     description="Historical backtest vs baselines.")
    parser.add_argument("--out", default="artifacts/backtest_report.json")
    args = parser.parse_args(argv)
    data_cfg = load_config("data")
    raw_dir = data_cfg["raw_dir"]
    report = run_report(
        dataset_path=f"{raw_dir}/{data_cfg['files']['results']}",
        editions=DEFAULT_EDITIONS,
        model_config=load_config("model"),
        elo_config=load_config("elo"),
        out_path=args.out,
    )
    for edition, data in report["aggregate"].items():
        print(f"{edition:8} acc={data['accuracy']:.3f} logloss={data['log_loss']:.3f} "
              f"brier={data['brier']:.3f} (n={data['n']})")
    print(f"Saved {args.out}")
    return 0


def main_backtest() -> None:
    sys.exit(run_backtest_cli(sys.argv[1:]))
```

Add to `pyproject.toml` under `[project.scripts]`:
```toml
backtest = "footy.cli:main_backtest"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_report.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/eval/report.py footy/cli.py pyproject.toml tests/test_report.py
git commit -m "feat: backtest report (cached JSON) + backtest CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Streamlit "Evaluación" tab

**Files:**
- Modify: `app/streamlit_app.py`
- Test: `tests/test_streamlit_app_imports.py` (must keep passing)

- [ ] **Step 1: Confirm the smoke test passes today**

Run: `python -m pytest tests/test_streamlit_app_imports.py -v`
Expected: PASS (1 passed)

- [ ] **Step 2: Edit `app/streamlit_app.py`**

Add near the top imports:
```python
import json as _json
from footy.eval.report import run_report, DEFAULT_EDITIONS

REPORT_PATH = ROOT / "artifacts" / "backtest_report.json"
```

Add this render function (anywhere among the other `_render_*` functions):
```python
def _render_eval_tab(base_predictor):
    st.caption("Compara el modelo contra baselines (Elo, naive, azar) y hace backtest: "
               "entrena antes de cada Mundial y evalúa en él. Así sabes si 60% es bueno.")
    if st.button("Recalcular backtest (~6 min)"):
        with st.spinner("Entrenando y evaluando por edición…"):
            data_cfg = load_config("data")
            raw = data_cfg["raw_dir"]
            run_report(dataset_path=f"{raw}/{data_cfg['files']['results']}",
                       editions=DEFAULT_EDITIONS, model_config=load_config("model"),
                       elo_config=load_config("elo"), out_path=REPORT_PATH)
        st.success("Backtest actualizado.")
    if not REPORT_PATH.exists():
        st.info("Aún no hay reporte. Aprieta **Recalcular backtest**.")
        return
    report = _json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    st.subheader("Resumen (todas las ediciones)")
    agg = report["aggregate"]
    st.dataframe(pd.DataFrame([
        {"Modelo": name, "Aciertos %": round(d["accuracy"] * 100, 1),
         "Log loss": d["log_loss"], "Brier": d["brier"], "Partidos": d["n"]}
        for name, d in agg.items()], ).sort_values("Aciertos %", ascending=False),
        width="stretch", hide_index=True)
    edition = st.selectbox("Ver edición", list(report["editions"].keys()))
    ed = report["editions"][edition]
    if ed["n"] == 0:
        st.info("Sin partidos para esa edición en el dataset.")
        return
    st.dataframe(pd.DataFrame([
        {"Modelo": name, "Aciertos %": round(m["accuracy"] * 100, 1),
         "Log loss": m["log_loss"], "Brier": m["brier"],
         "Error goles": m["goal_mae"], "Partidos": m["n"]}
        for name, m in ed["models"].items()],
    ).sort_values("Aciertos %", ascending=False), width="stretch", hide_index=True)
```

In `main`, change the tabs line to add the new tab and render it:
```python
    t1, t2, t3, t4, t5 = st.tabs(["Partido", "Mundial", "Apuestas", "Scoreboard", "Evaluación"])
    with t1:
        _render_match_tab(base_predictor)
    with t2:
        _render_tournament_tab(base_predictor)
    with t3:
        _render_betting_tab(base_predictor)
    with t4:
        _render_scoreboard_tab(base_predictor)
    with t5:
        _render_eval_tab(base_predictor)
```

- [ ] **Step 3: Run the smoke test**

Run: `python -m pytest tests/test_streamlit_app_imports.py -v`
Expected: PASS (1 passed — module imports, `main`/`build_engine` present)

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: all PASS (smoke ~1.5 min).

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: Evaluación tab (comparative backtest vs baselines)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §3 predictors (Dixon-Coles analytic, Elo favorite, naive, random, build_predictors; unknown→fallback) → Task 3 (+ Elo final_ratings Task 2). ✓
- §4 evaluate + calibration → Tasks 1, 4. ✓
- §5 backtest train-before-edition, aggregate, anti-leakage → Task 5. ✓
- §6 report JSON + CLI → Task 6. ✓
- §7 Evaluación tab (aggregate table, edition selector) → Task 7. ✓
- §1 WC2026 as an edition (out-of-sample live compare) → `DEFAULT_EDITIONS` includes 2026 (Task 6). ✓
- §1 FIFA/bookmaker N/A → simply not in `build_predictors` (documented); slots can be added later. ✓
- §9 testing (predictors≈sim, calibration ECE, evaluate, backtest no-leakage, report json, smoke) → all test tasks. ✓
- §8 error handling (empty edition → n=0; unknown team → fallback) → Tasks 3, 5. ✓

**Placeholder scan:** all code complete; cli/pyproject/app edits are additive with explicit anchors. No TODO/TBD.

**Type consistency:** `calibration_buckets(probs, actuals, bins)` (T1) used in `evaluate` (T4); `final_ratings(matches, config)` (T2) used in `build_predictors` (T3); predictor `.probs(a,b,neutral)`/`.goals(a,b,neutral)` (T3) consumed by `evaluate` (T4); `build_predictors(train, model_config, elo_config, as_of)` (T3) called by `backtest_edition` (T5); `run_backtest(dataset, editions, model_config, elo_config)` (T5) called by `run_report` (T6); `run_report(dataset_path, editions, model_config, elo_config, out_path)` (T6) called by CLI + Evaluación tab (T6/T7). Report JSON keys (`editions`/`aggregate`/`meta`, model dict keys) match the tab's reads (T7). App keeps `main`/`build_engine`. ✓

**Note:** the real `backtest` run fits one Dixon-Coles model per edition (~2 min each on the full dataset). Tests use tiny synthetic datasets so they stay fast and network-free; the real run is the user-triggered cached script/CLI/tab.
