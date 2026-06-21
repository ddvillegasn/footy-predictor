# Betting Markets Layer Implementation Plan (SP2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-match betting layer (markets, fair odds, optional value/EV) on top of the SP1 Dixon-Coles + Monte Carlo engine, deriving every market from the same MC samples.

**Architecture:** Refactor `montecarlo.py` to expose `simulate_goals()` (the raw goal arrays) without changing `simulate()`'s output. New `footy/betting/` package: `markets.py` counts events on the arrays, `odds.py` converts probabilities to fair odds, `value.py` compares against user-supplied bookmaker odds. `predict()` gains optional `include_markets`/`book_odds` and stays byte-identical to SP1 when they are unused.

**Tech Stack:** Python 3.10.6, numpy, pandas, scipy, pyyaml, pytest. Run tests from repo root with `python -m pytest`.

**Conventions:** branch `feature/baseline-v1`. TDD: failing test first. Commit per task, `git add` ONLY the named files (never `__pycache__`/`.pyc`). Commit trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## File Structure

| File | Responsibility |
|---|---|
| `footy/models/montecarlo.py` | MODIFY: add `simulate_goals`, `aggregate_outcomes`; `simulate` composes them |
| `footy/config.py` | MODIFY: add `config_fingerprint(name)` (sha1[:8] of a config file) |
| `footy/betting/__init__.py` | NEW: `BETTING_VERSION = "sp2-v1.0.0"` |
| `configs/betting.yaml` | NEW: O/U + handicap lines, top_scores, value thresholds |
| `footy/betting/markets.py` | NEW: `markets_from_samples(goals_a, goals_b, config)` |
| `footy/betting/odds.py` | NEW: `fair_odds`, `implied_prob`, `model_margin`, `market_margin`, `decorate_outcome`, `decorate_group` |
| `footy/betting/value.py` | NEW: `assess_value`, `assess_market` |
| `footy/predict.py` | MODIFY: `predict(include_markets, book_odds)` + market/value assembly helpers |
| `footy/cli.py` | MODIFY: `--markets`, `--book-odds` flags + parser |
| `tests/test_simulate_goals.py`, `tests/test_markets.py`, `tests/test_odds.py`, `tests/test_value.py`, `tests/test_predict_markets.py`, `tests/test_cli_markets.py` | NEW tests |

---

## Task 1: Refactor montecarlo — extract `simulate_goals` + `aggregate_outcomes`

**Files:**
- Modify: `footy/models/montecarlo.py`
- Test: `tests/test_simulate_goals.py`

- [ ] **Step 1: Write the failing test**

`tests/test_simulate_goals.py`:
```python
import numpy as np
import pytest

from footy.models.montecarlo import simulate_goals, aggregate_outcomes, simulate

CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 8}


def test_simulate_goals_shapes_and_meta():
    ga, gb, meta = simulate_goals(2.0, 0.5, CFG)
    assert len(ga) == 20000 and len(gb) == 20000
    assert meta["seed"] == 42 and meta["n_sims"] == 20000
    assert meta["lambda_a"] == 2.0 and meta["lambda_b"] == 0.5
    assert meta["dc_enabled"] is False
    assert meta["clip_max"] == 10


def test_simulate_goals_is_seed_deterministic():
    a1, b1, _ = simulate_goals(1.5, 1.2, CFG)
    a2, b2, _ = simulate_goals(1.5, 1.2, CFG)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_simulate_goals_respects_clip():
    ga, gb, _ = simulate_goals(5.0, 5.0, {**CFG, "max_goals": 3})
    assert ga.max() <= 3 and gb.max() <= 3


def test_simulate_goals_rejects_nonpositive_lambda():
    with pytest.raises(ValueError):
        simulate_goals(0.0, 1.0, CFG)
    with pytest.raises(ValueError):
        simulate_goals(1.0, -0.5, CFG)


def test_simulate_output_unchanged_after_refactor():
    # simulate() must equal aggregate_outcomes over the same goal arrays.
    ga, gb, _ = simulate_goals(1.7, 0.9, CFG)
    expected = aggregate_outcomes(ga, gb, 1.7, 0.9, CFG)
    assert simulate(1.7, 0.9, CFG) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simulate_goals.py -v`
Expected: FAIL with `ImportError: cannot import name 'simulate_goals'`

- [ ] **Step 3: Rewrite `footy/models/montecarlo.py`**

```python
import numpy as np


def simulate_goals(lambda_a: float, lambda_b: float, config: dict):
    """Seeded independent-Poisson goal samples for both teams.

    Returns (goals_a, goals_b, meta). meta records seed/n_sims/lambdas plus
    dc_enabled (False: sampling is plain Poisson; Dixon-Coles tau lives in the
    fit, not the sampler) and clip_max. Raises ValueError if a lambda <= 0.
    """
    if lambda_a <= 0 or lambda_b <= 0:
        raise ValueError(f"lambdas must be > 0, got {lambda_a}, {lambda_b}")
    n = int(config["n_sims"])
    seed = int(config["seed"])
    max_goals = int(config["max_goals"])

    rng = np.random.default_rng(seed)
    goals_a = np.clip(rng.poisson(lambda_a, n), 0, max_goals)
    goals_b = np.clip(rng.poisson(lambda_b, n), 0, max_goals)

    meta = {
        "seed": seed,
        "n_sims": n,
        "lambda_a": round(float(lambda_a), 4),
        "lambda_b": round(float(lambda_b), 4),
        "dc_enabled": False,
        "clip_max": max_goals,
    }
    return goals_a, goals_b, meta


def aggregate_outcomes(goals_a, goals_b, lambda_a: float, lambda_b: float, config: dict) -> dict:
    """1X2, score distribution, expected goals (= lambdas), and goal CI from samples."""
    n = len(goals_a)
    ci_level = float(config["ci_level"])
    top_scores = int(config.get("top_scores", 8))

    a_win = float(np.mean(goals_a > goals_b)) * 100.0
    draw = float(np.mean(goals_a == goals_b)) * 100.0
    b_win = float(np.mean(goals_a < goals_b)) * 100.0

    pair_counts = {}
    for ga, gb in zip(goals_a.tolist(), goals_b.tolist()):
        key = f"{ga}-{gb}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    ordered = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    most_likely = ordered[0][0]
    distribution = {k: round(v / n * 100.0, 2) for k, v in ordered[:top_scores]}

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


def simulate(lambda_a: float, lambda_b: float, config: dict) -> dict:
    """Backward-compatible: same output as before, now via the two helpers."""
    goals_a, goals_b, _ = simulate_goals(lambda_a, lambda_b, config)
    return aggregate_outcomes(goals_a, goals_b, lambda_a, lambda_b, config)
```

- [ ] **Step 4: Run tests to verify they pass (new + existing montecarlo + predict)**

Run: `python -m pytest tests/test_simulate_goals.py tests/test_montecarlo.py tests/test_predict.py -v`
Expected: all PASS (refactor preserves `simulate()` output, so SP1 tests stay green).

- [ ] **Step 5: Commit**

```bash
git add footy/models/montecarlo.py tests/test_simulate_goals.py
git commit -m "refactor: expose simulate_goals/aggregate_outcomes without changing simulate output

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: betting config + package marker + config fingerprint

**Files:**
- Create: `configs/betting.yaml`, `footy/betting/__init__.py`
- Modify: `footy/config.py`
- Test: `tests/test_config_fingerprint.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config_fingerprint.py`:
```python
from footy.config import load_config, config_fingerprint
from footy.betting import BETTING_VERSION


def test_betting_config_loads():
    cfg = load_config("betting")
    assert cfg["over_under_lines"][2] == 2.5
    assert cfg["top_scores"] == 5
    assert cfg["value"]["reliability_high"] == 0.70


def test_fingerprint_is_stable_8_char_hex():
    fp = config_fingerprint("betting")
    assert isinstance(fp, str) and len(fp) == 8
    assert fp == config_fingerprint("betting")  # deterministic
    int(fp, 16)  # valid hex


def test_betting_version_constant():
    assert BETTING_VERSION == "sp2-v1.0.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_fingerprint.py -v`
Expected: FAIL with `ImportError: cannot import name 'config_fingerprint'`

- [ ] **Step 3: Create config + implementation**

`configs/betting.yaml`:
```yaml
over_under_lines: [0.5, 1.5, 2.5, 3.5, 4.5]
handicap_lines: [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
top_scores: 5
value:
  threshold: 0.0
  ev_medium: 0.10
  reliability_low: 0.40
  reliability_high: 0.70
  kelly_quarter_divisor: 4
```

`footy/betting/__init__.py`:
```python
BETTING_VERSION = "sp2-v1.0.0"
```

Append to `footy/config.py` (keep existing content; add import at top and the function at the end):
```python
import hashlib


def config_fingerprint(name: str) -> str:
    """Stable 8-char sha1 of a config file's bytes (reproducibility tag)."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    return digest[:8]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_fingerprint.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add configs/betting.yaml footy/betting/__init__.py footy/config.py tests/test_config_fingerprint.py
git commit -m "feat: betting config, package marker, config fingerprint helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: markets engine from samples

**Files:**
- Create: `footy/betting/markets.py`
- Test: `tests/test_markets.py`

- [ ] **Step 1: Write the failing test**

`tests/test_markets.py`:
```python
import math

import numpy as np

from footy.models.montecarlo import simulate_goals
from footy.betting.markets import markets_from_samples

CFG = {
    "n_sims": 60000, "seed": 7, "max_goals": 10,
    "over_under_lines": [0.5, 1.5, 2.5, 3.5, 4.5],
    "handicap_lines": [-1.5, -0.5, 0.5, 1.5],
    "top_scores": 5,
}


def _markets(lam_a, lam_b):
    ga, gb, _ = simulate_goals(lam_a, lam_b, CFG)
    return markets_from_samples(ga, gb, CFG)


def test_1x2_and_double_chance_consistency():
    m = _markets(1.8, 1.0)
    o = m["1x2"]
    assert abs(o["home"] + o["draw"] + o["away"] - 1.0) < 1e-9
    dc = m["double_chance"]
    assert abs(dc["1X"] - (o["home"] + o["draw"])) < 1e-9
    assert abs(dc["X2"] - (o["draw"] + o["away"])) < 1e-9
    assert abs(dc["12"] - (o["home"] + o["away"])) < 1e-9


def test_over_under_sums_to_one_per_line():
    m = _markets(1.4, 1.1)
    for line, ou in m["over_under"].items():
        assert abs(ou["over"] + ou["under"] - 1.0) < 1e-9


def test_correct_score_top_and_mass():
    m = _markets(1.6, 0.8)
    cs = m["correct_score"]
    assert len(cs["top"]) == 5
    assert abs(cs["all_mass_check"] - 1.0) < 1e-9
    assert abs(cs["other_probability"] - (1.0 - sum(cs["top"].values()))) < 1e-6


def test_handicap_complementary():
    m = _markets(2.0, 0.7)
    for line, h in m["handicap"].items():
        assert abs(h["home"] + h["away"] - 1.0) < 1e-9


def test_cross_check_btts_and_over05_closed_form():
    # Valid ONLY for dc_enabled=False, independent Poisson sampling.
    lam_a, lam_b = 1.7, 1.1
    m = _markets(lam_a, lam_b)
    btts_closed = (1 - math.exp(-lam_a)) * (1 - math.exp(-lam_b))
    over05_closed = 1 - math.exp(-(lam_a + lam_b))
    assert abs(m["btts"]["yes"] - btts_closed) < 0.02
    assert abs(m["over_under"]["0.5"]["over"] - over05_closed) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_markets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.betting.markets'`

- [ ] **Step 3: Write implementation**

`footy/betting/markets.py`:
```python
import numpy as np


def markets_from_samples(goals_a, goals_b, config: dict) -> dict:
    """Compute raw market probabilities by counting events over goal samples.

    Pure counting: no model recomputation. Returns nested dict of probabilities
    (floats rounded to 4 decimals). Handicap covers Asian .5 lines only.
    """
    goals_a = np.asarray(goals_a)
    goals_b = np.asarray(goals_b)
    if len(goals_a) != len(goals_b) or len(goals_a) == 0:
        raise ValueError("goal arrays must be non-empty and equal length")
    n = len(goals_a)
    total = goals_a + goals_b

    def p(mask) -> float:
        return round(float(np.mean(mask)), 4)

    home = p(goals_a > goals_b)
    draw = p(goals_a == goals_b)
    away = p(goals_a < goals_b)

    over_under = {}
    for line in config["over_under_lines"]:
        over_under[str(line)] = {"over": p(total > line), "under": p(total < line)}

    handicap = {}
    for h in config["handicap_lines"]:
        a_eff = goals_a + h
        handicap[str(h)] = {"home": p(a_eff > goals_b), "away": p(a_eff < goals_b)}

    # Correct score: full counts -> top-N + remaining mass.
    pair_counts = {}
    for ga, gb in zip(goals_a.tolist(), goals_b.tolist()):
        key = f"{ga}-{gb}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    ordered = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_n = int(config["top_scores"])
    top = {k: round(v / n, 4) for k, v in ordered[:top_n]}
    all_mass = round(sum(v / n for v in pair_counts.values()), 4)
    other = round(1.0 - sum(top.values()), 4)

    return {
        "1x2": {"home": home, "draw": draw, "away": away},
        "double_chance": {
            "1X": p(goals_a >= goals_b),
            "12": p(goals_a != goals_b),
            "X2": p(goals_a <= goals_b),
        },
        "over_under": over_under,
        "btts": {"yes": p((goals_a > 0) & (goals_b > 0)), "no": p(~((goals_a > 0) & (goals_b > 0)))},
        "correct_score": {"top": top, "other_probability": other, "all_mass_check": all_mass},
        "handicap": handicap,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_markets.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/betting/markets.py tests/test_markets.py
git commit -m "feat: market probabilities from Monte Carlo samples

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: odds conversion

**Files:**
- Create: `footy/betting/odds.py`
- Test: `tests/test_odds.py`

- [ ] **Step 1: Write the failing test**

`tests/test_odds.py`:
```python
import pytest

from footy.betting.odds import (
    fair_odds, implied_prob, model_margin, market_margin,
    decorate_outcome, decorate_group,
)


def test_fair_odds_basic():
    assert fair_odds(0.5) == 2.0
    assert fair_odds(0.8) == 1.25
    assert fair_odds(1.0) == 1.0


def test_fair_odds_zero_is_none():
    assert fair_odds(0.0) is None


def test_implied_prob():
    assert implied_prob(2.0) == 0.5
    assert round(implied_prob(1.25), 4) == 0.8


def test_model_margin_from_raw_probs_is_zero():
    assert abs(model_margin([0.86, 0.09, 0.05])) < 1e-9


def test_market_margin_from_book_odds_positive():
    # Typical vig: implied probs sum above 1.
    assert market_margin([1.45, 4.2, 7.0]) > 0.0


def test_decorate_outcome_no_sim_status():
    d = decorate_outcome(0.0)
    assert d["fair_odds"] is None and d["status"] == "no_sim"


def test_decorate_group_adds_margin():
    g = decorate_group({"home": 0.86, "draw": 0.09, "away": 0.05})
    assert g["home"]["fair_odds"] == round(1 / 0.86, 2)
    assert abs(g["margin"]) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_odds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.betting.odds'`

- [ ] **Step 3: Write implementation**

`footy/betting/odds.py`:
```python
def fair_odds(prob: float):
    """Fair decimal odds = 1/prob. prob==0 -> None (no simulated occurrences)."""
    if prob <= 0:
        return None
    return round(1.0 / prob, 2)


def implied_prob(decimal_odds: float) -> float:
    """Implied probability of a decimal odd = 1/odds."""
    return 1.0 / decimal_odds


def model_margin(probs) -> float:
    """Overround of the model's fair odds, from RAW probabilities: sum(probs)-1.

    Uses raw (unrounded) probabilities so rounding never produces a false margin.
    """
    return sum(probs) - 1.0


def market_margin(book_odds) -> float:
    """Overround of real bookmaker odds: sum(1/odd) - 1 (the vig)."""
    return sum(1.0 / o for o in book_odds) - 1.0


def decorate_outcome(prob: float) -> dict:
    """One outcome -> {prob, fair_odds[, status]}."""
    odds = fair_odds(prob)
    out = {"prob": round(prob, 4), "fair_odds": odds}
    if odds is None:
        out["status"] = "no_sim"
    return out


def decorate_group(prob_dict: dict) -> dict:
    """Decorate every outcome in a group and attach the model margin (raw probs)."""
    result = {k: decorate_outcome(v) for k, v in prob_dict.items()}
    result["margin"] = round(model_margin(list(prob_dict.values())), 4)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_odds.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/betting/odds.py tests/test_odds.py
git commit -m "feat: fair odds, implied prob, margins, market decoration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: value / EV assessment

**Files:**
- Create: `footy/betting/value.py`
- Test: `tests/test_value.py`

- [ ] **Step 1: Write the failing test**

`tests/test_value.py`:
```python
import pytest

from footy.betting.value import assess_value, assess_market

VCFG = {"threshold": 0.0, "ev_medium": 0.10, "reliability_low": 0.40,
        "reliability_high": 0.70, "kelly_quarter_divisor": 4}


def test_positive_value_detected():
    v = assess_value(model_prob=0.60, book_odds=2.0, reliability=0.8, config=VCFG)
    assert v["ev_per_unit"] == round(0.60 * 2.0 - 1.0, 4)
    assert v["is_value"] is True
    assert v["kelly_fraction_quarter"] == round(v["kelly_fraction_raw"] / 4, 4)


def test_negative_value_is_skip():
    v = assess_value(model_prob=0.40, book_odds=2.0, reliability=0.9, config=VCFG)
    assert v["ev_per_unit"] < 0
    assert v["is_value"] is False
    assert v["stake_recommendation"] == "skip"
    assert v["kelly_fraction_raw"] == 0.0


def test_low_reliability_forces_skip_even_with_edge():
    v = assess_value(model_prob=0.70, book_odds=2.0, reliability=0.3, config=VCFG)
    assert v["ev_per_unit"] > 0
    assert v["stake_recommendation"] == "skip"


def test_high_ev_and_high_reliability_is_medium():
    v = assess_value(model_prob=0.80, book_odds=2.0, reliability=0.9, config=VCFG)
    assert v["stake_recommendation"] == "medium"


def test_invalid_book_odds_raises():
    with pytest.raises(ValueError):
        assess_value(model_prob=0.5, book_odds=1.0, reliability=0.8, config=VCFG)


def test_assess_market_only_priced_outcomes():
    probs = {"home": 0.60, "draw": 0.25, "away": 0.15}
    odds = {"home": 2.0}  # only home priced
    out = assess_market(probs, odds, reliability=0.8, config=VCFG)
    assert set(out.keys()) == {"home"}
    assert out["home"]["is_value"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_value.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.betting.value'`

- [ ] **Step 3: Write implementation**

`footy/betting/value.py`:
```python
from footy.betting.odds import implied_prob, fair_odds


def _stake_recommendation(ev: float, reliability: float, config: dict) -> str:
    if ev <= 0 or reliability < config["reliability_low"]:
        return "skip"
    if ev > config["ev_medium"] and reliability >= config["reliability_high"]:
        return "medium"
    return "small"


def assess_value(model_prob: float, book_odds: float, reliability: float, config: dict) -> dict:
    """Compare a model probability against a bookmaker decimal odd.

    EV assumes the model probability is correct: it is value relative to the
    model, not guaranteed profit. Reliability gates the stake recommendation.
    """
    if book_odds <= 1.0:
        raise ValueError(f"book_odds must be > 1.0, got {book_odds}")

    book_imp = implied_prob(book_odds)
    ev = model_prob * book_odds - 1.0
    kelly_raw = (model_prob * book_odds - 1.0) / (book_odds - 1.0)
    if kelly_raw < 0:
        kelly_raw = 0.0
    divisor = config.get("kelly_quarter_divisor", 4)

    return {
        "model_prob": round(model_prob, 4),
        "fair_odds": fair_odds(model_prob),
        "book_odds": round(book_odds, 2),
        "book_implied": round(book_imp, 4),
        "edge_pct": round(ev * 100.0, 2),
        "ev_per_unit": round(ev, 4),
        "kelly_fraction_raw": round(kelly_raw, 4),
        "kelly_fraction_quarter": round(kelly_raw / divisor, 4),
        "is_value": bool(ev > config.get("threshold", 0.0)),
        "stake_recommendation": _stake_recommendation(ev, reliability, config),
    }


def assess_market(prob_dict: dict, book_odds_dict: dict, reliability: float, config: dict) -> dict:
    """Assess value only for outcomes that have a supplied bookmaker odd."""
    out = {}
    for outcome, odd in book_odds_dict.items():
        if outcome not in prob_dict:
            continue  # odd for a nonexistent outcome -> ignored
        out[outcome] = assess_value(prob_dict[outcome], odd, reliability, config)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_value.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/betting/value.py tests/test_value.py
git commit -m "feat: value/EV/Kelly/stake assessment vs book odds

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: integrate markets + value into `predict()`

**Files:**
- Modify: `footy/predict.py`
- Test: `tests/test_predict_markets.py`

- [ ] **Step 1: Write the failing test**

`tests/test_predict_markets.py`:
```python
import pandas as pd

from footy.predict import Predictor

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 5}
BET_CFG = {"over_under_lines": [0.5, 1.5, 2.5, 3.5, 4.5],
           "handicap_lines": [-1.5, -0.5, 0.5, 1.5], "top_scores": 5,
           "value": {"threshold": 0.0, "ev_medium": 0.10, "reliability_low": 0.40,
                     "reliability_high": 0.70, "kelly_quarter_divisor": 4}}


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
        _matches(), model_config=MODEL_CFG, mc_config=MC_CFG, canonical=lambda x: x,
        as_of=pd.Timestamp("2020-01-01"), betting_config=BET_CFG,
        betting_config_version="testhash",
    )


def test_no_flags_matches_sp1_shape():
    out = _predictor().predict("Brazil", "Haiti", neutral=True)
    assert "markets" not in out and "value" not in out
    assert "simulation_meta" not in out


def test_include_markets_adds_markets_and_meta():
    out = _predictor().predict("Brazil", "Haiti", neutral=True, include_markets=True)
    assert out["betting_version"] == "sp2-v1.0.0"
    assert out["simulation_meta"]["betting_config_version"] == "testhash"
    assert out["simulation_meta"]["dc_enabled"] is False
    m = out["markets"]
    assert set(m) == {"1x2", "double_chance", "over_under", "btts", "correct_score", "handicap"}
    assert m["1x2"]["home"]["fair_odds"] is not None
    assert "value" not in out


def test_book_odds_adds_value_only_for_priced():
    book = {"1x2": {"home": 1.05}}
    out = _predictor().predict("Brazil", "Haiti", neutral=True, book_odds=book)
    assert set(out["value"].keys()) == {"1x2"}
    assert set(out["value"]["1x2"].keys()) == {"home"}
    assert "markets" in out  # book_odds implies markets


def test_book_odds_over_under_nested():
    book = {"over_under": {"2.5": {"over": 1.5}}}
    out = _predictor().predict("Brazil", "Haiti", neutral=True, book_odds=book)
    assert "over" in out["value"]["over_under"]["2.5"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predict_markets.py -v`
Expected: FAIL with `TypeError` (unexpected `betting_config`) or assertion failure.

- [ ] **Step 3: Rewrite `footy/predict.py`**

```python
from __future__ import annotations

import difflib

import pandas as pd

from footy.models.poisson import fit_dixon_coles, DixonColesModel
from footy.models.montecarlo import simulate_goals, aggregate_outcomes
from footy.features.strength import recent_form
from footy.reliability import compute_reliability
from footy.betting import BETTING_VERSION
from footy.betting.markets import markets_from_samples
from footy.betting.odds import decorate_outcome, decorate_group
from footy.betting.value import assess_market


def _decorate_markets(raw: dict) -> dict:
    """Walk raw market probabilities and attach fair odds (+ group margins)."""
    out = {
        "1x2": decorate_group(raw["1x2"]),
        "double_chance": decorate_group(raw["double_chance"]),
        "over_under": {line: decorate_group(g) for line, g in raw["over_under"].items()},
        "btts": decorate_group(raw["btts"]),
        "handicap": {line: decorate_group(g) for line, g in raw["handicap"].items()},
        "correct_score": {
            "top": {s: decorate_outcome(p) for s, p in raw["correct_score"]["top"].items()},
            "other_probability": raw["correct_score"]["other_probability"],
            "all_mass_check": raw["correct_score"]["all_mass_check"],
        },
    }
    return out


def _value_tree(raw: dict, book_odds: dict, reliability: float, value_config: dict) -> dict:
    """Assess value for each market/outcome the user priced. Flat and per-line markets."""
    out = {}
    for market, priced in book_odds.items():
        if market not in raw:
            continue
        if market in ("1x2", "double_chance", "btts"):
            out[market] = assess_market(raw[market], priced, reliability, value_config)
        elif market in ("over_under", "handicap"):
            per_line = {}
            for line, odds_group in priced.items():
                if line in raw[market]:
                    per_line[line] = assess_market(raw[market][line], odds_group, reliability, value_config)
            out[market] = per_line
        elif market == "correct_score":
            out[market] = assess_market(raw[market]["top"], priced, reliability, value_config)
    return out


class Predictor:
    """Bundles a fitted model + match history to answer predict() queries."""

    def __init__(self, model: DixonColesModel, matches: pd.DataFrame,
                 model_config: dict, mc_config: dict, canonical,
                 betting_config: dict | None = None, betting_config_version: str | None = None):
        self.model = model
        self.matches = matches
        self.model_config = model_config
        self.mc_config = mc_config
        self.canonical = canonical
        self.betting_config = betting_config
        self.betting_config_version = betting_config_version
        self._teams = set(model.attack.keys())

    @classmethod
    def from_matches(cls, matches: pd.DataFrame, model_config: dict, mc_config: dict,
                     canonical, as_of, betting_config: dict | None = None,
                     betting_config_version: str | None = None) -> "Predictor":
        canon = matches.copy()
        canon["home_team"] = canon["home_team"].map(canonical)
        canon["away_team"] = canon["away_team"].map(canonical)
        model = fit_dixon_coles(canon, model_config, as_of=as_of)
        return cls(model, canon, model_config, mc_config, canonical,
                   betting_config, betting_config_version)

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
                tournament: str = "Friendly", include_markets: bool = False,
                book_odds: dict | None = None) -> dict:
        a = self._resolve(team_a)
        b = self._resolve(team_b)

        lam_a, lam_b = self.model.rates(a, b, neutral=neutral)
        goals_a, goals_b, meta = simulate_goals(lam_a, lam_b, self.mc_config)
        sim = aggregate_outcomes(goals_a, goals_b, lam_a, lam_b, self.mc_config)

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

        result = {
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

        if not include_markets and book_odds is None:
            return result

        if self.betting_config is None:
            raise ValueError("betting_config is required for markets/value output")

        raw = markets_from_samples(goals_a, goals_b, self.betting_config)
        meta["betting_config_version"] = self.betting_config_version
        result["betting_version"] = BETTING_VERSION
        result["simulation_meta"] = meta
        result["markets"] = _decorate_markets(raw)

        if book_odds:
            result["value"] = _value_tree(
                raw, book_odds, reliability, self.betting_config["value"]
            )
        return result
```

- [ ] **Step 4: Run tests to verify they pass (new + SP1 predict still green)**

Run: `python -m pytest tests/test_predict_markets.py tests/test_predict.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add footy/predict.py tests/test_predict_markets.py
git commit -m "feat: optional markets/value output in predict() (SP1-compatible)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: CLI flags `--markets` and `--book-odds`

**Files:**
- Modify: `footy/cli.py`
- Test: `tests/test_cli_markets.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli_markets.py`:
```python
import json

import pandas as pd

from footy import cli
from footy.predict import Predictor

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 5}
BET_CFG = {"over_under_lines": [0.5, 1.5, 2.5, 3.5, 4.5],
           "handicap_lines": [-1.5, -0.5, 0.5, 1.5], "top_scores": 5,
           "value": {"threshold": 0.0, "ev_medium": 0.10, "reliability_low": 0.40,
                     "reliability_high": 0.70, "kelly_quarter_divisor": 4}}


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
        _matches(), model_config=MODEL_CFG, mc_config=MC_CFG, canonical=lambda x: x,
        as_of=pd.Timestamp("2020-01-01"), betting_config=BET_CFG,
        betting_config_version="testhash",
    )


def test_parse_book_odds_string():
    parsed = cli.parse_book_odds("1x2.home=1.45,1x2.draw=4.2,over_under.2.5.over=1.67")
    assert parsed == {"1x2": {"home": 1.45, "draw": 4.2},
                      "over_under": {"2.5": {"over": 1.67}}}


def test_cli_markets_flag(capsys):
    code = cli.run(["Brazil", "Haiti", "--neutral", "--markets"], predictor=_predictor())
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and "markets" in payload


def test_cli_book_odds_flag(capsys):
    code = cli.run(["Brazil", "Haiti", "--neutral", "--book-odds", "1x2.home=1.05"],
                   predictor=_predictor())
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["value"]["1x2"]["home"]["is_value"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_markets.py -v`
Expected: FAIL with `AttributeError: module 'footy.cli' has no attribute 'parse_book_odds'`

- [ ] **Step 3: Rewrite `footy/cli.py`**

```python
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from footy.config import load_config, config_fingerprint
from footy.data.loaders import load_results, load_former_names
from footy.data.clean import clean_results
from footy.data.names import NameCanonicalizer
from footy.predict import Predictor


def parse_book_odds(text: str) -> dict:
    """Parse "1x2.home=1.45,over_under.2.5.over=1.67" into a nested dict.

    Each entry is dotted-path=decimal. 2-level paths (market.outcome) and
    3-level paths (market.line.side) are supported.
    """
    result: dict = {}
    if not text:
        return result
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        path, _, raw_value = entry.partition("=")
        if not raw_value:
            raise ValueError(f"Malformed --book-odds entry {entry!r}; expected path=odds")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Bad odds in {entry!r}: {raw_value!r} is not a number") from exc
        parts = path.split(".")
        node = result
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value
    return result


def _build_default_predictor() -> Predictor:
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    mc_cfg = load_config("montecarlo")
    bet_cfg = load_config("betting")

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
        betting_config=bet_cfg, betting_config_version=config_fingerprint("betting"),
    )


def run(argv: list[str], predictor: Predictor | None = None) -> int:
    parser = argparse.ArgumentParser(prog="predict", description="Predict a national-team match.")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--neutral", action="store_true", help="Neutral venue (home_advantage=0)")
    parser.add_argument("--tournament", default="Friendly")
    parser.add_argument("--markets", action="store_true", help="Include betting markets")
    parser.add_argument("--book-odds", default=None,
                        help='Bookmaker odds, e.g. "1x2.home=1.45,1x2.draw=4.2"')
    args = parser.parse_args(argv)

    if predictor is None:
        predictor = _build_default_predictor()

    book_odds = parse_book_odds(args.book_odds) if args.book_odds else None
    include_markets = args.markets or book_odds is not None

    try:
        result = predictor.predict(
            args.team_a, args.team_b, neutral=args.neutral, tournament=args.tournament,
            include_markets=include_markets, book_odds=book_odds,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))
```

- [ ] **Step 4: Run tests to verify they pass (new + SP1 CLI still green)**

Run: `python -m pytest tests/test_cli_markets.py tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add footy/cli.py tests/test_cli_markets.py
git commit -m "feat: CLI --markets and --book-odds flags

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: full-suite verification

- [ ] **Step 1: Run the entire suite**

Run: `python -m pytest -q`
Expected: all tests PASS (SP1 + SP2). The real-data smoke test may take ~2 min.

- [ ] **Step 2: If anything fails**

Debug with superpowers:systematic-debugging. Do not weaken assertions. SP1 tests must remain green (the refactor preserves `simulate()` output).

- [ ] **Step 3: No commit needed** (verification only) unless a fix was required, in which case commit the fix with a descriptive message and the trailer.

---

## Self-Review

**Spec coverage:**
- §2 `simulate_goals` + meta (seed/n_sims/λ/dc_enabled/clip_max), `simulate` unchanged, λ≤0 error → Task 1. ✓
- §2 module layout, isolation → Tasks 2–5. ✓
- §3 all markets (1X2, double chance, O/U, BTTS, correct_score top+other+mass, asian handicap) → Task 3. ✓
- §4 fair odds, implied prob, model_margin (raw probs), market_margin, decoration, prob=0→None/no_sim → Task 4. ✓
- §5 value/EV/Kelly raw+quarter, stake_recommendation (EV+reliability), book_odds≤1 error, only-priced → Task 5. ✓
- §6 predict integration, compat-SP1 when no flags, betting_version, betting_config_version in simulation_meta, book_odds JSON nested → Task 6. ✓
- §6 CLI flags + string→nested parse → Task 7. ✓
- §7 betting.yaml + fingerprint → Task 2. ✓
- §8 testing A + cross-check C with dc_enabled=False note → Task 3 (`test_cross_check_*`), plus per-module tests. ✓

**Placeholder scan:** every code step is complete; no TODO/TBD. ✓

**Type consistency:** `simulate_goals -> (goals_a, goals_b, meta)` (T1) consumed in T3/T6; `aggregate_outcomes(goals_a, goals_b, lambda_a, lambda_b, config)` (T1) called in T6; `markets_from_samples(goals_a, goals_b, config)` (T3) consumed in T6 walkers; `decorate_group`/`decorate_outcome` (T4) used in T6 `_decorate_markets`; `assess_market(prob_dict, book_odds_dict, reliability, config)` (T5) used in T6 `_value_tree`; `Predictor.from_matches(..., betting_config, betting_config_version)` (T6) used in T7 default builder and tests; `parse_book_odds` (T7) returns the nested shape `_value_tree` expects. ✓

**Note:** Task 6 routes `correct_score` book odds against `raw["correct_score"]["top"]`; only top-N scores can be priced/valued (matches the spec's clean output). Documented in `_value_tree`.
