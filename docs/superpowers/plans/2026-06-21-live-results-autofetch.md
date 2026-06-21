# Live Auto-Fetch + Scoreboard Implementation Plan (SP4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-fetch official World Cup results from football-data.org, write them into the SP3 results file idempotently, re-run the tournament simulator, and report a predicted-vs-actual scoreboard — on demand or on a `--watch` interval.

**Architecture:** New `footy/live/` package on top of SP1 (`predict`, `metrics`) and SP3 (`structure`, `results`, `MatchSampler`, `simulate_tournaments`, `aggregate`). A pluggable `ResultsProvider` (concrete `FootballDataProvider` using `requests`) feeds `ingest`, which rewrites `wc2026_results.yaml`. A provider-agnostic `TournamentRunner` orchestrates ingest + simulate + scoreboard; `watch` loops it. The model is fit once and reused.

**Tech Stack:** Python 3.10.6, requests, pyyaml, numpy, pytest. Run from repo root: `python -m pytest`.

**Conventions:** branch `feature/baseline-v1`. TDD: failing test first. Commit per task, `git add` ONLY named files (never `__pycache__`/`.pyc`). No real network in tests (HTTP mocked / FakeProvider). Commit trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## File Structure

| File | Responsibility |
|---|---|
| `footy/live/__init__.py` | package marker |
| `footy/live/provider.py` | `ProviderMatch`, `ResultsProvider`, `FootballDataProvider` |
| `footy/live/name_map.py` | `load_name_map`, `map_team`, `map_stage` (hard errors) |
| `footy/live/ingest.py` | `build_played_matches`, `ingest` (idempotent writer) |
| `footy/live/scoreboard.py` | `scoreboard` (predicted vs actual) |
| `footy/live/runner.py` | `TournamentRunner`, `watch` |
| `footy/cli.py` | add `update-and-simulate` entry (`run_update`) |
| `configs/live.yaml`, `configs/name_map.yaml` | live config + name map |
| `pyproject.toml` | add `requests` dep + `update-and-simulate` script |
| `tests/test_*` | per-module, no network |

---

## Task 1: provider.py + requests dependency

**Files:**
- Create: `footy/live/__init__.py`, `footy/live/provider.py`
- Modify: `pyproject.toml`
- Test: `tests/test_football_data_adapter.py`

- [ ] **Step 1: Write the failing test**

`tests/test_football_data_adapter.py`:
```python
import pytest

import footy.live.provider as prov

PAYLOAD = {"matches": [
    {"id": 111, "homeTeam": {"name": "Brazil"}, "awayTeam": {"name": "Mexico"},
     "score": {"fullTime": {"home": 2, "away": 1}},
     "stage": "GROUP_STAGE", "group": "GROUP_A", "status": "FINISHED"},
    {"id": 112, "homeTeam": {"name": "USA"}, "awayTeam": {"name": "Iran"},
     "score": {"fullTime": {"home": None, "away": None}},
     "stage": "GROUP_STAGE", "group": "GROUP_B", "status": "FINISHED"},
]}


class FakeResp:
    def __init__(self, payload, json_exc=False):
        self._payload = payload
        self._json_exc = json_exc

    def raise_for_status(self):
        return None

    def json(self):
        if self._json_exc:
            raise ValueError("not json")
        return self._payload


def test_fetch_parses_and_skips_unscored(monkeypatch):
    monkeypatch.setattr(prov.requests, "get", lambda *a, **k: FakeResp(PAYLOAD))
    p = prov.FootballDataProvider("key", "https://x/v4/", "WC", 10)
    out = p.fetch_finished()
    assert len(out) == 1                       # the None-scored match is skipped
    m = out[0]
    assert m.api_match_id == "111" and isinstance(m.api_match_id, str)
    assert m.home_team == "Brazil" and m.away_team == "Mexico"
    assert m.home_score == 2 and m.away_score == 1
    assert m.stage == "GROUP_STAGE" and m.group == "GROUP_A"


def test_base_url_is_rstripped():
    p = prov.FootballDataProvider("key", "https://x/v4/", "WC", 10)
    assert p.base_url == "https://x/v4"


def test_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout"):
        prov.FootballDataProvider("key", "https://x/v4", "WC", 0)


def test_from_config_requires_env(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FOOTBALL_DATA_API_KEY"):
        prov.FootballDataProvider.from_config(
            {"base_url": "https://x/v4", "competition_code": "WC", "request_timeout": 10})


def test_bad_json_raises_clear(monkeypatch):
    monkeypatch.setattr(prov.requests, "get", lambda *a, **k: FakeResp(None, json_exc=True))
    p = prov.FootballDataProvider("key", "https://x/v4", "WC", 10)
    with pytest.raises(ValueError, match="not valid JSON"):
        p.fetch_finished()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_football_data_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.live'`

- [ ] **Step 3: Write implementation + add dependency**

`footy/live/__init__.py`:
```python
```
(empty — a single newline)

`footy/live/provider.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import requests


@dataclass
class ProviderMatch:
    api_match_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    stage: str
    group: str | None
    status: str


class ResultsProvider(Protocol):
    def fetch_finished(self) -> list[ProviderMatch]: ...


class FootballDataProvider:
    def __init__(self, api_key, base_url, competition_code, timeout):
        if timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.competition_code = competition_code
        self.timeout = timeout

    @classmethod
    def from_config(cls, live_cfg: dict) -> "FootballDataProvider":
        key = os.environ.get("FOOTBALL_DATA_API_KEY")
        if not key:
            raise ValueError("FOOTBALL_DATA_API_KEY env var not set")
        return cls(key, live_cfg["base_url"], live_cfg["competition_code"],
                   live_cfg["request_timeout"])

    def fetch_finished(self) -> list[ProviderMatch]:
        url = f"{self.base_url}/competitions/{self.competition_code}/matches"
        resp = requests.get(url, headers={"X-Auth-Token": self.api_key},
                            params={"status": "FINISHED"}, timeout=self.timeout)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ValueError("football-data response is not valid JSON") from exc
        out = []
        for m in payload.get("matches", []):
            ft = m["score"]["fullTime"]
            if ft.get("home") is None or ft.get("away") is None:
                continue
            out.append(ProviderMatch(
                api_match_id=str(m["id"]),
                home_team=m["homeTeam"]["name"], away_team=m["awayTeam"]["name"],
                home_score=int(ft["home"]), away_score=int(ft["away"]),
                stage=m["stage"], group=m.get("group"), status=m["status"]))
        return out
```

In `pyproject.toml`, add `"requests>=2.31"` to `[project].dependencies` (keep the existing
pandas/numpy/scipy/pyyaml/pyarrow entries), and add a second script line under
`[project.scripts]`:
```toml
update-and-simulate = "footy.cli:main_update"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_football_data_adapter.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/live/__init__.py footy/live/provider.py pyproject.toml tests/test_football_data_adapter.py
git commit -m "feat: football-data results provider (pluggable) + requests dep

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: name_map.py

**Files:**
- Create: `footy/live/name_map.py`
- Test: `tests/test_name_map.py`

- [ ] **Step 1: Write the failing test**

`tests/test_name_map.py`:
```python
import pytest

from footy.live.name_map import load_name_map, map_team, map_stage

KNOWN = {"United States", "South Korea", "Brazil"}
STAGE_MAP = {"GROUP_STAGE": "group", "LAST_16": "R16"}


def test_load_name_map_ok(tmp_path):
    import yaml
    p = tmp_path / "nm.yaml"
    p.write_text(yaml.safe_dump({"teams": {"USA": "United States"}}), encoding="utf-8")
    assert load_name_map(p) == {"USA": "United States"}


def test_load_name_map_non_dict_raises(tmp_path):
    p = tmp_path / "nm.yaml"
    p.write_text("teams: [1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_name_map(p)


def test_map_team_uses_mapping_then_passthrough():
    mapping = {"USA": "United States", "Korea Republic": "South Korea"}
    assert map_team("USA", mapping, KNOWN) == "United States"
    assert map_team("Brazil", mapping, KNOWN) == "Brazil"   # passthrough, already known


def test_map_team_unmapped_raises():
    with pytest.raises(ValueError, match="unmapped API team"):
        map_team("Atlantis", {}, KNOWN)


def test_map_stage_hit_and_miss():
    assert map_stage("GROUP_STAGE", STAGE_MAP) == "group"
    with pytest.raises(ValueError, match="unknown API stage"):
        map_stage("PENALTIES", STAGE_MAP)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_name_map.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.live.name_map'`

- [ ] **Step 3: Write implementation**

`footy/live/name_map.py`:
```python
from pathlib import Path

import yaml


def load_name_map(path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    teams = data.get("teams", {})
    if not isinstance(teams, dict):
        raise ValueError("configs/name_map.yaml: 'teams' must be a mapping")
    return teams


def map_team(api_name: str, mapping: dict, known_teams: set) -> str:
    name = mapping.get(api_name, api_name)
    if name not in known_teams:
        raise ValueError(
            f"unmapped API team '{api_name}' (resolved '{name}' not in tournament structure); "
            f"add an entry to configs/name_map.yaml")
    return name


def map_stage(raw_stage: str, stage_map: dict) -> str:
    if raw_stage not in stage_map:
        raise ValueError(f"unknown API stage '{raw_stage}'; add it to stage_map in configs/live.yaml")
    return stage_map[raw_stage]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_name_map.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/live/name_map.py tests/test_name_map.py
git commit -m "feat: exact name/stage mapping with hard errors

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: ingest.py (idempotent)

**Files:**
- Create: `footy/live/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ingest.py`:
```python
from types import SimpleNamespace

import pytest

from footy.live.provider import ProviderMatch
from footy.live.ingest import ingest, GENERATED_HEADER
from footy.tournament.results import load_results

STAGE_MAP = {"GROUP_STAGE": "group", "LAST_16": "R16"}


class FakeProvider:
    def __init__(self, matches):
        self._matches = matches

    def fetch_finished(self):
        return self._matches


def _structure():
    return SimpleNamespace(groups={"A": ["Brazil", "Mexico", "Ecuador", "Honduras"],
                                    "B": ["United States", "Iran", "Wales", "Ghana"]})


def _matches():
    return [
        ProviderMatch("111", "Brazil", "Mexico", 2, 1, "GROUP_STAGE", "GROUP_A", "FINISHED"),
        ProviderMatch("112", "United States", "Iran", 1, 0, "GROUP_STAGE", "GROUP_B", "FINISHED"),
    ]


def test_ingest_writes_sp3_compatible_yaml_with_header(tmp_path):
    out = tmp_path / "wc2026_results.yaml"
    n = ingest(FakeProvider(_matches()), _structure(), {}, STAGE_MAP, out)
    assert n == 2
    text = out.read_text(encoding="utf-8")
    assert text.startswith(GENERATED_HEADER)
    # SP3 loader parses it.
    res = load_results(out, _structure().groups)
    pm = res.lookup_group("A", "Brazil", "Mexico")
    assert pm.goals_a == 2 and pm.goals_b == 1


def test_ingest_is_idempotent(tmp_path):
    out = tmp_path / "r.yaml"
    ingest(FakeProvider(_matches()), _structure(), {}, STAGE_MAP, out)
    first = out.read_text(encoding="utf-8")
    ingest(FakeProvider(_matches()), _structure(), {}, STAGE_MAP, out)
    assert out.read_text(encoding="utf-8") == first


def test_group_stage_without_group_raises(tmp_path):
    bad = [ProviderMatch("1", "Brazil", "Mexico", 1, 0, "GROUP_STAGE", None, "FINISHED")]
    with pytest.raises(ValueError, match="no group"):
        ingest(FakeProvider(bad), _structure(), {}, STAGE_MAP, tmp_path / "r.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.live.ingest'`

- [ ] **Step 3: Write implementation**

`footy/live/ingest.py`:
```python
from pathlib import Path

import yaml

from footy.live.name_map import map_team, map_stage

GENERATED_HEADER = "# GENERATED by footy.live.ingest — do not edit manually\n"


def _group_letter(raw_group: str) -> str:
    return raw_group.split("_")[-1]


def build_played_matches(provider_matches, name_map, known_teams, stage_map) -> list:
    out = []
    for pm in provider_matches:
        if pm.home_score is None or pm.away_score is None:
            raise ValueError(f"match {pm.api_match_id} has no final score")
        team_a = map_team(pm.home_team, name_map, known_teams)
        team_b = map_team(pm.away_team, name_map, known_teams)
        stage = map_stage(pm.stage, stage_map)
        group = None
        if stage == "group":
            if pm.group is None:
                raise ValueError(f"group-stage match {pm.api_match_id} has no group")
            group = _group_letter(pm.group)
        out.append({"match_id": pm.api_match_id, "stage": stage, "group": group,
                    "team_a": team_a, "team_b": team_b,
                    "goals_a": pm.home_score, "goals_b": pm.away_score})
    return out


def ingest(provider, structure, name_map, stage_map, out_path) -> int:
    known_teams = {t for group in structure.groups.values() for t in group}
    played = build_played_matches(provider.fetch_finished(), name_map, known_teams, stage_map)
    dedup = {}
    for m in played:
        dedup[m["match_id"]] = m
    final = sorted(dedup.values(), key=lambda m: (m["stage"], m.get("group") or "", m["match_id"]))
    body = yaml.safe_dump({"played_matches": final}, sort_keys=False, allow_unicode=True)
    Path(out_path).write_text(GENERATED_HEADER + body, encoding="utf-8")
    return len(final)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/live/ingest.py tests/test_ingest.py
git commit -m "feat: idempotent ingest writing SP3-compatible results yaml

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: scoreboard.py

**Files:**
- Create: `footy/live/scoreboard.py`
- Test: `tests/test_scoreboard.py`

- [ ] **Step 1: Write the failing test**

`tests/test_scoreboard.py`:
```python
from footy.live.scoreboard import scoreboard


class FakePredictor:
    """Returns canned predictions keyed by (team_a, team_b)."""
    def __init__(self, table):
        self.table = table

    def predict(self, team_a, team_b, neutral=False):
        return self.table[(team_a, team_b)]


def test_empty_played_returns_none_metrics():
    board = scoreboard(FakePredictor({}), [])
    assert board["n"] == 0 and board["accuracy"] is None and board["matches"] == []


def test_scoreboard_metrics_and_details():
    table = {
        ("Brazil", "Mexico"): {"team_a_win": 70.0, "draw": 20.0, "team_b_win": 10.0,
                               "expected_goals_a": 2.0, "expected_goals_b": 0.5,
                               "most_likely_score": "2-0"},
        ("USA", "Iran"): {"team_a_win": 30.0, "draw": 30.0, "team_b_win": 40.0,
                          "expected_goals_a": 1.0, "expected_goals_b": 1.2,
                          "most_likely_score": "1-1"},
    }
    played = [
        {"team_a": "Brazil", "team_b": "Mexico", "goals_a": 2, "goals_b": 1},  # home win -> hit
        {"team_a": "USA", "team_b": "Iran", "goals_a": 3, "goals_b": 0},        # home win -> predicted away, miss
    ]
    board = scoreboard(FakePredictor(table), played)
    assert board["n"] == 2
    assert 0.0 <= board["accuracy"] <= 1.0
    assert board["accuracy"] == 0.5            # 1 of 2 correct
    first = board["matches"][0]
    assert first["hit"] is True
    assert first["predicted_outcome"] == "home" and first["actual_outcome"] == "home"
    assert first["predicted_prob"] == 0.7 and first["actual_prob"] == 0.7
    second = board["matches"][1]
    assert second["hit"] is False
    assert second["actual_outcome"] == "home" and second["actual_prob"] == 0.3
    assert board["goal_mae"] is not None and board["log_loss"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scoreboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.live.scoreboard'`

- [ ] **Step 3: Write implementation**

`footy/live/scoreboard.py`:
```python
from footy.metrics import log_loss_1x2, brier_1x2, accuracy_1x2


def _outcome(ga, gb):
    if ga > gb:
        return "home"
    if ga < gb:
        return "away"
    return "draw"


def scoreboard(predictor, played_matches: list) -> dict:
    """Compare the (pre-tournament) model's predictions to actual played results.

    Out-of-sample: the model is NOT retrained on tournament results (anti-leakage).
    """
    if not played_matches:
        return {"n": 0, "accuracy": None, "log_loss": None, "brier": None,
                "goal_mae": None, "matches": []}

    probs, actuals, details, goal_err = [], [], [], 0.0
    for m in played_matches:
        pred = predictor.predict(m["team_a"], m["team_b"], neutral=True)
        p = {"home": pred["team_a_win"] / 100.0, "draw": pred["draw"] / 100.0,
             "away": pred["team_b_win"] / 100.0}
        actual = _outcome(m["goals_a"], m["goals_b"])
        predicted = max(p, key=p.get)
        probs.append(p)
        actuals.append(actual)
        goal_err += abs(pred["expected_goals_a"] - m["goals_a"]) + abs(pred["expected_goals_b"] - m["goals_b"])
        details.append({
            "match": f"{m['team_a']} vs {m['team_b']}",
            "predicted_score": pred["most_likely_score"],
            "actual_score": f"{m['goals_a']}-{m['goals_b']}",
            "predicted_outcome": predicted,
            "actual_outcome": actual,
            "predicted_prob": round(p[predicted], 4),
            "actual_prob": round(p[actual], 4),
            "hit": predicted == actual,
        })

    n = len(played_matches)
    return {"n": n,
            "accuracy": accuracy_1x2(probs, actuals),
            "log_loss": log_loss_1x2(probs, actuals),
            "brier": brier_1x2(probs, actuals),
            "goal_mae": round(goal_err / (2 * n), 3),
            "matches": details}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scoreboard.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/live/scoreboard.py tests/test_scoreboard.py
git commit -m "feat: out-of-sample predicted-vs-actual scoreboard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: runner.py + watch

**Files:**
- Create: `footy/live/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

`tests/test_runner.py`:
```python
import numpy as np

from footy.tournament.structure import TournamentConfig
from footy.live.provider import ProviderMatch
from footy.live.runner import TournamentRunner, watch


class FakeProvider:
    def __init__(self, matches):
        self._matches = matches

    def fetch_finished(self):
        return self._matches


class FakeSampler:
    def __init__(self):
        self.strength = {"A1": 3.0, "A2": 2.0, "A3": 1.0, "A4": 0.5,
                         "B1": 3.0, "B2": 2.0, "B3": 1.0, "B4": 0.5}

    def lambdas(self, a, b, neutral):
        return self.strength[a], self.strength[b]

    def sample_goals(self, lam_a, lam_b, n, rng):
        return np.clip(rng.poisson(lam_a, n), 0, 10), np.clip(rng.poisson(lam_b, n), 0, 10)

    def scorelines(self, a, b, neutral, n, rng):
        la, lb = self.lambdas(a, b, neutral)
        return self.sample_goals(la, lb, n, rng)


class FakePredictor:
    def predict(self, team_a, team_b, neutral=False):
        return {"team_a_win": 60.0, "draw": 25.0, "team_b_win": 15.0,
                "expected_goals_a": 1.5, "expected_goals_b": 0.8, "most_likely_score": "1-0"}


def _struct():
    return TournamentConfig(
        name="Mini", neutral_default=True, points={"win": 3, "draw": 1, "loss": 0},
        groups={"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]},
        group_schedule="round_robin", per_group_advance=2, best_thirds=0,
        tiebreakers=["points", "goal_difference", "goals_for", "head_to_head", "fair_play", "drawing_of_lots"],
        thirds_ranking=["points", "goal_difference", "goals_for", "drawing_of_lots"],
        rounds=["SF", "F"],
        bracket_r32=[["winner_A", "runner_B"], ["winner_B", "runner_A"]],
        thirds_assignment="ranked_order",
    )


def test_cycle_returns_aggregate_scoreboard_meta(tmp_path):
    out = tmp_path / "r.yaml"
    runner = TournamentRunner(_struct(), {}, {"GROUP_STAGE": "group"}, out,
                              FakeSampler(), FakePredictor(), n=30, seed=1)
    provider = FakeProvider([ProviderMatch("1", "A1", "A2", 2, 0, "GROUP_STAGE", "GROUP_A", "FINISHED")])
    result = runner.cycle(provider)
    assert result["played"] == 1
    assert "teams" in result["aggregate"]
    assert result["scoreboard"]["n"] == 1
    assert result["meta"]["n_tournaments"] == 30


def test_watch_stops_on_keyboard_interrupt():
    class BoomRunner:
        def __init__(self):
            self.calls = 0

        def cycle(self, provider):
            self.calls += 1
            raise KeyboardInterrupt

    runner = BoomRunner()
    emitted = []
    watch(runner, provider=None, interval_minutes=0, emit=emitted.append)
    assert runner.calls == 1 and emitted == []   # raised before emit, loop exits cleanly
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.live.runner'`

- [ ] **Step 3: Write implementation**

`footy/live/runner.py`:
```python
import time

from footy.tournament.results import load_results
from footy.tournament.simulator import simulate_tournaments
from footy.tournament.aggregate import aggregate
from footy.live.ingest import ingest
from footy.live.scoreboard import scoreboard


class TournamentRunner:
    """Provider-agnostic orchestrator: ingest -> simulate -> scoreboard."""

    def __init__(self, structure, name_map, stage_map, results_path, sampler, predictor, n, seed):
        self.structure = structure
        self.name_map = name_map
        self.stage_map = stage_map
        self.results_path = results_path
        self.sampler = sampler
        self.predictor = predictor
        self.n = n
        self.seed = seed

    def cycle(self, provider) -> dict:
        n_written = ingest(provider, self.structure, self.name_map, self.stage_map, self.results_path)
        results = load_results(self.results_path, self.structure.groups)
        sims = simulate_tournaments(self.structure, results, self.sampler, self.n, self.seed)
        agg = aggregate(self.structure, sims)
        played_dicts = [
            {"team_a": pm.team_a, "team_b": pm.team_b,
             "goals_a": pm.goals_a, "goals_b": pm.goals_b}
            for pm in results.played
        ]
        board = scoreboard(self.predictor, played_dicts)
        return {"played": n_written, "aggregate": agg, "scoreboard": board,
                "meta": {"n_tournaments": self.n, "seed": self.seed,
                         "results_path": str(self.results_path)}}


def watch(runner, provider, interval_minutes, emit):
    """Run cycle -> emit -> sleep, forever, until KeyboardInterrupt."""
    interval_seconds = interval_minutes * 60
    try:
        while True:
            emit(runner.cycle(provider))
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/live/runner.py tests/test_runner.py
git commit -m "feat: provider-agnostic TournamentRunner + watch loop

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: CLI `update-and-simulate` + configs + full suite

**Files:**
- Modify: `footy/cli.py`
- Create: `configs/live.yaml`, `configs/name_map.yaml`
- Test: `tests/test_cli_update.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli_update.py`:
```python
import json
from pathlib import Path

from footy import cli
from footy.config import load_config

ROOT = Path(__file__).resolve().parent.parent


class FakeRunner:
    def cycle(self, provider):
        return {"played": 3, "aggregate": {"teams": {"Brazil": {"champion": 0.2}}},
                "scoreboard": {"n": 3, "accuracy": 0.66, "log_loss": 0.9, "brier": 0.5,
                               "goal_mae": 1.1, "matches": []},
                "meta": {"n_tournaments": 1000, "seed": 42, "results_path": "x"}}


def test_live_config_loads_and_stage_map_is_dict():
    cfg = load_config("live")
    assert isinstance(cfg["stage_map"], dict)
    assert cfg["competition_code"] == "WC"
    assert cfg["request_timeout"] > 0


def test_run_update_json_output(capsys):
    code = cli.run_update(["--json"], runner=FakeRunner(), provider=object())
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["played"] == 3
    assert payload["scoreboard"]["accuracy"] == 0.66


def test_run_update_summary_output(capsys):
    code = cli.run_update([], runner=FakeRunner(), provider=object())
    out = capsys.readouterr().out
    assert code == 0
    assert "Brazil" in out and "accuracy" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_update.py -v`
Expected: FAIL (`run_update` not defined, or `configs/live.yaml` missing)

- [ ] **Step 3: Create configs + implementation**

`configs/live.yaml`:
```yaml
base_url: "https://api.football-data.org/v4"
competition_code: "WC"
request_timeout: 10
watch_minutes: 15
stage_map:
  GROUP_STAGE: group
  LAST_32: R32
  LAST_16: R16
  QUARTER_FINALS: QF
  SEMI_FINALS: SF
  FINAL: F
```

`configs/name_map.yaml`:
```yaml
teams:
  "USA": "United States"
  "Korea Republic": "South Korea"
  "IR Iran": "Iran"
  "Côte d'Ivoire": "Ivory Coast"
```

Append to `footy/cli.py` (keep all existing SP1/SP2 code; add these imports near the top and
the functions at the end):
```python
import pandas as pd  # already imported above; do not duplicate

from footy.tournament.structure import load_structure
from footy.tournament.sampler import MatchSampler
from footy.live.provider import FootballDataProvider
from footy.live.name_map import load_name_map
from footy.live.runner import TournamentRunner, watch


def _build_runner() -> TournamentRunner:
    from footy.config import load_config, config_fingerprint
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    mc_cfg = load_config("montecarlo")
    live_cfg = load_config("live")
    sim_cfg = load_config("tournament_sim")
    if not isinstance(live_cfg.get("stage_map"), dict):
        raise ValueError("configs/live.yaml: 'stage_map' must be a mapping")

    base = "configs/tournaments"
    structure = load_structure(f"{base}/wc2026.yaml")
    predictor = _build_default_predictor()
    canon = predictor.canonical
    for g, teams in structure.groups.items():
        structure.groups[g] = [canon(t) for t in teams]

    sampler = MatchSampler(predictor.model, mc_cfg,
                           model_version=model_cfg["model_version"],
                           config_hash=config_fingerprint("montecarlo"))
    name_map = load_name_map("configs/name_map.yaml")
    return TournamentRunner(
        structure, name_map, live_cfg["stage_map"], f"{base}/wc2026_results.yaml",
        sampler, predictor, n=int(sim_cfg["n_tournaments"]), seed=int(sim_cfg["seed"]))


def _format_summary(result: dict) -> str:
    agg = result["aggregate"]; board = result["scoreboard"]
    champs = sorted(agg["teams"].items(), key=lambda kv: -kv[1]["champion"])[:8]
    lines = [f"Played matches: {result['played']}",
             f"Scoreboard: n={board['n']} accuracy={board['accuracy']} "
             f"log_loss={board['log_loss']} brier={board['brier']} goal_mae={board['goal_mae']}",
             "Top champion probabilities:"]
    for team, d in champs:
        lines.append(f"  {team:<18} {d['champion'] * 100:5.1f}%")
    return "\n".join(lines)


def run_update(argv, runner=None, provider=None, emit=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="update-and-simulate",
                                     description="Fetch official results and re-simulate the tournament.")
    parser.add_argument("--watch", type=int, default=None, metavar="MINUTES",
                        help="Re-run every MINUTES minutes")
    parser.add_argument("--json", action="store_true", help="Emit full JSON")
    args = parser.parse_args(argv)

    if runner is None:
        runner = _build_runner()
    if provider is None:
        provider = FootballDataProvider.from_config(load_config("live"))

    if emit is None:
        if args.json:
            emit = lambda result: print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            emit = lambda result: print(_format_summary(result))

    if args.watch is not None:
        watch(runner, provider, args.watch, emit)
    else:
        emit(runner.cycle(provider))
    return 0


def main_update() -> None:
    sys.exit(run_update(sys.argv[1:]))
```

Note: `footy/cli.py` already imports `json`, `sys`, `pandas as pd`, `load_config`, and
`_build_default_predictor` from SP1/SP2 — reuse them; do not redefine. Only add what is missing.

- [ ] **Step 4: Run the targeted test, then the full suite**

Run: `python -m pytest tests/test_cli_update.py -v`
Expected: PASS (3 passed)

Run: `python -m pytest -q`
Expected: all tests PASS (SP1+SP2+SP3+SP4). Smoke test may take ~2 min.

- [ ] **Step 5: Commit**

```bash
git add footy/cli.py configs/live.yaml configs/name_map.yaml tests/test_cli_update.py
git commit -m "feat: update-and-simulate CLI (--watch, --json) + live configs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §4 provider (ProviderMatch, ResultsProvider, FootballDataProvider, env key, timeout>0, rstrip, bad-JSON) → Task 1. ✓
- §5 name_map/stage_map hard errors + dict validation → Task 2 (+ live stage_map dict check in Task 6 `_build_runner`). ✓
- §6 ingest idempotent + header + score/group guards + SP3-compatible → Task 3. ✓
- §7 scoreboard out-of-sample, predicted_prob/actual_prob, goal_mae per team-match, empty→None → Task 4. ✓
- §8 runner provider-agnostic, cycle returns played/aggregate/scoreboard/meta, watch minutes + KeyboardInterrupt, fit-once (CLI builds predictor once) → Tasks 5, 6. ✓
- §9 error handling → guards across Tasks 1-3, 6. ✓
- §10 testing without real network (HTTP mock T1, FakeProvider T3/T5) → all test tasks. ✓
- requests dep + scripts entry → Task 1, Task 6. ✓

**Placeholder scan:** every code step is complete; the `pyproject.toml` and `cli.py` edits are described as additive with explicit "keep existing" notes (no ellipsis in code). No TODO/TBD.

**Type consistency:** `ProviderMatch` fields (T1) consumed by `ingest`/`build_played_matches` (T3) and `FakeProvider`/`TournamentRunner` (T5); `ingest(provider, structure, name_map, stage_map, out_path)` signature identical in T3 and `runner.cycle` (T5); `scoreboard(predictor, played_matches)` (T4) called in `runner.cycle` (T5) with the dict shape it expects; `TournamentRunner(structure, name_map, stage_map, results_path, sampler, predictor, n, seed)` constructor identical in T5 test and T6 `_build_runner`; `run_update(argv, runner, provider, emit)` (T6) matches its tests. `load_config("live")`/`config_fingerprint` reused from SP1/SP2. ✓

**Note:** `_build_runner` (Task 6) is exercised only via manual run (it fits the real model ~2 min); unit tests inject a `FakeRunner`/`FakeProvider`, so the suite stays fast and network-free. The existing `scripts/run_wc2026.py` demo predates this CLI; it can be deleted or kept — out of scope here.
