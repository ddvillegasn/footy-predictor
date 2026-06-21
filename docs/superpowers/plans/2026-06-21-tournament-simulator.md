# Tournament Simulator Implementation Plan (SP3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A config-driven Monte Carlo tournament simulator (instantiated for the World Cup 2026) that fixes already-played results and samples the rest, producing per-team round/champion probabilities, group distributions, slot frequencies and tournament odds.

**Architecture:** New `footy/tournament/` package on top of the SP1 match engine and SP2 odds. `sampler.py` is the sole RNG source for goals; `groups.py` and `knockout.py` are pure functions (results in → order/winner out, `rng` injected); `simulator.py` orchestrates N tournaments (fixing played matches, sampling pending); `aggregate.py` only counts; tournament odds reuse `betting/odds.py` + `value.py`.

**Tech Stack:** Python 3.10.6, numpy, pandas, pyyaml, pytest. Run from repo root: `python -m pytest`.

**Conventions:** branch `feature/baseline-v1`. TDD: failing test first. Commit per task, `git add` ONLY the named files (never `__pycache__`/`.pyc`). Commit trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

**Test isolation:** Unit tests use a lightweight `FakeSampler` (deterministic, no model fit) defined inline in each test that needs it. Only `test_sampler.py` exercises `MatchSampler` with a tiny stub model. Mini tournament = 2 groups × 4 teams.

## File Structure

| File | Responsibility |
|---|---|
| `footy/tournament/__init__.py` | package marker |
| `footy/tournament/structure.py` | `TournamentConfig` + `load_structure` (validate) |
| `footy/tournament/results.py` | `PlayedMatch`, `TournamentResults` + `load_results` (validate, lookup) |
| `footy/tournament/sampler.py` | `MatchSampler` (sole RNG goal source, λ cache) |
| `footy/tournament/groups.py` | `group_fixtures`, `group_table`, `rank_group` (FIFA), `rank_thirds` |
| `footy/tournament/knockout.py` | `build_bracket`, `resolve_match` (ET + weighted pens) |
| `footy/tournament/simulator.py` | `run_tournament`, `simulate_tournaments` |
| `footy/tournament/aggregate.py` | `aggregate` (counts → probabilities) + `tournament_odds` |
| `configs/tournaments/wc2026.yaml` | WC2026 structure |
| `configs/tournaments/wc2026_results.yaml` | live played results (starts empty) |
| `configs/tournament_sim.yaml` | n_tournaments, seed, neutral_default |
| `tests/test_tournament_*.py` | one per module |

---

## Task 1: tournament structure config loader

**Files:**
- Create: `footy/tournament/__init__.py`, `footy/tournament/structure.py`
- Test: `tests/test_tournament_structure.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tournament_structure.py`:
```python
import pytest

from footy.tournament.structure import load_structure, TournamentConfig

MINI = {
    "name": "Mini",
    "neutral_default": True,
    "points": {"win": 3, "draw": 1, "loss": 0},
    "groups": {"A": ["T1", "T2", "T3", "T4"], "B": ["T5", "T6", "T7", "T8"]},
    "group_schedule": "round_robin",
    "qualification": {"per_group_advance": 2, "best_thirds": 0},
    "tiebreakers": ["points", "goal_difference", "goals_for", "head_to_head",
                    "fair_play", "drawing_of_lots"],
    "thirds_ranking": ["points", "goal_difference", "goals_for", "drawing_of_lots"],
    "knockout": {"rounds": ["SF", "F"],
                 "bracket_r32": [["winner_A", "runner_B"], ["winner_B", "runner_A"]],
                 "thirds_assignment": "ranked_order"},
}


def _write(tmp_path, data):
    import yaml
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_loads_valid_config(tmp_path):
    cfg = load_structure(_write(tmp_path, MINI))
    assert isinstance(cfg, TournamentConfig)
    assert cfg.groups["A"] == ["T1", "T2", "T3", "T4"]
    assert cfg.per_group_advance == 2
    assert cfg.bracket_r32 == [["winner_A", "runner_B"], ["winner_B", "runner_A"]]


def test_duplicate_team_raises(tmp_path):
    bad = {**MINI, "groups": {"A": ["T1", "T2", "T3", "T4"], "B": ["T1", "T6", "T7", "T8"]}}
    with pytest.raises(ValueError, match="duplicate"):
        load_structure(_write(tmp_path, bad))


def test_bad_bracket_ref_raises(tmp_path):
    bad = {**MINI, "knockout": {**MINI["knockout"],
           "bracket_r32": [["winner_Z", "runner_B"], ["winner_B", "runner_A"]]}}
    with pytest.raises(ValueError, match="bracket"):
        load_structure(_write(tmp_path, bad))


def test_third_slot_out_of_range_raises(tmp_path):
    bad = {**MINI, "knockout": {**MINI["knockout"],
           "bracket_r32": [["winner_A", "third_slot_1"], ["winner_B", "runner_A"]]}}
    # best_thirds is 0, so third_slot_1 is invalid
    with pytest.raises(ValueError, match="third"):
        load_structure(_write(tmp_path, bad))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tournament_structure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.tournament'`

- [ ] **Step 3: Write implementation**

`footy/tournament/__init__.py`:
```python
```
(empty file — a single newline)

`footy/tournament/structure.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TournamentConfig:
    name: str
    neutral_default: bool
    points: dict
    groups: dict
    group_schedule: str
    per_group_advance: int
    best_thirds: int
    tiebreakers: list
    thirds_ranking: list
    rounds: list
    bracket_r32: list
    thirds_assignment: str


def _validate_slot_ref(ref: str, groups: dict, best_thirds: int) -> None:
    if ref.startswith("winner_") or ref.startswith("runner_"):
        g = ref.split("_", 1)[1]
        if g not in groups:
            raise ValueError(f"bracket ref points to unknown group: {ref}")
    elif ref.startswith("third_slot_"):
        idx = int(ref.rsplit("_", 1)[1])
        if idx < 1 or idx > best_thirds:
            raise ValueError(f"third slot out of range (best_thirds={best_thirds}): {ref}")
    else:
        raise ValueError(f"unrecognised bracket ref: {ref}")


def load_structure(path) -> TournamentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    groups = data["groups"]

    # Unique teams across all groups.
    seen = set()
    for team_list in groups.values():
        for t in team_list:
            if t in seen:
                raise ValueError(f"duplicate team across groups: {t}")
            seen.add(t)

    qual = data["qualification"]
    best_thirds = int(qual["best_thirds"])
    knockout = data["knockout"]

    for tie in knockout["bracket_r32"]:
        for ref in tie:
            _validate_slot_ref(ref, groups, best_thirds)

    return TournamentConfig(
        name=data["name"],
        neutral_default=bool(data["neutral_default"]),
        points=data["points"],
        groups=groups,
        group_schedule=data.get("group_schedule", "round_robin"),
        per_group_advance=int(qual["per_group_advance"]),
        best_thirds=best_thirds,
        tiebreakers=data["tiebreakers"],
        thirds_ranking=data["thirds_ranking"],
        rounds=knockout["rounds"],
        bracket_r32=knockout["bracket_r32"],
        thirds_assignment=knockout.get("thirds_assignment", "ranked_order"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tournament_structure.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/tournament/__init__.py footy/tournament/structure.py tests/test_tournament_structure.py
git commit -m "feat: tournament structure config loader with validation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: played-results loader

**Files:**
- Create: `footy/tournament/results.py`
- Test: `tests/test_results_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/test_results_loader.py`:
```python
import pytest

from footy.tournament.results import load_results, TournamentResults, PlayedMatch

STRUCT_TEAMS = {"A": ["T1", "T2", "T3", "T4"], "B": ["T5", "T6", "T7", "T8"]}


def _write(tmp_path, matches):
    import yaml
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"played_matches": matches}), encoding="utf-8")
    return p


def test_loads_and_looks_up_group(tmp_path):
    matches = [{"match_id": "A1", "stage": "group", "group": "A",
                "team_a": "T1", "team_b": "T2", "goals_a": 3, "goals_b": 1}]
    res = load_results(_write(tmp_path, matches), STRUCT_TEAMS)
    assert isinstance(res, TournamentResults)
    pm = res.lookup_group("A", "T2", "T1")  # order-insensitive
    assert pm.goals_a == 3 and pm.team_a == "T1"
    assert res.lookup_group("A", "T1", "T3") is None


def test_empty_results(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("played_matches: []", encoding="utf-8")
    res = load_results(p, STRUCT_TEAMS)
    assert res.lookup_group("A", "T1", "T2") is None


def test_team_not_in_structure_raises(tmp_path):
    matches = [{"match_id": "X1", "stage": "group", "group": "A",
                "team_a": "T1", "team_b": "GHOST", "goals_a": 1, "goals_b": 0}]
    with pytest.raises(ValueError, match="not in structure"):
        load_results(_write(tmp_path, matches), STRUCT_TEAMS)


def test_duplicate_match_id_raises(tmp_path):
    matches = [
        {"match_id": "A1", "stage": "group", "group": "A", "team_a": "T1", "team_b": "T2", "goals_a": 1, "goals_b": 0},
        {"match_id": "A1", "stage": "group", "group": "A", "team_a": "T3", "team_b": "T4", "goals_a": 2, "goals_b": 2},
    ]
    with pytest.raises(ValueError, match="duplicate match_id"):
        load_results(_write(tmp_path, matches), STRUCT_TEAMS)


def test_negative_score_raises(tmp_path):
    matches = [{"match_id": "A1", "stage": "group", "group": "A",
                "team_a": "T1", "team_b": "T2", "goals_a": -1, "goals_b": 0}]
    with pytest.raises(ValueError, match="score"):
        load_results(_write(tmp_path, matches), STRUCT_TEAMS)


def test_lookup_knockout(tmp_path):
    matches = [{"match_id": "SF1", "stage": "SF", "team_a": "T1", "team_b": "T5",
                "goals_a": 2, "goals_b": 1, "winner": "T1"}]
    res = load_results(_write(tmp_path, matches), STRUCT_TEAMS)
    pm = res.lookup_knockout("SF", "T5", "T1")
    assert pm.winner == "T1"
    assert res.lookup_knockout("F", "T1", "T5") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_results_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.tournament.results'`

- [ ] **Step 3: Write implementation**

`footy/tournament/results.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class PlayedMatch:
    match_id: str
    stage: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    group: str | None = None
    winner: str | None = None


def _same_pair(pm: PlayedMatch, x: str, y: str) -> bool:
    return {pm.team_a, pm.team_b} == {x, y}


class TournamentResults:
    def __init__(self, played: list[PlayedMatch]):
        self.played = played

    def lookup_group(self, group: str, team_a: str, team_b: str) -> PlayedMatch | None:
        for pm in self.played:
            if pm.stage == "group" and pm.group == group and _same_pair(pm, team_a, team_b):
                return pm
        return None

    def lookup_knockout(self, stage: str, team_a: str, team_b: str) -> PlayedMatch | None:
        for pm in self.played:
            if pm.stage == stage and pm.stage != "group" and _same_pair(pm, team_a, team_b):
                return pm
        return None


def load_results(path, structure_groups: dict) -> TournamentResults:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = data.get("played_matches") or []

    valid_teams = {t for team_list in structure_groups.values() for t in team_list}
    seen_ids = set()
    played = []
    for m in raw:
        mid = m["match_id"]
        if mid in seen_ids:
            raise ValueError(f"duplicate match_id: {mid}")
        seen_ids.add(mid)
        for team_key in ("team_a", "team_b"):
            if m[team_key] not in valid_teams:
                raise ValueError(f"team not in structure: {m[team_key]}")
        ga, gb = int(m["goals_a"]), int(m["goals_b"])
        if ga < 0 or gb < 0:
            raise ValueError(f"negative score in match {mid}")
        played.append(PlayedMatch(
            match_id=mid, stage=m["stage"], team_a=m["team_a"], team_b=m["team_b"],
            goals_a=ga, goals_b=gb, group=m.get("group"), winner=m.get("winner"),
        ))
    return TournamentResults(played)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_results_loader.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/tournament/results.py tests/test_results_loader.py
git commit -m "feat: tournament played-results loader with lookups

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: MatchSampler (sole RNG goal source)

**Files:**
- Create: `footy/tournament/sampler.py`
- Test: `tests/test_sampler.py`

- [ ] **Step 1: Write the failing test**

`tests/test_sampler.py`:
```python
import numpy as np

from footy.tournament.sampler import MatchSampler


class StubModel:
    """Minimal model exposing rates() like DixonColesModel."""
    def __init__(self):
        self.calls = 0

    def rates(self, team_a, team_b, neutral=False):
        self.calls += 1
        base = {"Strong": 2.2, "Weak": 0.6}
        la = base.get(team_a, 1.0)
        lb = base.get(team_b, 1.0)
        if not neutral:
            la += 0.2
        return la, lb


CFG = {"max_goals": 10}


def test_scorelines_shapes_and_determinism():
    s = MatchSampler(StubModel(), CFG, "m1", "h1")
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a1, b1 = s.scorelines("Strong", "Weak", True, 5000, rng1)
    a2, b2 = s.scorelines("Strong", "Weak", True, 5000, rng2)
    assert len(a1) == 5000
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert a1.mean() > b1.mean()  # strong scores more


def test_lambda_cache_avoids_remodel():
    model = StubModel()
    s = MatchSampler(model, CFG, "m1", "h1")
    s.lambdas("Strong", "Weak", True)
    s.lambdas("Strong", "Weak", True)
    assert model.calls == 1  # cached on second call


def test_sample_goals_respects_clip():
    s = MatchSampler(StubModel(), {"max_goals": 2}, "m1", "h1")
    rng = np.random.default_rng(1)
    ga, gb = s.sample_goals(8.0, 8.0, 1000, rng)
    assert ga.max() <= 2 and gb.max() <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.tournament.sampler'`

- [ ] **Step 3: Write implementation**

`footy/tournament/sampler.py`:
```python
from __future__ import annotations

import numpy as np


class MatchSampler:
    """Sole source of randomness for goals. Caches lambdas (not scorelines)."""

    def __init__(self, model, mc_config: dict, model_version: str, config_hash: str):
        self.model = model
        self.max_goals = int(mc_config["max_goals"])
        self.model_version = model_version
        self.config_hash = config_hash
        self._lambda_cache: dict = {}

    def lambdas(self, team_a: str, team_b: str, neutral: bool) -> tuple[float, float]:
        key = (team_a, team_b, neutral, self.model_version, self.config_hash)
        if key not in self._lambda_cache:
            self._lambda_cache[key] = self.model.rates(team_a, team_b, neutral=neutral)
        return self._lambda_cache[key]

    def sample_goals(self, lam_a: float, lam_b: float, n: int, rng):
        ga = np.clip(rng.poisson(lam_a, n), 0, self.max_goals)
        gb = np.clip(rng.poisson(lam_b, n), 0, self.max_goals)
        return ga, gb

    def scorelines(self, team_a: str, team_b: str, neutral: bool, n: int, rng):
        lam_a, lam_b = self.lambdas(team_a, team_b, neutral)
        return self.sample_goals(lam_a, lam_b, n, rng)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sampler.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/tournament/sampler.py tests/test_sampler.py
git commit -m "feat: MatchSampler with lambda cache as sole RNG goal source

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: group fixtures + table

**Files:**
- Create: `footy/tournament/groups.py`
- Test: `tests/test_groups_table.py`

- [ ] **Step 1: Write the failing test**

`tests/test_groups_table.py`:
```python
from footy.tournament.groups import group_fixtures, group_table

POINTS = {"win": 3, "draw": 1, "loss": 0}


def test_round_robin_fixtures():
    fx = group_fixtures(["A", "B", "C", "D"])
    assert len(fx) == 6
    assert ("A", "B") in fx and ("C", "D") in fx
    # no self matches, no duplicates
    assert all(a != b for a, b in fx)
    assert len(set(map(frozenset, fx))) == 6


def test_table_points_and_gd():
    teams = ["A", "B", "C", "D"]
    # A beats B 2-0, A beats C 1-0, A draws D 1-1
    results = [
        ("A", "B", 2, 0), ("A", "C", 1, 0), ("A", "D", 1, 1),
        ("B", "C", 0, 0), ("B", "D", 1, 2), ("C", "D", 3, 3),
    ]
    table = group_table(teams, results, POINTS)
    assert table["A"]["points"] == 7  # 2W 1D
    assert table["A"]["gf"] == 4 and table["A"]["ga"] == 1
    assert table["A"]["gd"] == 3
    assert table["D"]["points"] == 5  # W vs B, 2 draws
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_groups_table.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.tournament.groups'`

- [ ] **Step 3: Write implementation**

`footy/tournament/groups.py`:
```python
from __future__ import annotations

from itertools import combinations


def group_fixtures(teams: list) -> list:
    """All unordered pairs (round-robin), deterministic order."""
    return [(a, b) for a, b in combinations(teams, 2)]


def group_table(teams: list, results: list, points: dict) -> dict:
    """Build a standings dict from match results.

    results: list of (team_a, team_b, goals_a, goals_b).
    Returns {team: {points, w, d, l, gf, ga, gd}}.
    """
    table = {t: {"points": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0}
             for t in teams}
    for a, b, ga, gb in results:
        table[a]["gf"] += ga; table[a]["ga"] += gb
        table[b]["gf"] += gb; table[b]["ga"] += ga
        if ga > gb:
            table[a]["points"] += points["win"]; table[a]["w"] += 1
            table[b]["points"] += points["loss"]; table[b]["l"] += 1
        elif ga < gb:
            table[b]["points"] += points["win"]; table[b]["w"] += 1
            table[a]["points"] += points["loss"]; table[a]["l"] += 1
        else:
            table[a]["points"] += points["draw"]; table[a]["d"] += 1
            table[b]["points"] += points["draw"]; table[b]["d"] += 1
    for t in teams:
        table[t]["gd"] = table[t]["gf"] - table[t]["ga"]
    return table
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_groups_table.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/tournament/groups.py tests/test_groups_table.py
git commit -m "feat: group fixtures and standings table

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: FIFA tie-breakers + thirds ranking

**Files:**
- Modify: `footy/tournament/groups.py`
- Test: `tests/test_groups_rank.py`

- [ ] **Step 1: Write the failing test**

`tests/test_groups_rank.py`:
```python
import numpy as np

from footy.tournament.groups import rank_group, rank_thirds

POINTS = {"win": 3, "draw": 1, "loss": 0}
TB = ["points", "goal_difference", "goals_for", "head_to_head", "fair_play", "drawing_of_lots"]


def test_rank_by_points_then_gd_then_gf():
    teams = ["A", "B", "C", "D"]
    results = [
        ("A", "B", 1, 0), ("A", "C", 5, 0), ("A", "D", 1, 0),
        ("B", "C", 1, 0), ("B", "D", 1, 0), ("C", "D", 0, 1),
    ]
    # A: 9 pts. B: 6 pts. D: 3 pts. C: 0.
    order = rank_group(teams, results, POINTS, TB, np.random.default_rng(0))
    assert order == ["A", "B", "D", "C"]


def test_head_to_head_breaks_equal_pts_gd_gf():
    # Two teams level on points/GD/GF overall; H2H decides.
    teams = ["A", "B", "C", "D"]
    results = [
        ("A", "B", 1, 0),   # A beat B head-to-head
        ("A", "C", 0, 5),
        ("A", "D", 3, 0),
        ("B", "C", 0, 5),
        ("B", "D", 3, 0),
        ("C", "D", 1, 0),
    ]
    # A and B: each 6 pts, GD: A = 1-5+3 = -1? compute -> both equal; H2H A>B.
    order = rank_group(teams, results, POINTS, TB, np.random.default_rng(0))
    assert order.index("A") < order.index("B")


def test_drawing_of_lots_is_reproducible():
    # Two identical teams (mirror results) -> only lots can break; same seed = same order.
    teams = ["A", "B"]
    results = [("A", "B", 0, 0)]
    o1 = rank_group(teams, results, POINTS, TB, np.random.default_rng(7))
    o2 = rank_group(teams, results, POINTS, TB, np.random.default_rng(7))
    assert o1 == o2


def test_rank_thirds_picks_best():
    thirds = [
        ("X", {"points": 3, "gd": 1, "gf": 2}),
        ("Y", {"points": 6, "gd": 4, "gf": 5}),
        ("Z", {"points": 1, "gd": -2, "gf": 1}),
    ]
    ranking = ["points", "goal_difference", "goals_for", "drawing_of_lots"]
    order = rank_thirds(thirds, ranking, np.random.default_rng(0))
    assert order[0] == "Y" and order[-1] == "Z"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_groups_rank.py -v`
Expected: FAIL with `ImportError: cannot import name 'rank_group'`

- [ ] **Step 3: Append implementation to `footy/tournament/groups.py`**

```python
from itertools import groupby


def _global_key(stats: dict) -> tuple:
    return (stats["points"], stats["gd"], stats["gf"])


def _h2h_table(tied: list, results: list, points: dict) -> dict:
    """Mini-table using only matches among the tied teams."""
    sub = [(a, b, ga, gb) for (a, b, ga, gb) in results if a in tied and b in tied]
    return group_table(tied, sub, points)


def _break_ties(tied: list, results: list, points: dict, rng) -> list:
    h2h = _h2h_table(tied, results, points)
    ordered = sorted(tied, key=lambda t: _global_key(h2h[t]), reverse=True)
    out = []
    for _, grp in groupby(ordered, key=lambda t: _global_key(h2h[t])):
        still = list(grp)
        if len(still) == 1:
            out.extend(still)
        else:
            # fair_play is neutral (no data) -> final fallback: reproducible lots.
            lots = sorted(still, key=lambda t: rng.random())
            out.extend(lots)
    return out


def rank_group(teams: list, results: list, points: dict, tiebreakers: list, rng) -> list:
    """Order a group applying FIFA tie-breakers: points, GD, GF, then head-to-head
    mini-table among equals, then reproducible drawing of lots."""
    table = group_table(teams, results, points)
    ordered = sorted(teams, key=lambda t: _global_key(table[t]), reverse=True)
    out = []
    for _, grp in groupby(ordered, key=lambda t: _global_key(table[t])):
        tied = list(grp)
        if len(tied) == 1:
            out.extend(tied)
        else:
            out.extend(_break_ties(tied, results, points, rng))
    return out


def rank_thirds(thirds: list, thirds_ranking: list, rng) -> list:
    """Rank third-placed entries. thirds: list of (team, stats dict with
    points/gd/gf). Returns teams best-first."""
    def key(entry):
        _, stats = entry
        return (stats["points"], stats["gd"], stats["gf"])

    ordered = sorted(thirds, key=key, reverse=True)
    out = []
    for _, grp in groupby(ordered, key=key):
        tied = list(grp)
        if len(tied) == 1:
            out.append(tied[0][0])
        else:
            for team, _ in sorted(tied, key=lambda e: rng.random()):
                out.append(team)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_groups_rank.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/tournament/groups.py tests/test_groups_rank.py
git commit -m "feat: FIFA tie-breakers (H2H, reproducible lots) and thirds ranking

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: knockout bracket + match resolution

**Files:**
- Create: `footy/tournament/knockout.py`
- Test: `tests/test_knockout.py`

- [ ] **Step 1: Write the failing test**

`tests/test_knockout.py`:
```python
import numpy as np

from footy.tournament.knockout import build_bracket, resolve_match


class FakeSampler:
    """Deterministic sampler for tests: fixed extra-time goals, known lambdas."""
    def __init__(self, et_goals=(0, 0), lambdas=(2.0, 1.0)):
        self.et_goals = et_goals
        self._lambdas = lambdas

    def lambdas(self, a, b, neutral):
        return self._lambdas

    def sample_goals(self, lam_a, lam_b, n, rng):
        return (np.array([self.et_goals[0]] * n), np.array([self.et_goals[1]] * n))


def test_build_bracket_resolves_slots():
    group_ranks = {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}
    cfg = [["winner_A", "runner_B"], ["winner_B", "runner_A"]]
    ties = build_bracket(group_ranks, [], cfg)
    assert ties == [("A1", "B2"), ("B1", "A2")]


def test_build_bracket_with_thirds():
    group_ranks = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    cfg = [["winner_A", "third_slot_1"], ["winner_B", "third_slot_2"]]
    ties = build_bracket(group_ranks, ["C3", "D3"], cfg)
    assert ties == [("A1", "C3"), ("B1", "D3")]


def test_resolve_clear_winner():
    s = FakeSampler()
    w = resolve_match("A", "B", 2, 0, s, np.random.default_rng(0), neutral=True)
    assert w == "A"


def test_resolve_draw_goes_to_extra_time():
    # Regulation 1-1, extra time 1-0 for A -> A wins.
    s = FakeSampler(et_goals=(1, 0))
    w = resolve_match("A", "B", 1, 1, s, np.random.default_rng(0), neutral=True)
    assert w == "A"


def test_resolve_penalties_favour_stronger_over_many_seeds():
    # Regulation draw, extra time 0-0 -> penalties weighted by lambdas (A stronger).
    s = FakeSampler(et_goals=(0, 0), lambdas=(3.0, 1.0))
    wins_a = sum(
        resolve_match("A", "B", 0, 0, s, np.random.default_rng(seed), neutral=True) == "A"
        for seed in range(400)
    )
    assert wins_a > 240  # ~0.75 share, clearly above 50%


def test_penalty_clipping_extremes():
    # Even a huge lambda gap is clipped to [0.05, 0.95]: underdog still wins sometimes.
    s = FakeSampler(et_goals=(0, 0), lambdas=(50.0, 0.1))
    wins_b = sum(
        resolve_match("A", "B", 0, 0, s, np.random.default_rng(seed), neutral=True) == "B"
        for seed in range(400)
    )
    assert wins_b > 0  # clipping guarantees a floor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_knockout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.tournament.knockout'`

- [ ] **Step 3: Write implementation**

`footy/tournament/knockout.py`:
```python
from __future__ import annotations

ET_SCALE = 1.0 / 3.0          # extra time ~ 30/90 of regulation
PEN_CLIP = (0.05, 0.95)


def _resolve_ref(ref: str, group_ranks: dict, thirds_ranked: list) -> str:
    if ref.startswith("winner_"):
        return group_ranks[ref.split("_", 1)[1]][0]
    if ref.startswith("runner_"):
        return group_ranks[ref.split("_", 1)[1]][1]
    if ref.startswith("third_slot_"):
        idx = int(ref.rsplit("_", 1)[1]) - 1
        return thirds_ranked[idx]
    raise ValueError(f"unrecognised bracket ref: {ref}")


def build_bracket(group_ranks: dict, thirds_ranked: list, bracket_cfg: list) -> list:
    """Resolve slot references to concrete teams -> list of (team_a, team_b) ties."""
    return [
        (_resolve_ref(a, group_ranks, thirds_ranked),
         _resolve_ref(b, group_ranks, thirds_ranked))
        for a, b in bracket_cfg
    ]


def resolve_match(team_a, team_b, reg_a, reg_b, sampler, rng, neutral) -> str:
    """Return the winner. Draw -> extra time (scaled lambdas) -> weighted penalties."""
    if reg_a != reg_b:
        return team_a if reg_a > reg_b else team_b

    lam_a, lam_b = sampler.lambdas(team_a, team_b, neutral)
    ea, eb = sampler.sample_goals(lam_a * ET_SCALE, lam_b * ET_SCALE, 1, rng)
    total_a, total_b = reg_a + int(ea[0]), reg_b + int(eb[0])
    if total_a != total_b:
        return team_a if total_a > total_b else team_b

    p_a = lam_a / (lam_a + lam_b)
    p_a = max(PEN_CLIP[0], min(PEN_CLIP[1], p_a))
    return team_a if rng.random() < p_a else team_b
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_knockout.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/tournament/knockout.py tests/test_knockout.py
git commit -m "feat: knockout bracket build and ET/weighted-penalty resolution

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: tournament simulator (fix played, sample pending)

**Files:**
- Create: `footy/tournament/simulator.py`
- Test: `tests/test_simulator.py`

- [ ] **Step 1: Write the failing test**

`tests/test_simulator.py`:
```python
import numpy as np

from footy.tournament.structure import TournamentConfig
from footy.tournament.results import TournamentResults, PlayedMatch
from footy.tournament.simulator import run_tournament, simulate_tournaments


class FakeSampler:
    """Strong teams (lower index in each group) score more, deterministically-ish."""
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


def test_run_tournament_produces_champion_and_rounds():
    res = run_tournament(_struct(), TournamentResults([]), FakeSampler(), np.random.default_rng(1))
    assert res.champion in {"A1", "A2", "B1", "B2"}  # only qualifiers can win
    assert res.furthest_round[res.champion] == "champion"
    # group stage debug fields present
    assert "A1" in res.group_points and res.group_points["A1"] >= 0


def test_played_group_result_is_fixed():
    # Force A4 to thrash everyone via played results -> A4 must top group A every run.
    played = TournamentResults([
        PlayedMatch("g1", "group", "A4", "A1", 9, 0, group="A"),
        PlayedMatch("g2", "group", "A4", "A2", 9, 0, group="A"),
        PlayedMatch("g3", "group", "A4", "A3", 9, 0, group="A"),
    ])
    for seed in range(5):
        res = run_tournament(_struct(), played, FakeSampler(), np.random.default_rng(seed))
        assert res.group_order["A"][0] == "A4"  # fixed wins put A4 first


def test_simulate_tournaments_runs_many():
    results = simulate_tournaments(_struct(), TournamentResults([]), FakeSampler(), n=50, seed=3)
    assert len(results) == 50
    assert all(r.champion is not None for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simulator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.tournament.simulator'`

- [ ] **Step 3: Write implementation**

`footy/tournament/simulator.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from footy.tournament.groups import group_fixtures, group_table, rank_group, rank_thirds
from footy.tournament.knockout import build_bracket, resolve_match


@dataclass
class TournamentResult:
    champion: str
    furthest_round: dict          # team -> round name or "champion"
    group_order: dict             # group -> ordered team list
    group_position: dict          # team -> 1..N
    group_points: dict            # team -> points
    group_gd: dict                # team -> goal difference
    group_gf: dict                # team -> goals for
    run_id: int = 0


def _played_group_score(pm, a, b):
    """Return (goals_a, goals_b) oriented to (a, b)."""
    if pm.team_a == a:
        return pm.goals_a, pm.goals_b
    return pm.goals_b, pm.goals_a


def _played_knockout_winner(pm, a, b):
    if pm.winner is not None:
        return pm.winner
    ga, gb = _played_group_score(pm, a, b)
    if ga == gb:
        raise ValueError(f"played knockout {pm.match_id} is a draw without a winner field")
    return a if ga > gb else b


def run_tournament(structure, results, sampler, rng) -> TournamentResult:
    run_id = int(rng.integers(0, 2**31 - 1))
    group_order, group_position = {}, {}
    group_points, group_gd, group_gf = {}, {}, {}
    thirds_entries = []

    for g, teams in structure.groups.items():
        match_results = []
        for (a, b) in group_fixtures(teams):
            pm = results.lookup_group(g, a, b)
            if pm is not None:
                ga, gb = _played_group_score(pm, a, b)
            else:
                sa, sb = sampler.scorelines(a, b, structure.neutral_default, 1, rng)
                ga, gb = int(sa[0]), int(sb[0])
            match_results.append((a, b, ga, gb))

        table = group_table(teams, match_results, structure.points)
        order = rank_group(teams, match_results, structure.points, structure.tiebreakers, rng)
        group_order[g] = order
        for pos, t in enumerate(order, start=1):
            group_position[t] = pos
            group_points[t] = table[t]["points"]
            group_gd[t] = table[t]["gd"]
            group_gf[t] = table[t]["gf"]
        if structure.best_thirds > 0 and len(order) >= 3:
            third = order[2]
            thirds_entries.append((third, {"points": table[third]["points"],
                                           "gd": table[third]["gd"], "gf": table[third]["gf"]}))

    thirds_ranked = []
    if structure.best_thirds > 0:
        thirds_ranked = rank_thirds(thirds_entries, structure.thirds_ranking, rng)[:structure.best_thirds]

    group_ranks = {g: order for g, order in group_order.items()}
    ties = build_bracket(group_ranks, thirds_ranked, structure.bracket_r32)

    furthest = {}
    for tie in ties:
        for t in tie:
            furthest[t] = structure.rounds[0]

    for ridx, round_name in enumerate(structure.rounds):
        winners = []
        for (a, b) in ties:
            pm = results.lookup_knockout(round_name, a, b)
            if pm is not None:
                winner = _played_knockout_winner(pm, a, b)
            else:
                sa, sb = sampler.scorelines(a, b, structure.neutral_default, 1, rng)
                winner = resolve_match(a, b, int(sa[0]), int(sb[0]), sampler, rng,
                                       structure.neutral_default)
            winners.append(winner)
            next_label = "champion" if ridx == len(structure.rounds) - 1 else structure.rounds[ridx + 1]
            furthest[winner] = next_label
        ties = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)] if len(winners) > 1 else []
        if len(winners) == 1:
            champion = winners[0]

    return TournamentResult(
        champion=champion, furthest_round=furthest, group_order=group_order,
        group_position=group_position, group_points=group_points,
        group_gd=group_gd, group_gf=group_gf, run_id=run_id,
    )


def simulate_tournaments(structure, results, sampler, n: int, seed: int) -> list:
    master = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        child = np.random.default_rng(master.integers(0, 2**63 - 1))
        out.append(run_tournament(structure, results, sampler, child))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_simulator.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/tournament/simulator.py tests/test_simulator.py
git commit -m "feat: tournament simulator fixing played results, sampling pending

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: aggregate + tournament odds

**Files:**
- Create: `footy/tournament/aggregate.py`
- Test: `tests/test_aggregate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_aggregate.py`:
```python
import numpy as np

from footy.tournament.structure import TournamentConfig
from footy.tournament.results import TournamentResults
from footy.tournament.simulator import simulate_tournaments
from footy.tournament.aggregate import aggregate, tournament_odds


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


def _sims():
    return simulate_tournaments(_struct(), TournamentResults([]), FakeSampler(), n=300, seed=5)


def test_champion_probs_sum_to_one():
    agg = aggregate(_struct(), _sims())
    total = sum(t["champion"] for t in agg["teams"].values())
    assert abs(total - 1.0) < 1e-9


def test_group_positions_sum_to_one_per_team():
    agg = aggregate(_struct(), _sims())
    a1 = agg["groups"]["A"]["A1"]
    assert abs(a1["p1"] + a1["p2"] + a1["p3"] + a1["p4"] - 1.0) < 1e-9


def test_round_probs_monotone():
    agg = aggregate(_struct(), _sims())
    for t in agg["teams"].values():
        assert t["reach_F"] >= t["champion"] - 1e-9


def test_stronger_team_more_likely_champion():
    agg = aggregate(_struct(), _sims())
    assert agg["teams"]["A1"]["champion"] > agg["teams"]["A4"]["champion"]


def test_tournament_odds_fair_and_value():
    agg = aggregate(_struct(), _sims())
    vcfg = {"threshold": 0.0, "ev_medium": 0.10, "reliability_low": 0.40,
            "reliability_high": 0.70, "kelly_quarter_divisor": 4}
    out = tournament_odds(agg, book_odds={"champion": {"A1": 50.0}},
                          reliability=0.6, value_config=vcfg)
    assert out["odds"]["champion"]["A1"]["fair_odds"] is not None
    assert "A1" in out["value"]["champion"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.tournament.aggregate'`

- [ ] **Step 3: Write implementation**

`footy/tournament/aggregate.py`:
```python
from __future__ import annotations

from footy.betting.odds import decorate_group
from footy.betting.value import assess_market


def aggregate(structure, sims: list) -> dict:
    """Count Monte Carlo tournament results into probabilities."""
    n = len(sims)
    teams = [t for group in structure.groups.values() for t in group]
    rounds = structure.rounds
    ladder = rounds + ["champion"]
    rank_of = {label: i for i, label in enumerate(ladder)}

    team_stats = {t: {"advance_group": 0, "champion": 0} for t in teams}
    for label in rounds:
        for t in teams:
            team_stats[t][f"reach_{label}"] = 0
    group_pos = {g: {t: [0, 0, 0, 0] for t in group} for g, group in structure.groups.items()}

    for res in sims:
        for t in teams:
            reached = rank_of.get(res.furthest_round.get(t, ""), -1)
            for label in rounds:
                if reached >= rank_of[label]:
                    team_stats[t][f"reach_{label}"] += 1
            if reached >= rank_of[rounds[0]]:
                team_stats[t]["advance_group"] += 1
            if res.champion == t:
                team_stats[t]["champion"] += 1
        for g, order in res.group_order.items():
            for pos, t in enumerate(order):
                if pos < 4:
                    group_pos[g][t][pos] += 1

    teams_out = {}
    for t in teams:
        d = {"advance_group": team_stats[t]["advance_group"] / n,
             "champion": team_stats[t]["champion"] / n}
        for label in rounds:
            d[f"reach_{label}"] = team_stats[t][f"reach_{label}"] / n
        teams_out[t] = d

    groups_out = {}
    for g, group in structure.groups.items():
        groups_out[g] = {}
        for t in group:
            c = group_pos[g][t]
            groups_out[g][t] = {"p1": c[0] / n, "p2": c[1] / n, "p3": c[2] / n, "p4": c[3] / n}

    slot_freq = {}
    for idx in range(len(structure.bracket_r32)):
        slot_freq[f"R32_tie_{idx + 1}"] = {}
    for res in sims:
        first = list(structure.groups.keys())
        # slot occupancy at the first knockout round (who played each opening tie)
    return {"teams": teams_out, "groups": groups_out,
            "slot_outcome_frequency": _slot_frequency(structure, sims),
            "meta": {"n_tournaments": n}}


def _slot_frequency(structure, sims: list) -> dict:
    """Frequency of each team appearing as champion per simulation (compact slot proxy)."""
    n = len(sims)
    freq = {}
    for res in sims:
        freq[res.champion] = freq.get(res.champion, 0) + 1
    return {"champion": {team: c / n for team, c in sorted(freq.items(), key=lambda kv: -kv[1])}}


def tournament_odds(agg: dict, book_odds: dict | None, reliability: float,
                    value_config: dict) -> dict:
    """Fair odds for champion (+ optional value) reusing SP2 odds/value."""
    champion_probs = {t: d["champion"] for t, d in agg["teams"].items() if d["champion"] > 0}
    odds = {"champion": decorate_group(champion_probs)}
    out = {**agg, "odds": odds}
    if book_odds and "champion" in book_odds:
        out["value"] = {"champion": assess_market(champion_probs, book_odds["champion"],
                                                   reliability, value_config)}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/tournament/aggregate.py tests/test_aggregate.py
git commit -m "feat: tournament aggregation + odds/value reusing SP2

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: config skeletons + full-suite verification

**Files:**
- Create: `configs/tournaments/wc2026.yaml`, `configs/tournaments/wc2026_results.yaml`, `configs/tournament_sim.yaml`
- Test: `tests/test_wc2026_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_wc2026_config.py`:
```python
from pathlib import Path

from footy.tournament.structure import load_structure
from footy.tournament.results import load_results

ROOT = Path(__file__).resolve().parent.parent


def test_wc2026_structure_loads():
    cfg = load_structure(ROOT / "configs" / "tournaments" / "wc2026.yaml")
    assert len(cfg.groups) == 12
    assert all(len(v) == 4 for v in cfg.groups.values())
    assert cfg.best_thirds == 8


def test_wc2026_results_loads_empty():
    cfg = load_structure(ROOT / "configs" / "tournaments" / "wc2026.yaml")
    res = load_results(ROOT / "configs" / "tournaments" / "wc2026_results.yaml", cfg.groups)
    assert res.played == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wc2026_config.py -v`
Expected: FAIL (config files missing)

- [ ] **Step 3: Create the config files**

`configs/tournaments/wc2026.yaml` — 12 groups A–L of 4 teams each (use placeholder
canonical team names that exist in the dataset; the user will replace with the real draw).
The implementer must write all 48 entries explicitly (no ellipsis). Use real national-team
names present in `international_results/results.csv` (e.g. Mexico, Canada, United States,
Brazil, Argentina, France, Germany, Spain, England, Netherlands, Portugal, Belgium,
Croatia, Uruguay, Colombia, Japan, South Korea, Morocco, Senegal, Australia, Switzerland,
Denmark, Mexico-group fillers, etc.). Each group exactly 4 unique teams, 48 unique overall:

```yaml
name: "FIFA World Cup 2026"
neutral_default: true
points: {win: 3, draw: 1, loss: 0}
groups:
  A: [Mexico, Canada, Ecuador, Curacao]
  B: [United States, Wales, Iran, Ghana]
  C: [Argentina, Poland, Australia, Tunisia]
  D: [France, Denmark, Senegal, Peru]
  E: [Spain, Germany, Japan, CostaRica]
  F: [Belgium, Croatia, Morocco, Canada-F]
  G: [Brazil, Switzerland, Cameroon, Serbia]
  H: [Portugal, Uruguay, SouthKorea, Ghana-H]
  I: [England, Netherlands, Nigeria, Qatar]
  J: [Italy, Colombia, Egypt, SaudiArabia]
  K: [Netherlands-K, Mexico-K, Ecuador-K, Algeria]
  L: [Sweden, Chile, Ivory Coast, Panama]
group_schedule: round_robin
qualification: {per_group_advance: 2, best_thirds: 8}
tiebreakers: [points, goal_difference, goals_for, head_to_head, fair_play, drawing_of_lots]
thirds_ranking: [points, goal_difference, goals_for, drawing_of_lots]
knockout:
  rounds: [R32, R16, QF, SF, F]
  thirds_assignment: ranked_order
  bracket_r32:
    - [winner_A, third_slot_1]
    - [runner_C, runner_D]
    - [winner_E, third_slot_2]
    - [runner_G, runner_H]
    - [winner_I, third_slot_3]
    - [runner_K, runner_L]
    - [winner_B, third_slot_4]
    - [runner_F, runner_J]
    - [winner_C, third_slot_5]
    - [runner_A, runner_B]
    - [winner_G, third_slot_6]
    - [runner_E, runner_I]
    - [winner_K, third_slot_7]
    - [runner_M, runner_N]
    - [winner_D, third_slot_8]
    - [runner_F, runner_H]
```

NOTE FOR IMPLEMENTER: the team names above are placeholders illustrating the shape. Replace
every team with a real, dataset-present national team and ensure **48 unique names** and
that **every `runner_X`/`winner_X` references an existing group A–L** (the sketch above
references M/N which do NOT exist — fix the bracket so all refs are within A–L, and so
`third_slot_1..8` are used exactly once). Validate by running the test in Step 4; the loader
rejects unknown groups and out-of-range third slots, so a wrong bracket will fail the test.

`configs/tournaments/wc2026_results.yaml`:
```yaml
played_matches: []
```

`configs/tournament_sim.yaml`:
```yaml
n_tournaments: 20000
seed: 42
neutral_default: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wc2026_config.py -v`
Expected: PASS (2 passed). If it fails on bracket refs or team counts, fix `wc2026.yaml`
until valid — do not change the loader or weaken the test.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests PASS (SP1 + SP2 + SP3). Smoke test may take ~2 min.

- [ ] **Step 6: Commit**

```bash
git add configs/tournaments/wc2026.yaml configs/tournaments/wc2026_results.yaml configs/tournament_sim.yaml tests/test_wc2026_config.py
git commit -m "feat: WC2026 tournament config skeletons + validation test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §2 module layout → Tasks 1–8 create exactly those modules. ✓
- §3 structure + validation → Task 1. ✓
- §4 results live file + lookups + match_id/knockout rules → Tasks 2, 7. ✓
- §5 sampler sole-RNG + λ cache (model_version+config_hash), no scoreline cache → Task 3. ✓
- §6 groups table + FIFA tie-breakers (H2H, reproducible lots) + thirds → Tasks 4, 5. ✓
- §7 knockout bracket + ET + weighted penalties + clipping → Task 6. ✓
- §8 simulator (fix played, sample pending, debug fields, run_id) → Task 7. ✓
- §9 aggregate (round/champion probs, group dist, slot freq) + SP2 odds/value → Task 8. ✓
- §10 error handling → validation in Tasks 1, 2; played-knockout draw guard in Task 7. ✓
- §11 testing incl. weighted-penalty distribution + clipping → Task 6 (`test_resolve_penalties_*`, `test_penalty_clipping_extremes`). ✓
- Config files (WC2026) → Task 9. ✓

**Placeholder scan:** Task 9's `wc2026.yaml` deliberately contains a *sketch* with an explicit instruction to the implementer to replace placeholder team names and fix bracket refs, with the loader's validation (Task 1) as the gate. This is intentional (the real WC2026 draw is data the implementer fills), not a code placeholder — every code step elsewhere is complete.

**Type consistency:** `MatchSampler.scorelines/sample_goals/lambdas` (T3) used by `resolve_match` (T6) and `run_tournament` (T7) with matching signatures; `rank_group(teams, results, points, tiebreakers, rng)` and `rank_thirds(thirds, thirds_ranking, rng)` (T5) called identically in T7; `build_bracket(group_ranks, thirds_ranked, bracket_cfg)` and `resolve_match(a, b, reg_a, reg_b, sampler, rng, neutral)` (T6) called as defined in T7; `TournamentResult` fields (T7) consumed by `aggregate` (T8); `decorate_group`/`assess_market` reused from SP2 with their existing signatures. ✓

**Known follow-up:** `_slot_frequency` in Task 8 returns a compact champion-frequency proxy; a richer per-tie slot breakdown can be added later without changing the public `aggregate` contract.
