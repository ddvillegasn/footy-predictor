# Live World Cup Mode Implementation Plan (SP6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the app into a live WC2026 mode driven by real football-data: auto-build the real groups, fetch played results, compute live per-team tournament stats, and offer a refit "live model" for predictions alongside the out-of-sample baseline.

**Architecture:** Extend the SP4 `footy/live` provider with `fetch_structure`; add `structure_sync` (writes `wc2026.yaml` from the API, hard-erroring with the full list of unmapped names), `stats` (per-team tournament table), and a `build_live_predictor` in the UI service (base dataset + played matches, refit). The Streamlit app gains a refresh button, a groups/stats view, a markets-always match tab, and a live-model toggle. Two models stay separate: BASE (scoreboard, out-of-sample) vs LIVE (predictions).

**Tech Stack:** Python 3.10.6, requests, pyyaml, pandas, numpy, streamlit, pytest. Run from repo root: `python -m pytest`.

**Conventions:** branch `feature/baseline-v1`. TDD: failing test first. Commit per task, `git add` ONLY named files (never `__pycache__`/`.pyc`). No real network in tests (HTTP mock / FakeProvider). Commit trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## File Structure

| File | Responsibility |
|---|---|
| `footy/live/provider.py` | ADD `fetch_structure`; `from_config` env-or-file key |
| `footy/live/structure_sync.py` | NEW: `map_groups`, `write_structure_yaml`, `sync_structure`, `BRACKET_R32` |
| `footy/live/stats.py` | NEW: `team_stats` |
| `footy/ui/service.py` | ADD `build_live_predictor`; `match_prediction` markets-always |
| `app/streamlit_app.py` | refresh button, groups+stats tab, live toggle, secrets fallback |
| `configs/name_map.yaml` | add real WC2026 name mismatches |
| `.gitignore` | add `configs/secrets.local.yaml` |
| `tests/test_*` | per unit, mock/fake |

---

## Task 1: provider.fetch_structure + key-from-file fallback

**Files:**
- Modify: `footy/live/provider.py`
- Modify: `.gitignore`
- Test: `tests/test_fetch_structure.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fetch_structure.py`:
```python
import pytest

import footy.live.provider as prov

STRUCT_PAYLOAD = {"matches": [
    {"id": 1, "stage": "GROUP_STAGE", "group": "GROUP_A",
     "homeTeam": {"name": "Mexico"}, "awayTeam": {"name": "South Africa"},
     "score": {"fullTime": {"home": 2, "away": 0}}, "status": "FINISHED"},
    {"id": 2, "stage": "GROUP_STAGE", "group": "GROUP_A",
     "homeTeam": {"name": "South Korea"}, "awayTeam": {"name": "Czechia"},
     "score": {"fullTime": {"home": 2, "away": 1}}, "status": "FINISHED"},
    {"id": 3, "stage": "LAST_16", "group": None,
     "homeTeam": {"name": "Mexico"}, "awayTeam": {"name": "South Korea"},
     "score": {"fullTime": {"home": None, "away": None}}, "status": "TIMED"},
]}


class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_structure_groups_by_group(monkeypatch):
    monkeypatch.setattr(prov.requests, "get", lambda *a, **k: FakeResp(STRUCT_PAYLOAD))
    p = prov.FootballDataProvider("key", "https://x/v4", "WC", 10)
    groups = p.fetch_structure()
    assert set(groups.keys()) == {"GROUP_A"}                 # knockout match ignored
    assert groups["GROUP_A"] == ["Mexico", "South Africa", "South Korea", "Czechia"]


def test_from_config_reads_key_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    secrets = tmp_path / "secrets.local.yaml"
    secrets.write_text("football_data_api_key: filekey123\n", encoding="utf-8")
    p = prov.FootballDataProvider.from_config(
        {"base_url": "https://x/v4", "competition_code": "WC", "request_timeout": 10},
        secrets_path=secrets)
    assert p.api_key == "filekey123"


def test_from_config_missing_everywhere_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FOOTBALL_DATA_API_KEY"):
        prov.FootballDataProvider.from_config(
            {"base_url": "https://x/v4", "competition_code": "WC", "request_timeout": 10},
            secrets_path=tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fetch_structure.py -v`
Expected: FAIL (`fetch_structure` / `secrets_path` not defined)

- [ ] **Step 3: Edit `footy/live/provider.py`**

At the top, ensure these imports exist (add what is missing): `from pathlib import Path` and `import yaml`.

Replace the existing `from_config` classmethod with:
```python
    @classmethod
    def from_config(cls, live_cfg: dict, secrets_path="configs/secrets.local.yaml") -> "FootballDataProvider":
        key = os.environ.get("FOOTBALL_DATA_API_KEY")
        if not key:
            p = Path(secrets_path)
            if p.exists():
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                key = data.get("football_data_api_key")
        if not key:
            raise ValueError("FOOTBALL_DATA_API_KEY env var not set and no key in "
                             f"{secrets_path}")
        return cls(key, live_cfg["base_url"], live_cfg["competition_code"],
                   live_cfg["request_timeout"])
```

Add this method to `FootballDataProvider` (after `fetch_finished`):
```python
    def fetch_structure(self) -> dict:
        """Group-stage groups -> ordered team lists, from the matches endpoint.
        {'GROUP_A': ['Mexico', 'South Africa', ...], ...} (knockout matches ignored)."""
        url = f"{self.base_url}/competitions/{self.competition_code}/matches"
        resp = requests.get(url, headers={"X-Auth-Token": self.api_key}, timeout=self.timeout)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ValueError("football-data response is not valid JSON") from exc
        groups: dict = {}
        for m in payload.get("matches", []):
            if m.get("stage") != "GROUP_STAGE":
                continue
            g = m.get("group")
            if not g:
                continue
            bucket = groups.setdefault(g, [])
            for side in ("homeTeam", "awayTeam"):
                name = m[side]["name"]
                if name not in bucket:
                    bucket.append(name)
        return groups
```

In `.gitignore`, add a line:
```
configs/secrets.local.yaml
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fetch_structure.py tests/test_football_data_adapter.py -v`
Expected: all PASS (3 new + the existing 5 adapter tests stay green)

- [ ] **Step 5: Commit**

```bash
git add footy/live/provider.py .gitignore tests/test_fetch_structure.py
git commit -m "feat: provider.fetch_structure + key-from-file fallback

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: structure_sync.py

**Files:**
- Create: `footy/live/structure_sync.py`
- Modify: `configs/name_map.yaml`
- Test: `tests/test_structure_sync.py`

- [ ] **Step 1: Write the failing test**

`tests/test_structure_sync.py`:
```python
import pytest

from footy.live.structure_sync import map_groups, sync_structure
from footy.tournament.structure import load_structure

KNOWN = {"Mexico", "South Africa", "South Korea", "Czech Republic"}
NAME_MAP = {"Czechia": "Czech Republic"}


def test_map_groups_maps_and_letters():
    raw = {"GROUP_A": ["Mexico", "South Africa", "South Korea", "Czechia"]}
    groups = map_groups(raw, NAME_MAP, KNOWN)
    assert groups == {"A": ["Mexico", "South Africa", "South Korea", "Czech Republic"]}


def test_map_groups_collects_all_missing_in_one_error():
    raw = {"GROUP_A": ["Mexico", "Atlantis", "Wakanda", "Czechia"]}
    with pytest.raises(ValueError) as exc:
        map_groups(raw, NAME_MAP, KNOWN)
    msg = str(exc.value)
    assert "Atlantis" in msg and "Wakanda" in msg          # both listed, not just the first
    assert "2 team" in msg


class FakeProvider:
    def fetch_structure(self):
        # 2 full groups of 4 (A, B) using known/ mappable names.
        return {
            "GROUP_A": ["Mexico", "South Africa", "South Korea", "Czechia"],
            "GROUP_B": ["Mexico2", "South Africa2", "South Korea2", "Czech2"],
        }


def test_sync_structure_writes_loadable_yaml(tmp_path):
    known = KNOWN | {"Mexico2", "South Africa2", "South Korea2", "Czech2"}
    out = tmp_path / "wc.yaml"
    # Use only 2 groups, so the bracket template (which references A..L) must be trimmed
    # for this test by passing a 2-group bracket override.
    n = sync_structure(FakeProvider(), NAME_MAP, known, out,
                       bracket_r32=[["winner_A", "runner_B"], ["winner_B", "runner_A"]])
    assert n == 8
    cfg = load_structure(out)
    assert set(cfg.groups.keys()) == {"A", "B"}
    assert cfg.groups["A"][3] == "Czech Republic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_structure_sync.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.live.structure_sync'`

- [ ] **Step 3: Write implementation + extend name_map**

`footy/live/structure_sync.py`:
```python
from pathlib import Path

import yaml

# Official-style WC2026 R32 bracket template (group winners/runners + 8 ranked thirds).
BRACKET_R32 = [
    ["winner_A", "third_slot_1"], ["winner_B", "third_slot_2"],
    ["winner_C", "third_slot_3"], ["winner_D", "third_slot_4"],
    ["winner_E", "third_slot_5"], ["winner_F", "third_slot_6"],
    ["winner_G", "third_slot_7"], ["winner_H", "third_slot_8"],
    ["winner_I", "runner_A"], ["runner_B", "runner_C"],
    ["winner_J", "runner_D"], ["runner_E", "runner_F"],
    ["winner_K", "runner_G"], ["runner_H", "runner_I"],
    ["winner_L", "runner_J"], ["runner_K", "runner_L"],
]


def map_groups(raw_groups: dict, name_map: dict, known_teams: set) -> dict:
    """'GROUP_A' -> 'A' and API names -> canonical. Collects ALL teams that cannot be
    resolved into the dataset and raises ONE ValueError listing every one of them."""
    canonical_groups = {}
    missing = []
    for raw_label, teams in raw_groups.items():
        letter = raw_label.split("_")[-1]
        canon = []
        for t in teams:
            mapped = name_map.get(t, t)
            if mapped not in known_teams:
                missing.append((t, mapped))
            canon.append(mapped)
        canonical_groups[letter] = canon
    if missing:
        lines = "\n".join(f"  - API '{api}' -> '{res}' (add to configs/name_map.yaml)"
                          for api, res in missing)
        raise ValueError(f"{len(missing)} team(s) not in dataset:\n{lines}")
    return canonical_groups


def write_structure_yaml(groups: dict, out_path, bracket_r32=None) -> None:
    data = {
        "name": "FIFA World Cup 2026",
        "neutral_default": True,
        "points": {"win": 3, "draw": 1, "loss": 0},
        "groups": groups,
        "group_schedule": "round_robin",
        "qualification": {"per_group_advance": 2, "best_thirds": 8 if bracket_r32 is None else 0},
        "tiebreakers": ["points", "goal_difference", "goals_for", "head_to_head",
                        "fair_play", "drawing_of_lots"],
        "thirds_ranking": ["points", "goal_difference", "goals_for", "drawing_of_lots"],
        "knockout": {"rounds": ["R32", "R16", "QF", "SF", "F"],
                     "thirds_assignment": "ranked_order",
                     "bracket_r32": bracket_r32 if bracket_r32 is not None else BRACKET_R32},
    }
    Path(out_path).write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def sync_structure(provider, name_map: dict, known_teams: set, out_path, bracket_r32=None) -> int:
    """fetch_structure -> map_groups -> write wc2026.yaml. Returns team count."""
    raw = provider.fetch_structure()
    groups = map_groups(raw, name_map, known_teams)
    write_structure_yaml(groups, out_path, bracket_r32=bracket_r32)
    return sum(len(v) for v in groups.values())
```

In `configs/name_map.yaml`, extend the `teams:` mapping with the confirmed WC2026 mismatches
(keep the existing entries):
```yaml
  "Czechia": "Czech Republic"
  "Bosnia-Herzegovina": "Bosnia and Herzegovina"
  "Cabo Verde": "Cape Verde"
  "Türkiye": "Turkey"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_structure_sync.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/live/structure_sync.py configs/name_map.yaml tests/test_structure_sync.py
git commit -m "feat: structure sync builds wc2026.yaml from API (lists all unmapped)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: stats.py — live per-team tournament stats

**Files:**
- Create: `footy/live/stats.py`
- Test: `tests/test_stats.py`

- [ ] **Step 1: Write the failing test**

`tests/test_stats.py`:
```python
from types import SimpleNamespace

from footy.tournament.results import TournamentResults, PlayedMatch
from footy.live.stats import team_stats


def _structure():
    return SimpleNamespace(
        groups={"A": ["Mexico", "South Africa", "South Korea", "Czech Republic"]},
        points={"win": 3, "draw": 1, "loss": 0})


def test_team_stats_accumulates():
    results = TournamentResults([
        PlayedMatch("1", "group", "Mexico", "South Africa", 2, 0, group="A"),
        PlayedMatch("2", "group", "South Korea", "Mexico", 1, 1, group="A"),
    ])
    stats = team_stats(_structure(), results)
    mx = stats["Mexico"]
    assert mx["played"] == 2 and mx["points"] == 4          # win + draw
    assert mx["gf"] == 3 and mx["ga"] == 1 and mx["gd"] == 2
    assert mx["wins"] == 1 and mx["draws"] == 1 and mx["losses"] == 0
    assert mx["form"] == ["W", "D"]
    assert stats["Czech Republic"]["played"] == 0           # no matches yet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footy.live.stats'`

- [ ] **Step 3: Write implementation**

`footy/live/stats.py`:
```python
def team_stats(structure, results) -> dict:
    """Per-team tournament table from played matches (group + knockout):
    {team: {played, points, gf, ga, gd, wins, draws, losses, form}}.
    form = list of 'W'/'D'/'L', oldest first."""
    points = structure.points
    teams = {t for group in structure.groups.values() for t in group}
    stats = {t: {"played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0,
                 "wins": 0, "draws": 0, "losses": 0, "form": []} for t in teams}
    for pm in results.played:
        for team, gf, ga in ((pm.team_a, pm.goals_a, pm.goals_b),
                             (pm.team_b, pm.goals_b, pm.goals_a)):
            if team not in stats:
                continue
            s = stats[team]
            s["played"] += 1
            s["gf"] += gf
            s["ga"] += ga
            if gf > ga:
                s["points"] += points["win"]; s["wins"] += 1; s["form"].append("W")
            elif gf < ga:
                s["points"] += points["loss"]; s["losses"] += 1; s["form"].append("L")
            else:
                s["points"] += points["draw"]; s["draws"] += 1; s["form"].append("D")
    for t in stats:
        stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add footy/live/stats.py tests/test_stats.py
git commit -m "feat: live per-team tournament stats (PJ/Pts/GF/GA/GD/form)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: build_live_predictor + markets-always

**Files:**
- Modify: `footy/ui/service.py`
- Modify: `tests/test_ui_service.py`
- Test: `tests/test_live_predictor.py`

- [ ] **Step 1: Write the failing test**

`tests/test_live_predictor.py`:
```python
import pandas as pd

from footy.predict import Predictor
from footy.ui.service import build_live_predictor

MODEL_CFG = {"xi": 0.0, "max_goals": 10, "home_advantage_init": 0.25,
             "ridge": 0.01, "min_matches_reliable": 10, "model_version": "baseline-v1.0.0"}
MC_CFG = {"n_sims": 20000, "seed": 42, "max_goals": 10, "ci_level": 0.90, "top_scores": 5}


def _base_matches():
    rows = []
    for _ in range(12):
        rows.append(("2019-01-01", "Brazil", "Haiti", 1, 1))   # historically even-ish
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    return df


def test_live_refit_shifts_lambda_toward_recent_results():
    base = Predictor.from_matches(_base_matches(), model_config=MODEL_CFG, mc_config=MC_CFG,
                                  canonical=lambda x: x, as_of=pd.Timestamp("2020-01-01"))
    base_a, base_b = base.model.rates("Brazil", "Haiti", neutral=True)

    played = [{"team_a": "Brazil", "team_b": "Haiti", "goals_a": 5, "goals_b": 0}] * 6
    live = build_live_predictor(base, played, tournament_date="2026-06-15",
                                model_config=MODEL_CFG, mc_config=MC_CFG)
    live_a, live_b = live.model.rates("Brazil", "Haiti", neutral=True)
    # Brazil thrashed Haiti recently -> Brazil's rate should rise vs Haiti's.
    assert (live_a - live_b) > (base_a - base_b)
    assert isinstance(live, Predictor)
```

Also REPLACE in `tests/test_ui_service.py` the test
`test_match_prediction_without_book_odds_keeps_markets_off` with:
```python
def test_match_prediction_always_includes_markets():
    fp = FakePredictor()
    match_prediction(fp, "Brazil", "Haiti", neutral=True)
    assert fp.last_kwargs["include_markets"] is True
```
(Keep the other tests in that file unchanged.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_live_predictor.py tests/test_ui_service.py -v`
Expected: FAIL (`build_live_predictor` missing; markets-always test fails)

- [ ] **Step 3: Edit `footy/ui/service.py`**

Add `import pandas as pd` and `from footy.predict import Predictor` at the top. Replace
`match_prediction` and add `build_live_predictor`:
```python
def match_prediction(predictor, team_a, team_b, neutral=False, book_odds=None) -> dict:
    """Wrap predict(); markets are always included for the UI. Value/EV needs book_odds."""
    return predictor.predict(team_a, team_b, neutral=neutral,
                             include_markets=True, book_odds=book_odds)


def build_live_predictor(base_predictor, played_matches, tournament_date,
                         model_config, mc_config):
    """LIVE model: base dataset + played WC matches (recent date, neutral) refit.
    Reacts to tournament form (lightly, by design). Used for predictions, NOT scoreboard."""
    base = base_predictor.matches.copy()
    rows = [{"date": pd.Timestamp(tournament_date),
             "home_team": pm["team_a"], "away_team": pm["team_b"],
             "home_score": pm["goals_a"], "away_score": pm["goals_b"],
             "tournament": "FIFA World Cup", "neutral": True} for pm in played_matches]
    live_matches = pd.concat([base, pd.DataFrame(rows)], ignore_index=True) if rows else base
    as_of = live_matches["date"].max() + pd.Timedelta(days=1)
    return Predictor.from_matches(live_matches, model_config=model_config, mc_config=mc_config,
                                  canonical=base_predictor.canonical, as_of=as_of)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_live_predictor.py tests/test_ui_service.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add footy/ui/service.py tests/test_ui_service.py tests/test_live_predictor.py
git commit -m "feat: live-model refit + markets-always in UI service

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Streamlit app — refresh button, groups/stats tab, live toggle

**Files:**
- Modify: `app/streamlit_app.py`
- Test: `tests/test_streamlit_app_imports.py` (unchanged — must keep passing)

- [ ] **Step 1: Confirm the smoke test still describes the contract**

The existing `tests/test_streamlit_app_imports.py` imports the app module and asserts it has
`main` and `build_engine`. Keep both names. Run it now to see current state:

Run: `python -m pytest tests/test_streamlit_app_imports.py -v`
Expected: PASS (still the SP5 app)

- [ ] **Step 2: Rewrite `app/streamlit_app.py`**

```python
from pathlib import Path

import pandas as pd
import streamlit as st

from footy.cli import _build_default_predictor
from footy.config import load_config, config_fingerprint
from footy.tournament.structure import load_structure
from footy.tournament.results import load_results, TournamentResults
from footy.tournament.sampler import MatchSampler
from footy.live.provider import FootballDataProvider
from footy.live.name_map import load_name_map
from footy.live.ingest import ingest
from footy.live.structure_sync import sync_structure
from footy.live.stats import team_stats
from footy.ui.service import (team_list, match_prediction, tournament_probs,
                              live_scoreboard, build_live_predictor)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "configs" / "tournaments" / "wc2026_results.yaml"
STRUCTURE_PATH = ROOT / "configs" / "tournaments" / "wc2026.yaml"


@st.cache_resource
def build_engine():
    """BASE model (pre-tournament) + sampler + structure. Cached per server session."""
    predictor = _build_default_predictor()
    structure = load_structure(STRUCTURE_PATH)
    canon = predictor.canonical
    for g, teams in structure.groups.items():
        structure.groups[g] = [canon(t) for t in teams]
    mc_cfg = load_config("montecarlo")
    model_cfg = load_config("model")
    sampler = MatchSampler(predictor.model, mc_cfg,
                           model_version=model_cfg["model_version"],
                           config_hash=config_fingerprint("montecarlo"))
    return predictor, structure, sampler


@st.cache_resource
def build_live(_base_predictor, results_token):
    """LIVE model refit with played matches. results_token busts the cache when results change."""
    if not RESULTS_PATH.exists():
        return _base_predictor
    structure = load_structure(STRUCTURE_PATH)
    canon = _base_predictor.canonical
    for g, teams in structure.groups.items():
        structure.groups[g] = [canon(t) for t in teams]
    results = load_results(RESULTS_PATH, structure.groups)
    played = [{"team_a": pm.team_a, "team_b": pm.team_b,
               "goals_a": pm.goals_a, "goals_b": pm.goals_b} for pm in results.played]
    if not played:
        return _base_predictor
    return build_live_predictor(_base_predictor, played, tournament_date="2026-06-15",
                                model_config=load_config("model"), mc_config=load_config("montecarlo"))


def _results_token() -> str:
    return str(RESULTS_PATH.stat().st_mtime) if RESULTS_PATH.exists() else "none"


def _played_dicts(structure):
    if not RESULTS_PATH.exists():
        return []
    results = load_results(RESULTS_PATH, structure.groups)
    return [{"team_a": pm.team_a, "team_b": pm.team_b,
             "goals_a": pm.goals_a, "goals_b": pm.goals_b} for pm in results.played]


def _refresh_from_api(structure):
    try:
        provider = FootballDataProvider.from_config(load_config("live"))
        name_map = load_name_map("configs/name_map.yaml")
        known = {t for g in structure.groups.values() for t in g}
        sync_structure(provider, name_map, known, STRUCTURE_PATH)
        n = ingest(provider, structure, name_map, load_config("live")["stage_map"], RESULTS_PATH)
        st.success(f"Actualizado desde la API: {n} partidos jugados.")
        st.cache_resource.clear()
    except Exception as exc:  # noqa: BLE001 - surface any API/mapping problem to the user
        st.error(f"No se pudo actualizar desde la API: {exc}")


def _render_match_tab(base_predictor, structure):
    use_live = st.toggle("Usar modelo LIVE (con resultados del Mundial)", value=False)
    predictor = build_live(base_predictor, _results_token()) if use_live else base_predictor
    st.caption(f"Modelo en uso: {'LIVE (re-fit con jugados)' if use_live else 'BASE (histórico)'}")

    teams = team_list(base_predictor)
    c1, c2 = st.columns(2)
    team_a = c1.selectbox("Equipo A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    team_b = c2.selectbox("Equipo B", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
    neutral = st.checkbox("Cancha neutral", value=True)
    with st.expander("Cuotas de tu casa (opcional, para detectar valor)"):
        o1, o2, o3 = st.columns(3)
        odd_home = o1.number_input("Cuota A", min_value=0.0, value=0.0, step=0.05)
        odd_draw = o2.number_input("Cuota X", min_value=0.0, value=0.0, step=0.05)
        odd_away = o3.number_input("Cuota B", min_value=0.0, value=0.0, step=0.05)

    if st.button("Predecir", type="primary"):
        one = {}
        if odd_home > 1.0:
            one["home"] = odd_home
        if odd_draw > 1.0:
            one["draw"] = odd_draw
        if odd_away > 1.0:
            one["away"] = odd_away
        book = {"1x2": one} if one else None
        try:
            out = match_prediction(predictor, team_a, team_b, neutral=neutral, book_odds=book)
        except ValueError as exc:
            st.error(str(exc))
            return
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a}", f"{out['team_a_win']}%")
        m2.metric("Empate", f"{out['draw']}%")
        m3.metric(f"{team_b}", f"{out['team_b_win']}%")
        st.caption(f"xG {out['expected_goals_a']} - {out['expected_goals_b']} · "
                   f"marcador {out['most_likely_score']} · fiabilidad {out['prediction_reliability']}")
        st.bar_chart(pd.DataFrame({"prob %": [out["team_a_win"], out["draw"], out["team_b_win"]]},
                                  index=[team_a, "Empate", team_b]))
        mk = out["markets"]
        rows = [{"mercado": "1X2", "resultado": k, "prob": v["prob"], "cuota": v["fair_odds"]}
                for k, v in mk["1x2"].items() if isinstance(v, dict)]
        for line, ou in mk["over_under"].items():
            rows.append({"mercado": f"O/U {line}", "resultado": "over",
                         "prob": ou["over"]["prob"], "cuota": ou["over"]["fair_odds"]})
        rows.append({"mercado": "BTTS", "resultado": "sí",
                     "prob": mk["btts"]["yes"]["prob"], "cuota": mk["btts"]["yes"]["fair_odds"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        if "value" in out and out["value"].get("1x2"):
            vrows = [{"resultado": k, "edge %": v["edge_pct"], "EV": v["ev_per_unit"],
                      "stake": v["stake_recommendation"], "value": v["is_value"]}
                     for k, v in out["value"]["1x2"].items()]
            st.subheader("Valor vs tus cuotas")
            st.dataframe(pd.DataFrame(vrows), use_container_width=True)


def _render_tournament_tab(structure, sampler):
    played = _played_dicts(structure)
    if played:
        results = load_results(RESULTS_PATH, structure.groups)
        stats = team_stats(structure, results)
        st.subheader("Stats por selección (en el torneo)")
        sdf = pd.DataFrame([
            {"equipo": t, "PJ": s["played"], "Pts": s["points"], "GF": s["gf"],
             "GC": s["ga"], "DG": s["gd"], "forma": "".join(s["form"][-5:])}
            for t, s in sorted(stats.items(), key=lambda kv: -kv[1]["points"]) if s["played"] > 0])
        st.dataframe(sdf, use_container_width=True)
    else:
        st.info("Aún no hay partidos jugados. Usa 'Actualizar desde API' (barra lateral).")

    n = st.slider("Número de torneos a simular", 500, 10000, 1000, 500)
    if st.button("Simular Mundial", type="primary"):
        results = (load_results(RESULTS_PATH, structure.groups)
                   if RESULTS_PATH.exists() else TournamentResults([]))
        with st.spinner(f"Simulando {n} torneos…"):
            agg = tournament_probs(structure, results, sampler, n, seed=42)
        champ = sorted(agg["teams"].items(), key=lambda kv: -kv[1]["champion"])[:16]
        df = pd.DataFrame(
            {"campeón %": [round(d["champion"] * 100, 1) for _, d in champ],
             "avanza %": [round(d["advance_group"] * 100, 1) for _, d in champ]},
            index=[t for t, _ in champ])
        st.bar_chart(df[["campeón %"]])
        st.dataframe(df, use_container_width=True)


def _render_scoreboard_tab(base_predictor, structure):
    played = _played_dicts(structure)
    if not played:
        st.info("Aún no hay resultados cargados. Usa 'Actualizar desde API'.")
        return
    board = live_scoreboard(base_predictor, played)   # BASE model, out-of-sample
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", board["accuracy"])
    c2.metric("Log loss", board["log_loss"])
    c3.metric("Brier", board["brier"])
    st.caption(f"Partidos: {board['n']} · goal MAE {board['goal_mae']} · modelo BASE (out-of-sample)")
    st.dataframe(pd.DataFrame(board["matches"]), use_container_width=True)


def main():
    st.set_page_config(page_title="Footy — Mundial 2026", layout="wide")
    st.title("⚽ Footy — Mundial 2026 en vivo")
    st.caption("El modelo reacciona poco a resultados sueltos (es correcto). "
               "Cuotas/EV dependen del modelo; no son garantía.")
    with st.spinner("Cargando modelo (solo la primera vez)…"):
        base_predictor, structure, sampler = build_engine()
    with st.sidebar:
        st.header("Datos en vivo")
        if st.button("🔄 Actualizar desde API"):
            _refresh_from_api(structure)
    tab1, tab2, tab3 = st.tabs(["Predecir partido", "Mundial / Grupos", "Scoreboard"])
    with tab1:
        _render_match_tab(base_predictor, structure)
    with tab2:
        _render_tournament_tab(structure, sampler)
    with tab3:
        _render_scoreboard_tab(base_predictor, structure)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the smoke test**

Run: `python -m pytest tests/test_streamlit_app_imports.py -v`
Expected: PASS (1 passed — module imports, `main`/`build_engine` present)

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: all PASS (smoke ~2 min).

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: live WC2026 app (refresh button, stats, live toggle)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Final verification (boot the app)

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q`
Expected: all PASS.

- [ ] **Step 2: Boot the app headless to confirm it starts**

Run: `PYTHONPATH=. streamlit run app/streamlit_app.py --server.headless true --server.port 8524 &` then,
after ~10s, confirm the port is open and the log shows "You can now view your Streamlit app",
then stop the process. (No real API call is made on boot; build_engine fits the model on first
browser load.)

- [ ] **Step 3: No commit needed** (verification only) unless a fix was required.

---

## Self-Review

**Spec coverage:**
- §3 `fetch_structure` + key-from-file → Task 1. ✓
- §4 `structure_sync` (map all-missing, write wc2026.yaml) + name_map real names → Task 2. ✓
- §5 `stats.team_stats` → Task 3. ✓
- §6 `build_live_predictor` → Task 4. ✓
- §7 UI: refresh button, groups/stats tab, live toggle, markets-always, BASE scoreboard → Tasks 4, 5. ✓
- §8 error handling (sync full list, API failure st.error, missing key, no played → info) → Tasks 1, 2, 5. ✓
- §1/§10 two models (BASE vs LIVE, separate caches) → Tasks 4, 5. ✓
- §9 testing mock/fake → Tasks 1-4; app smoke Task 5. ✓

**Placeholder scan:** all code complete; `provider.py`, `name_map.yaml`, `service.py` edits are additive/replacements with explicit anchors. No TODO/TBD.

**Type consistency:** `fetch_structure` returns `{raw_group: [names]}` (T1) consumed by `map_groups`/`sync_structure` (T2) and `_refresh_from_api` (T5); `sync_structure(provider, name_map, known_teams, out_path, bracket_r32=None)` signature matches its test (T2) and app call (T5, default bracket); `team_stats(structure, results)` (T3) used in T5; `build_live_predictor(base_predictor, played_matches, tournament_date, model_config, mc_config)` (T4) used by `build_live` (T5); `match_prediction(..., book_odds=None)` always markets (T4) used in T5. App keeps `main`/`build_engine` for the smoke test. ✓

**Note:** the LIVE refit (~2 min) is cached separately via `build_live(_base_predictor, results_token)`; `results_token` (file mtime) busts it when results change. Real API sync happens only on the user's machine (they have the token); tests use FakeProvider/HTTP mock.
