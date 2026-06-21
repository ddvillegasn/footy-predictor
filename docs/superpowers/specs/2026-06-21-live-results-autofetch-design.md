# Spec — Auto-fetch de resultados en vivo + scoreboard · Sub-proyecto 4

- **Fecha:** 2026-06-21
- **Estado:** En revisión
- **Enfoque aprobado:** A — Provider pluggable + `FootballDataProvider` (football-data.org) + `name_map` + `ingest` idempotente + `runner` con modo `--watch`, usando `requests`. Incluye `scoreboard` (predicho vs real).
- **Depende de:** SP1 (`Predictor`/`predict`, `metrics.py`), SP3 (`structure`, `results`, `MatchSampler`, `simulate_tournaments`, `aggregate`). Todo verde (118 tests).

---

## 1. Contexto

SP3 simula el torneo condicionado a un archivo de resultados (`wc2026_results.yaml`) que
hoy se edita a mano. SP4 lo **automatiza**: trae los resultados oficiales del Mundial desde
**football-data.org**, actualiza ese archivo y re-corre el simulador; con `--watch N` repite
cada N minutos. Además, un **scoreboard** compara las predicciones del modelo contra los
resultados reales ya jugados (accuracy/log-loss/Brier en vivo, out-of-sample).

Arquitectura **pluggable** (interfaz `ResultsProvider`) para no casarse con football-data.
Reusa SP1/SP3; no los modifica. Errores **duros** ante cualquier desajuste de nombres/stage
(en datos deportivos, "casi coincide" corrompe en silencio).

**Aviso de datos:** `wc2026.yaml` debe reflejar los **grupos reales** del sorteo, y
`name_map.yaml`/`stage_map` cubrir los nombres/stages que devuelva la API. Si no, los errores
duros lo señalan; nunca se asigna mal en silencio.

---

## 2. Arquitectura y módulos

```
footy/
├── live/                        # NUEVO
│   ├── __init__.py
│   ├── provider.py              # ResultsProvider (Protocol) + ProviderMatch + FootballDataProvider
│   ├── name_map.py              # load_name_map, map_team, map_stage (error duro)
│   ├── ingest.py                # build_played_matches + ingest (idempotente, escribe results yaml)
│   ├── scoreboard.py            # compara predicho vs real (reusa predict + metrics)
│   └── runner.py                # TournamentRunner (provider-agnostic) + watch
├── cli.py                       # nuevo entry-point: update-and-simulate
configs/
├── name_map.yaml                # api_name -> canonical (mantenido por el usuario)
└── live.yaml                    # base_url, competition_code, request_timeout, watch_minutes, stage_map
tests/
├── test_name_map.py
├── test_provider_fake.py
├── test_football_data_adapter.py
├── test_ingest_runner.py
└── test_scoreboard.py
```

**Responsabilidades aisladas:**
- `provider.py` — normaliza datos crudos de la API a `ProviderMatch`. No mapea nombres/stage.
- `name_map.py` — mapeo exacto API→canónico + stage; error duro si falta.
- `ingest.py` — fetch→mapea→**sobrescribe** `wc2026_results.yaml` (idempotente). Dueño del archivo.
- `scoreboard.py` — predicho vs real (métricas SP1). Out-of-sample.
- `runner.py` — **provider-agnostic**: recibe un `ResultsProvider`; orquesta ingest+simula+scoreboard; `watch`.
- `cli.py` — ensambla provider (env+config), predictor real (fit 1 vez) y runner.

Nueva dependencia: **`requests`** (en `pyproject.toml`).

---

## 3. Flujo

```
football-data.org /v4/competitions/WC/matches?status=FINISHED
        │  header X-Auth-Token: $FOOTBALL_DATA_API_KEY ; timeout
        ▼
FootballDataProvider.fetch_finished() -> [ProviderMatch]
        ▼  name_map.map_team (error si falta) + map_stage (error si desconocido)
ingest -> [played_matches] -> sobrescribe configs/tournaments/wc2026_results.yaml (idempotente)
        ▼
TournamentRunner.cycle(provider):
   ingest -> load_results -> simulate_tournaments(sampler) -> aggregate    (probabilidades torneo)
                          \-> scoreboard(predictor, played)                (predicho vs real)
        ▼  {played, aggregate, scoreboard, meta}
CLI imprime (o --json); --watch N -> repite cada N min (fit del modelo 1 sola vez)
```

---

## 4. `provider.py`

```python
from dataclasses import dataclass
from typing import Protocol
import os, requests


@dataclass
class ProviderMatch:
    api_match_id: str          # string: dedup/logs/trazabilidad
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    stage: str                 # crudo: GROUP_STAGE, LAST_32, ...
    group: str | None          # crudo "GROUP_A" o None en knockout
    status: str                # "FINISHED"


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

Clave solo de env. `timeout>0` validado. `base_url` normalizado. JSON inválido → error claro.
`ResultsProvider` es `Protocol` → `FakeProvider` en tests sin herencia.

---

## 5. `name_map.py`

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

- Solo nombres que difieren van en `name_map.yaml`; el resto pasa y se valida contra los 48
  equipos de la estructura. Sin fuzzy: exacto y duro.
- `stage_map` se valida como dict al cargar `live.yaml` (en CLI/loader): si no es dict → ValueError.

`configs/name_map.yaml`:
```yaml
teams:
  "USA": "United States"
  "Korea Republic": "South Korea"
  "IR Iran": "Iran"
  "Côte d'Ivoire": "Ivory Coast"
```

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

---

## 6. `ingest.py` (idempotente)

La API es la fuente de verdad; `ingest` **reconstruye** la lista y **sobrescribe** el archivo
(no append/merge) → idempotente por construcción. Dedup por `api_match_id`, orden estable.

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

Output compatible con SP3 (`results.load_results` lo parsea, ignorando el comentario header).
El archivo es **propiedad de `ingest`** (sobrescribe entradas manuales). Header advierte no editar.

---

## 7. `scoreboard.py` (predicho vs real, en vivo)

Para cada partido jugado, el modelo (fit pre-torneo, **out-of-sample**) predice y se compara
con el real. Reusa `predict()` + `footy/metrics.py`. **No reentrenar con resultados del torneo**
(anti-leakage).

```python
from footy.metrics import log_loss_1x2, brier_1x2, accuracy_1x2


def _outcome(ga, gb):
    return "home" if ga > gb else "away" if ga < gb else "draw"


def scoreboard(predictor, played_matches: list) -> dict:
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
        probs.append(p); actuals.append(actual)
        goal_err += abs(pred["expected_goals_a"] - m["goals_a"]) + abs(pred["expected_goals_b"] - m["goals_b"])
        details.append({
            "match": f"{m['team_a']} vs {m['team_b']}",
            "predicted_score": pred["most_likely_score"],
            "actual_score": f"{m['goals_a']}-{m['goals_b']}",
            "predicted_outcome": predicted, "actual_outcome": actual,
            "predicted_prob": round(p[predicted], 4),   # confianza del pick
            "actual_prob": round(p[actual], 4),          # prob asignada al real (explica log-loss)
            "hit": predicted == actual,
        })
    n = len(played_matches)
    return {"n": n,
            "accuracy": accuracy_1x2(probs, actuals),
            "log_loss": log_loss_1x2(probs, actuals),
            "brier": brier_1x2(probs, actuals),
            "goal_mae": round(goal_err / (2 * n), 3),   # MAE por equipo-partido (2 equipos x n)
            "matches": details}
```

`goal_mae` es el **MAE promedio por equipo-partido** (suma de |λ−real| de ambos equipos ÷ 2n).
Sin partidos jugados → métricas `None`. Out-of-sample honesto.

---

## 8. `runner.py` (provider-agnostic) + CLI + watch

Fit del modelo real **una sola vez**; el watch reusa sampler (λ cacheadas) y predictor.

```python
import time
from footy.tournament.results import load_results
from footy.tournament.simulator import simulate_tournaments
from footy.tournament.aggregate import aggregate
from footy.live.ingest import ingest
from footy.live.scoreboard import scoreboard


class TournamentRunner:
    """Provider-agnostic: recibe un ResultsProvider en cada cycle."""
    def __init__(self, structure, name_map, stage_map, results_path, sampler, predictor, n, seed):
        self.structure = structure; self.name_map = name_map; self.stage_map = stage_map
        self.results_path = results_path; self.sampler = sampler; self.predictor = predictor
        self.n = n; self.seed = seed

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
    interval_seconds = interval_minutes * 60
    try:
        while True:
            emit(runner.cycle(provider))
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        return
```

`scoreboard` recibe los played como dicts (conversión inline de `PlayedMatch` arriba).

**CLI `update-and-simulate`:** construye predictor real (fit 1 vez) → `MatchSampler` →
canonicaliza equipos de la estructura → `TournamentRunner`; provider vía
`FootballDataProvider.from_config(live_cfg)`. Un disparo o `--watch N` (minutos). `--json`
imprime el dict completo; sin `--json`, resumen legible (top campeón + métricas scoreboard).

```
update-and-simulate
update-and-simulate --watch 15
update-and-simulate --json
```

---

## 9. Error handling

| Capa | Comportamiento |
|---|---|
| `provider` | falta `FOOTBALL_DATA_API_KEY` → ValueError; `timeout<=0` → ValueError; HTTP error → `raise_for_status`; JSON inválido → ValueError claro |
| `name_map` | `teams` no-dict → ValueError; equipo sin mapear → ValueError accionable |
| `stage_map` | no-dict → ValueError; stage desconocido → ValueError |
| `ingest` | score None → ValueError; grupo faltante en group-stage → ValueError; sobrescribe idempotente |
| `runner` | equipo canónico ausente del modelo → ValueError (vía predict/sampler) |
| `watch` | `KeyboardInterrupt` → salida limpia |

---

## 10. Testing (sin red real; determinismo por seed)

| Test | Verifica |
|---|---|
| `test_name_map` | carga (dict ok; no-dict → error); `map_team` hit/passthrough/unmapped→error; `map_stage` hit/desconocido→error |
| `test_provider_fake` | `FakeProvider` (canned) → `ingest` arma played list correcta; **idempotente** (2x → archivo idéntico, header incluido) |
| `test_football_data_adapter` | `requests.get` monkeypatched: campos `ProviderMatch` ok; `raise_for_status` propaga; JSON inválido → ValueError; `from_config` sin env → error; `timeout<=0` → error; `base_url` rstrip |
| `test_ingest_runner` | `ingest` escribe YAML que `results.load_results` parsea (contrato SP3) **y trae el header** `# GENERATED ...`; `TournamentRunner.cycle` (FakeProvider + FakeSampler + predictor fake + estructura mini) → `aggregate` + `scoreboard` + `meta`; `watch` con provider que lanza KeyboardInterrupt → retorna limpio |
| `test_scoreboard` | predictor fake + played → accuracy/log_loss/brier correctos; `predicted_prob`/`actual_prob`/`hit` por partido; lista vacía → métricas None |

---

## 11. Decisiones registradas

1. Enfoque A: provider pluggable + football-data adapter + name_map + ingest + runner watch, con `requests`.
2. `ResultsProvider` Protocol; `FakeProvider` en tests; adapter testeado con HTTP mock (sin red real).
3. Clave API solo por `FOOTBALL_DATA_API_KEY` (env). `timeout>0` obligatorio. `base_url.rstrip("/")`. JSON inválido → error claro.
4. `name_map.yaml` solo diferencias; mapeo exacto y duro (sin fuzzy). `stage_map` con error duro. Ambos validan ser dict.
5. `ProviderMatch.api_match_id: str` (dedup/traza).
6. `ingest` reconstruye y sobrescribe `wc2026_results.yaml` (idempotente), dueño del archivo, con header "do not edit"; orden estable `(stage, group, match_id)`; score None / grupo faltante → error.
7. `runner` provider-agnostic; fit del modelo 1 vez; `cycle` devuelve played/aggregate/scoreboard/meta; `watch` en minutos, corta con KeyboardInterrupt; CLI `--json`.
8. `scoreboard`: predicho vs real out-of-sample (no reentrenar con torneo); reusa `metrics.py`; `predicted_prob`/`actual_prob`; `goal_mae` = MAE por equipo-partido.
9. Aviso: `wc2026.yaml` (grupos reales) + `name_map`/`stage_map` deben mantenerse al día; los errores duros señalan desajustes.
