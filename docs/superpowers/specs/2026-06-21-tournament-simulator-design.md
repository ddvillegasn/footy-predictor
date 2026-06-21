# Spec — Simulador de torneo · Sub-proyecto 3

- **Fecha:** 2026-06-21
- **Estado:** En revisión
- **Enfoque aprobado:** A — Monte Carlo de torneo completo (jugar el torneo entero N veces, reusando el motor SP1 y las cuotas SP2)
- **Depende de:** SP1 (motor Dixon-Coles + `simulate_goals` + `Predictor`/`model.rates`) y SP2 (`betting/odds.py`, `betting/value.py`). Ambos completos, 83 tests verdes.

---

## 1. Contexto

El usuario coloca un torneo (genérico, instanciado con el Mundial 2026) y obtiene
probabilidades de avance y campeón, distribución de grupos, bracket más probable y cuotas
de torneo. El simulador es un **pronóstico condicionado**: si el torneo ya está en curso,
los **resultados oficiales jugados se fijan** y solo se samplean los partidos pendientes.

Motor genérico **config-driven** (no hardcodeado al Mundial): lee estructura y resultados
de archivos. Reusa el motor de partido SP1 (no se duplica) y las cuotas/valor SP2.

**No incluye:** tabla FIFA oficial de 495 combinaciones para terceros (se usa orden ranked
config-driven; tabla oficial = v2); factor must-win/importancia (research); scraping.

---

## 2. Arquitectura y módulos

```
footy/
├── tournament/                 # NUEVO
│   ├── __init__.py
│   ├── structure.py            # cargar/validar TournamentConfig (estructura)
│   ├── results.py              # cargar/validar TournamentResults (partidos jugados)
│   ├── sampler.py              # MatchSampler: marcadores por emparejamiento (unica fuente de azar)
│   ├── groups.py               # tabla + desempates FIFA + ranking mejores terceros (funciones puras)
│   ├── knockout.py             # bracket + resolucion de partido ET/penales (funciones puras)
│   ├── simulator.py            # corre N torneos (fija jugados, samplea pendientes)
│   └── aggregate.py            # conteo -> probabilidades (solo cuenta)
├── betting/odds.py, value.py   # REUSO para cuotas/valor de torneo
configs/
├── tournaments/
│   ├── wc2026.yaml             # estructura fija
│   └── wc2026_results.yaml     # estado vivo (partidos jugados)
└── tournament_sim.yaml         # n_tournaments, seed, neutral_default
tests/
├── test_tournament_structure.py
├── test_results_loader.py
├── test_groups.py
├── test_knockout.py
├── test_simulator.py
├── test_aggregate.py
└── test_tournament_odds.py
```

**Responsabilidades aisladas:**
- `structure.py` — parsea/valida `TournamentConfig`. No simula.
- `results.py` — parsea/valida `TournamentResults` (partidos jugados). No simula.
- `sampler.py` — única fuente de aleatoriedad; marcadores desde `model.rates` + Poisson seedeado.
- `groups.py`, `knockout.py` — funciones puras (entran resultados, sale orden/ganador); rng inyectado.
- `simulator.py` — orquesta N torneos; fija jugados, samplea pendientes.
- `aggregate.py` — solo cuenta muestras → probabilidades.
- Cuotas: `betting/odds.py` + `value.py` de SP2.

---

## 3. `TournamentConfig` (`configs/tournaments/wc2026.yaml`)

```yaml
name: "FIFA World Cup 2026"
neutral_default: true
points: {win: 3, draw: 1, loss: 0}
groups:
  A: [Mexico, TeamA2, TeamA3, TeamA4]
  # ... hasta L (12 grupos x 4 = 48)
group_schedule: round_robin        # auto-genera los 6 partidos por grupo
qualification:
  per_group_advance: 2             # 1ro y 2do (24)
  best_thirds: 8                   # 8 mejores terceros de 12
tiebreakers:                       # FIFA, en orden
  - points
  - goal_difference
  - goals_for
  - head_to_head                   # mini-tabla entre empatados (pts, GD, GF)
  - fair_play                      # NEUTRAL: sin datos de tarjetas, no desempata
  - drawing_of_lots                # fallback final REPRODUCIBLE (seedeado), no aleatorio libre
thirds_ranking: [points, goal_difference, goals_for, drawing_of_lots]
knockout:
  rounds: [R32, R16, QF, SF, F]    # sin 3er puesto
  bracket_r32:                     # 16 cruces por referencia de slot
    - [winner_A, third_slot_1]
    - [runner_A, runner_B]
    # ...
  thirds_assignment: ranked_order  # terceros rankeados -> third_slot_1..8 (tabla FIFA 495 = v2)
```

Validación: 12 grupos × 4, equipos únicos y **canónicos/en dataset**, refs de bracket
resolubles. `fair_play` documentado como neutral (no computable). `drawing_of_lots` es el
fallback final y es **reproducible** (depende del seed del torneo), no azar libre.

---

## 4. `TournamentResults` (`configs/tournaments/wc2026_results.yaml`)

Estado vivo, separado de la estructura. Se actualiza cada jornada sin tocar `wc2026.yaml`.

```yaml
played_matches:
  - match_id: A1            # clave primaria cuando exista
    stage: group            # group | R32 | R16 | QF | SF | F
    group: A                # solo para stage=group
    team_a: Germany
    team_b: CostaRica
    goals_a: 3
    goals_b: 1
  - match_id: A2
    stage: group
    group: A
    team_a: Ecuador
    team_b: Curacao
    goals_a: 2
    goals_b: 0
```

**Regla de fijado (en `simulator.py`):**
- Un partido jugado se identifica por **`match_id` cuando existe**.
- Para grupos: el fixture es fijo → siempre matchea por (grupo, par de equipos).
- Para knockout: validar **`stage` + `teams`**, porque el cruce depende del bracket
  simulado. Un played result de knockout se aplica **solo** en torneos donde esos dos
  equipos llegan a ese cruce/etapa; si no, se simula.
- Si el partido está en results → usar marcador real (no llamar a `sampler`); si no → simular.

Validación: marcador mal formado → error; equipos fuera de su grupo/estructura → error;
`match_id` duplicado → error.

---

## 5. `sampler.py` — única fuente de azar

- `MatchSampler(model, mc_config, model_version, config_hash)`.
- `.scorelines(team_a, team_b, neutral, n) -> (goals_a[], goals_b[])`: usa `model.rates`
  + Poisson seedeado (reusa la lógica de `simulate_goals`).
- **Cache: solo λ**, por `(team_a, team_b, neutral, model_version, config_hash)`. **No se
  cachean marcadores** (evita correlaciones espurias entre torneos).
- `.lambdas(team_a, team_b, neutral) -> (λ_a, λ_b)`: para escalar prórroga y ponderar penales.

---

## 6. `groups.py` — tabla + desempates FIFA + terceros (puro)

- `group_table(results, points_cfg) -> standings[]`: W/D/L, GF, GA, GD, Pts por equipo.
- `rank_group(standings, results, tiebreakers, rng) -> [equipos ordenados]`:
  1. Pts → GD → GF (global).
  2. Empate persistente → mini-tabla **head-to-head** solo entre empatados (Pts→GD→GF).
  3. `fair_play` → neutral (no desempata).
  4. `drawing_of_lots` → sorteo seedeado vía `rng` (fallback final reproducible).
- `rank_thirds(thirds_standings, thirds_ranking_cfg, rng) -> [terceros ordenados]`:
  ordena los 12 terceros y devuelve los 8 mejores en orden ranked (para `third_slot_i`).

---

## 7. `knockout.py` — bracket + resolución (puro)

- `build_bracket(group_ranks, thirds_ranked, bracket_cfg) -> [cruces R32]`: resuelve refs
  de slot (`winner_A`, `runner_B`, `third_slot_1..8`) a equipos; terceros en orden ranked.
- `resolve_match(team_a, team_b, sampled_regulation, lambdas, rng) -> ganador`:
  1. Marcador reglamentario (provisto por `sampler`); si no es empate → gana el de más goles.
  2. Empate → **prórroga**: goles extra sampleados con λ escalada (≈ 30/90 de la λ); sumar.
  3. Sigue empate → **penales ponderados por fuerza**: `P(A) = λ_a/(λ_a+λ_b)` con clipping
     a `[0.05, 0.95]`, decidido vía `rng`.
- `play_round(cruces, sampler, rng) -> [ganadores]`; encadenar R32→R16→QF→SF→F → campeón.

---

## 8. `simulator.py` — N torneos

- `run_tournament(structure, results, sampler, rng) -> TournamentResult`: juega grupos
  (fija jugados, samplea pendientes), arma tabla (`groups.py`), clasificados + mejores
  terceros, bracket (`knockout.py`), knockout → campeón + ronda alcanzada por equipo +
  posiciones de grupo. Guarda `run_id`/`tournament_seed` reproducible.
- `TournamentResult` incluye, por equipo: ronda alcanzada, posición de grupo, y
  `group_stage_points`, `group_stage_goal_diff`, `group_stage_goals_for` (debug/explicabilidad).
- `simulate_tournaments(structure, results, sampler, n, seed) -> [TournamentResult]`.
- **Rendimiento:** grupos vectorizados (N marcadores por fixture pendiente de una vez);
  knockout iterado por torneo con λ cacheada. Vectorizar knockout = mejora posterior.

---

## 9. `aggregate.py` — solo cuenta

Sobre los N `TournamentResult`:
- Por equipo: `P(pasa de grupo)`, `P(llega a R16/QF/SF/Final)`, `P(campeón)` = conteo/N.
- Por grupo: `P(1ro/2do/3ro/4to)` por equipo.
- **`slot_outcome_frequency`**: por slot/ronda/cruce, frecuencia de cada resultado (no el
  bracket completo más frecuente, que es muy disperso).
- Consistencia (tests): posiciones de grupo suman 1; `sum P(campeón)=1`; `P(ronda)`
  monótona decreciente por equipo.

### Cuotas de torneo (reuso SP2)
- Probabilidades → `fair_odds`: campeón outright, "llega a la final", "pasa de grupo".
- Valor opcional: cuota outright de bookie → `assess_value` (edge%, EV, stake). Contrato
  honesto de SP2 (EV relativo al modelo; mostrar fiabilidad).

### Salida final (dict)
```json
{
  "teams": {"Brazil": {"advance_group": 0.91, "reach_R16": 0.78, "reach_QF": 0.55,
                       "reach_SF": 0.34, "reach_final": 0.21, "champion": 0.12}, "...": {}},
  "groups": {"A": {"Germany": {"p1": 0.62, "p2": 0.24, "p3": 0.10, "p4": 0.04}}},
  "slot_outcome_frequency": {"R32_tie_1": {"Germany": 0.55, "...": 0.45}},
  "odds": {"champion": {"Brazil": {"prob": 0.12, "fair_odds": 8.33}}},
  "value": {"champion": {"Brazil": {"book_odds": 9.0, "ev_per_unit": 0.08, "is_value": true,
                                    "stake_recommendation": "small"}}},
  "meta": {"n_tournaments": 20000, "seed": 42, "model_version": "baseline-v1.0.0",
           "betting_config_version": "a1b2c3d4", "played_matches_count": 18}
}
```

---

## 10. Error handling

| Capa | Comportamiento |
|---|---|
| `structure` | 12 grupos × 4, equipos únicos/canónicos, refs de bracket resolubles → error si no |
| `results` | marcador mal formado / equipos fuera de estructura / `match_id` duplicado → error |
| `sampler` | equipo desconocido → `ValueError`; λ≤0 ya cubierto |
| resolución | played result aplicado por `match_id` (o grupo+par; knockout por stage+teams) |
| `aggregate` | clasificados incompletos (config rota) → error antes de contar |

---

## 11. Testing

Determinismo: todo vía `rng`/seed. Tests usan torneo **mini** (p. ej. 2 grupos × 4) para
velocidad, no el Mundial completo.

| Test | Verifica |
|---|---|
| `test_tournament_structure` | config válida carga; tamaño de grupo malo / equipo duplicado / slot ref roto → error |
| `test_results_loader` | parsea played_matches; marcador inválido / equipo fuera de estructura / id duplicado → error |
| `test_groups` | tabla; **cada** desempate FIFA (Pts→GD→GF, mini-tabla H2H, sorteo seedeado reproducible); ranking terceros + selección de 8 |
| `test_knockout` | `build_bracket` resuelve slots (incl. orden terceros); `resolve_match`: ganador claro; empate→prórroga; empate+prórroga→penales; determinismo dado rng |
| `test_simulator` | torneo mini corre; **played result fija marcador** (determinista, refleja en tabla); resto sampleado; `run_id` reproducible; campos de debug presentes |
| `test_aggregate` | posiciones suman 1; `sum P(campeón)=1`; `P(ronda)` monótona; `slot_outcome_frequency` cuenta |
| `test_tournament_odds` | probs → cuotas justas (SP2); valor con cuota outright de bookie |

**Penales ponderados:** no probar un solo caso — probar **distribución sobre muchos seeds**
(el favorito pasa más que 50%, y el clipping `[0.05,0.95]` se respeta en extremos).

---

## 12. Decisiones registradas

1. Enfoque A: Monte Carlo de torneo completo, reusa motor SP1 y cuotas SP2.
2. Motor genérico config-driven; instanciado con WC2026 (12 grupos × 4, 32 a knockout).
3. Desempates FIFA completos con mini-tabla H2H; `fair_play` neutral (sin datos); sorteo = fallback reproducible seedeado.
4. Mejores terceros: orden ranked config-driven (tabla FIFA 495 = v2).
5. Knockout: empate → prórroga (λ escalada) → penales ponderados `λ_a/(λ_a+λ_b)` con clipping `[0.05,0.95]`.
6. **Resultados en vivo**: `TournamentResults` en archivo aparte; jugados se fijan (no se samplean); identificación por `match_id` (knockout valida stage+teams).
7. `sampler` cachea solo λ (clave incluye model_version + config_hash); nunca marcadores.
8. `TournamentResult` guarda puntos/GD/GF de grupo (debug/explicabilidad).
9. Agregación = `slot_outcome_frequency` (no bracket completo más frecuente).
10. `aggregate.py` solo cuenta; `groups.py`/`knockout.py` puras; `sampler.py` única fuente de azar.
11. Cuotas/valor de torneo reusan `betting/odds.py` y `value.py` de SP2 (contrato honesto).
12. Sin 3er puesto, sin must-win, sin scraping en este sub-proyecto.
