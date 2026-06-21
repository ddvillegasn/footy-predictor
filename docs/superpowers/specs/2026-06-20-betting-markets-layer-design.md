# Spec — Capa de apuestas por partido · Sub-proyecto 2

- **Fecha:** 2026-06-20
- **Estado:** En revisión
- **Enfoque aprobado:** A (mercados desde muestras Monte Carlo) en producción + C (cross-check analítico) en testing
- **Depende de:** SP1 baseline (motor Dixon-Coles + Monte Carlo + `predict_match`), ya implementado y testeado (48 tests verdes).

---

## 1. Contexto

SP1 entrega un motor que predice **un partido suelto** entre dos selecciones (λ_a, λ_b
vía Dixon-Coles + Monte Carlo → 1X2, marcador, fiabilidad). Este sub-proyecto añade una
**capa de apuestas por partido** encima de ese motor, sin rehacerlo:

- El usuario coloca un partido (cualquiera) y obtiene, además de la predicción SP1,
  **mercados de apuestas** derivados de las mismas muestras MC.
- Cuotas justas, probabilidad implícita, márgenes.
- Detección de **valor/EV opcional** si el usuario pega la cuota de su bookie (BetPlay, etc.).

**No** incluye: simulador de torneo/grupos/bracket (sub-proyecto 3), scraping de cuotas
de casas, ni τ en el sampleo Monte Carlo (v2).

### Una sola fuente de verdad
`predict()` llama `simulate_goals()` **una vez**; los mismos arrays de goles alimentan el
1X2 de SP1 **y** todos los mercados. Mercados garantizados coherentes con la predicción.

---

## 2. Arquitectura y módulos

```
footy/
├── models/
│   └── montecarlo.py          # REFACTOR: extraer simulate_goals()
├── betting/                   # NUEVO paquete
│   ├── __init__.py
│   ├── markets.py             # mercados desde muestras MC (solo cuenta eventos)
│   ├── odds.py                # cuota justa, prob implicita, margen
│   └── value.py               # edge%, EV, kelly, stake vs cuota bookie opcional
├── predict.py                 # extender output con "markets" y "value" opcionales
└── cli.py                     # flags --markets y --book-odds
configs/
└── betting.yaml               # lineas O/U, lineas handicap, top_scores, umbrales valor
tests/
├── test_simulate_goals.py
├── test_markets.py            # consistencia + cross-check analitico (enfoque C)
├── test_odds.py
├── test_value.py
└── test_predict_markets.py    # integracion
```

**Responsabilidades aisladas (cada módulo testeable solo):**
- `markets.py` — recibe `(goals_a, goals_b, config)` → dict de probabilidades por mercado.
  **No sabe de cuotas. No recalcula nada del modelo: solo opera sobre arrays ya generados.**
- `odds.py` — prob → cuota justa, prob implícita, margen. **No sabe de MC ni de bookies.**
- `value.py` — prob del modelo + cuota bookie → edge%, EV, kelly, stake. **No sabe de MC ni mercados.**

Flujo: `montecarlo.py` produce realidad simulada → `markets.py` cuenta eventos →
`odds.py` convierte probabilidades → `value.py` compara contra bookie.

### Refactor de `montecarlo.py`
```python
simulate_goals(lambda_a, lambda_b, config) -> (goals_a, goals_b, meta)
simulate(lambda_a, lambda_b, config) -> dict   # ahora usa simulate_goals internamente
```
- `meta = {"seed", "n_sims", "lambda_a", "lambda_b", "dc_enabled", "clip_max"}`.
- `dc_enabled` = si el **sampleo** aplica la corrección Dixon-Coles τ. **Hoy `False`**: las
  muestras son Poisson independiente; τ vive en el ajuste de λ (SP1), no en el sampleo.
  Documentado explícito para no engañar. τ en sampleo = v2.
- `simulate()` mantiene su salida actual EXACTA → los 11 tests de SP1 de Monte Carlo
  siguen verdes (refactor sin cambio de comportamiento observable).
- `λ_a ≤ 0` o `λ_b ≤ 0` → `ValueError`. (Se mantiene como error por ahora; permitir λ==0
  sería v2 si el modelo lo justificara.)

---

## 3. Motor de mercados (`markets.py`)

Recibe `(goals_a, goals_b, config)`. `total = goals_a + goals_b`, `n = len(goals_a)`.
`prob(evento) = mean(mascara_booleana)`. **Solo cuenta eventos sobre los arrays.**

| Mercado | Cálculo |
|---|---|
| **1X2** | `home = mean(a>b)` · `draw = mean(a==b)` · `away = mean(a<b)` |
| **Doble oportunidad** | `1X = mean(a>=b)` · `12 = mean(a!=b)` · `X2 = mean(a<=b)` |
| **Over/Under** (cada línea L) | `over = mean(total>L)` · `under = mean(total<L)`. Líneas .5 ⇒ sin push |
| **BTTS** | `yes = mean((a>0)&(b>0))` · `no = 1-yes` |
| **Marcador exacto** | conteo de `"x-y"`; se exponen los `top_scores` más probables |
| **Hándicap** (cada línea h, asiática .5) | `a_eff = a + h` · `home = mean(a_eff>b)` · `away = mean(a_eff<b)` |

**Hándicap — semántica explícita:** para línea `h` (ej. `-1.5`):
`home` = el equipo A cubre su hándicap `h` (ej. gana por 2+); `away` = el equipo B cubre
su contra-hándicap `+|h|`. Son complementarios. **Solo líneas asiáticas .5 (sin push).**
Líneas enteras (−1, 0, +1) = **v2** (requieren salida win/push/lose).

**Marcador exacto — salida** (output limpio, chequeo interno de masa):
```json
"correct_score": {
  "top": {"2-0": 0.12, "1-0": 0.11, "2-1": 0.09, "3-0": 0.08, "0-0": 0.07},
  "other_probability": 0.53,
  "all_mass_check": 1.0
}
```
`top` = `top_scores` más probables; `other_probability` = masa fuera del top-N
(`1 − sum(top)`), ayuda a leer cuánto peso quedó fuera; `all_mass_check` = suma de la
matriz completa (≈1.0), verificable en tests.

**Salida de `markets.py`** (probabilidades crudas, sin cuotas):
```json
{
  "1x2": {"home": 0.86, "draw": 0.09, "away": 0.05},
  "double_chance": {"1X": 0.95, "12": 0.91, "X2": 0.14},
  "over_under": {"0.5": {"over": 0.97, "under": 0.03}, "2.5": {"over": 0.71, "under": 0.29}},
  "btts": {"yes": 0.42, "no": 0.58},
  "correct_score": {"top": {"3-0": 0.12, "2-0": 0.11}, "other_probability": 0.77, "all_mass_check": 1.0},
  "handicap": {"-1.5": {"home": 0.55, "away": 0.45}}
}
```

**Consistencia (verificada en tests):** `home+draw+away=1`; `1X=home+draw`,
`X2=draw+away`, `12=home+away`; `over+under=1` por línea; `all_mass_check≈1`.

---

## 4. Cuotas (`odds.py`)

Pura aritmética. No sabe de MC ni de mercados.

```python
fair_odds(prob) -> float | None      # 1/prob; prob=0 -> None
implied_prob(decimal_odds) -> float  # 1/odds
model_margin(prob_list) -> float     # sum(probs_crudas) - 1  (fair odds del modelo)
market_margin(odds_list) -> float    # sum(1/odd) - 1          (cuotas reales de bookie)
decorate_market(prob_dict) -> dict   # cada outcome -> {prob, fair_odds}; agrega "margin"
```

- **Cuota justa** = `1/prob` (decimal). **Prob implícita** = `1/cuota`.
- **Margen del modelo** (fair) = `sum(probs_crudas) − 1` → se calcula con probabilidades
  **sin redondear** (evita falsos `0.0001` por rounding). Para fair odds ≈ 0.
- **Margen de bookie** = `sum(1/oddᵢ) − 1`, solo para cuotas reales (vig 5-7%).
- **Casos borde:** `prob=0` → `fair_odds=None`, `status="no_sim"`. `prob=1` → `1.0`.
  Probabilidades redondeadas a 4 decimales; cuotas a 2 (el redondeo es solo de
  presentación; el margen usa crudas).

**Salida decorada (ej. 1X2):**
```json
"1x2": {
  "home": {"prob": 0.86, "fair_odds": 1.16},
  "draw": {"prob": 0.09, "fair_odds": 11.11},
  "away": {"prob": 0.05, "fair_odds": 20.0},
  "margin": 0.0
}
```

---

## 5. Valor / EV (`value.py`)

Compara prob del modelo contra la cuota de bookie del usuario. Solo corre para outcomes
con cuota dada. No sabe de MC ni mercados.

```python
assess_value(model_prob, book_odds, reliability, config) -> dict
assess_market(prob_dict, book_odds_dict, reliability, config) -> dict
```

**Matemática (cuota decimal):**
- `book_implied = 1 / book_odds`.
- `prob_edge = model_prob − book_implied`.
- `ev_per_unit = model_prob × book_odds − 1`. `edge_pct = ev_per_unit × 100`.
- `kelly_fraction_raw = (model_prob × book_odds − 1) / (book_odds − 1)`; ≤0 → 0.
- `kelly_fraction_quarter = kelly_fraction_raw / 4` (más seguro para uso real).
- `is_value = ev_per_unit > threshold` (`threshold` en `betting.yaml`, default 0).
- **`stake_recommendation`** ∈ `{"skip","small","medium"}`, basado en **EV + fiabilidad**
  (no solo EV):
  - `skip`: `ev_per_unit ≤ 0` **o** `reliability < reliability_low`.
  - `small`: `ev_per_unit > 0` con fiabilidad media (o EV pequeño bajo `ev_medium`).
  - `medium`: `ev_per_unit > ev_medium` **y** `reliability ≥ reliability_high`.
  - Umbrales `reliability_low`, `reliability_high`, `ev_medium` en `betting.yaml`.

**Salida por outcome con cuota:**
```json
"home": {
  "model_prob": 0.784, "fair_odds": 1.28,
  "book_odds": 1.45, "book_implied": 0.690,
  "edge_pct": 13.7, "ev_per_unit": 0.137,
  "kelly_fraction_raw": 0.30, "kelly_fraction_quarter": 0.075,
  "is_value": true, "stake_recommendation": "small"
}
```

**Honestidad (en spec y salida):**
- EV asume que la prob del modelo es correcta → es valor *relativo al modelo*, NO ganancia
  garantizada. Se muestra junto al `prediction_reliability` de SP1; fiabilidad baja ⇒ el
  valor vale poco.
- Kelly raw puede salir alto; por eso se expone también `quarter`. Nota de gestión de
  riesgo, no consejo financiero.

**Error handling:** `book_odds ≤ 1.0` → `ValueError`. Outcome sin cuota → omitido (queda
fair only). Cuota para mercado/outcome inexistente → ignorada + warning.

---

## 6. Integración en `predict()` y salida completa

`predict()` gana 2 params opcionales: `include_markets=False`, `book_odds=None`.
- Sin `include_markets` → salida **idéntica a SP1** (compat total; no aparecen `markets`
  ni `value`).
- `include_markets=True`, sin `book_odds` → aparece `markets` + `simulation_meta`, sin `value`.
- Con `book_odds` (implica markets) → aparece `value` solo en los outcomes/mercados dados.
- `predict()` usa `simulate_goals()` una vez; reusa los arrays para 1X2 (SP1) y mercados.

**Versionado y trazabilidad** (separa motor predictivo de capa de apuestas):
- `betting_version` = `"sp2-v1.0.0"` (SemVer propio de la capa; cambia con líneas, Kelly,
  hándicaps o nuevos mercados, independiente de `model_version`).
- `simulation_meta.betting_config_version` = hash corto (sha1, 8 chars) del contenido
  resuelto de `betting.yaml`. Permite reproducir un resultado aunque `betting.yaml` cambie
  después. Solo aparece cuando hay `markets`.

**`book_odds` acepta dict JSON anidado** (y el string CLI se parsea a esa estructura):
```json
{ "1x2": {"home": 1.45, "draw": 4.2}, "over_under": {"2.5": {"over": 1.67}} }
```

**Salida completa (ejemplo Alemania vs Costa de Marfil, neutral, con book_odds en 1X2):**
```json
{
  "team_a": "Germany", "team_b": "Ivory Coast",
  "team_a_win": 78.4, "draw": 14.1, "team_b_win": 7.5,
  "expected_goals_a": 2.1, "expected_goals_b": 0.6,
  "most_likely_score": "2-0",
  "prediction_reliability": 0.74,
  "model_version": "baseline-v1.0.0",
  "betting_version": "sp2-v1.0.0",
  "simulation_meta": {"seed": 42, "n_sims": 100000, "lambda_a": 2.1,
    "lambda_b": 0.6, "dc_enabled": false, "clip_max": 10,
    "betting_config_version": "a1b2c3d4"},
  "markets": {
    "1x2": {"home": {"prob": 0.784, "fair_odds": 1.28},
            "draw": {"prob": 0.141, "fair_odds": 7.09},
            "away": {"prob": 0.075, "fair_odds": 13.33}, "margin": 0.0},
    "double_chance": {"1X": {"prob": 0.925, "fair_odds": 1.08}, "...": "..."},
    "over_under": {"2.5": {"over": {"prob": 0.61, "fair_odds": 1.64},
                           "under": {"prob": 0.39, "fair_odds": 2.56}}},
    "btts": {"yes": {"prob": 0.41, "fair_odds": 2.44},
             "no": {"prob": 0.59, "fair_odds": 1.69}},
    "correct_score": {"top": {"2-0": {"prob": 0.16, "fair_odds": 6.25}}, "all_mass_check": 1.0},
    "handicap": {"-1.5": {"home": {"prob": 0.52, "fair_odds": 1.92},
                          "away": {"prob": 0.48, "fair_odds": 2.08}}}
  },
  "value": {
    "1x2": {
      "home": {"model_prob": 0.784, "fair_odds": 1.28, "book_odds": 1.45,
        "book_implied": 0.690, "edge_pct": 13.7, "ev_per_unit": 0.137,
        "kelly_fraction_raw": 0.30, "kelly_fraction_quarter": 0.075,
        "is_value": true, "stake_recommendation": "small"}
    }
  }
}
```

**CLI:**
```bash
predict Germany "Ivory Coast" --markets
predict Germany "Ivory Coast" --markets --book-odds 1x2.home=1.45,1x2.draw=4.2
```

---

## 7. Configuración (`configs/betting.yaml`)

```yaml
over_under_lines: [0.5, 1.5, 2.5, 3.5, 4.5]
handicap_lines: [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
top_scores: 5
value:
  threshold: 0.0          # EV minimo para is_value
  ev_medium: 0.10         # EV por encima del cual stake puede ser "medium"
  reliability_low: 0.40   # debajo -> stake "skip"
  reliability_high: 0.70  # encima (con EV) -> "medium"
  kelly_quarter_divisor: 4
```

---

## 8. Testing (A en producción + C cross-check en tests)

Fixtures: arrays deterministas (seed fijo) y λ conocidos.

| Test | Verifica |
|---|---|
| `test_simulate_goals` | largo = n_sims; meta correcta; seed → determinismo; clip respetado; `λ≤0` lanza |
| `test_markets` (consistencia) | `home+draw+away≈1`; identidades doble oportunidad; `over+under=1`; `all_mass_check≈1` |
| `test_markets` (**cross-check C**) | MC vs cerrado: `btts.yes ≈ (1−e^−λa)(1−e^−λb)`; `over 0.5 ≈ 1−e^−(λa+λb)`; U/O via `total~Poisson(λa+λb)`; tolerancia 0.02 |
| `test_odds` | `fair_odds=1/prob`; `implied_prob=1/cuota`; `model_margin=sum(probs)−1≈0`; `prob=0→None/no_sim`; `prob=1→1.0` |
| `test_value` | edge/EV; `kelly_quarter=raw/4`; `stake_recommendation` por EV+fiabilidad; `book_odds≤1` lanza; `is_value` |
| `test_predict_markets` (integración) | `include_markets` añade `markets`+`simulation_meta`; `book_odds` añade `value` solo en outcomes dados; sin flags = salida SP1 idéntica; `book_odds` acepta dict JSON y string CLI |

**Validez del cross-check analítico:** las fórmulas cerradas asumen **Poisson independiente
con `dc_enabled=False`** (sampleo actual). Si v2 introduce τ en el sampleo, estos tests
deben **migrar de fórmula cerrada a consistencia interna** (comparar mercados entre sí y
contra el 1X2, sin asumir Poisson independiente). Marcado explícito en el test.

---

## 9. Decisiones registradas

1. Enfoque A (mercados desde muestras MC) en producción; C (cross-check analítico) en tests.
2. `simulate_goals()` devuelve `(goals_a, goals_b, meta)`; `simulate()` lo reusa sin cambiar su salida.
3. `dc_enabled=False` hoy (sampleo Poisson independiente); τ en sampleo = v2.
4. Mercados: 1X2, doble oportunidad, O/U, BTTS, marcador exacto (top + mass check), hándicap asiático .5.
5. Hándicap solo líneas .5 (sin push); enteras = v2 (win/push/lose).
6. Cuota justa = 1/prob; margen del modelo desde probs crudas; margen de bookie aparte.
7. Valor opcional: requiere cuota del usuario; edge/EV/kelly(raw+quarter)/stake_recommendation (EV+fiabilidad).
8. `predict()` extendido con flags opcionales; sin ellos, compat SP1 total.
9. `book_odds` acepta dict JSON y string CLI parseado a la misma estructura.
10. `λ ≤ 0` = error por ahora.
11. SP1 docs + hold-out (Task 17/18) + Gate 4 quedaron diferidos por el pivote; retomar tras SP2.
12. `betting_version` separado de `model_version`; `betting_config_version` = hash de `betting.yaml` en `simulation_meta`.
13. `correct_score` expone `other_probability` (masa fuera del top-N) además de `top` y `all_mass_check`.
14. Cross-check analítico migra a consistencia interna si v2 mete τ en el sampleo.
