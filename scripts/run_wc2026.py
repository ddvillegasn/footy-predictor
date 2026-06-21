"""One-off runner: real WC2026 tournament simulation.

Wires the real Dixon-Coles model (SP1) + MatchSampler + simulator + aggregate (SP3).
Not a committed feature yet; a proper `simulate-tournament` CLI would wrap this.
"""
from pathlib import Path

from footy.cli import _build_default_predictor
from footy.config import load_config, config_fingerprint
from footy.tournament.structure import load_structure
from footy.tournament.results import load_results
from footy.tournament.sampler import MatchSampler
from footy.tournament.simulator import simulate_tournaments
from footy.tournament.aggregate import aggregate, tournament_odds

ROOT = Path(__file__).resolve().parent.parent
N = 1500

print("Fitting real model (Dixon-Coles on full dataset)...")
predictor = _build_default_predictor()

structure = load_structure(ROOT / "configs" / "tournaments" / "wc2026.yaml")
# Canonicalize team names so they match the fitted model's keys.
model_teams = set(predictor.model.attack.keys())
missing = []
for g, teams in structure.groups.items():
    canon = []
    for t in teams:
        c = predictor.canonical(t)
        if c not in model_teams:
            missing.append((g, t, c))
        canon.append(c)
    structure.groups[g] = canon
if missing:
    print("WARNING: teams not in model (will error):", missing)

results = load_results(ROOT / "configs" / "tournaments" / "wc2026_results.yaml", structure.groups)

mc_cfg = load_config("montecarlo")
sampler = MatchSampler(predictor.model, mc_cfg,
                       model_version=load_config("model")["model_version"],
                       config_hash=config_fingerprint("montecarlo"))

print(f"Simulating {N} tournaments...")
sims = simulate_tournaments(structure, results, sampler, n=N, seed=42)
agg = aggregate(structure, sims)
out = tournament_odds(agg, book_odds=None, reliability=0.6, value_config=load_config("betting")["value"])

champ = sorted(agg["teams"].items(), key=lambda kv: -kv[1]["champion"])
print("\n=== Top 12 champion probabilities ===")
for team, d in champ[:12]:
    fair = out["odds"]["champion"].get(team, {}).get("fair_odds")
    print(f"{team:<18} champ {d['champion']*100:5.1f}%  reach_final {d['reach_F']*100:5.1f}%  "
          f"advance_group {d['advance_group']*100:5.1f}%  fair_odds {fair}")

print(f"\n=== Group A advance_group probs (played_matches={len(results.played)}) ===")
for t in structure.groups["A"]:
    print(f"{t:<18} advance {agg['teams'][t]['advance_group']*100:5.1f}%")
