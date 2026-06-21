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
