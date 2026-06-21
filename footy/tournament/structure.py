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
