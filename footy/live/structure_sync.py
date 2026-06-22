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
