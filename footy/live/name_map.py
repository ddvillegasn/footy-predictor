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
