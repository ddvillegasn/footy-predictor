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
