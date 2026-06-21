from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import requests


@dataclass
class ProviderMatch:
    api_match_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    stage: str
    group: str | None
    status: str


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
