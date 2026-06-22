import pytest

import footy.live.provider as prov

STRUCT_PAYLOAD = {"matches": [
    {"id": 1, "stage": "GROUP_STAGE", "group": "GROUP_A",
     "homeTeam": {"name": "Mexico"}, "awayTeam": {"name": "South Africa"},
     "score": {"fullTime": {"home": 2, "away": 0}}, "status": "FINISHED"},
    {"id": 2, "stage": "GROUP_STAGE", "group": "GROUP_A",
     "homeTeam": {"name": "South Korea"}, "awayTeam": {"name": "Czechia"},
     "score": {"fullTime": {"home": 2, "away": 1}}, "status": "FINISHED"},
    {"id": 3, "stage": "LAST_16", "group": None,
     "homeTeam": {"name": "Mexico"}, "awayTeam": {"name": "South Korea"},
     "score": {"fullTime": {"home": None, "away": None}}, "status": "TIMED"},
]}


class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_structure_groups_by_group(monkeypatch):
    monkeypatch.setattr(prov.requests, "get", lambda *a, **k: FakeResp(STRUCT_PAYLOAD))
    p = prov.FootballDataProvider("key", "https://x/v4", "WC", 10)
    groups = p.fetch_structure()
    assert set(groups.keys()) == {"GROUP_A"}                 # knockout match ignored
    assert groups["GROUP_A"] == ["Mexico", "South Africa", "South Korea", "Czechia"]


def test_from_config_reads_key_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    secrets = tmp_path / "secrets.local.yaml"
    secrets.write_text("football_data_api_key: filekey123\n", encoding="utf-8")
    p = prov.FootballDataProvider.from_config(
        {"base_url": "https://x/v4", "competition_code": "WC", "request_timeout": 10},
        secrets_path=secrets)
    assert p.api_key == "filekey123"


def test_from_config_missing_everywhere_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FOOTBALL_DATA_API_KEY"):
        prov.FootballDataProvider.from_config(
            {"base_url": "https://x/v4", "competition_code": "WC", "request_timeout": 10},
            secrets_path=tmp_path / "nope.yaml")
