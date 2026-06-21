import pytest

import footy.live.provider as prov

PAYLOAD = {"matches": [
    {"id": 111, "homeTeam": {"name": "Brazil"}, "awayTeam": {"name": "Mexico"},
     "score": {"fullTime": {"home": 2, "away": 1}},
     "stage": "GROUP_STAGE", "group": "GROUP_A", "status": "FINISHED"},
    {"id": 112, "homeTeam": {"name": "USA"}, "awayTeam": {"name": "Iran"},
     "score": {"fullTime": {"home": None, "away": None}},
     "stage": "GROUP_STAGE", "group": "GROUP_B", "status": "FINISHED"},
]}


class FakeResp:
    def __init__(self, payload, json_exc=False):
        self._payload = payload
        self._json_exc = json_exc

    def raise_for_status(self):
        return None

    def json(self):
        if self._json_exc:
            raise ValueError("not json")
        return self._payload


def test_fetch_parses_and_skips_unscored(monkeypatch):
    monkeypatch.setattr(prov.requests, "get", lambda *a, **k: FakeResp(PAYLOAD))
    p = prov.FootballDataProvider("key", "https://x/v4/", "WC", 10)
    out = p.fetch_finished()
    assert len(out) == 1                       # the None-scored match is skipped
    m = out[0]
    assert m.api_match_id == "111" and isinstance(m.api_match_id, str)
    assert m.home_team == "Brazil" and m.away_team == "Mexico"
    assert m.home_score == 2 and m.away_score == 1
    assert m.stage == "GROUP_STAGE" and m.group == "GROUP_A"


def test_base_url_is_rstripped():
    p = prov.FootballDataProvider("key", "https://x/v4/", "WC", 10)
    assert p.base_url == "https://x/v4"


def test_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout"):
        prov.FootballDataProvider("key", "https://x/v4", "WC", 0)


def test_from_config_requires_env(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FOOTBALL_DATA_API_KEY"):
        prov.FootballDataProvider.from_config(
            {"base_url": "https://x/v4", "competition_code": "WC", "request_timeout": 10})


def test_bad_json_raises_clear(monkeypatch):
    monkeypatch.setattr(prov.requests, "get", lambda *a, **k: FakeResp(None, json_exc=True))
    p = prov.FootballDataProvider("key", "https://x/v4", "WC", 10)
    with pytest.raises(ValueError, match="not valid JSON"):
        p.fetch_finished()
