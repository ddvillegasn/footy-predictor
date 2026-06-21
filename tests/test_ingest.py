from types import SimpleNamespace

import pytest

from footy.live.provider import ProviderMatch
from footy.live.ingest import ingest, GENERATED_HEADER
from footy.tournament.results import load_results

STAGE_MAP = {"GROUP_STAGE": "group", "LAST_16": "R16"}


class FakeProvider:
    def __init__(self, matches):
        self._matches = matches

    def fetch_finished(self):
        return self._matches


def _structure():
    return SimpleNamespace(groups={"A": ["Brazil", "Mexico", "Ecuador", "Honduras"],
                                    "B": ["United States", "Iran", "Wales", "Ghana"]})


def _matches():
    return [
        ProviderMatch("111", "Brazil", "Mexico", 2, 1, "GROUP_STAGE", "GROUP_A", "FINISHED"),
        ProviderMatch("112", "United States", "Iran", 1, 0, "GROUP_STAGE", "GROUP_B", "FINISHED"),
    ]


def test_ingest_writes_sp3_compatible_yaml_with_header(tmp_path):
    out = tmp_path / "wc2026_results.yaml"
    n = ingest(FakeProvider(_matches()), _structure(), {}, STAGE_MAP, out)
    assert n == 2
    text = out.read_text(encoding="utf-8")
    assert text.startswith(GENERATED_HEADER)
    # SP3 loader parses it.
    res = load_results(out, _structure().groups)
    pm = res.lookup_group("A", "Brazil", "Mexico")
    assert pm.goals_a == 2 and pm.goals_b == 1


def test_ingest_is_idempotent(tmp_path):
    out = tmp_path / "r.yaml"
    ingest(FakeProvider(_matches()), _structure(), {}, STAGE_MAP, out)
    first = out.read_text(encoding="utf-8")
    ingest(FakeProvider(_matches()), _structure(), {}, STAGE_MAP, out)
    assert out.read_text(encoding="utf-8") == first


def test_group_stage_without_group_raises(tmp_path):
    bad = [ProviderMatch("1", "Brazil", "Mexico", 1, 0, "GROUP_STAGE", None, "FINISHED")]
    with pytest.raises(ValueError, match="no group"):
        ingest(FakeProvider(bad), _structure(), {}, STAGE_MAP, tmp_path / "r.yaml")
