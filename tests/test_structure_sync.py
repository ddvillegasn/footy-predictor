import pytest

from footy.live.structure_sync import map_groups, sync_structure
from footy.tournament.structure import load_structure

KNOWN = {"Mexico", "South Africa", "South Korea", "Czech Republic"}
NAME_MAP = {"Czechia": "Czech Republic"}


def test_map_groups_maps_and_letters():
    raw = {"GROUP_A": ["Mexico", "South Africa", "South Korea", "Czechia"]}
    groups = map_groups(raw, NAME_MAP, KNOWN)
    assert groups == {"A": ["Mexico", "South Africa", "South Korea", "Czech Republic"]}


def test_map_groups_collects_all_missing_in_one_error():
    raw = {"GROUP_A": ["Mexico", "Atlantis", "Wakanda", "Czechia"]}
    with pytest.raises(ValueError) as exc:
        map_groups(raw, NAME_MAP, KNOWN)
    msg = str(exc.value)
    assert "Atlantis" in msg and "Wakanda" in msg          # both listed, not just the first
    assert "2 team" in msg


class FakeProvider:
    def fetch_structure(self):
        # 2 full groups of 4 (A, B) using known/ mappable names.
        return {
            "GROUP_A": ["Mexico", "South Africa", "South Korea", "Czechia"],
            "GROUP_B": ["Mexico2", "South Africa2", "South Korea2", "Czech2"],
        }


def test_sync_structure_writes_loadable_yaml(tmp_path):
    known = KNOWN | {"Mexico2", "South Africa2", "South Korea2", "Czech2"}
    out = tmp_path / "wc.yaml"
    # Use only 2 groups, so the bracket template (which references A..L) must be trimmed
    # for this test by passing a 2-group bracket override.
    n = sync_structure(FakeProvider(), NAME_MAP, known, out,
                       bracket_r32=[["winner_A", "runner_B"], ["winner_B", "runner_A"]])
    assert n == 8
    cfg = load_structure(out)
    assert set(cfg.groups.keys()) == {"A", "B"}
    assert cfg.groups["A"][3] == "Czech Republic"
