import pytest

from footy.live.name_map import load_name_map, map_team, map_stage

KNOWN = {"United States", "South Korea", "Brazil"}
STAGE_MAP = {"GROUP_STAGE": "group", "LAST_16": "R16"}


def test_load_name_map_ok(tmp_path):
    import yaml
    p = tmp_path / "nm.yaml"
    p.write_text(yaml.safe_dump({"teams": {"USA": "United States"}}), encoding="utf-8")
    assert load_name_map(p) == {"USA": "United States"}


def test_load_name_map_non_dict_raises(tmp_path):
    p = tmp_path / "nm.yaml"
    p.write_text("teams: [1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_name_map(p)


def test_map_team_uses_mapping_then_passthrough():
    mapping = {"USA": "United States", "Korea Republic": "South Korea"}
    assert map_team("USA", mapping, KNOWN) == "United States"
    assert map_team("Brazil", mapping, KNOWN) == "Brazil"   # passthrough, already known


def test_map_team_unmapped_raises():
    with pytest.raises(ValueError, match="unmapped API team"):
        map_team("Atlantis", {}, KNOWN)


def test_map_stage_hit_and_miss():
    assert map_stage("GROUP_STAGE", STAGE_MAP) == "group"
    with pytest.raises(ValueError, match="unknown API stage"):
        map_stage("PENALTIES", STAGE_MAP)
