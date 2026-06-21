import pytest

from footy.tournament.results import load_results, TournamentResults, PlayedMatch

STRUCT_TEAMS = {"A": ["T1", "T2", "T3", "T4"], "B": ["T5", "T6", "T7", "T8"]}


def _write(tmp_path, matches):
    import yaml
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"played_matches": matches}), encoding="utf-8")
    return p


def test_loads_and_looks_up_group(tmp_path):
    matches = [{"match_id": "A1", "stage": "group", "group": "A",
                "team_a": "T1", "team_b": "T2", "goals_a": 3, "goals_b": 1}]
    res = load_results(_write(tmp_path, matches), STRUCT_TEAMS)
    assert isinstance(res, TournamentResults)
    pm = res.lookup_group("A", "T2", "T1")  # order-insensitive
    assert pm.goals_a == 3 and pm.team_a == "T1"
    assert res.lookup_group("A", "T1", "T3") is None


def test_empty_results(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("played_matches: []", encoding="utf-8")
    res = load_results(p, STRUCT_TEAMS)
    assert res.lookup_group("A", "T1", "T2") is None


def test_team_not_in_structure_raises(tmp_path):
    matches = [{"match_id": "X1", "stage": "group", "group": "A",
                "team_a": "T1", "team_b": "GHOST", "goals_a": 1, "goals_b": 0}]
    with pytest.raises(ValueError, match="not in structure"):
        load_results(_write(tmp_path, matches), STRUCT_TEAMS)


def test_duplicate_match_id_raises(tmp_path):
    matches = [
        {"match_id": "A1", "stage": "group", "group": "A", "team_a": "T1", "team_b": "T2", "goals_a": 1, "goals_b": 0},
        {"match_id": "A1", "stage": "group", "group": "A", "team_a": "T3", "team_b": "T4", "goals_a": 2, "goals_b": 2},
    ]
    with pytest.raises(ValueError, match="duplicate match_id"):
        load_results(_write(tmp_path, matches), STRUCT_TEAMS)


def test_negative_score_raises(tmp_path):
    matches = [{"match_id": "A1", "stage": "group", "group": "A",
                "team_a": "T1", "team_b": "T2", "goals_a": -1, "goals_b": 0}]
    with pytest.raises(ValueError, match="score"):
        load_results(_write(tmp_path, matches), STRUCT_TEAMS)


def test_lookup_knockout(tmp_path):
    matches = [{"match_id": "SF1", "stage": "SF", "team_a": "T1", "team_b": "T5",
                "goals_a": 2, "goals_b": 1, "winner": "T1"}]
    res = load_results(_write(tmp_path, matches), STRUCT_TEAMS)
    pm = res.lookup_knockout("SF", "T5", "T1")
    assert pm.winner == "T1"
    assert res.lookup_knockout("F", "T1", "T5") is None
