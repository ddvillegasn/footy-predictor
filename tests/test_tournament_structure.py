import pytest

from footy.tournament.structure import load_structure, TournamentConfig

MINI = {
    "name": "Mini",
    "neutral_default": True,
    "points": {"win": 3, "draw": 1, "loss": 0},
    "groups": {"A": ["T1", "T2", "T3", "T4"], "B": ["T5", "T6", "T7", "T8"]},
    "group_schedule": "round_robin",
    "qualification": {"per_group_advance": 2, "best_thirds": 0},
    "tiebreakers": ["points", "goal_difference", "goals_for", "head_to_head",
                    "fair_play", "drawing_of_lots"],
    "thirds_ranking": ["points", "goal_difference", "goals_for", "drawing_of_lots"],
    "knockout": {"rounds": ["SF", "F"],
                 "bracket_r32": [["winner_A", "runner_B"], ["winner_B", "runner_A"]],
                 "thirds_assignment": "ranked_order"},
}


def _write(tmp_path, data):
    import yaml
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_loads_valid_config(tmp_path):
    cfg = load_structure(_write(tmp_path, MINI))
    assert isinstance(cfg, TournamentConfig)
    assert cfg.groups["A"] == ["T1", "T2", "T3", "T4"]
    assert cfg.per_group_advance == 2
    assert cfg.bracket_r32 == [["winner_A", "runner_B"], ["winner_B", "runner_A"]]


def test_duplicate_team_raises(tmp_path):
    bad = {**MINI, "groups": {"A": ["T1", "T2", "T3", "T4"], "B": ["T1", "T6", "T7", "T8"]}}
    with pytest.raises(ValueError, match="duplicate"):
        load_structure(_write(tmp_path, bad))


def test_bad_bracket_ref_raises(tmp_path):
    bad = {**MINI, "knockout": {**MINI["knockout"],
           "bracket_r32": [["winner_Z", "runner_B"], ["winner_B", "runner_A"]]}}
    with pytest.raises(ValueError, match="bracket"):
        load_structure(_write(tmp_path, bad))


def test_third_slot_out_of_range_raises(tmp_path):
    bad = {**MINI, "knockout": {**MINI["knockout"],
           "bracket_r32": [["winner_A", "third_slot_1"], ["winner_B", "runner_A"]]}}
    # best_thirds is 0, so third_slot_1 is invalid
    with pytest.raises(ValueError, match="third"):
        load_structure(_write(tmp_path, bad))
