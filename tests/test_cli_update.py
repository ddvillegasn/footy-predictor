import json
from pathlib import Path

from footy import cli
from footy.config import load_config

ROOT = Path(__file__).resolve().parent.parent


class FakeRunner:
    def cycle(self, provider):
        return {"played": 3, "aggregate": {"teams": {"Brazil": {"champion": 0.2}}},
                "scoreboard": {"n": 3, "accuracy": 0.66, "log_loss": 0.9, "brier": 0.5,
                               "goal_mae": 1.1, "matches": []},
                "meta": {"n_tournaments": 1000, "seed": 42, "results_path": "x"}}


def test_live_config_loads_and_stage_map_is_dict():
    cfg = load_config("live")
    assert isinstance(cfg["stage_map"], dict)
    assert cfg["competition_code"] == "WC"
    assert cfg["request_timeout"] > 0


def test_run_update_json_output(capsys):
    code = cli.run_update(["--json"], runner=FakeRunner(), provider=object())
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["played"] == 3
    assert payload["scoreboard"]["accuracy"] == 0.66


def test_run_update_summary_output(capsys):
    code = cli.run_update([], runner=FakeRunner(), provider=object())
    out = capsys.readouterr().out
    assert code == 0
    assert "Brazil" in out and "accuracy" in out.lower()
