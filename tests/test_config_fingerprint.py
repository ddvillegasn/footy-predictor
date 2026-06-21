from footy.config import load_config, config_fingerprint
from footy.betting import BETTING_VERSION


def test_betting_config_loads():
    cfg = load_config("betting")
    assert cfg["over_under_lines"][2] == 2.5
    assert cfg["top_scores"] == 5
    assert cfg["value"]["reliability_high"] == 0.70


def test_fingerprint_is_stable_8_char_hex():
    fp = config_fingerprint("betting")
    assert isinstance(fp, str) and len(fp) == 8
    assert fp == config_fingerprint("betting")  # deterministic
    int(fp, 16)  # valid hex


def test_betting_version_constant():
    assert BETTING_VERSION == "sp2-v1.0.0"
