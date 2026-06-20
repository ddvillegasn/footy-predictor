from footy.config import load_config


def test_load_model_config_has_version():
    cfg = load_config("model")
    assert cfg["model_version"] == "baseline-v1.0.0"
    assert cfg["max_goals"] >= 1


def test_load_missing_config_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")
