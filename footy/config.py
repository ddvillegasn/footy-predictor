from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml as a dict. Raise FileNotFoundError if absent."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
