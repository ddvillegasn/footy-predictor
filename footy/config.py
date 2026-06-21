from pathlib import Path
import hashlib

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml as a dict. Raise FileNotFoundError if absent."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def config_fingerprint(name: str) -> str:
    """Stable 8-char sha1 of a config file's bytes (reproducibility tag)."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    return digest[:8]
