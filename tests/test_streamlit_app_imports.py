import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_module_imports_without_running_main():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # runs top-level; main() is guarded by __main__
    assert hasattr(mod, "main") and hasattr(mod, "build_engine")
