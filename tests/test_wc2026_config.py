from pathlib import Path

from footy.tournament.structure import load_structure
from footy.tournament.results import load_results

ROOT = Path(__file__).resolve().parent.parent


def test_wc2026_structure_loads():
    cfg = load_structure(ROOT / "configs" / "tournaments" / "wc2026.yaml")
    assert len(cfg.groups) == 12
    assert all(len(v) == 4 for v in cfg.groups.values())
    assert cfg.best_thirds == 8


def test_wc2026_results_loads_empty():
    cfg = load_structure(ROOT / "configs" / "tournaments" / "wc2026.yaml")
    res = load_results(ROOT / "configs" / "tournaments" / "wc2026_results.yaml", cfg.groups)
    assert res.played == []
