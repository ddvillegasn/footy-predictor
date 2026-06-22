from pathlib import Path

from footy.tournament.structure import load_structure
from footy.tournament.results import load_results

ROOT = Path(__file__).resolve().parent.parent


def test_wc2026_structure_loads():
    cfg = load_structure(ROOT / "configs" / "tournaments" / "wc2026.yaml")
    assert len(cfg.groups) == 12
    assert all(len(v) == 4 for v in cfg.groups.values())
    assert cfg.best_thirds == 8


def test_wc2026_results_loads():
    # wc2026_results.yaml is live state (owned by footy.live.ingest): it may be empty
    # before the tournament or hold real played matches once results are fetched. The
    # contract is only that it parses against the structure and every result references
    # teams that exist in the structure.
    cfg = load_structure(ROOT / "configs" / "tournaments" / "wc2026.yaml")
    res = load_results(ROOT / "configs" / "tournaments" / "wc2026_results.yaml", cfg.groups)
    known = {t for group in cfg.groups.values() for t in group}
    assert isinstance(res.played, list)
    for pm in res.played:
        assert pm.team_a in known and pm.team_b in known
