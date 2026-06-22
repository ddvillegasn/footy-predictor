from footy.ui.styles import CSS


def test_css_defines_core_classes():
    for cls in [".fty-card", ".fty-card.fav", ".fty-bar-fill", ".fty-chip",
                ".fty-badge.alta", ".fty-badge.valor", ".fty-badge.novalor",
                ".fty-odds-row"]:
        assert cls in CSS
    assert CSS.strip().startswith("<style>") and CSS.strip().endswith("</style>")
