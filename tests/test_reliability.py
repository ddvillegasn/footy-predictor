from footy.reliability import compute_reliability


def test_more_matches_yields_higher_reliability():
    low = compute_reliability(
        matches_a=2, matches_b=2, recent_a=1, recent_b=1,
        data_age_days=4000, dispersion=2.0, missing_rankings=2, min_matches=10,
    )
    high = compute_reliability(
        matches_a=120, matches_b=110, recent_a=10, recent_b=10,
        data_age_days=120, dispersion=1.0, missing_rankings=0, min_matches=10,
    )
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_missing_rankings_lowers_reliability():
    base = dict(matches_a=50, matches_b=50, recent_a=8, recent_b=8,
                data_age_days=200, dispersion=1.0, min_matches=10)
    with_rank = compute_reliability(missing_rankings=0, **base)
    without = compute_reliability(missing_rankings=2, **base)
    assert with_rank > without
