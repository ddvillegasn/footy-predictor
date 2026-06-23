from footy.metrics import calibration_buckets


def test_perfect_confidence_correct_is_zero_ece():
    probs = [{"home": 1.0, "draw": 0.0, "away": 0.0}] * 5
    actuals = ["home"] * 5
    c = calibration_buckets(probs, actuals, bins=10)
    assert c["ece"] == 0.0
    assert sum(b["n"] for b in c["bins"]) == 5


def test_overconfident_wrong_has_high_ece():
    probs = [{"home": 1.0, "draw": 0.0, "away": 0.0}] * 4
    actuals = ["home", "home", "away", "away"]   # conf 1.0 but only 50% correct
    c = calibration_buckets(probs, actuals, bins=10)
    assert c["ece"] >= 0.49


def test_bins_record_prob_and_frequency():
    probs = [{"home": 0.6, "draw": 0.2, "away": 0.2},
             {"home": 0.6, "draw": 0.2, "away": 0.2}]
    actuals = ["home", "away"]   # one hit, one miss; conf 0.6 bucket
    c = calibration_buckets(probs, actuals, bins=10)
    bucket = next(b for b in c["bins"] if b["n"] == 2)
    assert bucket["prob_media"] == 0.6 and bucket["frecuencia"] == 0.5
