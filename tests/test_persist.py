from footy.persist import load_or_build, files_fingerprint


def test_load_or_build_builds_once_then_loads(tmp_path):
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return {"v": 42}

    cache = tmp_path / "obj.pkl"
    a = load_or_build(cache, "fp1", build)
    b = load_or_build(cache, "fp1", build)
    assert a == b == {"v": 42}
    assert calls["n"] == 1            # second call loaded from disk, did not rebuild


def test_load_or_build_rebuilds_on_fingerprint_change(tmp_path):
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return calls["n"]

    cache = tmp_path / "obj.pkl"
    load_or_build(cache, "fp1", build)
    out = load_or_build(cache, "fp2", build)   # changed fingerprint -> rebuild
    assert out == 2 and calls["n"] == 2


def test_files_fingerprint_missing_vs_present(tmp_path):
    f = tmp_path / "a.txt"
    fp_missing = files_fingerprint([f])
    f.write_text("x", encoding="utf-8")
    fp_present = files_fingerprint([f])
    assert fp_missing != fp_present
