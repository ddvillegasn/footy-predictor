"""Disk persistence for expensive-to-build objects (the fitted model).

Avoids re-fitting the Dixon-Coles model on every app start: the fitted object is
pickled with a fingerprint of its inputs; if the fingerprint still matches, it loads
from disk instantly instead of refitting.
"""
import hashlib
import pickle
from pathlib import Path


def files_fingerprint(paths) -> str:
    """Stable short hash of a set of files, derived from their CONTENT.

    Content rather than modification time. An mtime is rewritten by a git clone, a
    file copy or a container build without a single byte changing, which would
    invalidate a perfectly good cache on every deployment and force an 80-second
    refit. It also fails the other way: an edit can preserve mtime and slip past.

    Hashing the bytes costs a few milliseconds on these inputs, against the refit it
    avoids, and is correct in both directions.
    """
    h = hashlib.sha1()
    for p in paths:
        p = Path(p)
        h.update(str(p).encode())
        if not p.exists():
            h.update(b"missing")
            continue
        h.update(b"present")
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def load_or_build(cache_path, fingerprint: str, build_fn):
    """Return the cached pickled object if its stored fingerprint matches; otherwise
    build it, pickle it alongside the fingerprint, and return it."""
    cache_path = Path(cache_path)
    fp_path = cache_path.with_suffix(cache_path.suffix + ".fp")
    if (cache_path.exists() and fp_path.exists()
            and fp_path.read_text(encoding="utf-8").strip() == fingerprint):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    obj = build_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(obj, fh)
    fp_path.write_text(fingerprint, encoding="utf-8")
    return obj
