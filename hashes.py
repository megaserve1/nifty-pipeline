"""
hashes.py -- content hashing. Used by the feature cache, the manifest and the lock.

WHY CONTENT HASH AND NOT FILE TIME:
    The old cache asked "is the notebook newer than its cached parquet?" (mtime).
    Copying a file from Windows bumps its mtime although the bytes are identical
    -> a needless re-run; and restoring an old file can make it LOOK older than a
    stale cache -> a silently stale feature reaching the live model.
    Hashing the bytes answers the honest question: "did the content change?"
"""
import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path) -> str:
    """Streaming sha256 of a file (works on a 95 MB parquet without loading it)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_notebook_code(path) -> str:
    """Hash a notebook's CODE ONLY -- ignoring outputs, execution counts and metadata.

    A notebook re-run with identical code produces different bytes (new outputs,
    new execution_count), so hashing the raw .ipynb would say 'changed' every time.
    We hash only the source of the code cells: the thing that decides the result.
    Plain .py files are hashed whole.
    """
    p = Path(path)
    if p.suffix != ".ipynb":
        return sha256_file(p)
    import json
    nb = json.loads(p.read_text())
    code = "\n".join(
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    )
    return sha256_text(code)
