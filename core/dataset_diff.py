"""
dataset_diff.py -- compare dataset versions by their frozen recipes
===================================================================
Reads the version yamls (not the giant parquets), so comparing 2 or 20
versions is instant.

Run:
    python dataset_diff.py v1 v2          # pairwise diff
    python dataset_diff.py v1 v2 v3 v4    # feature matrix across many
    python dataset_diff.py --all          # matrix of every version
"""
import argparse

import yaml

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


def _recipe_files():
    """the frozen recipes -- and ONLY the recipes. dataset_v2.lock.yaml matches the same glob,
    and its stem parsed as int('2.lock') crashed --all the moment anything was ever published.
    make_version._version_files() got this right; this file had forgotten the filter."""
    return [p for p in C.VERSIONS_DIR.glob("dataset_v*.yaml")
            if not p.name.endswith(".lock.yaml")]


def _sort_key(v: str) -> tuple:
    """'v2' -> (2, 0), 'v2.1' -> (2, 1). int(v.lstrip('v')) crashed on every minor version --
    int('2.1') is a ValueError, and minor versions are the whole point of the ablation design."""
    parts = v.lstrip("v").split(".")
    return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def load(v: str) -> dict:
    p = C.VERSIONS_DIR / f"dataset_{v}.yaml"
    if not p.exists():
        have = sorted(x.stem.replace("dataset_", "") for x in _recipe_files())
        raise SystemExit(f"unknown version '{v}'. existing: {have or 'none'}")
    return yaml.safe_load(p.read_text())


def pairwise(a: str, b: str):
    da, db = load(a), load(b)
    fa, fb = set(da["features"]), set(db["features"])
    print(f"diff {a} -> {b}")
    print(f"  features added   : {sorted(fb - fa) or '-'}")
    print(f"  features removed : {sorted(fa - fb) or '-'}")
    print(f"  features kept    : {len(fa & fb)}")
    for key in ("labels_file", "date_range"):
        va, vb = da.get(key), db.get(key)
        print(f"  {key:16s} : {'unchanged' if va == vb else f'{va}  ->  {vb}'}")
    print(f"  created          : {da.get('created')}  vs  {db.get('created')}")
    print(f"  author           : {da.get('author')}  vs  {db.get('author')}")


def matrix(versions: list):
    docs = {v: load(v) for v in versions}
    all_feats = sorted(set().union(*(d["features"] for d in docs.values())))
    w = max((len(f) for f in all_feats), default=10) + 2
    print(f"{'FEATURE':{w}s} " + " ".join(f"{v:>4s}" for v in versions))
    for f in all_feats:
        row = " ".join(f"{'x' if f in docs[v]['features'] else '-':>4s}" for v in versions)
        print(f"{f:{w}s} {row}")
    print(f"\n{'total':{w}s} " +
          " ".join(f"{len(docs[v]['features']):>4d}" for v in versions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="*", help="e.g. v1 v2 [v3 ...]")
    ap.add_argument("--all", action="store_true", help="compare every version")
    a = ap.parse_args()

    if a.all:
        vs = sorted((p.stem.replace("dataset_", "") for p in _recipe_files()),
                    key=_sort_key)
        if len(vs) < 2:
            raise SystemExit("need at least 2 frozen versions")
        matrix(vs)
    elif len(a.versions) == 2:
        pairwise(*a.versions)
    elif len(a.versions) > 2:
        matrix(a.versions)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
