"""
core/make_version.py -- SELECT features -> freeze a DATASET VERSION (the recipe).

  1. print the ballot     python core/make_version.py --new
        -> selection_sheet.yaml lists every registered feature, none ticked.
  2. the expert ticks     open it, put  x  after his picks. He types no feature names.
  3. freeze               python core/make_version.py --from-sheet
        -> versions/dataset_vN.yaml   (immutable; N auto-increments)

Shortcuts:
    --all                            take every registered feature
    --from v1 --drop a,b --add c     derive a new version from an old one

THE RECIPE IS IMMUTABLE. Nothing is ever written back into it -- the manifest certifies
sha256(recipe), so a lock written inside would break its own hash on the next read. The
lock lives beside it, in dataset_vN.lock.yaml, written by core/publish_version.py.

The recipe records a LOGICAL labels name, never a filesystem path, so this runs fine on a
machine that has no labels CSV. It also freezes each feature's CLOCK (bar period), which
bridge/build_dataset.py needs in order to align on bar-close without lookahead.
"""
import argparse
import datetime as dt
import getpass
import re
import sys
import pathlib

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


# ------------------------------------------------------------------ helpers
def load_registry() -> dict:
    if not C.REGISTRY.exists():
        raise SystemExit("no registry.yaml -- run  python bridge/register.py  first")
    return yaml.safe_load(C.REGISTRY.read_text()) or {}


# ------------------------------------------------------------------ version numbers
#
# THE NUMBER TELLS YOU HOW THE VERSION WAS MADE, AND THAT IS THE WHOLE POINT.
#
#   v1, v2, v3      A MAJOR. A fresh, considered human choice -- the expert ticked a ballot
#                   (or --all took everything). A new starting point.
#
#   v2.1, v2.2      A MINOR. A VARIATION on v2 -- v2 with a feature dropped or added.
#
# why it matters: a comparison only means something if ONE thing changed.
#
#     v2 vs v2.1    only stress_signal is missing -> if the score drops, THAT FEATURE MATTERS
#     v2.1 vs v2.2  same base, one feature removed each -> WHICH DROP HURT MORE?
#     v2 vs v7      two different selections -> too much changed. the comparison teaches nothing.
#
# so: SAME MAJOR = a fair comparison. DIFFERENT MAJOR = don't bother.
#
# deriving from a sub-version stays FLAT: v2.1 with another feature dropped becomes v2.3, not
# v2.1.1. it is still a variation on v2. the recipe records `parent: v2.1`, so the full lineage
# is kept -- we just do not drown in dots.
#
# and it makes the ClearML mapping fall out for free: v2 -> "2.0", v2.1 -> "2.1". valid PEP440,
# sorts correctly (2.10 > 2.9), and the pipeline version IS the ClearML version. no translation.

VERSION_RE = re.compile(r"dataset_v(\d+)(?:\.(\d+))?\.yaml$")


def _version_files():
    return [p for p in C.VERSIONS_DIR.glob("dataset_v*.yaml")
            if not p.name.endswith(".lock.yaml")]


def _parse(name: str) -> tuple[int, int]:
    """'v2'   -> (2, 0)      a major
       'v2.3' -> (2, 3)      the 3rd variation on v2"""
    m = re.fullmatch(r"v?(\d+)(?:\.(\d+))?", str(name).strip())
    if not m:
        raise SystemExit(f"'{name}' is not a version. expected v2 or v2.1")
    return int(m.group(1)), int(m.group(2) or 0)


def existing_versions() -> list[tuple[int, int]]:
    out = []
    for p in _version_files():
        m = VERSION_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2) or 0)))
    return sorted(out)


def next_major() -> str:
    """a brand new starting point: v1, v2, v3 ..."""
    majors = [maj for maj, _ in existing_versions()]
    return f"v{max(majors) + 1 if majors else 1}"


def next_minor(parent: str) -> str:
    """the next variation on the parent's MAJOR: v2.1, v2.2, v2.3 ...

    deriving from v2.1 gives v2.3 (the next free slot under major 2), NOT v2.1.1 -- it is still
    a variation on v2. the parent is written into the recipe, so nothing is lost.
    """
    maj, _ = _parse(parent)
    minors = [mn for m, mn in existing_versions() if m == maj]
    return f"v{maj}.{max(minors) + 1 if minors else 1}"


def load_version(v: str) -> dict:
    p = C.VERSIONS_DIR / f"dataset_{v}.yaml"
    if not p.exists():
        have = sorted(x.stem.replace("dataset_", "") for x in _version_files())
        raise SystemExit(f"unknown version '{v}'. existing: {have or 'none'}")
    return yaml.safe_load(p.read_text())


def freeze(features: list, source: str, parent: str | None = None) -> str:
    """write the frozen recipe. THE ONE EXIT -- every mode ends here, so no route can skip the
    checks. a MAJOR if it came from a ballot; a MINOR if it was derived from a parent."""
    reg = load_registry()
    unknown = [f for f in features if f not in reg]
    if unknown:
        raise SystemExit(f"unknown feature(s): {unknown}\n"
                         f"(check spelling:  python bridge/register.py --list)")
    if not features:
        raise SystemExit("selection is EMPTY -- tick at least one feature.")

    # A VARIATION MUST INHERIT ITS PARENT'S FROZEN CLOCKS.
    # the whole point of v2 vs v2.1 is that ONE thing changed -- the dropped feature. deriving
    # used to re-read clocks from the CURRENT registry, so if the registry moved between the
    # freezes (a rescan, a human edit), v2.1 could differ from v2 in feature selection AND in
    # alignment -- and the comparison would silently teach nothing. the parent's word wins for
    # every feature it knew; only genuinely NEW features take from today's registry.
    parent_clocks = {}
    if parent:
        parent_clocks = (load_version(parent) or {}).get("feature_clocks") or {}

    v = next_minor(parent) if parent else next_major()
    doc = {
        "name":        f"dataset_{v}",
        "kind":        "variation" if parent else "selection",
        "parent":      parent,                 # None for a major; the exact parent for a minor
        "created":     dt.datetime.now().isoformat(timespec="seconds"),
        "author":      getpass.getuser(),
        "selected_by": source,
        "labels_name": C.labels_name(),        # derived from the FILE, so it cannot go stale
        "date_range":  "full",
        "features":    sorted(features),
        # ONLY the clocks a HUMAN actually declared. Not the machine's summary.
        #
        # registry.yaml's `clock:` is written BY register.py as infer_bar_minutes() -- the MAX
        # across the file's columns, i.e. THE SLOWEST COLUMN. It is a one-line summary for the
        # human reading the ballot, nothing more.
        #
        # This used to freeze that number into the recipe unconditionally, and build_dataset
        # applies a frozen clock as a FLOOR on every column of the feature. merged_raw_1min has
        # 182 columns; one of them (gap_state) measures 60min, so the file's summary is 60min --
        # and `close`, which genuinely updates every minute, would have been held back to
        # max(1, 60) = 60 MINUTES STALE. 181 columns crippled because one column is slow.
        #
        # A number the machine wrote is not a declaration. So we freeze `clock:` only when a
        # human has actually CHANGED it -- i.e. when it differs from `clock_measured`, which
        # register.py writes beside it and never touches again. Otherwise we freeze nothing, and
        # build_dataset measures each column on its own terms (which is deterministic: the
        # parquet's sha256 is in the manifest, so the same bytes always give the same clocks).
        # a registry entry that has `clock` but NO `clock_measured` (hand-written, or from an
        # older registry format) tells us nothing about who set the number -- treating it as a
        # human declaration would freeze a machine summary as a 60-minute floor on every
        # column. no clock_measured = no evidence = freeze nothing.
        "feature_clocks": {
            **{f: reg[f]["clock"] for f in sorted(features)
               if reg[f].get("clock") and reg[f].get("clock_measured")
               and reg[f]["clock"] != reg[f]["clock_measured"]},
            # the parent's frozen word wins for every feature it covered (see note above)
            **{f: c for f, c in parent_clocks.items() if f in features},
        },
    }
    out = C.VERSIONS_DIR / f"dataset_{v}.yaml"
    out.write_text("# FROZEN RECIPE -- immutable. Never edit; make a new version instead.\n"
                   "# The lock (hashes, clearml id) is written beside this as *.lock.yaml\n"
                   + yaml.safe_dump(doc, sort_keys=False))

    tag = f"VARIATION on {parent}" if parent else "NEW SELECTION"
    print(f"FROZEN -> {out}   [{tag}]")
    print(f"   {len(features)} features: {', '.join(sorted(features)[:8])}"
          f"{' ...' if len(features) > 8 else ''}")
    if parent:
        print(f"   compare them:  python core/dataset_diff.py {parent} {v}")
    print(f"\nnext:  python bridge/build_dataset.py --version {v}")
    return v


# ------------------------------------------------------------------ modes
def _sheet_has_ticks() -> bool:
    if not C.SELECTION_SHEET.exists():
        return False
    for line in C.SELECTION_SHEET.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        if line.partition(":")[2].split("#", 1)[0].strip():
            return True
    return False


def mode_new_ballot(fresh: bool = False):
    if _sheet_has_ticks() and not fresh:
        raise SystemExit("selection_sheet.yaml already has ticks -- refusing to wipe it.\n"
                         "  freeze it first:  python core/make_version.py --from-sheet\n"
                         "  or force a blank: python core/make_version.py --new --fresh")
    reg = load_registry()
    lines = [
        "# ================= FEATURE SELECTION SHEET (the ballot) =================",
        "# put  x  after a feature to INCLUDE it.  leave blank to exclude.",
        "# do NOT add or rename lines -- this menu is printed from registry.yaml.",
        "# when done:  python core/make_version.py --from-sheet",
        "# ========================================================================",
    ]
    for name in sorted(reg):
        desc = reg[name].get("desc") or reg[name]["file"]
        clock = reg[name].get("clock", "?")
        lines.append(f"{name}:{' ' * max(1, 44 - len(name))}# [{clock}] {desc}")
    C.SELECTION_SHEET.write_text("\n".join(lines) + "\n")
    print(f"ballot written -> {C.SELECTION_SHEET}")
    print(f"   {len(reg)} features listed, none ticked.")
    print("   tick your picks, then:  python core/make_version.py --from-sheet")


def mode_from_sheet():
    if not C.SELECTION_SHEET.exists():
        raise SystemExit("no ballot -- run  python core/make_version.py --new  first")
    # Deliberately NOT yaml.safe_load: 'name:x' (no space after the colon) is invalid YAML,
    # and a human ticking a form should not have to know that. Anything after the colon,
    # before the # comment, counts as a tick.
    picked, total = [], 0
    for line in C.SELECTION_SHEET.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, _, rest = line.partition(":")
        total += 1
        if rest.split("#", 1)[0].strip():
            picked.append(name.strip())
    print(f"ballot: {len(picked)} of {total} features ticked")
    freeze(picked, source="ballot")


def derive(base: str, drop: list, add: list, quiet: bool = False) -> str:
    """v2 minus a feature, plus a feature -> a VARIATION: v2.1, v2.2, ..."""
    doc = load_version(base)
    have = set(doc["features"])

    # a drop that removes nothing, or an add that adds nothing, is almost always a typo.
    # say so -- a silently empty change would give you a version identical to its parent.
    no_op_drop = [d for d in drop if d not in have]
    no_op_add = [a for a in add if a in have]
    if no_op_drop:
        print(f"      !! {base} does not contain {no_op_drop} -- nothing to drop")
    if no_op_add:
        print(f"      !! {base} already contains {no_op_add} -- nothing to add")

    feats = (have - set(drop)) | set(add)
    if feats == have:
        raise SystemExit(f"this would produce a version IDENTICAL to {base}. "
                         f"check the names against:  python bridge/register.py --list")

    if not quiet:
        print(f"derive from {base}: {len(have)} features "
              f"- {len(have - feats)} + {len(feats - have)} -> {len(feats)}")
    src = (f"derived from {base}"
           + (f" -drop {','.join(drop)}" if drop else "")
           + (f" +add {','.join(add)}" if add else ""))
    return freeze(sorted(feats), source=src, parent=base)


def mode_all():
    reg = load_registry()
    print(f"selecting ALL {len(reg)} registered features")
    freeze(sorted(reg), source="--all")          # a MAJOR -- it is a fresh whole-menu choice


def mode_from_plan(path):
    """run a PLAN FILE: many variations at once.

    the point is the ABLATION -- 'make one version per feature, each with that feature dropped,
    so I can see which one actually matters'. by hand that is fourteen commands. here it is one
    file and one command, and it is reviewable in git BEFORE any compute happens.

    THE PLAN IS AN INPUT, NEVER A RECORD. versions/dataset_vN.yaml is always the truth about
    what a version IS. the plan is only one of the ways to ask for one.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"no plan at {path}\n\nexample:\n"
                         "  versions:\n"
                         "    - {parent: v1, drop: [stress_signal]}\n"
                         "    - {parent: v1, drop: [gap_final_features]}\n"
                         "    - {parent: v1, add: [momentum_state]}\n")
    plan = yaml.safe_load(path.read_text()) or {}
    items = plan.get("versions") or []
    if not items:
        raise SystemExit(f"{path} has no 'versions:' list")

    print(f"plan: {len(items)} variation(s) from {path}\n")
    made, failed = [], []
    for i, item in enumerate(items, 1):
        parent = item.get("parent")
        drop = list(item.get("drop") or [])
        add = list(item.get("add") or [])
        if not parent:
            failed.append((i, "no parent given"))
            print(f"  [{i}] SKIPPED -- no 'parent'")
            continue
        label = (f"-{','.join(drop)}" if drop else "") + (f" +{','.join(add)}" if add else "")
        print(f"  [{i}] {parent} {label}")
        try:
            v = derive(parent, drop, add, quiet=True)
            made.append(v)
        except SystemExit as e:
            failed.append((i, str(e).strip().splitlines()[0]))
            print(f"      FAILED: {str(e).strip().splitlines()[0]}")
        print()

    # BE LOUD. a plan runner that quietly says 'skipping' is how you lose a day wondering
    # whether anything happened at all.
    print("=" * 66)
    print(f"PLAN DONE   created: {len(made)}   failed: {len(failed)}")
    print("=" * 66)
    for v in made:
        print(f"  CREATED  {v}")
    for i, why in failed:
        print(f"  FAILED   item {i}: {why}")
    if made:
        print(f"\nbuild them:")
        for v in made:
            print(f"  python bridge/build_dataset.py --version {v}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", action="store_true", help="print a fresh ballot")
    ap.add_argument("--fresh", action="store_true", help="with --new: overwrite a ticked ballot")
    ap.add_argument("--from-sheet", action="store_true", help="freeze the ticked ballot")
    ap.add_argument("--from", dest="base", metavar="vN",
                    help="derive a VARIATION from an existing version (v2 -> v2.1)")
    ap.add_argument("--drop", default="", metavar="a,b")
    ap.add_argument("--add", default="", metavar="c,d")
    ap.add_argument("--all", action="store_true", help="select every registered feature")
    ap.add_argument("--from-plan", nargs="?", const=str(C.CONFIGS_DIR / "version_plan.yaml"),
                    metavar="FILE",
                    help="run a plan file: many variations at once (an ablation sweep)")
    ap.add_argument("--list", action="store_true", help="show the version tree")
    a = ap.parse_args()

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    if a.new:
        mode_new_ballot(fresh=a.fresh)
    elif a.from_sheet:
        mode_from_sheet()
    elif a.base:
        derive(a.base, split(a.drop), split(a.add))
    elif a.all:
        mode_all()
    elif a.from_plan:
        mode_from_plan(a.from_plan)
    elif a.list:
        mode_list()
    else:
        ap.print_help()


def mode_list():
    """show the version tree, so you can SEE which comparisons are fair."""
    vers = existing_versions()
    if not vers:
        print("no versions yet")
        return
    print("versions  (same MAJOR = a fair comparison; different major = too much changed)\n")
    for maj in sorted({m for m, _ in vers}):
        doc = load_version(f"v{maj}")
        print(f"  {'v' + str(maj):<8s} {len(doc['features']):3d} features   "
              f"{doc.get('selected_by', '?')}")
        for m, mn in vers:
            if m != maj or mn == 0:
                continue
            d = load_version(f"v{maj}.{mn}")
            print(f"    +-- {'v' + str(maj) + '.' + str(mn):<6s} {len(d['features']):3d} features   "
                  f"{d.get('selected_by', '?')}")
        print()


if __name__ == "__main__":
    main()
