"""
bridge/register.py -- catalogue the feature parquets into the MENU (registry.yaml).

the feature team hands us ONE PARQUET PER FEATURE. we do not run notebooks any more.
this script just looks at each parquet and writes down what it is.

for every feature it records:
    file        which parquet it came from
    clock       its bar period in minutes -- MEASURED from the data, not guessed
    columns     the columns it contributes
    na_policy   what its NaN MEANS (the feature team should set this -- see below)
    rows, span  a sanity check you can eyeball

it never runs anything and never changes a parquet. it only reads and writes the menu.

run:
    python bridge/register.py            # scan + add anything new
    python bridge/register.py --list     # just show the menu
    python bridge/register.py --rescan   # re-measure everything, even known features
"""
import argparse
import datetime as dt
import re
import sys
import pathlib

import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C                          # noqa: E402
from bridge.align import (            # noqa: E402
    infer_bar_minutes, column_clocks, clock_report,
)
from bridge import leak_guard         # noqa: E402


def load_registry() -> dict:
    # read the menu we already have. if there is no menu yet, start with an empty one.
    if C.REGISTRY.exists():
        return yaml.safe_load(C.REGISTRY.read_text()) or {}
    return {}


def save_registry(reg: dict):
    # write the menu back out, sorted, so it diffs cleanly in git
    header = (
        "# FEATURE REGISTRY -- the MENU. written by bridge/register.py.\n"
        "#\n"
        "# clock      : the feature's bar period. MEASURED from the data. if the feature team\n"
        "#              tells you a different number, put THEIRS here -- build_dataset re-measures\n"
        "#              and shouts if they disagree.\n"
        "# na_policy  : what this feature's NaN MEANS. the feature team should set this.\n"
        "#     sentinel  NaN is a real state, not missing data (e.g. 'there is no gap').\n"
        "#               kept as NaN for xgboost/catboost; given an impossible value for\n"
        "#               random forest so it becomes its own branch. NEVER 0, never the mean.\n"
        "#     zero      NaN honestly means zero (a count of nothing).\n"
        "#     ffill     a slow value on fast rows -- carry the last one forward.\n"
        "#     drop      the row is unusable; exclude it.\n"
        "# desc       : a one-line note for the human reading the ballot.\n"
    )
    C.REGISTRY.write_text(header + yaml.safe_dump(reg, sort_keys=True, allow_unicode=True))


def feature_name(path) -> str:
    """turn a filename into a clean feature name.

    'gap_final_features - Copy.parquet'  ->  'gap_final_features'
    the name is what the expert ticks on the ballot, so it must be tidy and stable.
    """
    stem = path.stem                                # drop the .parquet
    stem = stem.split(" - Copy")[0]                 # windows' ' - Copy' suffix
    # anything that is not a letter or digit -> underscore. this kills parentheses, brackets,
    # spaces and dashes in one go -- e.g. 'Bucket_Bucket_Raw(V4)' -> 'bucket_bucket_raw_v4'.
    # the name becomes a column prefix (feature__col), so a '(' here ends up in every column
    # name, and some downstream tools choke on that. keep it strictly [a-z0-9_].
    stem = re.sub(r"[^0-9A-Za-z]+", "_", stem)
    while "__" in stem:                             # collapse any doubled underscores
        stem = stem.replace("__", "_")
    return stem.strip("_").lower()


def feature_group(path) -> str:
    """which sub-folder of data/features this parquet lives in -- that folder IS the group.

    data/features/Bucket_Raw_Features/gap_state_raw.parquet  ->  'Bucket_Raw_Features'
    a file dropped straight into data/features (no sub-folder) has no group -> '_root'.
    the group lets make_version pick a whole folder at once, instead of ticking 30+ boxes.
    """
    rel = path.relative_to(C.FEATURES_DIR)
    return rel.parts[0] if len(rel.parts) > 1 else "_root"


def read_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """make the timestamp the index. it may arrive as the index already, or as a column."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for c in ("datetime", "timestamp", "ts", "date"):
        if c in df.columns:
            out = df.copy()
            out[c] = pd.to_datetime(out[c])
            return out.set_index(c).sort_index()
    raise ValueError("no timestamp -- the parquet needs a datetime index or a "
                     "datetime/timestamp column")


def inspect(path, allow: list | None = None) -> dict:
    """open one feature parquet and work out everything we need to know about it."""
    df = read_time_index(pd.read_parquet(path))

    # ---- THE LEAK GUARD. before anything else. -------------------------------------
    # a column that holds the FUTURE, the ANSWER, or the CALENDAR never reaches the ballot.
    # it is not offered to the expert, it is not written to registry.yaml, and build_dataset
    # drops it again on the way through. see bridge/leak_guard.py for why this exists.
    # `allow` is what a human already looked at and cleared -- it survives a re-scan.
    screen = leak_guard.screen(df, allow=allow)
    leak_guard.report(screen, where=path.name)
    if screen["banned"]:
        df = df[screen["ok"]]
        if not len(df.columns):
            raise SystemExit(
                f"\n  every column in {path.name} was refused. this is not a feature file --\n"
                f"  it is a working file. move it out of {C.FEATURES_DIR}.")

    # which columns are numbers, and which are text (a text column is a CATEGORY --
    # catboost eats it as-is, random forest and xgboost need it turned into numbers)
    cat_cols, num_cols = [], []
    for c in df.columns:
        t = str(df[c].dtype)
        (cat_cols if t in ("object", "str", "string", "category", "bool") else num_cols).append(c)

    # measure the real bar period. a 5-min value written onto five 1-min rows repeats across
    # those rows -- that repetition is what gives the clock away.
    #
    # EVERY column, not just the numbers. this used to pass cols=num_cols, so a TEXT column was
    # never measured at all -- and then the registry printed a confident `clock:` for a feature
    # whose one categorical column had never been looked at. gap_state is exactly that column.
    # a clock that was never measured is not a measurement, it is a guess wearing a lab coat.
    clock = infer_bar_minutes(df) if len(df.columns) else 1

    # per-column, so a mixed parquet (a 5-min signal next to the 1-min price it came from) is
    # visible to the human reading the ballot, instead of being flattened to one number.
    per_col = {c: f"{m}min" for c, m in column_clocks(df).items()}

    # is the measurement WORTH anything? a column that hardly ever changes cannot be told apart
    # from a column on a slow clock -- so we say so, instead of pretending we know.
    reports = {c: clock_report(df[c]) for c in df.columns}
    unsure = sorted(c for c, r in reports.items() if not r["confident"])

    # count the NaN per column, so the human can see whether a policy is even needed
    na = {c: int(df[c].isna().sum()) for c in df.columns}
    has_na = any(v > 0 for v in na.values())

    return {
        "clock": f"{clock}min",
        "clock_measured": f"{clock}min",
        "clock_per_column": per_col,
        "clock_unsure": unsure,          # these are GUESSES, and they guess SLOW
        "columns": list(df.columns),     # the SAFE ones. the banned ones are already gone.
        "banned": screen["banned"],      # on the record, with the reason, in registry.yaml
        "suspect": screen["suspect"],
        "categorical": cat_cols,
        "rows": int(len(df)),
        "span": [str(df.index.min()), str(df.index.max())],
        "na_counts": {c: v for c, v in na.items() if v > 0},
        "has_na": has_na,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the menu and stop")
    ap.add_argument("--rescan", action="store_true",
                    help="re-measure every feature, even ones already in the menu")
    a = ap.parse_args()

    reg = load_registry()

    # ---- just show the menu ------------------------------------------------
    if a.list:
        if not reg:
            print("the menu is empty -- run  python bridge/register.py  first")
            return
        print(f"{'FEATURE':30s} {'GROUP':22s} {'CLOCK':7s} {'COLS':>4s}  FILE")
        for name, m in sorted(reg.items(), key=lambda kv: (kv[1].get('group', ''), kv[0])):
            print(f"{name:30s} {m.get('group','_root'):22s} {m.get('clock','?'):7s} "
                  f"{len(m.get('columns', [])):4d}  {m['file']}")
        # a per-group tally, so you can confirm the counts before building a group dataset
        from collections import Counter
        tally = Counter(m.get('group', '_root') for m in reg.values())
        print(f"\n{len(reg)} features registered   groups: "
              + ", ".join(f"{g}={n}" for g, n in sorted(tally.items())))
        return

    # ---- the drop folder must exist. we do NOT create it. -------------------
    # if it is missing, somebody gave the wrong address. silently making an empty folder
    # would report '0 features, done!' as if everything were fine.
    if not C.FEATURES_DIR.exists():
        raise SystemExit(f"feature folder not found: {C.FEATURES_DIR}\n"
                         f"the feature team drops one parquet per feature there.")

    # RECURSE into sub-folders. the feature team groups parquets by folder
    # (Bucket_Raw_Features, Bucket_Features, Raw_Computed_Faetures ...). each folder is a GROUP.
    # .iterdir() only saw the top level, so a grouped drop was INVISIBLE -- 0 files, "done!".
    files = sorted(p for p in C.FEATURES_DIR.rglob("*.parquet")
                   if not p.name.startswith(".") and "__pycache__" not in p.parts)
    if not files:
        raise SystemExit(f"no .parquet files under {C.FEATURES_DIR} (searched sub-folders too)")

    print(f"scanning {C.FEATURES_DIR}  ->  {len(files)} feature parquets\n")

    added, updated, skipped = 0, 0, 0
    seen_this_scan = {}
    for p in files:
        name = feature_name(p)
        rel_file = p.relative_to(C.FEATURES_DIR).as_posix()   # e.g. Bucket_Features/gap_state_bucket.parquet
        group = feature_group(p)                              # the sub-folder this came from

        # two different files cannot own one name. if that happens the second one would be
        # silently ignored for ever, and the team would think their feature was in the
        # pipeline when it was not. so we refuse, loudly.
        # two different files cleaning to one name is refused ALWAYS -- --rescan used to
        # bypass this, so on a rescan the last file scanned silently took the name over and the
        # team thought BOTH features were in the pipeline. (seen_this_scan catches two NEW files
        # colliding within one scan, which the registry alone cannot see.)
        if name in seen_this_scan:
            print(f"  DUPLICATE refused: '{name}' claimed by both "
                  f"{seen_this_scan[name]} and {rel_file}. rename one of them.")
            continue
        if name in reg and reg[name]["file"] != rel_file:
            print(f"  DUPLICATE refused: '{name}' is already {reg[name]['file']}, "
                  f"now also {rel_file}. rename one of them (or delete the registry entry "
                  f"if the file was renamed on purpose).")
            continue
        seen_this_scan[name] = rel_file

        if name in reg and not a.rescan:
            skipped += 1
            print(f"  KNOWN  {name:32s} {reg[name].get('clock','?'):7s} "
                  f"{reg[name].get('na_policy','?')}")
            continue

        # a per-file heartbeat. clock measurement is a groupby over ~513k rows PER COLUMN, so a
        # wide base table (170 cols) is silent for a minute or two. without this line the whole
        # scan looks frozen for 25-40 min and the temptation is to kill it -- which is the one
        # thing that actually breaks it. flush so it shows immediately, not buffered.
        print(f"  scan   {name:32s} [{group}] reading + measuring clocks ...", flush=True)
        try:
            # anything a human already cleared in allow_columns survives the rescan
            info = inspect(p, allow=(reg.get(name, {}) or {}).get("allow_columns"))
        except Exception as e:
            print(f"  BAD    {name:32s} {type(e).__name__}: {e}")
            continue

        old = reg.get(name, {})
        # NEVER clobber a clock a human set. register.py's own header tells the feature team to
        # put THEIR number here -- and --rescan used to silently throw it away, which killed the
        # only guard against a mis-measured clock (= lookahead). measured goes in a separate
        # field so a disagreement is visible instead of erased.
        keep_clock = old.get("clock") or info["clock"]
        reg[name] = {
            "file": rel_file,
            # the sub-folder this feature came from. make_version --group <name> selects a whole
            # group in one shot. a human never has to tick 30+ boxes to build a group dataset.
            "group": group,
            "clock": keep_clock,
            "clock_measured": info["clock"],
            # keep whatever the human already set; only default it the first time
            "na_policy": old.get("na_policy", C.DEFAULT_NA_POLICY),
            "desc": old.get("desc", ""),
            "columns": info["columns"],
            # the refused columns, with the reason, kept on the record. a human reading the
            # ballot can see WHAT was taken away and WHY -- silence would just look like the
            # file had fewer columns than it does.
            "banned": info["banned"] or None,
            # a human writes column names in here, with a comment saying why. they then survive
            # the leak guard. it is explicit, it is in the file, and it has a name against it.
            "allow_columns": old.get("allow_columns") or None,
            # the human's per-column clock claims survive a rescan too -- --rescan used to
            # silently delete them, which un-fixed gap_state on the next routine scan.
            "clock_override": old.get("clock_override") or None,
            # the per-column evidence, ON THE RECORD. these were computed, printed, and then
            # thrown away -- the ballot reader had to take the one-number summary on faith.
            "clock_per_column": info.get("clock_per_column") or None,
            "clock_unsure": info.get("clock_unsure") or None,
            "categorical": info["categorical"],
            "rows": info["rows"],
            "span": info["span"],
            "na_counts": info["na_counts"],
            "added": old.get("added", dt.date.today().isoformat()),
        }
        tag = "UPDATE" if old else "+ NEW "
        added += (0 if old else 1)
        updated += (1 if old else 0)
        print(f"  {tag} {name:32s} {info['clock']:7s} {len(info['columns'])} cols"
              f"{'  CATEGORICAL: ' + ','.join(info['categorical']) if info['categorical'] else ''}")
        if info["has_na"]:
            worst = max(info["na_counts"].items(), key=lambda kv: kv[1])
            print(f"         NaN in {len(info['na_counts'])} col(s), worst = {worst[0]} "
                  f"({worst[1]:,} rows, {worst[1]/info['rows']*100:.1f}%)  "
                  f"-> na_policy: {reg[name]['na_policy']}")

    save_registry(reg)
    print(f"\ndone. {added} new, {updated} re-measured, {skipped} already known, "
          f"{len(reg)} total  ->  {C.REGISTRY.name}")
    print("\nNEXT: open registry.yaml and set  na_policy  +  desc  for anything new.")
    print("      only the people who wrote the feature know what its NaN means.")
    print("      then:  python core/make_version.py --new")


if __name__ == "__main__":
    main()
