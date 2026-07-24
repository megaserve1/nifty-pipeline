"""
bridge/build_dataset.py -- build ONE dataset version + its certificate.

    versions/dataset_vN.yaml        the recipe: which features the expert picked
    data/features/*.parquet         the feature tables the feature team handed us
    the labels csv                  one label + one weight per minute
              |
              v
    datasets/vN/dataset_vN.parquet  the training table -- one row per minute
    datasets/vN/manifest.json       the certificate: schema, checksums, provenance, and the
                                    sworn statement that the no-lookahead rule was applied

WHAT THIS DOES, IN ORDER
    1. read the labels. they are the SPINE -- one row per minute, and nothing changes that.
    2. for each picked feature: read its parquet, apply its NaN policy, and ALIGN it to the
       spine so that no minute ever sees a bar that has not closed yet.
    3. glue them side by side. one concat, not a chain of merges.
    4. write the parquet and the certificate.

THE NO-LOOKAHEAD RULE lives in bridge/align.py. It is the most important line in the project.
This file's job is to feed it the right clock for every feature and to record what it did.

this is the LAST file on the temporary side. everything after it (dvc push, ClearML publish,
training) is the permanent core, which never imports this module.

run:
    python bridge/build_dataset.py --version v3
"""
import argparse
import getpass
import sys
import pathlib

import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C          # noqa: E402
import contract             # noqa: E402
import hashes               # noqa: E402
from bridge.align import align_feature_to_labels, column_clocks, infer_bar_minutes  # noqa: E402
from bridge import leak_guard  # noqa: E402
from na_policy import apply_policy                            # noqa: E402
from bridge.register import read_time_index                          # noqa: E402


def load_recipe(version: str) -> dict:
    p = C.VERSIONS_DIR / f"dataset_{version}.yaml"
    if not p.exists():
        have = sorted(x.stem.replace("dataset_", "") for x in C.VERSIONS_DIR.glob("dataset_v*.yaml")
                      if not x.name.endswith(".lock.yaml"))
        raise SystemExit(f"unknown version '{version}'. existing: {have or 'none'}")
    return yaml.safe_load(p.read_text())


def load_registry() -> dict:
    if not C.REGISTRY.exists():
        raise SystemExit("no registry.yaml -- run  python bridge/register.py  first")
    return yaml.safe_load(C.REGISTRY.read_text()) or {}


def load_labels() -> pd.DataFrame:
    """the spine. one row per minute. every feature gets bent to fit THIS."""
    labels_path = C.labels_csv()       # resolved HERE, the one place that needs it
    lab = pd.read_csv(labels_path, low_memory=False)
    raw_ts = lab[C.LABEL_TS_COL]
    ts = pd.to_datetime(raw_ts, format=C.LABEL_TS_FORMAT, errors="coerce")
    # THE FIXED FORMAT IS FOR ONE LABELLING RUN, AND A NEW LABELS FILE CAN USE A DIFFERENT ONE.
    # the earlier set was %d-%m-%Y; the anchor set is ISO (2020-01-01 09:15:00). parsed with the
    # WRONG fixed format, every timestamp became NaT and the next line dropped ALL 513k rows --
    # then the build wrote a 0-row parquet that PASSED the manifest check ("OK 0 rows") and was
    # one dry-run away from training three models on nothing. so: if the fixed format fails to
    # parse most rows, fall back to inference; and if inference ALSO can't, STOP loudly instead of
    # silently dropping the dataset to nothing.
    if ts.isna().mean() > 0.01:
        ts = pd.to_datetime(raw_ts, errors="coerce")
    bad = int(ts.isna().sum())
    if bad and bad / max(len(lab), 1) > 0.01:
        raise SystemExit(
            f"{bad:,} of {len(lab):,} label timestamps ({bad / len(lab) * 100:.1f}%) could not be "
            f"parsed. neither C.LABEL_TS_FORMAT={C.LABEL_TS_FORMAT!r} nor inference matched them. "
            f"first raw values: {list(raw_ts.head(3))}. fix the format or the labels file -- "
            f"refusing to build a dataset that silently drops most of its rows.")
    lab[C.LABEL_TS_COL] = ts
    lab = lab.dropna(subset=[C.LABEL_TS_COL])
    # 6 of the 7 label strings carry a trailing space in the raw file. strip them, or every
    # equality check and the label encoder break in a way that is very hard to see.
    lab[C.LABEL_COL] = lab[C.LABEL_COL].astype(str).str.strip()
    for w in (C.WEIGHT_COL, C.WEIGHT_RAW_COL):
        if w in lab.columns:
            lab[w] = pd.to_numeric(lab[w], errors="coerce")
    lab = lab.sort_values(C.LABEL_TS_COL).reset_index(drop=True)

    dup = int(lab[C.LABEL_TS_COL].duplicated().sum())
    if dup:
        raise SystemExit(f"the labels have {dup:,} duplicate timestamps -- refusing to build")
    return lab


def _as_minutes(v) -> int:
    if not v:
        return 0
    try:
        return int(str(v).replace("min", "").strip())
    except ValueError:
        return 0


def resolve_declared_clock(name: str, recipe: dict, reg: dict) -> int:
    """the clock a HUMAN actually declared for this feature. 0 means nobody did.

    THE TRAP THIS AVOIDS, AND IT IS A BAD ONE.

    registry.yaml's `clock:` is written BY register.py, as infer_bar_minutes() -- which is the
    MAX over the file's columns, i.e. THE SLOWEST COLUMN IN THE FILE. It is a one-number summary
    for a human reading the ballot. It is NOT a declaration.

    But this function used to hand it straight to build_dataset, which applies it as a FLOOR:

        safe = max(measured, declared)

    merged_raw_1min has 182 columns. One of them (gap_state) measures 60min. So the file's
    summary `clock:` is 60min -- and every other column, including `close`, which genuinely
    updates every single minute, would have been held back to  max(1, 60) = 60 MINUTES STALE.
    181 columns crippled because one column is slow. The whole dataset, an hour behind.

    That is exactly the failure column_clocks() exists to prevent, and its own docstring names:
        "the SLOWEST clock -> the 1-minute column is held back = needlessly stale"

    So: a number the MACHINE wrote is not a declaration. We only treat `clock:` as a human's
    word when a human has actually CHANGED it -- i.e. when it differs from `clock_measured`,
    which register.py writes alongside it and never touches again. Same value in both fields
    means nobody has said anything, and the per-column measurement stands on its own.
    """
    # the frozen recipe wins -- that clock was chosen deliberately when the version was made
    frozen = _as_minutes((recipe.get("feature_clocks") or {}).get(name))
    if frozen:
        return frozen

    meta = reg.get(name, {}) or {}
    clock = _as_minutes(meta.get("clock"))
    measured = _as_minutes(meta.get("clock_measured"))

    # clock == clock_measured  ->  register.py wrote both. no human has spoken. ignore it.
    if not clock or clock == measured:
        return 0
    return clock


def resolve_clock_override(name: str, reg: dict) -> dict:
    """the ONE way to serve a column FASTER than the data can prove is safe.

    WHY THIS EXISTS, AND WHY IT IS DELIBERATELY UGLY TO USE.

    The measurement can only give us an UPPER BOUND on a column's clock. It works by looking for
    repetition -- a 5-minute value smeared over five 1-minute rows repeats, and that repetition
    is the fingerprint. But a column that simply DOES NOT CHANGE MUCH repeats too, and there is
    no way on earth to tell those two apart from the data alone.

    gap_state is the live example. It is NO_GAP about 86% of the time, so it sits perfectly still
    inside a 60-minute bar and the detector calls it a 60-minute column. That answer is SAFE
    (stale, never early) but it is wrong, and it is expensive: gap_state would be served an hour
    late and blanked for the first hour of every session -- which is precisely the hour when the
    gap matters most. It is really a 1-minute column. It is knowable AT THE OPEN, because it is
    computed from yesterday's close and today's open, and neither of those is in the future.

    Only the feature team knows that. No measurement will ever discover it. So there has to be a
    way for a human to say it -- but it must be a DELIBERATE, SEPARATE, RECORDED act, not a
    quiet edit to a `clock:` field that register.py itself wrote. A declaration the machine
    seeded is not a declaration; it is the measurement wearing a hat, and the old code let it
    silence its own alarm.

    So:  clock:          the safe default. the measurement. we never go faster than this.
         clock_override: a human, on the record, saying "I know how this is computed, and it is
                         point-in-time." It lands in the manifest, so the claim has a name on it.

    registry.yaml:
        gap_final_features:
          clock: 60min                       # <- what the data looks like
          clock_override:
            gap_state: 1min                  # <- what it IS. set by a human, on purpose.
            _why: "gap state is fixed at the open from yesterday's close. not a bar."
    """
    ov = (reg.get(name, {}) or {}).get("clock_override") or {}
    return {c: _as_minutes(v) for c, v in ov.items()
            if not str(c).startswith("_") and _as_minutes(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, metavar="vN")
    a = ap.parse_args()
    version = a.version

    recipe_path = C.VERSIONS_DIR / f"dataset_{version}.yaml"
    recipe = load_recipe(version)
    reg = load_registry()
    feats = recipe["features"]
    print(f"building {recipe['name']}  ({len(feats)} features)")

    # ---- 1. the spine -------------------------------------------------------
    print(f"[1/5] labels  <- {C.labels_csv().name}")
    lab = load_labels()
    label_ts = lab[C.LABEL_TS_COL]
    print(f"      {len(lab):,} minutes  ({label_ts.min()} -> {label_ts.max()})")

    # ---- 2. one feature at a time: NaN policy, then align -------------------
    print(f"[2/5] aligning {len(feats)} features   (no-lookahead rule: bar_close)")
    pieces, per_feature, drop_rows = [], [], pd.Series(False, index=lab.index)

    for name in feats:
        meta = reg.get(name)
        if meta is None:
            raise SystemExit(f"'{name}' is in the recipe but not in registry.yaml -- "
                             f"run  python bridge/register.py")
        path = C.FEATURES_DIR / meta["file"]
        if not path.exists():
            raise SystemExit(f"feature '{name}' has no parquet at {path}\n"
                             f"the feature team must drop it in {C.FEATURES_DIR}")

        raw = read_time_index(pd.read_parquet(path))

        # ---- THE LEAK GUARD, A SECOND TIME. ----------------------------------------
        # register.py already refused these, but this file reads the PARQUET, not the registry's
        # column list -- so a ban that only lived in registry.yaml would be decoration. The
        # parquet is the thing we actually ingest, so the parquet is the thing we screen.
        # It also covers the case where someone hand-edits registry.yaml to put a banned column
        # back, and the case where the feature team quietly re-drops a file with a new column in
        # it and nobody re-runs register.py.
        screen = leak_guard.screen(raw, allow=meta.get("allow_columns"))
        # a raw date/timestamp is NEVER a feature, in either mode -- it lets the model memorise
        # which day it is. dropped up front so 'allow everything' can never let it back in.
        cal = [c for c in raw.columns if c.lower() in C.CALENDAR_ALWAYS_DROP]
        if screen["banned"] or cal:
            leak_guard.report(screen, where=f"{name} ({meta['file']})")
            if C.LEAK_GUARD_ENFORCE:
                keep = [c for c in screen["ok"] if c not in cal]      # guard armed: drop all bans
            else:
                # report-only (feature team confirmed no lookahead): keep the flagged columns,
                # but still drop the raw calendar columns above.
                keep = [c for c in raw.columns if c not in cal]
                dropped = [c for c in raw.columns if c not in keep]
                if dropped:
                    print(f"      LEAK_GUARD_ENFORCE=False -> keeping flagged columns; "
                          f"still dropping calendar {dropped}")
            raw = raw[keep]
            if not len(raw.columns):
                raise SystemExit(f"every column of '{name}' was refused -- it is a working file, "
                                 f"not a feature file. take it out of {C.FEATURES_DIR}.")

        cat_cols = [c for c in raw.columns
                    if str(raw[c].dtype) in ("object", "str", "string", "category", "bool")]
        num_cols = [c for c in raw.columns if c not in cat_cols]

        policy = meta.get("na_policy", C.DEFAULT_NA_POLICY)

        # EVERY COLUMN GETS ITS OWN CLOCK.
        # one parquet can hold a 5-minute signal next to the 1-minute price it was computed
        # from. if we forced ONE clock on the file, the fast column would drag the slow one
        # down to a 1-minute clock -- and the slow column would then be served BEFORE ITS BAR
        # CLOSED. that is the lookahead leak, straight back in. measured: it hits 4 of every
        # 5 minutes. so each column is aligned on its own terms.
        #
        # MEASURED ON THE RAW FRAME. NEVER THE TREATED ONE. this used to run AFTER apply_policy
        # -- and a 'zero' policy fills NaN with 0.0, so a slow column with missing minutes
        # suddenly MOVES inside its own bar (value, 0, value). movement reads as a fast clock,
        # and a fast clock is the leak direction. the NaN policy is about what missing MEANS;
        # the clock is about when a value is KNOWABLE. they must not contaminate each other.
        measured = column_clocks(raw)
        declared = resolve_declared_clock(name, recipe, reg)
        overrides = resolve_clock_override(name, reg)

        # an override naming a column that does not exist is a typo waiting to hide a leak fix.
        # it must not be a silent no-op -- the human thinks the override is live.
        for c in overrides:
            if c not in measured:
                print(f"      !! clock_override names a column '{c}' that '{name}' does not "
                      f"have. columns are: {sorted(measured)[:8]}... FIX registry.yaml -- "
                      f"this override is doing NOTHING.")

        # keep the NaN. xgboost and catboost learn a branch for it; the trainer gives random
        # forest a sentinel at fit time, per model, so all three see the same feature list.
        # the ffill cap is in BARS, so hand it this file's real (slowest) bar period -- it was
        # hardwired to 1, which capped 'ffill' at 3 source ROWS instead of 3 bars.
        treated, na_note = apply_policy(
            raw, policy, for_model="native",
            bar_minutes=max(measured.values(), default=1),
            tolerance_bars=C.STALE_TOLERANCE_BARS,
            sentinel_margin=C.SENTINEL_MARGIN,
            categorical_na_label=C.CATEGORICAL_NA_LABEL,
            fixed_sentinel=C.NA_FIXED_SENTINEL,   # -999 for ALL models when set; None = per-column
        )

        # THE MEASUREMENT IS THE CEILING, NOT A SUGGESTION.
        #
        # column_clocks() returns the LARGEST period a column never moves inside. That is an
        # UPPER BOUND on its true clock: the real clock is that, or faster. So serving at the
        # measured clock is ALWAYS safe (stale at worst), and serving faster than it is a claim
        # nothing in the data can back up.
        #
        # This used to be the other way round: a human's `clock:` was trusted and the
        # measurement was only an "alarm". But register.py WROTE that `clock:` field from the
        # very same measurement -- so the alarm was comparing the measurement against a copy of
        # itself, and could never fire. The guard was dead by construction.
        #
        # Now: we take the SLOWER of (declared, measured), always. The one and only way to go
        # faster is an explicit clock_override, set by a human on purpose, which we shout about
        # and write into the manifest so the claim has a name on it. See resolve_clock_override.
        clocks, used_overrides = {}, {}
        for c, m in measured.items():
            safe = max(m, declared) if declared else m

            if c in overrides:
                want = overrides[c]
                clocks[c] = want
                used_overrides[c] = {"used": want, "measured": m}
                if want < safe:
                    print(f"      !! CLOCK OVERRIDE  {name}__{c}: the data looks like {m}min, "
                          f"but a human has declared {want}min.")
                    print(f"         using {want}min. THIS IS A HUMAN CLAIM, NOT A MEASUREMENT --")
                    print(f"         it says this column is point-in-time and does not wait for a")
                    print(f"         bar to close. if that is wrong, it is a LOOKAHEAD LEAK.")
                    print(f"         it is recorded in the manifest under clock_overrides.")
                continue

            if declared and m > declared:
                print(f"      !! CLOCK ALARM  {name}__{c}: declared {declared}min, but the data "
                      f"looks like {m}min.")
                print(f"         using {m}min (the slower, safe one). if the column is really "
                      f"{declared}min it just changes slowly -- and if so, say it on purpose "
                      f"with a clock_override in registry.yaml. do NOT edit `clock:`.")
            clocks[c] = safe

        if not declared and not overrides:
            # nobody declared anything. what the data says, which errs SLOW -- stale is honest,
            # early is lookahead.
            print(f"      !! no clock declared for {name} -- using the measurement. "
                  f"ask the feature team to confirm it in registry.yaml.")
            clocks = measured

        groups = {}                                  # clock -> the columns on that clock
        for c, mins in clocks.items():
            groups.setdefault(mins, []).append(c)

        parts = []
        for mins in sorted(groups):
            sub = treated[groups[mins]]
            a_sub = align_feature_to_labels(sub, label_ts, bar_minutes=mins,
                                            tolerance_bars=C.STALE_TOLERANCE_BARS)
            parts.append(a_sub)
        aligned = pd.concat(parts, axis=1)[list(treated.columns)]   # keep the original order
        aligned.columns = [f"{name}__{c}" for c in aligned.columns]

        if policy == "drop":
            # this feature says some minutes are genuinely unusable. remove THE LABEL MINUTES
            # THAT WOULD HAVE BEEN SERVED THE UNUSABLE BAR -- not the minutes of its stamp.
            #
            # THE OFF-BY-ONE-BAR THIS USED TO HAVE: na_note hands back the FEATURE-row stamps
            # of the unusable bars. an unusable bar STAMPED 09:15 on a 5-min clock is SERVED at
            # 09:20..09:24 -- so dropping label minute 09:15 deleted a minute that was fed a
            # perfectly good (09:10) bar, and KEPT 09:20-09:24, the minutes actually poisoned.
            # the stamps must be pushed through bar_close first, per column-clock group, and the
            # whole serving window (close .. close + tolerance) removed.
            #
            # note we use the mask na_policy handed back, NOT "the aligned row is all-NaN".
            # an all-NaN aligned row only tells you no bar had closed yet; it says nothing about
            # whether the underlying data was unusable. and we must never DELETE the source rows,
            # because a hole in the feature table gets forward-filled by align() -- which turns
            # 'drop' into 'quietly serve a stale value'. exactly the opposite of what it means.
            bad_ts = na_note.get("unusable_index")
            if bad_ts is not None and len(bad_ts):
                from bridge.align import bar_close
                bad_idx = pd.DatetimeIndex(bad_ts)
                worst = max(clocks.values(), default=1)      # the slowest clock serves longest
                closes = bar_close(bad_idx, worst)
                lts = pd.DatetimeIndex(label_ts)
                poisoned = pd.Series(False, index=range(len(lts)))
                horizon = pd.Timedelta(minutes=worst * C.STALE_TOLERANCE_BARS)
                for cl in pd.DatetimeIndex(closes).unique():
                    poisoned |= pd.Series((lts >= cl) & (lts < cl + horizon))
                drop_rows |= poisoned.to_numpy()
                na_note["unusable_label_minutes"] = int(poisoned.sum())

        pieces.append(aligned)

        no_bar = float(aligned.isna().all(axis=1).mean() * 100)
        clock_str = "/".join(f"{m}min" for m in sorted(groups))
        src = "declared+measured" if declared else "measured"
        print(f"      +  {name:28s} clock={clock_str:12s} ({src:17s}) na={policy:8s} "
              f"{aligned.shape[1]:3d} cols  {no_bar:5.2f}% minutes with no closed bar")
        if len(groups) > 1:
            for m in sorted(groups):
                print(f"           {m:>2}min: {', '.join(groups[m])}")
        if cat_cols:
            print(f"         categorical: {', '.join(cat_cols)}  "
                  f"(catboost reads these as-is; RF/XGB get them encoded)")

        per_feature.append({
            "name": name,
            "file": meta["file"],
            "column_clocks": {f"{name}__{c}": m for c, m in clocks.items()},
            "clock_minutes": max(clocks.values()) if clocks else 1,
            "clock_source": src,
            "clock_measured": {f"{name}__{c}": m for c, m in measured.items()},
            # every place a HUMAN overrode the measurement to serve a column FASTER than the data
            # can prove is safe. this is the only door the leak can come back through, so it is
            # written on the certificate, by name, every time it is used. an empty dict is the
            # normal, boring, safe case.
            "clock_overrides": used_overrides,
            "na_policy": policy,
            # THE CERTIFICATE DESCRIBES THE ARTIFACT, NOT ITS INGREDIENTS.
            # na_note carries the SOURCE file's NaN counts (useful context) -- but a reader of
            # the manifest needs to know what is missing in the parquet THEY WERE HANDED, which
            # also includes the NaN that ALIGNMENT introduces (no bar closed yet). both are
            # recorded, clearly named. and a raw DatetimeIndex must never go into json -- it
            # serialized as a TRUNCATED python repr ('...' in the middle of the certificate).
            "na_treatment": {
                **{k: (v if not isinstance(v, pd.DatetimeIndex) else
                       {"n": int(len(v)),
                        "first": str(v.min()) if len(v) else None,
                        "last": str(v.max()) if len(v) else None})
                   for k, v in na_note.items()},
                "na_in_built_output": {c: int(aligned[c].isna().sum())
                                       for c in aligned.columns
                                       if int(aligned[c].isna().sum()) > 0},
            },
            "categorical": [f"{name}__{c}" for c in cat_cols],
            "parquet_sha256": hashes.sha256_file(path),
            "columns": list(aligned.columns),
        })

    # ---- 3. glue side by side ----------------------------------------------
    print("[3/5] assembling")
    X = pd.concat(pieces, axis=1)
    keep_label = [c for c in (C.LABEL_TS_COL, C.LABEL_COL, C.WEIGHT_COL, C.WEIGHT_RAW_COL)
                  if c in lab.columns]
    df = pd.concat([lab[keep_label].reset_index(drop=True), X], axis=1)

    if bool(drop_rows.any()):
        n = int(drop_rows.sum())
        df = df[~drop_rows.values].reset_index(drop=True)
        print(f"      dropped {n:,} minutes (a feature with na_policy=drop could not serve them)")

    # ---- the row id --------------------------------------------------------
    # stamped HERE, after the merge on timestamp and after any drops, so it is a clean 0..n-1 on
    # the FINAL rows. it is an IDENTIFIER, NOT A FEATURE: it is added to df but never to
    # feature_columns below (which is list(X.columns) -- features only), and it carries no '__',
    # so the trainer's feature selection (manifest feature_columns, or the '__' fallback) can never
    # pick it up. the model never sees it, so it cannot memorise row position -- which is exactly
    # why leak_guard bans an id COUNTER when it arrives as a FEATURE. as a downstream JOIN KEY it
    # is safe, and it is what the scored tables and the next team merge on (an integer is safer to
    # join than a datetime). timestamp is the merge key; this is the id stamped on the result.
    df.insert(0, C.INDEX_COL, range(len(df)))
    print(f"      stamped {C.INDEX_COL} (id column, not a feature): 0..{len(df)-1:,}")

    feature_columns = list(X.columns)
    categorical_columns = [c for f in per_feature for c in f["categorical"]]
    complete = int(X.notna().all(axis=1).sum())
    print(f"      {len(df):,} rows x {df.shape[1]} cols")
    print(f"      {complete:,} rows have every feature present "
          f"({complete/max(len(df),1)*100:.1f}%)")

    if C.NA_FIXED_SENTINEL is not None:
        # FLAT SENTINEL (project decision): fill EVERY remaining NaN -- the feature's own missing
        # AND the no-closed-bar minutes alignment leaves at each session open -- so the parquet
        # carries no NaN at all. numeric -> the fixed value; text -> the categorical marker (a
        # number cannot live in a text column). done AFTER assembly so alignment NaN is caught too.
        catset = set(categorical_columns)
        num_feat = [c for c in feature_columns if c not in catset]
        cat_feat = [c for c in feature_columns if c in catset]
        n_num = int(df[num_feat].isna().sum().sum()) if num_feat else 0
        if n_num:
            df[num_feat] = df[num_feat].fillna(C.NA_FIXED_SENTINEL)
        n_cat = 0
        for c in cat_feat:
            m = df[c].isna()
            if m.any():
                n_cat += int(m.sum())
                df[c] = df[c].astype("object").where(~m, C.CATEGORICAL_NA_LABEL)
        print(f"      NA_FIXED_SENTINEL: filled {n_num:,} numeric NaN with {C.NA_FIXED_SENTINEL} "
              f"and {n_cat:,} categorical NaN with '{C.CATEGORICAL_NA_LABEL}' -> no NaN left")
    else:
        print(f"      NOTE: rows with NaN are KEPT. the NaN is information (see na_policy), and")
        print(f"            xgboost/catboost learn from it. dropping them would delete the dataset.")

    # ---- 4. write the parquet ----------------------------------------------
    #
    # BOOL + NaN = A MIXED object COLUMN, AND PARQUET REFUSES IT.
    #
    #     ArrowTypeError: Expected bytes, got a 'bool' object
    #     Conversion failed for column adiii_merged_ml__pat_is_green
    #
    # pat_is_green is a bool. Alignment then puts NaN in the first minute of every session (no
    # bar has closed yet -- correct, and deliberate). But numpy bool cannot hold NaN, so pandas
    # silently upcasts the column to `object`, and it ends up holding True, False AND NaN
    # together. pyarrow inspects it, guesses one type, meets the other, and dies.
    #
    # The fix is not to fill the NaN -- that NaN is real information and na_policy already had
    # its say. It is to give the column ONE type. These are all categorical columns anyway
    # (register.py files bool under categorical, because a bool IS a two-valued category), and
    # catboost/the encoders want strings. So: every non-null value becomes a string, the NaN
    # stays NaN, and the column has a single, honest type.
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    mixed = []
    for c in obj_cols:
        kinds = {type(v).__name__ for v in df[c].dropna().head(5000)}
        if len(kinds) > 1 or "bool" in kinds:
            df[c] = df[c].map(lambda v: v if pd.isna(v) else str(v))
            mixed.append(c)
    if mixed:
        print(f"      {len(mixed)} bool/mixed column(s) cast to string so parquet can hold them")
        print(f"      (a bool cannot hold NaN, so pandas made them `object`. the NaN is KEPT.)")

    out_dir = C.DATASETS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / f"dataset_{version}.parquet"
    print(f"[4/5] writing {parquet}")
    df.to_parquet(parquet, index=False)

    # THE SCHEMA IN THE CERTIFICATE MUST DESCRIBE THE FILE AS A READER WILL SEE IT.
    #
    # It used to be taken from the in-memory frame (`df.dtypes`) -- and the parquet round-trip
    # CHANGES dtypes. Measured on v2: 11 columns held as `object` in memory (strings + NaN)
    # came back as `str` after write+read (pandas 3's arrow-backed string type). The manifest
    # said 'object', the parquet said 'str', and contract.assert_schema -- which compares
    # strictly, as it should -- would have refused the trainer's own certified dataset with
    # "SCHEMA DRIFT: retyped". A certificate that describes the ingredients instead of the cake.
    #
    # So: write the file, read it back, and certify WHAT CAME BACK. The re-read costs a few
    # seconds once per build and makes the schema true by construction.
    df_check = pd.read_parquet(parquet)
    if len(df_check) != len(df) or list(df_check.columns) != list(df.columns):
        raise SystemExit("the parquet did not read back with the same shape it was written -- "
                         "refusing to certify it")
    schema_as_read = [{"col": c, "dtype": str(t)}
                      for c, t in zip(df_check.columns, df_check.dtypes)]
    del df_check

    # ---- 5. the certificate -------------------------------------------------
    print("[5/5] manifest")
    dist = df[C.LABEL_COL].value_counts().to_dict()

    # the weight column ranks CONVICTION, not rarity. surface the zero-weight classes here,
    # because a class at weight 0 can never be learned and the trainer must be able to say so.
    zero_w = []
    if C.WEIGHT_COL in df.columns:
        mw = df.groupby(C.LABEL_COL)[C.WEIGHT_COL].mean()
        zero_w = sorted(c for c, m in mw.items() if float(m) == 0.0)

    contract.write_manifest(
        out_dir / "manifest.json",
        version=version,
        recipe_sha256=hashes.sha256_file(recipe_path),
        parquet_sha256=hashes.sha256_file(parquet),
        rows=int(len(df)),
        n_features=len(feature_columns),
        schema=schema_as_read,          # from the READ-BACK, never from the in-memory frame
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        label_col=C.LABEL_COL,
        weight_cols=[c for c in (C.WEIGHT_COL, C.WEIGHT_RAW_COL) if c in df.columns],
        per_feature=per_feature,
        labels_name=recipe.get("labels_name", C.labels_name()),
        labels_sha256=hashes.sha256_file(C.labels_csv()),
        class_distribution={str(k): int(v) for k, v in dist.items()},
        zero_weight_classes=zero_w,
        no_peek={"applied": True, "rule": "bar_close",
                 "tolerance_bars": C.STALE_TOLERANCE_BARS},
        built_by=getpass.getuser(),
        status="ready",
    )

    print("\nlabel distribution:")
    print(df[C.LABEL_COL].value_counts().to_string())
    if zero_w:
        print(f"\n  !! ZERO-WEIGHT CLASSES: {zero_w}")
        print(f"  !! these contribute NOTHING to a weighted fit, so no model can learn them.")
        print(f"  !! with NO_TRADE at weight 0 the model will want to trade every single minute.")
        print(f"  !! fix it in the LABEL POLICY (give it ~0.1-0.2), not in the trainer.")
    print(f"\nnext:  python core/publish_version.py --version {version}")


if __name__ == "__main__":
    main()
