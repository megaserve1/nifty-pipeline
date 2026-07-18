"""
final_pipeline / config.py
==========================
ONE settings file, split by OWNERSHIP into two fenced sections:

  [CORE]   -- the PERMANENT pipeline: select -> version -> dvc push -> publish -> train.
  [BRIDGE] -- the TEMPORARY ingest side. Moves to the feature team. Delete on handover day.

Per-feature settings never live here. Each feature declares its own clock and NaN policy in
registry.yaml, because only the people who WROTE the feature know what its NaN means.
This file does not grow when you go from 14 features to 400.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# =============================================================================
# [CORE]  -- permanent. core/ and trainer/ use only what is below.
# =============================================================================

VERSIONS_DIR = ROOT / "versions"        # frozen recipes (git-tracked, immutable)
DATASETS_DIR = ROOT / "datasets"        # the handover surface: datasets/vN/{parquet, manifest.json}
REGISTRY     = ROOT / "registry.yaml"   # the MENU of features
SELECTION_SHEET = ROOT / "selection_sheet.yaml"   # the expert's ballot
CONFIGS_DIR  = ROOT / "configs"

# The LOGICAL name of the label set. The recipe records this string, never a filesystem path,
# so make_version.py runs fine on a machine that has no labels CSV at all.
#
# DERIVED FROM THE FILE NAME. It used to be a hand-typed constant -- and when the labels file
# changed (7class -> 7class_notrade7k, the NO_TRADE weight fix), nobody updated the constant.
# Result: v1 and v2 are certified under the SAME labels_name while their labels_sha256 prove
# they were built from DIFFERENT files. A lineage record that lies is worse than none. The name
# now moves when the file does, and it cannot be forgotten because there is nothing to remember.
# (LABELS_FILE is defined below, in the [BRIDGE] section -- Python resolves this at call time.)
def labels_name() -> str:
    import re
    stem = LABELS_FILE.rsplit(".", 1)[0]
    # strip ANY  YYYY-MM-DD_HHMM  stamp so the logical name is stable across re-labels. hard-coding
    # one date meant a new labels file kept its full dated name -- the exact staleness this guards
    # against. the labels_sha256 in the manifest is what proves two files differ, not this name.
    return re.sub(r"^(labels_1min_)\d{4}-\d{2}-\d{2}_\d{4}_", r"\1", stem)

# ---- the three models ---------------------------------------------------------
MODEL_TYPES = ["random_forest", "xgboost", "catboost"]

# ---- ClearML ------------------------------------------------------------------
CLEARML_PROJECT   = "Nifty Production"
CLEARML_DATASET   = "nifty_signal_dataset"
TRAIN_QUEUE       = "training"          # a clearml-agent must listen here, or nothing runs
SHAP_QUEUE        = "training"          # SHAP runs on the same queue by default

def base_trainer_name(model_type: str) -> str:
    """The name of the base task publish_version.py clones for each model."""
    return f"train_{model_type} (base)"

BASE_SHAP_NAME     = "shap_explain (base)"
BASE_CHAMPION_NAME = "select_champion (base)"

# ============================================================================
# STORAGE MODE  -- THE ONE SWITCH. flip this word to move all data storage.
# ============================================================================
#   "gcs"    the production path. dataset bytes -> your GCS bucket via DVC; ClearML holds only
#            a POINTER (add_external_files). app.clear.ml sees metadata only. data on your GCP.
#
#   "local"  the all-local path, for 2 machines beside each other with a SELF-HOSTED ClearML
#            server. dataset bytes are uploaded to that server's fileserver (add_files), which
#            lives on YOUR machine. no GCS, no DVC. data never leaves your PCs.
#            >> only safe with a SELF-HOSTED server. pointing "local" at app.clear.ml (SaaS)
#               would upload your data to ClearML's cloud -- which breaks data residency. <<
#
# TO REVERT after the local run: set this back to "gcs". nothing else to touch.
STORAGE_MODE = "gcs"

# Where ClearML writes model artifacts. In "gcs" mode this MUST be your own GCS bucket --
# app.clear.ml holds only metadata, so no market data or model bytes ever leave company GCP.
# In "local" mode it is None: artifacts go to the self-hosted fileserver (see model_output_uri).
GCS_BUCKET     = "live-nifty-pipeline"          # production bucket for the live run (set 2026-07-16)
GCS_OUTPUT_URI = f"gs://{GCS_BUCKET}/clearml"


def model_output_uri():
    """where trained models are written. None in local mode = the self-hosted fileserver."""
    return None if STORAGE_MODE == "local" else GCS_OUTPUT_URI


# ---- DVC (only used in "gcs" mode) --------------------------------------------
DVC_REMOTE_NAME = "gcs"
DVC_REMOTE_URL  = f"gs://{GCS_BUCKET}/final_pipeline/dvc"

# ---- training ------------------------------------------------------------------
# The data is cut by TIME into three slices:
#
#     |<------- train ------->|xxx|<-- val -->|xxx|<---- test ---->|
#      oldest                                                 newest
#
# VAL is what the hyper-parameter search is allowed to look at. TEST is opened ONCE, at the
# end, on the winner. If we tuned on TEST and then reported TEST, the number would be the best
# of N noisy draws -- biased low by construction, and there would be nothing honest left to
# measure with. See trainer/objective.py for the worked example.
TEST_FRACTION = 0.30     # the most RECENT 30% of time is the test set. NEVER a random split.

# The tuning set. Set it to 0 to turn it OFF and get a plain two-way TRAIN | embargo | TEST.
#
#   VAL_FRACTION = 0.0   -> train 70 / test 30.  A SMOKE RUN. hpo.py cannot run (it refuses).
#   VAL_FRACTION = 0.15  -> train 55 / val 15 / test 30.  Required before you tune anything.
#
# WHY YOU CANNOT TUNE WITHOUT IT. The moment hpo.py runs 30 trials and keeps the best TEST
# score, the test set has entered the training loop. You did not fit the model on it -- you
# fitted the SETTINGS on it, and settings are parameters too. The winning number is then the
# best of 30 noisy draws: biased low by construction, and there is nothing honest left to
# report. So HPO tunes on VAL, and TEST is opened once, at the end, on the winner.
# Fine to leave at 0 while you are just watching the pipeline run.
VAL_FRACTION  = 0.15

# The gap thrown away between two slices (the "xxx" above).
#
# WHY IT EXISTS. A row just before a cut and a row just after it share almost all of their
# rolling-window history -- they are near-duplicates. Without a gap, the later slice is partly
# made of the earlier one and its score is flattered.
#
# WHY IT IS COUNTED IN SESSIONS, NOT CALENDAR DAYS. This was wrong until 2026-07-14. The old
# setting was EMBARGO_DAYS = 21, applied as CALENDAR days, to cover a feature lookback of
# "20 days". But "20 days" means 20 TRADING SESSIONS -- ret_20d is 20 bars of a DAILY series,
# and a daily series has one bar per session. The market is shut at weekends and on holidays,
# so 21 calendar days spans only ~14 sessions. Measured on the real label file: mean 14.1,
# min 9, max 16 -- NEVER 20, at any cut point in five years. You need 25-39 calendar days
# (mean 29.8) to span 20 sessions. So the embargo was ~30% short, and the first ~6 sessions of
# every test set still carried features built from training-period prices.
#
# tests/test_purged_cv.py::test_calendar_days_are_NOT_trading_sessions proves this on the real
# data. We now count the thing the feature counts: sessions, read off the timestamps
# themselves (no holiday table -- see trainer/purged_cv.sessions_of).
EMBARGO_SESSIONS = 25    # >= the longest feature lookback (20 sessions) + 5 sessions of margin

# The label's own horizon: how far FORWARD a label at time t has to look before it knows its
# own answer. It sets the purge on the OTHER side -- the last training rows must not have
# labels whose outcome resolves inside the test slice.
#
# !! THIS NUMBER IS NOT CONFIRMED. !!
# The label file is named ..._A20_Ex_z2_En_zNA_7class.csv and nobody has written down what the
# 20 in A20 is. 1 session is the conservative reading for an INTRADAY signal (a trade opened
# today is resolved today). If the label actually looks 20 sessions forward, this must become
# 20 and every purge in trainer/purged_cv.py changes with it.
# ASK THE PERSON WHO BUILT THE LABELS, then delete this warning.
LABEL_HORIZON_SESSIONS = 1

# The market's opening minute, as minutes past midnight. NSE opens at 09:15 -> 555.
# bar_close() anchors its bars HERE, not at midnight. This matters for any clock that does not
# divide 555 evenly: a 30-minute bar of the session runs 09:15-09:44 and closes at 09:45, but
# flooring from MIDNIGHT would put its close at 09:30 -- serving it 15 minutes early, which is
# a lookahead leak that no assert would catch. 555 = 3 x 5 x 37, so 1/3/5/15 divide it and
# 30/60 do not. See bridge/align.py.
SESSION_ANCHOR_MINUTES = 9 * 60 + 15     # 555

# The severity matrix: what each kind of mistake actually COSTS in trading terms.
SEVERITY_FILE = CONFIGS_DIR / "severity_7class.json"


# =============================================================================
# [BRIDGE]  -- temporary ingest. Delete this whole section on handover day.
# =============================================================================

DATA_DIR     = ROOT / "data"
OUT_DIR      = ROOT / "output"

# >>> THE FEATURE TEAM DROPS ONE PARQUET PER FEATURE HERE <<<
# Each file: data/features/<feature_name>.parquet
#   - the timestamp is the INDEX (or a datetime/timestamp column)
#   - the columns are the feature's own columns, nothing else
# We do NOT run notebooks any more. We ingest the parquets they hand us.
FEATURES_DIR = DATA_DIR / "features"

# the labels + weights (bridge-only; core never opens this file)
#
# THIS USED TO BE  Path.home() / "Downloads" / "labels_....csv"  AND THAT WAS A BUG.
#
# It is hard-wired to ONE laptop. Clone the repo onto the VM, or onto a colleague's machine, and
# there is no ~/Downloads/labels...csv -- the pipeline dies on the first line of load_labels().
# Worse, the file lived OUTSIDE the repo, so it was not versioned and not DVC-tracked: the
# manifest swears to a labels_sha256 for a file sitting in a Downloads folder that anybody could
# overwrite or delete. That is a lineage hole with the ground truth in it.
#
# The labels now belong IN the project, at data/labels/, and DVC tracks them like everything
# else. One `dvc pull` on any machine and they are there.
LABELS_DIR  = DATA_DIR / "labels"

# 2026-07-14: the NO_TRADE weight is no longer ZERO.
#
#   old file (..._7class.csv)          NO_TRADE weight 0.000  ->  0.0% of the loss
#   this file (..._notrade7k.csv)      NO_TRADE weight 0.064  -> 12.4% of the loss
#
# NO_TRADE is 53% of every row. With a weight of 0 those rows contributed NOTHING to the loss,
# so the model could not learn to stay out and wanted to trade every single minute. That is
# fixed. 12.4% is on the LOW side (20-40% was the aim -- weight_raw 12,000-33,000 rather than
# 7,000), so expect it still to lean toward over-trading. Do not argue about it: run
# trainer/local_check.py and read the "minutes it would trade" line. That is the evidence.
LABELS_FILE = "labels_1min_2026-07-17_1139_A20_Ex_z2_En_zNA_anchor_7class.csv"

# the previous file, kept so a result can be reproduced against it.
LABELS_FILE_ZERO_NOTRADE = "labels_1min_2026-07-03_1535_A20_Ex_z2_En_zNA_7class.csv"


def _find_labels() -> Path:
    """where the labels are, on THIS machine. searched in order, and it says so if it fails."""
    import os
    tried = []

    env = os.environ.get("NIFTY_LABELS")           # 1. an explicit override, for odd setups
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
        tried.append(f"$NIFTY_LABELS -> {p}")

    p = LABELS_DIR / LABELS_FILE                   # 2. THE RIGHT PLACE. in the repo, DVC-tracked.
    if p.exists():
        return p
    tried.append(str(p))

    p = Path.home() / "Downloads" / LABELS_FILE    # 3. the old spot. works, but only on ONE box.
    if p.exists():
        return p
    tried.append(str(p))

    # Do not return a path that does not exist -- that produces a FileNotFoundError 200 lines
    # later, from inside pandas, and it tells you nothing about what to do.
    raise SystemExit(
        "\n  the labels file was not found. looked in:\n    "
        + "\n    ".join(tried)
        + f"\n\n  put it in the project so every machine can find it:\n"
          f"      mkdir -p {LABELS_DIR}\n"
          f"      mv ~/Downloads/{LABELS_FILE} {LABELS_DIR}/\n"
          f"      dvc add {LABELS_DIR / LABELS_FILE}\n"
          f"      git add {LABELS_DIR / LABELS_FILE}.dvc && git commit -m 'labels'\n"
          f"      dvc push\n\n"
          f"  then on any other machine:  git clone ... && dvc pull\n")


# THE LABELS PATH IS RESOLVED LAZILY. NOT AT IMPORT. THIS WAS A CRASH.
#
# It used to say `LABELS_CSV = _find_labels()` right here, at module level -- which ran the
# search (and its SystemExit) the moment ANYTHING imported config. Every trainer, every core
# script, every test imports config. So a ClearML agent on a fresh VM clone -- which needs the
# labels for NOTHING (it fetches the built parquet from the bucket) -- died at `import config`
# before it could do a single thing. Proven by repro: `python -c "import config"` on a
# labels-less machine exits 1. It also contradicted this file's own promise that core never
# opens the labels.
#
# Only bridge/build_dataset.py may call this, and only when it actually builds.
def labels_csv() -> Path:
    """the labels file on THIS machine -- resolved when asked for, never at import."""
    return _find_labels()

LABEL_TS_COL    = "timestamp"
LABEL_COL       = "primary_label"
WEIGHT_COL      = "weight"
WEIGHT_RAW_COL  = "weight_raw"
LABEL_TS_FORMAT = "%d-%m-%Y %H:%M"      # e.g. 01-01-2020 09:15

# ---- CLASS WEIGHTS: a fixed weight per class, instead of the labels' per-row `weight` ---------
#
# Set to a dict -> every row is weighted by its CLASS (these numbers).
# Set to None    -> fall back to the labels file's per-row `weight` column.
#
# >>> KEYED BY CLASS NAME ON PURPOSE. THIS IS NOT A STYLE CHOICE. <<<
# The feature team's version used integer keys 0..6 in THEIR ordering (0 = No Trade). But sklearn's
# LabelEncoder sorts the class names ALPHABETICALLY, so in this pipeline:
#       0=ENTRY_SMALL 1=ENTRY_SUB 2=ENTRY_SUPER 3=EXIT_SMALL 4=EXIT_SUB 5=EXIT_SUPER 6=NO_TRADE
# Feeding their integer dict in directly would have given NO_TRADE the TOP weight (1.00) and the
# entries the BOTTOM (0.05) -- the exact opposite of the intent -- and nothing would have errored.
# A name -> weight map cannot be silently reordered, so that class of bug is impossible here.
# THESE ARE BALANCED (INVERSE-FREQUENCY) WEIGHTS -- sklearn's class_weight="balanced" formula:
#
#         weight_i = N / (K * n_i)        N = 513,611 rows, K = 7 classes, n_i = rows in class i
#
# computed on the anchor labels and rounded to 2dp. the property that makes them worth using:
# weight_i * n_i is CONSTANT, so every class commands the SAME ~14.3% share of the loss. that lifts
# the ENTRY classes from ~20% of the loss (under conviction weights) to 42.9% -- the rare classes
# the models were completely blind to.
#
# >>> DO NOT ROUND THESE TO INTEGERS. <<<  NO_TRADE (0.21) would round to 0, and 353,407 rows --
# 69% of the dataset -- would contribute NOTHING to the loss. the model could then never learn
# "don't trade" and would try to trade every minute. 2dp is the floor.
#
# NOTE: derived from the CURRENT label counts. if the labels change, recompute:
#         weight_i = len(labels) / (7 * rows_in_class_i)
CLASS_WEIGHTS = {
    "ENTRY_SUB":   12,      # rarest      (5,895 rows)   exact 12.4466
    "ENTRY_SMALL":  5,      #            (13,614 rows)   exact  5.3895
    "ENTRY_SUPER":  4,      #            (19,603 rows)   exact  3.7429
    "EXIT_SUB":     3,      #            (25,320 rows)   exact  2.8978
    "EXIT_SMALL":   2,      #            (34,830 rows)   exact  2.1066
    "EXIT_SUPER":   1,      #            (60,942 rows)   exact  1.2040
    "NO_TRADE":     0.2,    # most common (353,407 rows) exact  0.2076
                            # ^ STAYS DECIMAL ON PURPOSE. rounded to 0 it would delete 353,407 rows
                            #   (69% of the data) from the loss and the model could never learn
                            #   "don't trade". every other class is a whole number; this one is not.
}

# ---- what a feature's NaN MEANS ------------------------------------------------
# NaN does not mean the same thing in every feature, so there is no blanket rule. The team
# that wrote the feature declares its policy in registry.yaml. We obey it and record what we did.
#
#   sentinel  NaN is INTENTIONAL -- it is a real state, not missing data.
#             e.g. gap_fill_ratio is NaN when there is NO GAP: the question has no answer.
#             XGBoost + CatBoost get the real NaN (they learn a branch for it natively).
#             RandomForest cannot take NaN, so it gets a value BELOW ANYTHING REAL (see below),
#             which makes one clean cut separating "no gap" from every real value.
#             Never 0 and never the mean -- both are values the feature can genuinely take, so
#             they would COLLIDE with real rows and destroy the distinction.
#
#   zero      NaN honestly means zero (a count of nothing). Filled with 0 for all models.
#   ffill     a slow value carried on fast rows. Forward-filled, but bounded so it cannot leap
#             the overnight gap.
#   drop      the row is genuinely unusable. Those rows are excluded from the dataset.
#
NA_POLICIES = ("sentinel", "zero", "ffill", "drop")
DEFAULT_NA_POLICY = "sentinel"          # the safe default: assume the NaN means something

# The sentinel is COMPUTED PER COLUMN, never hardcoded. A fixed -999 would be fine for a 0..1
# ratio but a feature measured in points could legitimately BE -999 -- and then we would have
# recreated the very collision we are avoiding. So:
#       sentinel = column_min - (column_max - column_min) - 1
# which is guaranteed to sit below every value the feature can actually take. The exact value
# used is written into the manifest, so it is never a mystery later.
SENTINEL_MARGIN = 1.0

# forward-fill cap: how many bar-periods a value may be carried before it is called stale.
STALE_TOLERANCE_BARS = 3

# a missing CATEGORY keeps its own identity rather than being blended into a real one
CATEGORICAL_NA_LABEL = "MISSING"

# ---- leak-guard enforcement (project decision, 2026-07-17) ---------------------
# The feature team has CONFIRMED the current feature set has no walk-forward/lookahead, so the
# leak-guard runs in REPORT-ONLY mode: it still prints what it WOULD flag (kept on the record),
# but build_dataset does NOT drop those columns. Set back to True to re-arm the guard.
LEAK_GUARD_ENFORCE = False
# ...EXCEPT these. A raw DATE or TIMESTAMP is never a feature -- it lets the model memorise which
# day it is and fail on every unseen date. These are dropped even in report-only mode. (matched
# case-insensitively against the exact column name.)
CALENDAR_ALWAYS_DROP = ("session", "t5", "date", "datetime", "timestamp", "expiry_date")

# ---- fixed NaN sentinel (project decision, 2026-07-17) ------------------------
# When set, EVERY numeric NaN is filled with this one value for ALL models (no drop, no 0). The
# per-column formula above is safer (it can never collide), but a flat -999 is simpler to explain
# and is out of range for this feature set. Set to None to go back to the per-column sentinel.
# (Text/categorical NaN still becomes CATEGORICAL_NA_LABEL -- a number can't live in a text column.)
NA_FIXED_SENTINEL = -999.0


# --- make sure the folders exist ------------------------------------------------
for _d in (VERSIONS_DIR, DATASETS_DIR, CONFIGS_DIR, DATA_DIR, FEATURES_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
