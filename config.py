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
import os
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
def label_menu() -> dict:
    """the label sets that EXIST, read from data/labels/registry.yaml -- the one source of truth.

    config does not keep its own copy of this. a hard-coded default here (a filename, or even a
    handle) goes stale the moment the label team swaps a file, and every recipe frozen without an
    explicit --labels then names something that is not there. the menu is the register; this reads
    it. read lazily, on call, so config stays importable even if the menu is missing.
    """
    p = LABELS_DIR / "registry.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def labels_name() -> str:
    """the DEFAULT label handle. DERIVED, never hard-coded.

    the rules, in order:
      1. a handle marked `default: true` in the menu wins.
      2. exactly one label set in the menu -> that one, obviously.
      3. several, none marked -> REFUSE. picking one silently would decide something the person
         freezing the version is supposed to decide (same features + a different label is a
         DIFFERENT version), and a wrong guess is invisible until the numbers disagree.
    """
    # the override hook wins, for both this and _find_labels -- one pin, one answer. a pinned
    # value may be a handle ("L1") or a literal filename ("labels_mini.csv"); the stem is the name.
    if LABELS_FILE:
        return LABELS_FILE.rsplit(".", 1)[0] if "." in LABELS_FILE else LABELS_FILE

    menu = label_menu()
    if not menu:
        raise SystemExit(
            f"no label menu at {LABELS_DIR/'registry.yaml'} -- add one entry per label set:\n"
            f"  L1:\n    file: L1.parquet\n    note: \"what it is, in plain words\"")
    marked = [k for k, v in menu.items() if isinstance(v, dict) and v.get("default")]
    if marked:
        return marked[0]
    if len(menu) == 1:
        return next(iter(menu))
    raise SystemExit(
        f"{len(menu)} label sets exist ({', '.join(sorted(menu))}) and none is marked default.\n"
        f"  say which one:   python core/make_version.py --from-sheet --labels L1\n"
        f"  or mark one in {LABELS_DIR/'registry.yaml'} with  default: true")

# ---- train / val / test ------------------------------------------------------
# "time"          TRAIN | gap | VAL | gap | TEST, oldest to newest. one boundary, one gap.
# "bundle_random" whole 15-min candles assigned at RANDOM. no minute is split across slices, but
#                 there is NO embargo and none is possible -- train and test end up interleaved
#                 minutes apart while the features look back 20 sessions. that is the open risk.
# chosen 2026-07-31 on the forward hold-out: macro_f1 0.139 vs time's 0.135, and NO inflation
# (dev 0.115 < forward 0.139; a leaking split shows the opposite). scripts/forward_holdout_test.py
SPLIT_STRATEGY = "bundle_random"
BUNDLE_MINUTES = 15        # one bundle = one candle. only used by bundle_random.
SPLIT_SEED     = 42        # recorded in the model bundle: a different seed = a DIFFERENT split.

# ---- the three models ---------------------------------------------------------
MODEL_TYPES = ["xgboost", "catboost"]   # random_forest turned OFF -- xgboost + catboost only

# ---- ClearML ------------------------------------------------------------------
# project name. defaults to production; set NIFTY_PROJECT to point a whole run at a throwaway test
# project (register/build/publish/train all follow it). unset it to go back -- no code change.
CLEARML_PROJECT   = os.environ.get("NIFTY_PROJECT", "nifty_main")
CLEARML_DATASET   = "nifty_signal_dataset"
# the OOS (out-of-sample) set the backtest scores against. IT MUST NOT SHARE THE TRAINING DATASET
# NAME: auto_trigger watches CLEARML_DATASET, so publishing OOS rows under that name would start
# TRAINING on out-of-sample data.
CLEARML_OOS_DATASET = "nifty_oos_dataset"
CLEARML_PRICE_DATASET = "nifty_ohlcv"       # the OHLCV the backtest prices trades against


def resolve_dataset_id(dataset_name: str, pinned: str = "") -> str:
    """the id of a ClearML dataset, resolved BY NAME at call time.

    WHY NOT PIN THE ID IN THIS FILE. an id is a fact that lives in ClearML, and a copy of it here
    goes stale the moment the dataset is re-uploaded. that is not hypothetical: on 2026-07-31 the
    OHLCV was replaced with a longer file, the old dataset was deleted, and config still named the
    dead id -- every worker would have failed to fetch prices. a NAME survives a re-upload; the id
    does not. same reasoning as labels_name() reading the label menu instead of holding a filename.

    `pinned` is the escape hatch: set the *_DATASET_ID constant to freeze one exact version (for
    reproducing an old result). empty = always take the newest under that name.
    """
    if pinned:
        return pinned
    from clearml import Dataset               # imported lazily: config must stay usable offline
    try:
        return Dataset.get(dataset_project=CLEARML_PROJECT, dataset_name=dataset_name).id
    except Exception:
        # PRICES AND THE OOS SET ARE REFERENCE DATA, NOT PROJECT DATA. they are uploaded once and
        # used by every project (production, a throwaway test project, whatever NIFTY_PROJECT is
        # today), so refusing to look outside the current project would mean re-uploading the same
        # 40 MB per project. fall back to a name search across projects.
        return Dataset.get(dataset_name=dataset_name).id
# the master OOS currently registered (MASTER_OOS 2025-01-01 -> 2026-02-24, 593 feature columns,
# NO labels -- by design: the backtest evaluates from predictions, it does not need true labels).
# it matches the V7/V8/V9 feature naming, NOT our v1-v6 datasets, so only a model trained on those
# can score it. bytes live in gs://<bucket>/datasets.
OOS_DATASET_ID    = ""      # empty = resolve CLEARML_OOS_DATASET by name (the newest).
                            # set an id here only to FREEZE one exact version for reproducing.
TRAIN_QUEUE       = "training"          # a clearml-agent must listen here, or nothing runs
SHAP_QUEUE        = "training"          # SHAP runs on the same queue by default
EXPORT_QUEUE      = "training"          # scored-tables export runs on the same queue by default
OOS_QUEUE         = "training"          # OOS scoring runs on the same queue by default

def base_trainer_name(model_type: str) -> str:
    """The name of the base task publish_version.py clones for each model."""
    return f"train_{model_type} (base)"

BASE_SHAP_NAME     = "shap_explain (base)"
BASE_EXPORT_NAME   = "export_scored_tables (base)"
BASE_OOS_NAME      = "score_oos (base)"
# the short name that rides in the OOS task name and the table filename, e.g.
# scored_oos_2025_2026_xgboost v7. change it when the OOS window moves, so two runs against
# DIFFERENT out-of-sample periods can never be mistaken for each other in ClearML.
OOS_TAG            = "2025_2026"
BASE_DEEPCHECKS_NAME = "deepchecks_report (base)"
DEEPCHECKS_QUEUE   = "training"     # runs on the same queue by default
# deepchecks on 513k rows x 500 cols is very slow. it SAMPLES this many rows per split -- the
# checks and their conditions are unchanged, only the runtime is. 0 = use every row.
DEEPCHECKS_SAMPLE  = 50_000

# ---- backtest ------------------------------------------------------------------
# set BOTH and every finished model automatically gets a backtest on its TEST table, printed into
# that task's ClearML console. leave either blank and you just get the tables (no backtest).
# the script must be COMMITTED in the repo -- an agent runs the repo snapshot, not your laptop.
BACKTEST_SCRIPT   = "scripts/backtest_single.py"

# BACKTEST THE OOS SET ONLY -- never the train/test tables.
# a backtest walks a table row by row and only means something on CONTINUOUS minutes. under
# bundle_random the test table is scattered 15-min bundles: measured on v7, 274,605 rows with
# 13,088 BREAKS and a largest gap of 6 days. an equity curve stitched across those is untradeable.
# True is honest only under SPLIT_STRATEGY = "time", where test IS one block at the end.
BACKTEST_ON_TEST_TABLE = False
PRICE_DATASET_ID  = ""      # empty = resolve CLEARML_PRICE_DATASET ("nifty_ohlcv") by name,
                            # so a re-upload is picked up automatically instead of leaving a dead
                            # id here. set an id only to freeze one exact version.
PRICE_FILE        = ""                  # OR a LOCAL path to the OHLCV parquet. only works while
                                        #   the agent runs on THIS machine -- a worker on another
                                        #   box has no such file and the backtest is skipped.
                                        #   PRICE_DATASET_ID wins if both are set.
BASE_CHAMPION_NAME = "select_champion (base)"
# CHAMPION IS OFF. select_champion is not run -- it was judged useless (2026-07-22). publish never
# queues it regardless of flags. flip this to True (and pass --champion) only if you ever want the
# "pick the best of the trained models" step back.
RUN_CHAMPION       = False

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
GCS_OUTPUT_URI = f"gs://{GCS_BUCKET}/clearml"   # LEGACY -- old artifacts already live here; we do not move them

# NEW STRUCTURE (2026-07-24): each kind of output gets its own folder, so the bucket is browsable
# by type. OLD artifacts stay under clearml/ (kept, not migrated); everything from now writes here.
#     artifacts/models   trained model bundles
#     artifacts/shap     SHAP outputs
#     tables             the scored tables
#     dvc/               datasets (unchanged -- DVC content-addressed blobs)
GCS_ARTIFACTS_URI = f"gs://{GCS_BUCKET}/artifacts"
GCS_TABLES_URI    = f"gs://{GCS_BUCKET}/tables"


def model_output_uri():
    """where trained MODELS are written. None in local mode = the self-hosted fileserver."""
    return None if STORAGE_MODE == "local" else f"{GCS_ARTIFACTS_URI}/models"


def shap_output_uri():
    """where SHAP outputs are written. None in local mode = the self-hosted fileserver."""
    return None if STORAGE_MODE == "local" else f"{GCS_ARTIFACTS_URI}/shap"


def tables_output_uri():
    """where the scored tables are written. None in local mode = the self-hosted fileserver."""
    return None if STORAGE_MODE == "local" else GCS_TABLES_URI


def deepchecks_output_uri():
    """where the deepchecks HTML reports are written -> gs://<bucket>/artifacts/deepchecks .
    they land as task artifacts, so anyone can download and open them from the ClearML UI."""
    return None if STORAGE_MODE == "local" else f"{GCS_ARTIFACTS_URI}/deepchecks"


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

# The gap thrown away between two slices. a row just before a cut and one just after share almost
# all of their rolling-window history -- near-duplicates. without a gap the later slice is partly
# a copy of the earlier one and the score is flattered. must be >= the longest feature lookback.
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

# ---------------------------------------------------------------------------------------------
# THE FEATURE SOURCES -- where a column's source name ends and its own name begins.
#
# the feature team's handed-over matrices name columns <source><sep><column>. the separator USED
# to be '__', and two things read it:
#     leak_guard  -- its patterns are anchored (^session$, ^label_int), so they must be applied to
#                    the COLUMN part. given 'vwap_microstructure_session' with no separator it
#                    tests the whole string, matches nothing, and passes a calendar column.
#     register.py -- "does almost every column carry '__'" is how it recognises an already merged
#                    and aligned matrix (pre_aligned), so it does not shift the data a second time.
#
# the 2026-08-01 drop removed the separator (bar_conviction__bar_range -> bar_conviction_bar_range).
# rather than rewrite 1.6 GB of their parquets on every drop, we list the sources here and strip
# the prefix ourselves. this works with '__' AND with '_', so it survives either convention.
#
# WHY NOT JUST SPLIT ON '_' AND TEST EVERY PIECE: 'candle_pattern_live_label' would end at the
# piece 'label', match ^label$, and be banned -- and it is a real feature that has always been
# trained on. knowing the source is the only way to find the true column name.
#
# ADD A NAME HERE when the feature team adds a source. anything that matches none of these is
# reported, never guessed.
FEATURE_SOURCES = [
    "bar_conviction", "breakout_state", "candle_pattern", "fade_risk", "flow_state",
    "gap_state", "intraday_positioning", "level_proximity", "momentum_state",
    "pullback_severity", "smc_structure", "stress_signal", "structural_position",
    "trend_direction", "trend_maturity", "vol_level", "volume_quality", "vwap_microstructure",
    # the extra sources that arrived with V9 -- these still carry '__' and work either way
    "features_47", "nifty_features_v3", "nifty_technical_features", "rrg_features",
    "nifty_context_features", "dc_pos_275_bucket_features", "vsa_dev_bucket_features",
]


def column_part(name: str) -> str:
    """'bar_conviction_bar_range' -> 'bar_range'.  '...__session' -> 'session'.

    returns the name UNCHANGED when no source matches -- the caller then tests the whole string,
    which is the old behaviour and never silently drops a check.
    """
    if "__" in name:
        return name.split("__", 1)[1]
    for s in sorted(FEATURE_SOURCES, key=len, reverse=True):
        if name.startswith(s + "_") and len(name) > len(s) + 1:
            return name[len(s) + 1:]
    return name

# the labels + weights (bridge-only; core never opens this file)
#
# never a path under Downloads: an agent on another machine has no such folder, and the run dies
# there instead of here.
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
# NOTHING IS HARD-CODED HERE ANY MORE.
# this used to be the literal filename of whichever labels were current. that is a copy of a fact
# that lives in data/labels/registry.yaml, and a copy goes stale: the label team swapped the file
# and every recipe frozen without --labels pointed at something that no longer existed. the menu
# is the register; labels_name() reads it. LABELS_FILE is kept ONLY as an override hook (tests
# monkeypatch it, and an odd setup can pin one file) -- empty means "ask the menu".
LABELS_FILE = ""


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

    # the handle comes from the MENU unless something pinned LABELS_FILE. a handle has no
    # extension, so try both; a pinned literal filename is used exactly as given.
    want = LABELS_FILE or labels_name()
    names = ([want] if "." in want
             else [f"{want}.parquet", f"{want}.csv", want])

    for base in (LABELS_DIR,                       # 2. THE RIGHT PLACE. in the repo, DVC-tracked.
                 Path.home() / "Downloads"):       # 3. the old spot. works, but only on ONE box.
        for n in names:
            p = base / n
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
          f"      mv ~/Downloads/{names[0]} {LABELS_DIR}/\n"
          f"      dvc add {LABELS_DIR / names[0]}\n"
          f"      git add {LABELS_DIR / names[0]}.dvc && git commit -m 'labels'\n"
          f"      dvc push\n\n"
          f"  then on any other machine:  git clone ... && dvc pull\n")


# RESOLVED LAZILY, never at import. as a module-level constant its SystemExit fired the moment
# anything imported config -- including --help and every test collection.
def labels_csv() -> Path:
    """the labels file on THIS machine -- resolved when asked for, never at import."""
    return _find_labels()


def labels_csv_for(labels_name):
    """resolve the labels file for a version's CHOSEN label set (recipe['labels_name']).

    THIS is how multiple label sets work. A version records which label it picked -- 'L1', 'L2',
    'L3', ... -- and the build resolves + hashes EXACTLY that file, so labels_name and
    labels_sha256 in the certificate always describe the SAME file (that was the old bug: the name
    came from the recipe, the sha from the single configured file, and they could disagree).

    Looks for data/labels/<labels_name>.csv (a $NIFTY_LABELS override still wins). Returns the Path,
    or None if nothing matches -- the caller then falls back to the single LABELS_FILE, which keeps
    pre-multi-label recipes building exactly as before.
    """
    import os
    if not labels_name:
        return None
    name = str(labels_name)
    env = os.environ.get("NIFTY_LABELS")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser()
    # a bare handle (L1) -> L1.csv or L1.parquet; an explicit name (L1.parquet) is tried as-is.
    names = [name] if name.endswith((".csv", ".parquet")) else [f"{name}.csv", f"{name}.parquet"]
    for base in (LABELS_DIR, Path.home() / "Downloads"):
        for n in names:
            if (base / n).exists():
                return base / n
    return None

LABEL_TS_COL    = "timestamp"
LABEL_COL       = "primary_label"
WEIGHT_COL      = "weight"
WEIGHT_RAW_COL  = "weight_raw"
LABEL_TS_FORMAT = "%d-%m-%Y %H:%M"      # e.g. 01-01-2020 09:15

# THE ROW ID. assigned at dataset build (build_dataset.py), one per merged row. it is an
# IDENTIFIER, never a feature -- the trainer builds X from feature_columns (or the '__' fallback),
# and this name has no '__', so the model can never see it or memorise row position. it exists as
# a stable JOIN KEY for the scored tables and the next team (an integer is safer to join on than a
# datetime). timestamp is the merge key; this is stamped on the result. see [[unique-index-in-dataset]].
INDEX_COL       = "unique_index"

# ---- CLASS WEIGHTS -- per class, and they OVERRIDE the labels' per-row `weight` --------------
# set {} to fall back to that column instead (conviction: SUPER > SUB > SMALL).
# balanced/inverse-frequency:  weight_i = len(labels) / (7 * rows_in_class_i)
# every class then commands the same ~14.3% of the loss, lifting the ENTRY classes from a few
# percent to 42.9% between them. recompute after ANY label change -- these are L1's counts.
# KEYED BY NAME, never index: LabelEncoder sorts alphabetically, so an integer dict would hand
#   NO_TRADE the top weight and the entries the bottom, silently.
# NO_TRADE STAYS DECIMAL: round it to 0 and 74.5% of rows stop contributing to the loss, and the
#   model can never learn "don't trade".
CLASS_WEIGHTS = {
    "ENTRY_SUB":    19,     # rarest        (  7,845 rows,  0.8%)  exact 18.6078
    "ENTRY_SMALL":  14,     #               ( 10,665 rows,  1.0%)  exact 13.6876
    "EXIT_SMALL":    8,     #               ( 18,135 rows,  1.8%)  exact  8.0495
    "ENTRY_SUPER":   4,     #               ( 34,380 rows,  3.4%)  exact  4.2460
    "EXIT_SUB":      2,     #               ( 65,578 rows,  6.4%)  exact  2.2260
    "EXIT_SUPER":    1,     #               (123,509 rows, 12.1%)  exact  1.1819
    "NO_TRADE":      0.192,  # most common  (761,735 rows, 74.5%)  exact  0.1916
                            # ^ STAYS DECIMAL ON PURPOSE (see above).
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

# ---- fixed NaN sentinel -------------------------------------------------------
# None (2026-07-24): NaN is KEPT in the dataset, and each model handles it the way it is meant to:
#   xgboost / catboost  keep the NaN and learn a branch for "missing" NATIVELY (a per-split
#                       decision -- strictly better than one fixed number).
#   random forest       gets a PER-COLUMN sentinel at train time (compute_sentinel, fit on the
#                       TRAIN rows only -- can never collide with a real value; safer than -999).
# a flat value here (e.g. -999.0) instead fills EVERY numeric NaN for ALL models at BUILD time --
# simpler to explain, but it takes native NaN handling AWAY from xgboost/catboost (they'd just see
# -999). so it is off. (Text/categorical NaN always becomes CATEGORICAL_NA_LABEL -- see train.py.)
NA_FIXED_SENTINEL = None


# --- make sure the folders exist ------------------------------------------------
for _d in (VERSIONS_DIR, DATASETS_DIR, CONFIGS_DIR, DATA_DIR, FEATURES_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
