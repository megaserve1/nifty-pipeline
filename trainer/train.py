"""
trainer/train.py -- train ONE model on ONE published dataset version.

three model types share this one script:
    random_forest   xgboost   catboost
publish_version.py enqueues all three at once, and they train IN PARALLEL, one per agent.

read it top to bottom like a story. every non-obvious line has a plain-english comment.

WHY THE THREE MODELS ARE NOT TREATED THE SAME
    they disagree about two things, and both matter here:

    missing values (NaN):
        xgboost   handles NaN itself. at every split it tries sending the missing rows left,
                  then right, and keeps whichever predicts better. so "no gap" gets its own
                  branch and the model LEARNS what a missing value means.
        catboost  same -- handles NaN natively.
        random forest  CANNOT TRAIN WITH NaN AT ALL. it crashes. so we give it a sentinel:
                  a number the feature can never really take (see bridge/na_policy.py).
                  the tree then makes one cut and separates the missing rows itself, which is
                  the same outcome, done by hand.

    text columns (like gap_state = NO_GAP / GAP_UP / GAP_DOWN):
        catboost  reads text natively (cat_features). this is what catboost is FOR.
        xgboost + random forest  need the text turned into numbers first. the mapping is SAVED
                  with the model, so live data is encoded the same way for ever.

THE WEIGHT WARNING YOU MUST NOT IGNORE
    the labels file gives NO_TRADE a weight of 0, and NO_TRADE is 53% of all rows. a row with
    weight 0 contributes NOTHING to the loss. so the model never learns to stay out, and it
    will want to trade EVERY SINGLE MINUTE. this was measured on all three libraries: none of
    them ever predicts a zero-weight class. the trainer shouts about it. the fix belongs in the
    LABEL POLICY (give NO_TRADE ~0.1-0.2), not in here.

run: never by hand. core/publish_version.py clones the base task and enqueues it.
"""
import argparse
import json
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C      # noqa: E402
import contract         # noqa: E402


# ------------------------------------------------------------------ the split
#
# THE SPLIT LIVES IN trainer/objective.py NOW. This file used to own a two-way time_split() with
# a CALENDAR-day embargo. Both halves of that were wrong:
#
#   1. TWO-WAY. train | embargo | test, scored on TEST. Honest for ONE model with fixed
#      settings. But the moment hpo.py runs 30 trials and keeps the best TEST score, the test
#      set has entered the training loop -- we did not fit the model on it, we fitted the
#      SETTINGS on it, and settings are parameters too. The winning number is then the best of
#      30 noisy draws, biased low by construction, and there is nothing honest left to report.
#      So there are THREE slices now: the optimiser tunes on VAL, and TEST is opened once, at
#      the end, on the winner. That number is the one you are allowed to say out loud.
#
#   2. CALENDAR DAYS. `cut + Timedelta(days=21)` to cover a "20-day" feature lookback -- but
#      20 days means 20 TRADING SESSIONS, and 21 calendar days spans only ~14 of them. The
#      embargo was ~30% short. It is counted in sessions now. See purged_cv.embargo_end.
from trainer.objective import (        # noqa: E402
    three_way_split, trading_cost, report_trading_cost, series_for,
)
from trainer import hyperparams        # noqa: E402  -- configs/hyperparams.yaml, the ONE source


# ---------------------------------------------------------------- fetch the data file
def find_dataset_parquet(local, dataset_id: str):
    """the parquet inside a downloaded ClearML dataset -- WHATEVER it is named.

    THE TRAP: in gcs mode the dataset is an EXTERNAL-FILE pointer to the DVC blob, and DVC stores
    content-addressed -- the downloaded file is named by its md5 (e.g. '9015ad71...'), with NO
    '.parquet' extension. So the old `glob("*.parquet")` found nothing and the trainer died with
    "no parquet inside" even though the file was right there. pandas reads parquet by CONTENT, not
    by name, so we just find the real data file (the biggest non-hidden file) and read it.
    """
    import pathlib
    p = next(pathlib.Path(local).glob("*.parquet"), None)   # local mode: named normally
    if p is not None:
        return p
    files = [f for f in pathlib.Path(local).rglob("*")      # gcs/dvc mode: md5-named blob
             if f.is_file() and not f.name.startswith(".") and f.name != "manifest.json"]
    if not files:
        raise SystemExit(f"no data file inside dataset {dataset_id} (downloaded to {local})")
    return max(files, key=lambda f: f.stat().st_size)       # the parquet is the biggest file


def load_model_bundle(path):
    """joblib.load a model bundle, REBUILDING the xgboost model from its portable UBJ.

    WHY THIS EXISTS. XGBoost pickles its booster as a VERSION-SPECIFIC binary buffer. a bundle
    pickled under one xgboost dies with "XGBoostError: input stream corrupted" the moment another
    xgboost tries to unpickle it -- and clearml-agents build a FRESH venv per task, so the SHAP or
    champion task easily resolves a different xgboost than the trainer did. pinning is fragile
    against that. so the trainer no longer pickles the xgboost object at all: it stores the model's
    NATIVE UBJ (cross-version-portable) under 'xgb_model_ubj' and leaves 'model' None. here we load
    the (now version-safe) bundle and rebuild the classifier from that UBJ. non-xgboost bundles are
    untouched. older bundles (no 'xgb_model_ubj') load exactly as before.
    """
    import joblib, tempfile, pathlib
    b = joblib.load(path)
    if b.get("model") is None and b.get("xgb_model_ubj") is not None:
        import xgboost as xgb
        tf = tempfile.NamedTemporaryFile(suffix=".ubj", delete=False); tf.write(b["xgb_model_ubj"]); tf.close()
        clf = xgb.XGBClassifier()
        clf.load_model(tf.name)                 # restores num_class/objective -> predict_proba works
        pathlib.Path(tf.name).unlink()
        b["model"] = clf
    elif b.get("model") is None and b.get("cb_model_cbm") is not None:
        from catboost import CatBoostClassifier
        tf = tempfile.NamedTemporaryFile(suffix=".cbm", delete=False); tf.write(b["cb_model_cbm"]); tf.close()
        clf = CatBoostClassifier()
        clf.load_model(tf.name, format="cbm")   # restores classes + cat features -> ShapValues works
        pathlib.Path(tf.name).unlink()
        b["model"] = clf
    return b


# ------------------------------------------------------------------ the models
def parse_max_features(v):
    """the forest's max_features arrives as a STRING, because "sqrt" and "0.3" travel down the
    same argparse argument. sklearn wants the word as a word and the fraction as a float."""
    if v in (None, "", "none", "None"):
        return None                       # None = use every feature at every split
    s = str(v).strip()
    if s in ("sqrt", "log2"):
        return s
    try:
        return float(s)
    except ValueError:
        raise SystemExit(f"max_features must be sqrt / log2 / a fraction, got {v!r}")


def build_model(model_type: str, n_classes: int, params: dict, cat_idx=None, has_val: bool = False):
    """make one model. each library gets the settings that suit it.

    has_val: whether a validation split exists. early stopping (xgboost/catboost) needs an eval_set
    at fit time, so it is only ARMED when there is a val split -- otherwise the library raises
    'need at least one validation set'. with early stopping on, n_estimators is a CEILING.
    """
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            # 0 means NO LIMIT -- a fully grown forest, which is sklearn's own default and what
            # a forest actually wants. it used to arrive here as 6, because ONE argparse default
            # was shared by all three models and 6 is a BOOSTING depth. a depth-6 forest is a
            # crippled forest: boosting builds hundreds of shallow trees that CORRECT each other,
            # a forest builds deep trees that are individually strong and get AVERAGED. capping
            # it at 6 gives you the weakness of both and the strength of neither.
            max_depth=params["max_depth"] or None,
            min_samples_leaf=params["min_samples_leaf"],
            # THIS, not max_depth, is the forest's real regulariser, and it is what decorrelates
            # the trees. averaging only helps if the trees disagree.
            max_features=parse_max_features(params.get("max_features", "sqrt")),
            # bootstrap this fraction of rows per tree (feature-team set 0.7). None = all rows.
            max_samples=params.get("max_samples", None),
            n_jobs=-1,                       # use every core on the machine
            random_state=params["seed"],
            # per-row sample_weight (from the labels) already encodes class importance -- and it
            # matches the team's class_weight dict but finer-grained -- so this stays None.
            class_weight=None,
        )

    if model_type == "xgboost":
        import xgboost as xgb
        # GROWTH POLICY -- the depth-vs-loss switch.
        #   "depthwise" (default): grow level by level, capped by max_depth = DEPTH-BASED.
        #   "lossguide"          : split whichever leaf cuts loss most, capped by max_leaves
        #                          = LOSS-BASED (leaf-wise).
        grow_policy = str(params.get("grow_policy", "depthwise"))
        # max_depth 0 means "no depth cap". what that SHOULD do depends on the policy:
        #   lossguide: 0 = correct, "let max_leaves rule" -- pass it through.
        #   depthwise: 0 = an UNBOUNDED level-wise booster, the exact boosting anti-pattern the
        #              old `or 6` guarded against. keep the fall-back to 6 there.
        # None (genuinely unset) -> 6 in both. this is why it is not just `params["max_depth"] or 6`
        # (that turns lossguide's deliberate 0 into 6 and silently caps the overfit).
        md = params["max_depth"]
        if md is None:
            max_depth = 6
        elif int(md) == 0:
            max_depth = 0 if grow_policy == "lossguide" else 6
        else:
            max_depth = int(md)
        return xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=max_depth,
            # max_leaves=0 means "no leaf limit" and is ignored under depthwise, so it is safe in
            # both modes -- the yaml decides which mode by setting grow_policy.
            grow_policy=grow_policy,
            max_leaves=int(params.get("max_leaves", 0) or 0),
            # max_bin: histogram resolution. more bins = finer split thresholds = MORE capacity
            # to fit (and more overfit). xgboost default is 256; higher is the overfit direction.
            max_bin=int(params.get("max_bin", 256)),
            learning_rate=params["learning_rate"],
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            # min_child_weight is a floor on the SUM OF WEIGHTS in a leaf, not a row count. that
            # matters enormously here: our weights run 0.00-0.91 and NO_TRADE is 0.00, so a leaf
            # holding 500 NO_TRADE rows has a child weight of ZERO. keep this floor low or
            # xgboost refuses to split anywhere near the rare classes.
            min_child_weight=params.get("min_child_weight", 1.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            reg_alpha=params.get("reg_alpha", 0.0),    # L1: prunes useless features to zero
            gamma=params.get("gamma", 0.0),            # min loss drop to allow a split
            objective="multi:softprob",       # 7 classes -> a probability for each
                                              # (NOT binary:logistic from the pdf -- that is a
                                              #  2-class metric; we have 7 classes)
            num_class=n_classes,
            tree_method="hist",               # fast, and handles NaN natively
            n_jobs=-1,
            random_state=params["seed"],
            eval_metric="mlogloss",
            # stop early once val mlogloss stops improving. only ARMED when a val split exists
            # (the fit passes eval_set); else None = train all n_estimators.
            early_stopping_rounds=(int(params["early_stopping_rounds"])
                                   if has_val and params.get("early_stopping_rounds") else None),
            verbosity=0,
        )

    if model_type == "catboost":
        from catboost import CatBoostClassifier
        kw = dict(
            iterations=params["n_estimators"],
            depth=params["max_depth"] or 6,   # catboost HARD CAPS depth at 16 (oblivious trees)
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),   # L2 on leaf values -- the main brake
            min_data_in_leaf=int(params.get("min_data_in_leaf", 1)),
            rsm=params.get("rsm", 1.0),                   # column sampling (= colsample_bytree)
            random_strength=params.get("random_strength", 1.0),  # noise added when scoring splits
            loss_function="MultiClass",
            random_seed=params["seed"],
            cat_features=cat_idx or None,     # catboost reads text columns as-is
            verbose=0,
            allow_writing_files=False,        # do not litter the agent's disk
        )
        # GROWTH POLICY -- the depth-vs-loss switch for catboost.
        #   "SymmetricTree" (default): oblivious trees, capped by depth = DEPTH-BASED. this is the
        #                              MOST depth-based form there is (same split across a level).
        #   "Lossguide"             : split the highest-loss leaf, capped by max_leaves = LOSS-BASED.
        #   "Depthwise"             : level by level but non-oblivious.
        # max_leaves is ONLY legal when the policy is NOT SymmetricTree -- passing it under
        # SymmetricTree makes catboost refuse to start. so add it only for the leaf-wise modes.
        gp = str(params.get("grow_policy", "SymmetricTree"))
        kw["grow_policy"] = gp
        if gp != "SymmetricTree" and params.get("max_leaves"):
            kw["max_leaves"] = int(params["max_leaves"])
        # BOOTSTRAP: `subsample` is only valid for a NON-Bayesian bootstrap. the default (Bayesian)
        # uses bagging_temperature and silently IGNORES subsample -- so pass exactly the matching
        # pair, never both. feature-team set bootstrap_type=Bernoulli + subsample=0.7.
        bt = str(params.get("bootstrap_type", "Bayesian"))
        kw["bootstrap_type"] = bt
        if bt.lower() == "bayesian":
            kw["bagging_temperature"] = params.get("bagging_temperature", 1.0)
        else:
            kw["subsample"] = params.get("subsample", 1.0)
        # early stopping needs an eval_set at fit time -- only arm it when a val split exists.
        if has_val and params.get("early_stopping_rounds"):
            kw["early_stopping_rounds"] = int(params["early_stopping_rounds"])
        return CatBoostClassifier(**kw)

    raise SystemExit(f"unknown model type {model_type!r}. one of: {C.MODEL_TYPES}")


# ------------------------------------------------------------------ metrics
def full_proba(model, proba, n_classes: int):
    """predict_proba, guaranteed to have ONE COLUMN PER LABEL-ENCODER CLASS.

    sklearn's RandomForest returns columns only for the classes it SAW IN TRAINING. if a class
    is missing from the train slice (a short --rows run, a degenerate split), proba has fewer
    columns than the label encoder has classes -- and report_metrics' `proba[:, i]` silently
    reads the WRONG class's probabilities, or IndexErrors. xgboost is immune (num_class is
    pinned), which makes the bug worse: it appears only for one model type, only on small runs.
    """
    import numpy as np
    proba = np.asarray(proba)
    if proba.shape[1] == n_classes:
        return proba
    seen = getattr(model, "classes_", None)
    if seen is None:
        raise SystemExit(f"predict_proba returned {proba.shape[1]} columns for {n_classes} "
                         f"classes and the model has no classes_ to map them with")
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(np.asarray(seen).astype(int)):
        out[:, cls] = proba[:, j]
    return out


def report_metrics(y_true, y_pred, proba, classes, logger, split: str):
    """report the numbers that are HONEST for this problem.

    accuracy is banned. NO_TRADE is 53% of the data, so a model that always says NO_TRADE
    scores 53% and is completely useless. per-class recall is what tells you whether the model
    can actually find the rare ENTRY signals -- which is the only thing that makes money.

    EVERY NUMBER CARRIES ITS SPLIT IN ITS NAME. THIS IS NOT TIDINESS.
        this used to report bare "macro_f1". with one split that was fine. now we score TWO
        splits, and calling it twice would put both on the SAME series -- "Summary"/"macro_f1".
        clearml's last_metrics keeps only the LAST value written, so val and test would silently
        overwrite each other and you could not tell which number you were looking at.

        worse: point the optimiser at "Summary"/"macro_f1" and it would read whichever split was
        written last -- which, in the order this file runs them, is TEST. you would be tuning on
        the test set, and nothing about the run would look wrong.

        so: "val/macro_f1" and "test/macro_f1". series_for() is the ONE place these names are
        built, and hpo.py searches by the same function -- so the two cannot drift apart.
    """
    from sklearn.metrics import (classification_report, average_precision_score,
                                 confusion_matrix)

    rep = classification_report(y_true, y_pred, labels=range(len(classes)),
                                target_names=classes, zero_division=0, output_dict=True)
    text = classification_report(y_true, y_pred, labels=range(len(classes)),
                                 target_names=classes, zero_division=0)
    print(f"\n--- per-class report [{split}]")
    print(text)
    if logger:
        # print_console=False: we already print(text) above. without this, clearml's report_text
        # ALSO echoes the whole table to the console, so every split printed TWICE. this keeps it
        # in the ClearML web UI (report_text) but not a second time on the terminal.
        logger.report_text(f"[{split}]\n{text}", print_console=False)

    # PR-AUC per class: how well the model ranks a rare class, ignoring the threshold.
    # for a rare class this is far more informative than precision/recall at one cut-off.
    onehot = np.eye(len(classes))[y_true]
    pr = {}
    for i, cls in enumerate(classes):
        if onehot[:, i].sum() == 0:
            continue                                   # this class never appears in this slice
        pr[cls] = float(average_precision_score(onehot[:, i], proba[:, i]))
        if logger:
            logger.report_single_value(series_for(split, f"PR_AUC/{cls}"), round(pr[cls], 4))

    # macro-F1: the average F1 across classes, treating a 1.2% class as equal to a 53% class.
    # that is exactly what we want -- we care about the rare signals.
    macro_f1 = float(rep["macro avg"]["f1-score"])
    mean_pr = float(np.mean(list(pr.values()))) if pr else 0.0

    # the classes the model NEVER predicts. if a class is here, the model is blind to it.
    never = [c for i, c in enumerate(classes) if (y_pred == i).sum() == 0]

    if logger:
        logger.report_single_value(series_for(split, "macro_f1"), round(macro_f1, 4))
        logger.report_single_value(series_for(split, "mean_pr_auc"), round(mean_pr, 4))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                         columns=[f"pred_{c}" for c in classes])
    if logger:
        logger.report_table("Confusion matrix", split, table_plot=cm_df)

    return {"per_class": rep, "pr_auc": pr, "macro_f1": macro_f1,
            "mean_pr_auc": mean_pr, "never_predicted": never,
            "confusion": cm_df.to_dict()}


# ------------------------------------------------------------------ main
def main():
    # CLEARML MUST BE IMPORTED BEFORE argparse.parse_args() RUNS. THIS IS NOT STYLE.
    # clearml hooks argparse BY MONKEY-PATCHING IT AT IMPORT TIME (clearml/binding/args.py) --
    # that is how Task.init later knows which arguments were parsed, and how a CLONED task's
    # Args/ overrides reach the script. Import clearml AFTER parsing and the parse is invisible:
    # every Args/ value publish_version or hpo sets on a clone is silently ignored, the script
    # runs with argparse defaults, sees no --dataset_id, prints "registration run" and exits
    # GREEN having trained NOTHING. All three models. The same silent-failure shape as
    # finalize()-without-publish(). Verified against clearml 2.1.10 source + executed repro.
    from clearml import Dataset, Task    # noqa: F401  -- BEFORE the parser. see note above.

    ap = argparse.ArgumentParser()
    # these land in ClearML's "Args/" section. publish_version.py writes Args/dataset_id --
    # the section name must match or the override is silently ignored and the trainer would
    # quietly train on the wrong dataset. that exact bug already cost a day once.
    ap.add_argument("--model_type", default="xgboost", choices=C.MODEL_TYPES)
    ap.add_argument("--dataset_id", default="", help="the EXACT ClearML dataset id to train on")
    ap.add_argument("--dataset_version", default="")
    # THE LABEL MUST TRAVEL WITH THE VALUES, OR IT CAN LIE.
    # the numeric params are resolved on the CONTROLLER (publish_version reads hyperparams.yaml
    # and sets each one as Args/<name>) precisely so a stale file on the agent cannot change what
    # gets trained. but the LABEL was still read from the agent's own copy of hyperparams.yaml --
    # which is the file at the base task's git commit, not the one you just edited. so you could
    # train with h2's numbers and have the run tagged h1, with nothing anywhere disagreeing.
    # same shape as every silent bug in this project. the controller now sends the label too.
    ap.add_argument("--hyperparams_version", default="")

    # ---- EVERY HYPERPARAMETER COMES FROM configs/hyperparams.yaml. NONE IS WRITTEN HERE. ----
    #
    # they used to be argparse literals, right here, and that is exactly how the max_depth bug
    # happened: ONE default (6) shared by all three models, and 6 is a BOOSTING depth -- so
    # RandomForest was silently built as a depth-6 forest. crippled. it took a day to find,
    # because the number was buried in code nobody reads, and the test named after the bug
    # tested build_model() rather than the default that actually reaches it.
    #
    # now: to change a setting you edit ONE readable YAML file. you never touch Python.
    #
    # argparse still exists, and it has to -- ClearML's optimiser overrides task parameters BY
    # NAME (Args/max_depth), so the names must be real argparse arguments or the override is a
    # SILENT no-op. But it now accepts EVERY name any model uses, and each model ignores the
    # ones that are not its own. hpo.py could not even start before, because it searched over
    # names this parser had never heard of.
    for name in hyperparams.all_param_names():
        ap.add_argument(f"--{name}", default=None,
                        help=f"overrides configs/hyperparams.yaml -> <model>.default.{name}")
    a = ap.parse_args()

    import joblib

    task = Task.init(project_name=C.CLEARML_PROJECT,
                     task_name=C.base_trainer_name(a.model_type),
                     task_type=Task.TaskTypes.training,
                     output_uri=C.model_output_uri())  # gcs mode: your bucket; local mode: the self-hosted fileserver
    logger = task.get_logger()

    # no dataset id -> this is the one-off registration run that creates the base task shape.
    if not a.dataset_id:
        print("no --dataset_id: this is a base-task registration run. exiting cleanly.")
        task.close()
        return

    # ---- 1. fetch EXACTLY the dataset we were told --------------------------
    print(f"[1/6] fetching dataset {a.dataset_id}")
    ds = Dataset.get(dataset_id=a.dataset_id, alias="training_data")   # never "the latest"
    local = pathlib.Path(ds.get_local_copy())
    parquet = find_dataset_parquet(local, a.dataset_id)
    df = pd.read_parquet(parquet)
    print(f"      {parquet.name}  ({len(df):,} rows)")

    # ---- 2. the certificate must still describe this file -------------------
    man = ds.get_metadata("manifest")
    if man is None and (local / "manifest.json").exists():
        man = json.loads((local / "manifest.json").read_text())
    if man:
        contract.assert_schema(df, man)
        print(f"[2/6] schema matches the certificate "
              f"({man['n_features']} features, no-lookahead rule = {man['no_peek']['rule']})")
    else:
        print("[2/6] WARNING: no certificate attached -- training on an uncertified dataset")

    feat_cols = man["feature_columns"] if man else [c for c in df.columns if "__" in c]
    cat_cols = (man or {}).get("categorical_columns", [])
    y_raw = df[C.LABEL_COL].astype(str)
    # THE ROW WEIGHTS. two sources, config decides which:
    #   config.CLASS_WEIGHTS set -> a fixed weight per CLASS, mapped BY NAME (never by class index:
    #                               LabelEncoder sorts alphabetically, so an index-keyed dict would
    #                               hand NO_TRADE the top weight and the entries the bottom).
    #   otherwise                -> the labels file's own per-row `weight` column.
    if getattr(C, "CLASS_WEIGHTS", None):
        w = y_raw.map(C.CLASS_WEIGHTS)
        unmapped = sorted(set(y_raw[w.isna()]))
        if unmapped:
            raise SystemExit(f"config.CLASS_WEIGHTS has no weight for {unmapped}. every class in "
                             f"the labels needs one, or those rows would train at weight NaN.")
        w = w.astype(float)
        print(f"[2/6] weights <- config.CLASS_WEIGHTS (per class, by name): {C.CLASS_WEIGHTS}")
    else:
        w = (df[C.WEIGHT_COL].fillna(0.0) if C.WEIGHT_COL in df.columns
             else pd.Series(1.0, index=df.index))
        print(f"[2/6] weights <- the labels file's per-row '{C.WEIGHT_COL}' column")

    # ---- 3. split by time: TRAIN | embargo | VAL | embargo | TEST -------------
    # VAL is what hpo.py tunes on. TEST is opened once, at the end. see objective.py.
    ts = pd.to_datetime(df[C.LABEL_TS_COL])
    tr, va, te, split_info = three_way_split(
        ts, C.VAL_FRACTION, C.TEST_FRACTION, C.EMBARGO_SESSIONS)
    print(f"[3/6] time split, embargo = {C.EMBARGO_SESSIONS} TRADING SESSIONS "
          f"(not calendar days -- 21 calendar days is only ~14 sessions)")
    print(f"      train {int(tr.sum()):>9,}   <= {split_info['train_end']}")
    if split_info["val_enabled"]:
        print(f"      val   {int(va.sum()):>9,}   {split_info['val_start']} .. "
              f"{split_info['val_end']}   <- hpo tunes on THIS")
    else:
        print(f"      val         OFF   (config.VAL_FRACTION = 0)  -- plain train/test. "
              f"hpo.py will refuse to run.")
    print(f"      test  {int(te.sum()):>9,}   >= {split_info['test_start']}")
    print(f"      thrown away in the embargo(es) {split_info['n_embargoed']:,}")

    # ---- 4. the weight problem, said out loud -------------------------------
    # computed from the weights ACTUALLY being used (class-weights or the labels column), so this
    # guard cannot look at one source while the model trains on the other.
    by_class = w[tr].groupby(y_raw[tr]).mean()
    dead = sorted(c for c, m in by_class.items() if float(m) == 0.0)
    zero_pct = float((w[tr] == 0).mean() * 100)
    print(f"[4/6] sample_weight: {zero_pct:.1f}% of training rows have weight 0")
    if dead:
        msg = (f"UNLEARNABLE CLASSES (mean weight 0): {dead}\n"
               f"  rows with weight 0 contribute NOTHING to the loss.\n"
               f"  this model will NEVER predict them, and with NO_TRADE at 0 it will want to\n"
               f"  trade every single minute. fix it in the LABEL POLICY (weight ~0.1-0.2),\n"
               f"  not here.")
        print("      !! " + msg.replace("\n", "\n      !! "))
        logger.report_text(msg)
        task.add_tags(["UNLEARNABLE_CLASS"])

    # ---- 5. prepare the features, the way THIS model needs them -------------
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_raw)
    classes = list(le.classes_)

    X = df[feat_cols].copy()
    cat_maps = {}

    if a.model_type == "catboost":
        # catboost reads text as text. it just needs the missing ones to be a real category
        # and the column to be a string.
        for c in cat_cols:
            X[c] = X[c].astype(str).fillna(C.CATEGORICAL_NA_LABEL)
        cat_idx = [X.columns.get_loc(c) for c in cat_cols]
        print(f"[5/6] catboost: {len(cat_idx)} text column(s) read natively; NaN kept as-is")
    else:
        # random forest and xgboost need numbers.
        #
        # EVERYTHING BELOW IS FITTED ON THE TRAINING ROWS ONLY, THEN APPLIED TO BOTH.
        # this is not fussiness. if we built the category mapping, or worked out the sentinel,
        # by looking at the WHOLE dataset, then facts about the test period would have shaped
        # how the training data was prepared. that is a leak. it never crashes, it never shows
        # up in a metric, and it quietly flatters the score. so: fit on train, apply to test --
        # exactly as we would have to do in production, where the future does not exist yet.
        from na_policy import encode_categoricals, compute_sentinel

        _, cat_maps = encode_categoricals(X.loc[tr])          # LEARN the mapping from train only
        X, _ = encode_categoricals(X, mapping=cat_maps)       # APPLY it to everything
        # a category that appears only in the test period becomes -1 -- which is exactly what
        # would happen live when a genuinely new category shows up.
        cat_idx = None

        if a.model_type == "random_forest":
            # random forest cannot train with NaN. give the missing rows a value nothing real
            # can reach, so the tree separates them with one cut. never 0, never the mean --
            # both are values a feature can genuinely take, so they would collide with real rows.
            sentinels = {}
            for c in X.columns:
                if X[c].isna().any():
                    sv = compute_sentinel(X.loc[tr, c], C.SENTINEL_MARGIN)   # TRAIN only
                    X[c] = X[c].fillna(sv)
                    sentinels[c] = sv
            cat_maps["_sentinels"] = sentinels
            print(f"[5/6] random forest: {len(sentinels)} column(s) had NaN -> given a sentinel "
                  f"below every real value (computed from the TRAIN rows only)")
        else:
            print(f"[5/6] xgboost: NaN kept -- it learns a branch for missing on its own")

    Xtr, Xva, Xte = X.loc[tr], X.loc[va], X.loc[te]
    ytr = le.transform(y_raw[tr])
    yva = le.transform(y_raw[va])
    yte = le.transform(y_raw[te])
    wtr = w[tr].to_numpy()

    # the defaults come from configs/hyperparams.yaml. anything the CLI or ClearML's optimiser
    # actually set overrides them, cast to the type the YAML says it is. no number lives in code.
    cli = {k: getattr(a, k, None) for k in hyperparams.all_param_names()}
    params = hyperparams.merge(a.model_type, cli)
    changed = {k: v for k, v in cli.items() if v is not None and k in params}
    print(f"      hyperparams <- configs/hyperparams.yaml [{a.model_type}]")
    print(f"      {params}")
    if changed:
        print(f"      overridden: {changed}")
    # WHICH SETTINGS PRODUCED THIS RESULT. the dataset half was already tagged (v5); this adds the
    # hyperparameter half (h2), so the board reads "v5 + h2" and two runs are only comparable when
    # BOTH match. params_sha is measured from the values, so a stale hand-set label is detectable.
    # prefer the label the controller sent (it read the same yaml it read the VALUES from).
    # fall back to this machine's yaml for a plain local `python trainer/train.py` run.
    hp_version = (a.hyperparams_version or "").strip() or hyperparams.version()
    # MEASURE THE SHA FROM THE PARAMS ACTUALLY TRAINED WITH, not from this machine's yaml file.
    #
    # hyperparams.params_sha() re-reads configs/hyperparams.yaml -- but a clearml AGENT runs a
    # code snapshot at the BASE TASK's git commit, so its yaml is whatever was committed then,
    # while the VALUES arrive separately as Args/* resolved on the controller from the yaml as it
    # is NOW. edit the yaml, publish without committing, and the two disagree: the run trains on
    # h2's numbers, is labelled h2, and reports h1's sha. the one number whose whole job is to
    # catch a lying label would itself be lying.
    #
    # `params` below is the merged, cast, final dict this run trains on. hashing THAT is true on
    # every machine regardless of what any file says -- same principle as measuring a feature's
    # clock instead of trusting registry.yaml's declaration.
    hp_sha = hyperparams.sha_of(params)
    print(f"      hyperparams set: {hp_version}   (values sha {hp_sha})")
    task.connect({**params, "model_type": a.model_type, "dataset_id": a.dataset_id,
                  "dataset_version": a.dataset_version or ds.version,
                  "hyperparams_version": hp_version, "params_sha": hp_sha,
                  "n_features": len(feat_cols), "test_fraction": C.TEST_FRACTION,
                  "val_fraction": C.VAL_FRACTION, "embargo_sessions": C.EMBARGO_SESSIONS})
    # THE h2 TAG MUST NOT GO ON AN HPO TRIAL. a trial's numbers come from the SEARCH SPACE
    # (max_depth picked from [3,7,10,14], etc.), not from the h2 fixed set -- but the search never
    # overrides hyperparams_version, so without this guard every trial reads 'h2' off the yaml and
    # tags itself h2. a board full of "h2" trials that are not h2 is exactly the kind of thing a
    # cross-examination catches. same detection as the SHAP guard below: a clone still carrying
    # "(base)" in its name is a trial (or the base task itself), not a real fixed-number run.
    # trials are already marked 'optimization' + 'opt:<id>' by clearml, so tag them 'hpo' and drop
    # the misleading version label.
    is_hpo_trial = "(base)" in (task.name or "")
    version_tag = "hpo" if is_hpo_trial else hp_version
    task.add_tags([a.model_type, f"v{a.dataset_version or ds.version}", version_tag])

    # ---- 6. train, score, save ----------------------------------------------
    print(f"[6/6] training {a.model_type} on {len(Xtr):,} rows x {len(feat_cols)} features")
    model = build_model(a.model_type, len(classes), params, cat_idx, has_val=bool(len(Xva)))
    # boosters early-stop on the val split; the forest fits all trees at once (no early stopping).
    if len(Xva) and a.model_type == "xgboost":
        model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
    elif len(Xva) and a.model_type == "catboost":
        model.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xva, yva))
    else:
        model.fit(Xtr, ytr, sample_weight=wtr)      # the per-row weight, in every library

    # --- TRAINING (in-sample). PRINTED, not logged. this is the OVERFIT CHECK: train scoring far
    # above val/test means the model memorised rather than learned. skipped on HPO trials -- dozens
    # of throwaways do not each need a full train-set predict. logger=None so it only PRINTS (and so
    # series_for, which knows only val/test, is never asked for a 'train' series).
    if not is_hpo_trial:
        tr_proba = full_proba(model, model.predict_proba(Xtr), len(classes))
        tr_pred = tr_proba.argmax(axis=1)
        report_metrics(ytr, tr_pred, tr_proba, classes, logger=None, split="train")

    # --- VALIDATION. this is the number hpo.py optimises. ---------------------
    # WITHOUT THIS BLOCK THE WHOLE SEARCH IS THEATRE. clearml's optimiser does not call a score
    # function of ours -- it lets the trial finish and then READS ONE SCALAR off the task, by
    # title and series. if the scalar is not there, Objective.get_objective() swallows the miss
    # and returns None. None is not an error. every trial scores the same nothing, and hours
    # later the search hands back the parameters it happened to sample first. green all the way,
    # no traceback. it is the same SHAPE of bug as finalize()-without-publish().
    if len(Xva):
        va_proba = full_proba(model, model.predict_proba(Xva), len(classes))
        va_pred = va_proba.argmax(axis=1)
        va_metrics = report_metrics(yva, va_pred, va_proba, classes, logger, split="val")
        va_cost = trading_cost(yva, va_pred, classes)
        report_trading_cost(logger, "val", va_cost)     # -> "Summary"/"val/trading_cost"
        print(f"\n      val trading_cost = {va_cost:.4f}   <- this is what hpo.py minimises")
    else:
        va_metrics, va_cost = None, None
        print("\n      val is OFF (VAL_FRACTION=0) -- no tuning scalar reported. "
              "hpo.py cannot run until you turn it on.")

    # --- TEST. opened ONCE, and never tuned against. --------------------------
    proba = full_proba(model, model.predict_proba(Xte), len(classes))
    pred = proba.argmax(axis=1)
    metrics = report_metrics(yte, pred, proba, classes, logger, split="test")
    te_cost = trading_cost(yte, pred, classes)
    report_trading_cost(logger, "test", te_cost)
    metrics["trading_cost"] = te_cost
    metrics["val"] = None if va_metrics is None else {
        "macro_f1": va_metrics["macro_f1"],
        "mean_pr_auc": va_metrics["mean_pr_auc"],
        "trading_cost": va_cost,
        "never_predicted": va_metrics["never_predicted"]}
    metrics["split"] = split_info

    if metrics["never_predicted"]:
        print(f"\n      !! this model NEVER predicts: {metrics['never_predicted']}")

    # HPO TRIALS ARE SEARCH ONLY. the val/trading_cost the optimiser reads was already reported
    # above, so a trial STOPS HERE -- it does not save or upload a model. otherwise every one of
    # 20-40 throwaway trials would push a ~100 MB bundle into your GCS bucket. only a real run
    # (and the promoted winner, retrained by publish) writes a model. this is what keeps HPO off
    # your bucket entirely.
    if is_hpo_trial:
        print("  HPO trial -- search only; no model saved or uploaded (keeps GCP + local clean).")
        task.close()
        return

    # THE BUNDLE. shap_explain.py and select_champion.py both open this, so its shape is a
    # contract. an early trainer once set output_uri but never actually saved a model -- the
    # task looked green and the "production" model had no file in it. so we save, then check.
    out = pathlib.Path(f"model_{a.model_type}.joblib")
    # XGBoost's pickled booster is a VERSION-SPECIFIC binary buffer -- a bundle saved under one
    # xgboost dies with "input stream corrupted" when an agent-built venv on a different xgboost
    # tries to unpickle it (which just happened, twice). so keep the xgboost OBJECT OUT of the
    # pickle and store its NATIVE, cross-version-portable UBJ instead; load_model_bundle rebuilds
    # it. everything else in the bundle pickles fine across versions.
    # BOTH xgboost AND catboost pickle a version-specific binary buffer -- a bundle saved under one
    # version dies with a corrupted-load when an agent-built venv on a different version unpickles
    # it. so we keep those objects OUT of the pickle and store each library's NATIVE, portable
    # format instead (xgboost UBJ, catboost CBM); load_model_bundle rebuilds them. random_forest is
    # pure sklearn/numpy and pickles fine across versions, so it stays as-is.
    import tempfile
    model_field, xgb_ubj, cb_cbm = model, None, None
    if a.model_type == "xgboost":
        _tf = tempfile.NamedTemporaryFile(suffix=".ubj", delete=False); _tf.close()
        model.save_model(_tf.name)              # sklearn wrapper: preserves num_class/objective
        xgb_ubj = pathlib.Path(_tf.name).read_bytes()
        pathlib.Path(_tf.name).unlink()
        model_field = None
    elif a.model_type == "catboost":
        _tf = tempfile.NamedTemporaryFile(suffix=".cbm", delete=False); _tf.close()
        model.save_model(_tf.name, format="cbm")   # catboost native format -- portable across versions
        cb_cbm = pathlib.Path(_tf.name).read_bytes()
        pathlib.Path(_tf.name).unlink()
        model_field = None
    joblib.dump({
        "model": model_field,
        "xgb_model_ubj": xgb_ubj,               # portable formats; None for the models not using them
        "cb_model_cbm": cb_cbm,
        "label_encoder": le,
        "features": feat_cols,
        "categorical": cat_cols,
        "cat_maps": cat_maps,              # includes RF's sentinels
        "model_type": a.model_type,
        "dataset_id": a.dataset_id,
        "dataset_version": a.dataset_version or ds.version,
        "metrics": metrics,
        # THE SPLIT TRAVELS WITH THE MODEL. shap_explain used to rebuild the test mask from
        # CURRENT config -- so a config edit between training and explaining (it happened the
        # very day this was written: TEST_FRACTION 0.2->0.3) silently made SHAP explain rows
        # the model had TRAINED on. the bundle now carries the cut, and consumers use IT.
        "split": {"test_start": split_info["test_start"],
                  "val_fraction": C.VAL_FRACTION, "test_fraction": C.TEST_FRACTION,
                  "embargo_sessions": C.EMBARGO_SESSIONS},
    }, out)
    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit("the model file was not written -- refusing to report success")
    task.upload_artifact("model", str(out))        # -> lands in the GCS bucket via output_uri

    print(f"\ndone. {a.model_type}")
    if va_metrics is not None:
        print(f"  val   trading_cost={va_cost:.4f}  macro_f1={va_metrics['macro_f1']:.4f}"
              f"   <- tuned on")
    print(f"  test  trading_cost={te_cost:.4f}  macro_f1={metrics['macro_f1']:.4f}"
          f"  mean_pr_auc={metrics['mean_pr_auc']:.4f}   <- report THIS one")
    print(f"trained on dataset {a.dataset_id} (v{ds.version})")

    # ---- now, and ONLY now, ask for this model to be explained -------------
    # SHAP is queued HERE rather than up-front by publish_version.py. the reason is a race:
    # if all three SHAP tasks were queued at the same time as the three models, an agent that
    # finished one model early would grab the next queued task -- which might be the SHAP task
    # for a model that is STILL TRAINING. it would find no model artifact and die.
    # queueing it from inside the finished trainer means the model provably exists.
    #
    # BUT NOT DURING A HYPERPARAMETER SEARCH. an HPO run trains DOZENS of throwaway trial models,
    # and each one reaching this line would queue its own SHAP task -- 20 trials -> 20 SHAP tasks,
    # all piling onto the same queue, explaining models nobody will keep. that is exactly what
    # flooded the board on 2026-07-21 (3 finished trials -> 3 queued shap_xgboost v4). SHAP is for
    # the model you PROMOTE, not for every guess along the way.
    #
    # how we know it is a trial: the optimizer CLONES the base task ("train_xgboost (base)") and
    # names the clone "train_xgboost (base): Args/...". a real run is renamed "train_xgboost v5"
    # by publish_version. so a name still carrying "(base)" is either an HPO trial or the base
    # registration itself -- neither wants SHAP.
    is_hpo_trial = "(base)" in (task.name or "")
    if is_hpo_trial:
        print("  HPO trial (or base task) -- NOT queuing SHAP. explain the promoted winner, "
              "not every trial.")
    else:
        queue_shap_for_me(task.id, a.model_type, a.dataset_version or ds.version)

    task.close()


def queue_shap_for_me(model_task_id: str, model_type: str, version: str):
    """ask ClearML to run shap_explain against THIS model, now that it is saved."""
    from clearml import Task
    base = Task.get_task(project_name=C.CLEARML_PROJECT, task_name=C.BASE_SHAP_NAME)
    if base is None:
        print("  (no SHAP base task registered -- skipping. "
              "run: python trainer/register_base_trainer.py)")
        return
    run = Task.clone(source_task=base, name=f"shap_{model_type} v{version}")
    run.set_parameters({
        "Args/model_task_id": model_task_id,
        "Args/model_type": model_type,
        "Args/dataset_version": version,
    })
    Task.enqueue(run, queue_name=C.SHAP_QUEUE)
    print(f"  queued shap_{model_type} v{version}  (explains THIS model)")


if __name__ == "__main__":
    main()
