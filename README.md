# NIFTY dataset and model-training pipeline

This repository turns feature parquet files and one-minute labels into a frozen,
hash-verified dataset version. It publishes that dataset with DVC and ClearML, trains
Random Forest, XGBoost, and CatBoost, explains each model with SHAP, and selects the model
whose mistakes have the lowest trading-severity cost.

## Pipeline flow

```text
feature parquets + labels CSV
          |
          v
bridge/register.py          -> registry.yaml
          |
          v
core/make_version.py        -> selection_sheet.yaml -> versions/dataset_vN.yaml
          |
          v
bridge/build_dataset.py     -> datasets/vN/dataset_vN.parquet
                               datasets/vN/manifest.json
          |
          v
core/publish_version.py     -> verifies recipe, schema and hashes
                            -> DVC/GCS data + ClearML dataset + lock YAML
                            -> queues model training and champion selection
          |
          +--> trainer/train.py -> model artifact + metrics
          |                         |
          |                         +--> trainer/shap_explain.py
          |
          +--> trainer/select_champion.py
```

Important supporting files:

- `contract.py` checks that the recipe, parquet, and manifest agree before publishing.
- `hashes.py` calculates the SHA-256 values used for integrity and lineage checks.
- `bridge/align.py` serves a feature only after its bar has closed.
- `bridge/leak_guard.py` rejects suspicious columns such as forward returns and running IDs.
- `na_policy.py` and `bridge/na_policy.py` apply each feature's declared missing-value rule.
- `trainer/objective.py` creates the time split and calculates trading cost.
- `trainer/purged_cv.py` supplies session-aware embargo helpers and optional purged K-fold CV.
- `configs/hyperparams.yaml` is the single source for model defaults and HPO search spaces.
- `configs/severity_7class.json` defines how costly each kind of classification mistake is.

## Clone and install

```bash
git clone <YOUR-GIT-REPOSITORY-URL>
cd final_pipeline

python3 -m venv final_venv
source final_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the offline verification suite immediately after cloning:

```bash
python -m pytest -q
```

The tests use synthetic fixtures, so they can run without downloading the market dataset.

## Restore the published v3 data

Git contains only `datasets/v3/dataset_v3.parquet.dvc`, not the 201 MB parquet itself.
Authenticate to the configured GCS bucket and let DVC restore it:

```bash
# Laptop authentication; a GCE VM should use its attached service account instead.
gcloud auth application-default login

dvc pull datasets/v3/dataset_v3.parquet.dvc
```

If a clone should use another bucket, keep that change local:

```bash
dvc remote modify --local gcs url gs://<bucket>/final_pipeline/dvc
dvc pull datasets/v3/dataset_v3.parquet.dvc
```

The current certified v3 contains 513,611 rows and 282 feature columns. A user needs GCS
permission to pull its bytes. Raw source features and labels are intentionally not stored in
Git; the feature team must supply them when building a new version.

## Configuration before a real run

Review `config.py`:

- replace the demo `GCS_BUCKET` before production use;
- choose `STORAGE_MODE = "gcs"` or `"local"`;
- confirm `LABELS_FILE`, `VAL_FRACTION`, `TEST_FRACTION`, and `EMBARGO_SESSIONS`;
- confirm `LABEL_HORIZON_SESSIONS` with the label author;
- review each feature's `clock`, `clock_per_column`, and `na_policy` in `registry.yaml`.

For ClearML, copy the template outside the repository and insert your own credentials:

```bash
cp clearml.conf.example ~/clearml.conf
```

Never commit the completed `clearml.conf`; it is ignored by Git.

## Build a new dataset version

Place source files here:

```text
data/features/<one logical feature>.parquet
data/labels/<labels file>.csv
```

Then run:

```bash
# 1. Inspect feature files and create/update registry.yaml.
python bridge/register.py
python bridge/register.py --list

# 2a. Interactive selection.
python core/make_version.py --new
# Change x: false to x: true in selection_sheet.yaml.
python core/make_version.py --from-sheet

# 2b. Or select every registered feature directly.
python core/make_version.py --all

# 3. Build the exact version printed by make_version.py.
python bridge/build_dataset.py --version vN

# 4. Rehearse every publication check without changing Git, GCS, or ClearML.
python core/publish_version.py --version vN --dry-run

# 5. Publish and queue training only after the dry-run passes.
python core/publish_version.py --version vN

# 6. Preserve the publication receipt and push both dataset commits to the Git remote.
git add versions/dataset_vN.lock.yaml
git commit -m "lock dataset vN publication"
git push
```

Useful version commands:

```bash
python core/make_version.py --from v3 --drop feature_a --add feature_b
python core/make_version.py --from-plan
python core/make_version.py --list
python core/dataset_diff.py v3 v3.1
python core/dataset_diff.py --all
```

Frozen `versions/dataset_vN.yaml` files are records: do not edit one after creation. Create a
new major or minor version instead.

## ClearML training setup

Register the reusable base tasks after the first setup and whenever their captured code changes:

```bash
python trainer/register_base_trainer.py --force
clearml-agent daemon --queue training
```

`publish_version.py` clones one base task per requested model. Each completed trainer uploads a
Joblib model bundle to `model_output_uri()`, queues its own SHAP task, and reports validation/test
metrics. The champion task waits for the requested models and ranks them by trading cost.

Examples:

```bash
# Publish data without training.
python core/publish_version.py --version vN --no-train

# Train only selected model types.
python core/publish_version.py --version vN --models random_forest,xgboost

# Local smoke check using an already restored parquet.
python trainer/local_check.py --version v3 --models xgboost --rows 50000
```

Do not run `core/auto_trigger.py` beside the normal publish path: both can enqueue training.
`auto_trigger.py` is only for a dataset published outside `publish_version.py`.

## Time split and evaluation

The default split is chronological:

```text
oldest                                                   newest
|------ train ------| 25-session gap |-- validation --| gap |--- test ---|
```

With the current config, train/validation/test use 55%/15%/30% of the time range. HPO sees only
validation trading cost; test is reserved for the final measurement. The full `PurgedKFold`
class is available and tested, but normal training uses the single chronological split above.

Reported evaluation includes per-class precision/recall/F1, per-class PR-AUC, macro F1, mean
PR-AUC, confusion matrix, never-predicted classes, and severity-weighted trading cost. Accuracy
is deliberately not the headline because the classes are imbalanced.

## Hyperparameter search

Integrated publish-and-tune path:

```bash
python core/publish_version.py --version vN --tune --hpo-trials 15
```

Manual path:

```bash
python trainer/hpo.py \
  --model_type xgboost \
  --dataset_id <CLEARML-DATASET-ID> \
  --dataset_version vN \
  --trials 30 \
  --concurrent 2

python trainer/apply_hpo.py best_params_xgboost.json
```

The unattended three-model helper requires the dataset ID instead of embedding a machine-specific
ID in Git:

```bash
./run_tuned_overnight.sh <CLEARML-DATASET-ID> vN
```

## What is and is not in Git

Committed:

- pipeline source, configuration, tests, and documentation;
- `registry.yaml` and frozen version recipes;
- v3 manifest, publication lock, and DVC pointer.

Ignored:

- credentials and local ClearML configuration;
- virtual environments, caches, logs, notebooks, and generated reports;
- raw feature/label files, dataset parquet bytes, and trained model artifacts;
- temporary selection sheets and HPO result files.

## Open production work

- Deepchecks and Evidently are planned in `docs/PLAN_deepchecks_evidently_gcp.md`; their gates
  are not implemented yet.
- Replace the demo GCS bucket and rotate any previously exposed ClearML credential.
- Resolve the feature-team questions recorded in `docs/AUDIT_2026-07-15.md`, especially duplicate
  columns, high-NaN columns, clock overrides, and the confirmed label horizon.
- v3 fixes the zero-weight `NO_TRADE` issue, but its final business weight should still be judged
  from local/validation trading behavior.
