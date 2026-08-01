# Where the data lives (for the visualizer)

Nothing is on your disk. Everything comes from our GCP bucket, fetched through ClearML by **name**
(not by id — ids change every time we re-upload, names do not).

One-time setup on your machine:

```bash
pip install clearml pandas pyarrow
clearml-init          # we will give you the keys
gcloud auth application-default login
gcloud auth application-default set-quota-project mega-ml
```

ClearML project: **`nifty_main`**

---

## 1. Candles — the master OHLCV

```python
from clearml import Dataset
import pandas as pd, pathlib

d = Dataset.get(dataset_project="nifty_main", dataset_name="nifty_ohlcv")
folder = pathlib.Path(d.get_local_copy())
df = pd.read_parquet(next(folder.rglob("*.parquet")))
df = df.reset_index()          # time is the INDEX, named 'datetime'
```

1,021,847 rows · 2015-01-09 09:15 → 2026-02-24 15:29 · 1-minute bars.

Columns:

| column | |
|---|---|
| `datetime` | the index — reset_index() to get it as a column |
| `Nifty_Futures_Open` / `High` / `Low` / `Close` | OHLC |
| `close` | duplicate of close, kept for older scripts |
| `Nifty_Futures_Volume`, `Nifty_Futures_Open_Interest` | |
| `VIX_Close`, `Time_Counter`, `ATR_14_NEW_FUTURES` | |

Your `schema.detect_ohlc()` already maps this correctly — we tested it. It picks
`datetime` / `Nifty_Futures_Open` / `_High` / `_Low` / `close`.

---

## 2. Predictions — the scored table

One per model, per run. They are **artifacts on a task**, not files in a bucket.

```python
from clearml import Task

# the model you want (they are named  train_xgboost v7  etc.)
model = Task.get_task(project_name="nifty_main", task_name="train_xgboost v7")

# its scored-tables task carries Args/model_task_id == model.id  -- that is the exact link.
# (or just open ClearML and copy the scored_tables task id)
t = Task.get_task(task_id="<the scored_tables task id>")
path = t.artifacts["scored_test_v7"].get_local_copy()
df = pd.read_parquet(path)
```

Artifact names: `scored_train_v7`, `scored_test_v7` (and `scored_oos_v7` on an OOS run).
**`scored_test_*` is the honest one** — train is what the model already saw.

Columns:

| column | what your code calls it |
|---|---|
| `timestamp` | time |
| `true_label` | target |
| `predicted_label` | predicted |
| `unique_index` | our row id (candle bundle id) |
| `split` | train / val / test |
| `correct` | bool |
| `proba_NO_TRADE`, `proba_ENTRY_SUPER`, … | 7 probability columns |
| `model_task_id`, `model_type`, `train_dataset_id`, `train_dataset_version`, `scored_at` | provenance |

Labels are the 7 strings: `NO_TRADE, ENTRY_SUB, ENTRY_SMALL, ENTRY_SUPER, EXIT_SUB, EXIT_SMALL,
EXIT_SUPER`.

Your `schema.detect_labels()` finds `timestamp` / `true_label` / `predicted_label` on its own.

---

## 3. Features — the dataset the model trained on

The model names its own dataset, so you never have to guess the version:

```python
import joblib
bundle = joblib.load(model.artifacts["model"].get_local_copy())
ds = Dataset.get(dataset_id=bundle["dataset_id"])
feat = pd.read_parquet(next(pathlib.Path(ds.get_local_copy()).rglob("*.parquet")))
```

Feature columns are named `source__column` (e.g. `vwap_microstructure_raw__vwap_dist`).
`timestamp` is a column here, not the index. `primary_label` and `weight` are also in this file.

---

## 4. The OOS set

Same pattern, name `nifty_oos_dataset`. Held-out period, no labels used for scoring — the model
predicts and the backtest evaluates.

---

## Notes

- `Dataset.get_local_copy()` caches under `~/.clearml/cache/`, so the second call is instant.
- Please **do not rename any feature column**. Downstream code matches on those names.
- Give us back the edited `app.py` / `src/` and we merge it into
  `final_pipeline/ui/visualizer/` — the page wrapper around it is ours and does not need touching.
