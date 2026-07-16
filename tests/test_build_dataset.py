"""
tests/test_build_dataset.py -- the whole ingest chain, on synthetic data.

builds a fake feature parquet + fake labels, runs the real registry -> align -> NaN-policy ->
join path, and proves:
  * every label minute survives (the labels are the spine)
  * a 5-min feature is never visible before its bar closes  (no lookahead, end to end)
  * the NaN policy was actually applied
  * the certificate matches the parquet that was written
"""
import json
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from bridge.align import align_feature_to_labels          # noqa: E402
from na_policy import apply_policy                 # noqa: E402
from bridge.register import feature_name, read_time_index # noqa: E402
import contract, hashes                                   # noqa: E402


# ------------------------------------------------------------------ the name rule
def test_feature_name_cleans_the_windows_copy_suffix():
    p = pathlib.Path("gap_final_features - Copy.parquet")
    assert feature_name(p) == "gap_final_features"


def test_feature_name_is_tidy():
    assert feature_name(pathlib.Path("My Feature-Name.parquet")) == "my_feature_name"


# ------------------------------------------------------------------ time index
def test_time_can_arrive_as_the_index_or_as_a_column():
    idx = pd.date_range("2020-01-01 09:15", periods=5, freq="1min")
    a = pd.DataFrame({"x": range(5)}, index=idx)                 # index
    b = pd.DataFrame({"datetime": idx, "x": range(5)})           # column
    assert isinstance(read_time_index(a).index, pd.DatetimeIndex)
    assert isinstance(read_time_index(b).index, pd.DatetimeIndex)


def test_a_parquet_with_no_timestamp_is_refused():
    with pytest.raises(ValueError, match="no timestamp"):
        read_time_index(pd.DataFrame({"x": [1, 2, 3]}))


# ------------------------------------------------------------------ the full chain
def _spine(n=40):
    return pd.date_range("2020-01-01 09:15", periods=n, freq="1min")


def test_end_to_end_no_lookahead_survives_the_whole_chain():
    """a 5-min feature, NaN in it, ingested exactly the way build_dataset does it."""
    labels = _spine(40)

    # a 5-min value smeared across five 1-min rows -- the real shape of the data.
    # bucket 09:15 (knowable only at 09:20) carries 100.
    bucket = labels.floor("5min")
    val = pd.Series(bucket).map({t: 100.0 * (i + 1) for i, t in enumerate(sorted(set(bucket)))})
    feat = pd.DataFrame({"v": val.values, "state": "NO_GAP"}, index=labels)
    feat.loc[feat.index[:3], "v"] = np.nan            # intentional NaN at the start

    # 1. NaN policy: sentinel, kept as NaN for xgboost/catboost
    treated, note = apply_policy(feat, "sentinel", for_model="native", bar_minutes=5)
    assert treated["v"].isna().sum() == 3, "the NaN must be preserved for native-NaN models"

    # 2. align to the spine
    out = align_feature_to_labels(treated, labels, bar_minutes=5)
    got = pd.Series(out["v"].values, index=labels)

    # the spine is untouched: one row per label minute, in order
    assert len(out) == len(labels)

    # THE INVARIANT: 09:16 must NOT see the 09:15 bucket (it closes at 09:20)
    assert pd.isna(got.loc["2020-01-01 09:16"]), "LEAK: saw a bar that had not closed"
    # 09:20 is the close -> it becomes legal exactly then
    assert got.loc["2020-01-01 09:25"] == 200.0      # the 09:20 bar, now closed


def test_random_forest_gets_no_nan_but_xgboost_does():
    """the same feature, two treatments -- one per model family."""
    idx = _spine(10)
    feat = pd.DataFrame({"v": [np.nan, 0.0, 0.5, np.nan, 1.0, 0.2, np.nan, 0.9, 0.1, 0.3]},
                        index=idx)

    native, _ = apply_policy(feat, "sentinel", for_model="native")
    rf, note = apply_policy(feat, "sentinel", for_model="no_nan")

    assert native["v"].isna().sum() == 3, "xgboost/catboost keep the NaN"
    assert rf["v"].isna().sum() == 0, "random forest cannot take NaN"

    sv = note["columns"]["v"]["sentinel"]
    assert sv < feat["v"].min(skipna=True), "and the sentinel cannot collide with a real value"
    assert rf["v"].iloc[1] == 0.0, "the real 0.0 must survive untouched"


# ------------------------------------------------------------------ the certificate
def test_manifest_certifies_the_parquet_it_describes(tmp_path):
    recipe = tmp_path / "dataset_v1.yaml"
    recipe.write_text("name: dataset_v1\nfeatures: [f]\n")
    parquet = tmp_path / "dataset_v1.parquet"
    pd.DataFrame({"a": [1.0, 2.0]}).to_parquet(parquet)

    contract.write_manifest(
        tmp_path / "manifest.json",
        version="v1",
        recipe_sha256=hashes.sha256_file(recipe),
        parquet_sha256=hashes.sha256_file(parquet),
        rows=2, n_features=1,
        schema=[{"col": "a", "dtype": "float64"}],
        feature_columns=["a"], label_col="primary_label", weight_cols=["weight"],
        per_feature=[], labels_name="lbl", labels_sha256="x",
        class_distribution={"NO_TRADE": 2},
        no_peek={"applied": True, "rule": "bar_close", "tolerance_bars": 3},
        built_by="test", status="ready",
    )
    man = contract.load(tmp_path / "manifest.json")
    contract.validate_manifest(man, recipe, parquet)          # must not raise

    # tamper with the parquet -> the gate must refuse it
    pd.DataFrame({"a": [9.9]}).to_parquet(parquet)
    with pytest.raises(contract.ContractError, match="PARQUET MISMATCH"):
        contract.validate_manifest(man, recipe, parquet)
