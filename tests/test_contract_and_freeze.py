"""
tests/test_contract_and_freeze.py -- the publish GATE and the selection GATE.

These are the two places where a bad thing must be STOPPED rather than passed along:
  * freeze()           refuses a typo'd or empty feature selection
  * validate_manifest() refuses a parquet that does not match the recipe that ordered it,
                        or that does not certify the no-peek rule
"""
import json
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import contract  # noqa: E402
import hashes    # noqa: E402


# ------------------------------------------------------------------ manifest gate
def _good_manifest(recipe: pathlib.Path, parquet: pathlib.Path) -> dict:
    return {
        "manifest_schema_version": contract.MANIFEST_SCHEMA_VERSION,
        "version": "v9",
        "recipe_sha256": hashes.sha256_file(recipe),
        "parquet_sha256": hashes.sha256_file(parquet),
        "rows": 100, "n_features": 2,
        "schema": [{"col": "a__x", "dtype": "float64"}, {"col": "b__y", "dtype": "float64"}],
        "feature_columns": ["a__x", "b__y"], "label_col": "primary_label",
        "weight_cols": ["weight"],
        "per_feature": [], "labels_name": "labels_x", "labels_sha256": "deadbeef",
        "class_distribution": {"NO_TRADE": 10},
        "no_peek": {"applied": True, "rule": "bar_close", "tolerance_bars": 3},
        "built_at": "2026-01-01T00:00:00+00:00", "built_by": "test", "status": "ready",
    }


@pytest.fixture
def files(tmp_path):
    recipe = tmp_path / "dataset_v9.yaml"
    recipe.write_text("name: dataset_v9\nfeatures: [a]\n")
    parquet = tmp_path / "dataset_v9.parquet"
    # a REAL parquet, matching the manifest's rows/schema. it used to be junk bytes ("just
    # hashable"), which was enough while the gate only hashed the file -- the gate now reads
    # the parquet FOOTER to verify rows and column names, so the certificate's own test data
    # must be honest too.
    import pandas as pd
    pd.DataFrame({"a__x": [1.0] * 100, "b__y": [2.0] * 100}).to_parquet(parquet, index=False)
    return recipe, parquet


def test_valid_manifest_passes(files):
    recipe, parquet = files
    contract.validate_manifest(_good_manifest(recipe, parquet), recipe, parquet)


def test_rejects_parquet_built_from_a_different_recipe(files):
    recipe, parquet = files
    man = _good_manifest(recipe, parquet)
    recipe.write_text("name: dataset_v9\nfeatures: [a, b]\n")   # recipe edited after build
    with pytest.raises(contract.ContractError, match="RECIPE MISMATCH"):
        contract.validate_manifest(man, recipe, parquet)


def test_rejects_parquet_that_changed_after_certification(files):
    recipe, parquet = files
    man = _good_manifest(recipe, parquet)
    parquet.write_bytes(b"tampered")
    with pytest.raises(contract.ContractError, match="PARQUET MISMATCH"):
        contract.validate_manifest(man, recipe, parquet)


def test_refuses_to_publish_a_dataset_that_does_not_certify_no_peek(files):
    recipe, parquet = files
    man = _good_manifest(recipe, parquet)
    man["no_peek"] = {"applied": False, "rule": "bar_close"}
    with pytest.raises(contract.ContractError, match="lookahead"):
        contract.validate_manifest(man, recipe, parquet)


def test_refuses_an_unknown_no_peek_rule(files):
    recipe, parquet = files
    man = _good_manifest(recipe, parquet)
    man["no_peek"] = {"applied": True, "rule": "shift1"}   # the OLD, leaky rule
    with pytest.raises(contract.ContractError, match="only trusts 'bar_close'"):
        contract.validate_manifest(man, recipe, parquet)


def test_refuses_a_half_written_build(files):
    recipe, parquet = files
    man = _good_manifest(recipe, parquet)
    man["status"] = "building"
    with pytest.raises(contract.ContractError, match="did not finish"):
        contract.validate_manifest(man, recipe, parquet)


# ------------------------------------------------------------------ schema drift
def test_assert_schema_catches_a_renamed_column(files):
    import pandas as pd
    recipe, parquet = files
    man = _good_manifest(recipe, parquet)
    df = pd.DataFrame({"a_renamed": [1.0, 2.0]})
    with pytest.raises(contract.ContractError, match="SCHEMA DRIFT"):
        contract.assert_schema(df, man)


# ------------------------------------------------------------------ selection gate
def test_freeze_refuses_unknown_and_empty(monkeypatch, tmp_path):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "core"))
    import config as C
    monkeypatch.setattr(C, "VERSIONS_DIR", tmp_path)
    monkeypatch.setattr(C, "REGISTRY", tmp_path / "registry.yaml")
    C.REGISTRY.write_text("real_feature:\n  file: r.py\n  clock: 1min\n")

    import importlib
    mv = importlib.import_module("make_version")
    importlib.reload(mv)
    monkeypatch.setattr(mv.C, "VERSIONS_DIR", tmp_path)
    monkeypatch.setattr(mv.C, "REGISTRY", tmp_path / "registry.yaml")

    with pytest.raises(SystemExit, match="unknown feature"):
        mv.freeze(["typo_feature"], source="test")
    with pytest.raises(SystemExit, match="EMPTY"):
        mv.freeze([], source="test")
