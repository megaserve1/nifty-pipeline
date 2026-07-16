"""
tests/test_versioning.py -- the version numbering, and why it is shaped this way.

    v1, v2, v3      a MAJOR -- a fresh human choice (a ballot, or --all)
    v2.1, v2.2      a MINOR -- a VARIATION on v2 (a feature dropped or added)

THE POINT: a comparison only means something if ONE thing changed.
    v2 vs v2.1   -> only stress_signal is missing. score drops? THAT FEATURE MATTERS.
    v2 vs v7     -> two different selections. too much changed. teaches nothing.
So: same major = a fair comparison.
"""
import sys
import pathlib

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "core"))


@pytest.fixture
def mv(monkeypatch, tmp_path):
    """make_version, pointed at a throwaway registry + versions dir."""
    import importlib
    import config as C

    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump({
        "alpha": {"file": "a.parquet", "clock": "1min"},
        "beta":  {"file": "b.parquet", "clock": "5min"},
        "gamma": {"file": "c.parquet", "clock": "1min"},
    }))
    monkeypatch.setattr(C, "REGISTRY", reg)
    monkeypatch.setattr(C, "VERSIONS_DIR", tmp_path)
    monkeypatch.setattr(C, "CONFIGS_DIR", tmp_path)

    m = importlib.import_module("make_version")
    importlib.reload(m)
    monkeypatch.setattr(m.C, "REGISTRY", reg)
    monkeypatch.setattr(m.C, "VERSIONS_DIR", tmp_path)
    monkeypatch.setattr(m.C, "CONFIGS_DIR", tmp_path)
    return m


# ------------------------------------------------------------------ majors
def test_a_ballot_makes_a_MAJOR(mv):
    v = mv.freeze(["alpha", "beta"], source="ballot")
    assert v == "v1", "a fresh human choice is a new starting point"
    v2 = mv.freeze(["alpha"], source="ballot")
    assert v2 == "v2"


def test_a_major_records_that_it_is_a_selection(mv):
    mv.freeze(["alpha", "beta"], source="ballot")
    doc = mv.load_version("v1")
    assert doc["kind"] == "selection"
    assert doc["parent"] is None


# ------------------------------------------------------------------ minors
def test_deriving_makes_a_MINOR_under_the_parents_major(mv):
    mv.freeze(["alpha", "beta", "gamma"], source="ballot")      # v1
    v = mv.derive("v1", drop=["beta"], add=[])
    assert v == "v1.1", "a variation on v1, not a whole new selection"

    v = mv.derive("v1", drop=["gamma"], add=[])
    assert v == "v1.2", "the next variation on the same parent"


def test_deriving_from_a_SUB_version_stays_FLAT(mv):
    """v1.1 with another feature dropped becomes v1.3 -- NOT v1.1.1.
    it is still a variation on v1. the parent is recorded, so nothing is lost."""
    mv.freeze(["alpha", "beta", "gamma"], source="ballot")      # v1
    mv.derive("v1", drop=["beta"], add=[])                      # v1.1
    v = mv.derive("v1.1", drop=["gamma"], add=[])
    assert v == "v1.2", "flat under the major. no v1.1.1."

    doc = mv.load_version("v1.2")
    assert doc["parent"] == "v1.1", "...but the true parent IS recorded"


def test_a_new_ballot_after_variations_starts_a_new_MAJOR(mv):
    mv.freeze(["alpha", "beta"], source="ballot")     # v1
    mv.derive("v1", drop=["beta"], add=[])            # v1.1
    v = mv.freeze(["gamma"], source="ballot")
    assert v == "v2", "a fresh choice is a new major, not v1.2"


def test_a_minor_records_its_parent_and_how_it_was_made(mv):
    mv.freeze(["alpha", "beta", "gamma"], source="ballot")
    mv.derive("v1", drop=["beta"], add=[])
    doc = mv.load_version("v1.1")
    assert doc["kind"] == "variation"
    assert doc["parent"] == "v1"
    assert "-drop beta" in doc["selected_by"]
    assert doc["features"] == ["alpha", "gamma"]


# ------------------------------------------------------------------ the guards
def test_a_change_that_changes_nothing_is_refused(mv):
    """dropping a feature that is not there would give you a copy of the parent under a new
    number -- which looks like an experiment and is not one."""
    mv.freeze(["alpha", "beta"], source="ballot")
    with pytest.raises(SystemExit, match="IDENTICAL"):
        mv.derive("v1", drop=["gamma"], add=[])       # gamma was never in v1


def test_a_typo_in_a_feature_name_stops_everything(mv):
    mv.freeze(["alpha"], source="ballot")
    with pytest.raises(SystemExit, match="unknown feature"):
        mv.derive("v1", drop=[], add=["alphaa"])


def test_an_unknown_parent_is_refused(mv):
    with pytest.raises(SystemExit, match="unknown version"):
        mv.derive("v9", drop=["alpha"], add=[])


# ------------------------------------------------------------------ the plan file
def test_a_plan_file_makes_many_variations_at_once(mv, tmp_path, capsys):
    """THE ABLATION. one file, one command, one version per feature dropped."""
    mv.freeze(["alpha", "beta", "gamma"], source="ballot")      # v1

    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump({"versions": [
        {"parent": "v1", "drop": ["alpha"]},
        {"parent": "v1", "drop": ["beta"]},
        {"parent": "v1", "drop": ["gamma"]},
    ]}))

    mv.mode_from_plan(plan)
    out = capsys.readouterr().out
    assert "created: 3" in out, "and it must SAY so -- a quiet plan runner loses you a day"

    for v, gone in [("v1.1", "alpha"), ("v1.2", "beta"), ("v1.3", "gamma")]:
        doc = mv.load_version(v)
        assert gone not in doc["features"]
        assert doc["parent"] == "v1"
    # all three share a parent -> they are FAIR to compare with each other


def test_a_plan_reports_its_failures_loudly(mv, tmp_path, capsys):
    mv.freeze(["alpha", "beta"], source="ballot")
    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump({"versions": [
        {"parent": "v1", "drop": ["alpha"]},          # fine
        {"parent": "v1", "drop": ["nonexistent"]},    # changes nothing -> refused
        {"drop": ["beta"]},                            # no parent
    ]}))
    mv.mode_from_plan(plan)
    out = capsys.readouterr().out
    assert "created: 1" in out
    assert "failed: 2" in out


# ------------------------------------------------------------------ ClearML mapping
def test_our_version_IS_the_clearml_version():
    """no translation layer to get wrong. and it sorts correctly under PEP440."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "core"))
    from publish_version import semver

    assert semver("v2") == "2.0"
    assert semver("v2.1") == "2.1"
    assert semver("v2.10") == "2.10"

    from packaging.version import Version
    assert Version("2.10") > Version("2.9"), "PEP440 sorts numerically, not as text"
    assert Version("2.1") > Version("2.0")
    assert Version("3.0") > Version("2.99")
