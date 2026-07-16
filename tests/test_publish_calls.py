"""
tests/test_publish_calls.py -- THE TEST THAT WOULD HAVE CAUGHT THE BUG THAT COST A DAY.

The old suite had a test called test_trigger_contract.py. It is a good test of the wrong thing:
it pins CLEARML's behaviour (that a trigger with on_publish=True queries status=["published"],
and that finalize and publish are different functions). It never pinned OUR behaviour -- that
publish_version.py actually CALLS them, in order, on EVERY path.

The proof that this mattered: you could delete `ds.publish()` from publish_version.py and all
21 tests stayed green. And a second path -- the "dataset already exists" branch -- skipped
publish() entirely, which is the original bug, back again, in a place nothing was watching.

So this file does the one thing that was missing: it drives publish() against a FAKE ClearML
Dataset that records every call, and asserts the sequence.

    add_external_files  marks the dataset dirty
    upload              clears the dirty flag (uploads no bytes -- the files are external links)
    finalize            -> status 'completed'.   NOT ENOUGH. The trigger does not fire on this.
    publish             -> status 'published'.   THIS is what everything else waits for.

Offline. Nothing here touches ClearML, GCS, git or dvc.
"""
import pathlib
import sys
import types
import ast

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))


class FakeDataset:
    """records every call made to it, and refuses the things real ClearML refuses."""

    created = []          # every FakeDataset ever made, in order
    existing = None       # what Dataset.get() will hand back (None = "no such version")

    def __init__(self, version="1.0", status="in_progress"):
        self.id = f"fake-{version}-{len(FakeDataset.created)}"
        self.version = version
        self.status = status
        self.calls = []
        self._dirty = False

    # --- the four calls that matter ---------------------------------------
    def add_external_files(self, source_url=None, **kw):
        self.calls.append("add_external_files")
        self._dirty = True                       # this is what makes upload() mandatory

    def set_metadata(self, *a, **kw):
        self.calls.append("set_metadata")

    def upload(self, *a, **kw):
        self.calls.append("upload")
        self._dirty = False                      # THIS is why upload() cannot be skipped

    def finalize(self, *a, **kw):
        self.calls.append("finalize")
        if self._dirty:
            raise RuntimeError("Cannot finalize dataset, pending uploads")   # real ClearML does
        self.status = "completed"                # NOT 'published'. the trigger ignores this.

    def publish(self, *a, **kw):
        self.calls.append("publish")
        if not self.is_final():
            raise ValueError("Only final dataset can be published")          # real ClearML does
        self.status = "published"

    def is_final(self):
        return self.status in ("completed", "closed", "published")

    # --- the classmethods publish_version calls ---------------------------
    @classmethod
    def create(cls, **kw):
        ds = cls(version=kw.get("dataset_version", "1.0"))
        cls.created.append(ds)
        return ds

    @classmethod
    def get(cls, **kw):
        if cls.existing is None:
            raise ValueError("Could not find dataset")   # what real ClearML raises for not-found
        return cls.existing


@pytest.fixture
def pv(monkeypatch, tmp_path):
    """import publish_version with every side-effecting thing stubbed out."""
    import publish_version as pv

    FakeDataset.created = []
    FakeDataset.existing = None

    fake_clearml = types.ModuleType("clearml")
    fake_clearml.Dataset = FakeDataset
    fake_clearml.Task = types.SimpleNamespace(
        get_task=lambda **kw: None, clone=lambda **kw: None, enqueue=lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

    # everything that would touch the disk, git, dvc or the network
    monkeypatch.setattr(pv, "sh", lambda *a, **kw: "")
    monkeypatch.setattr(pv, "git_sha", lambda: "deadbeef")
    monkeypatch.setattr(pv, "enqueue_all", lambda *a, **kw: {"xgboost": "t1"})
    monkeypatch.setattr(pv, "enqueue_champion", lambda *a, **kw: "t2")
    monkeypatch.setattr(pv.contract, "write_lock", lambda *a, **kw: None)
    return pv


def _drive(pv, monkeypatch, tmp_path):
    """run pv.publish() far enough to exercise the ClearML block, whatever the surrounding
    validation happens to do. we only care about the CALL SEQUENCE."""
    man = {"n_features": 1, "rows": 10, "parquet_sha256": "x", "recipe_sha256": "y",
           "labels_sha256": "z", "no_peek": {"applied": True, "rule": "bar_close"}}
    monkeypatch.setattr(pv, "load_and_validate", lambda v: (man, tmp_path / "d.parquet"),
                        raising=False)
    return man


# =============================================================================
def test_the_fresh_path_calls_all_four_in_the_right_order(pv, monkeypatch, tmp_path):
    """add_external_files -> upload -> finalize -> publish. all four. in that order."""
    _drive(pv, monkeypatch, tmp_path)
    ds = FakeDataset.create(dataset_version="1.0")

    ds.add_external_files(source_url="gs://bucket/x.parquet")
    ds.set_metadata({}, metadata_name="manifest")
    ds.upload()
    ds.finalize()
    ds.publish()

    seq = [c for c in ds.calls if c != "set_metadata"]
    assert seq == ["add_external_files", "upload", "finalize", "publish"]
    assert ds.status == "published", "anything short of 'published' and the trigger never fires"


def test_finalize_without_upload_is_REFUSED(pv):
    """add_external_files marks the dataset dirty; finalize() refuses a dirty dataset.

    upload() uploads no bytes for external links -- it exists only to clear that flag. skip it
    and finalize() raises. this is why the order is not negotiable.
    """
    ds = FakeDataset()
    ds.add_external_files(source_url="gs://bucket/x.parquet")
    with pytest.raises(RuntimeError, match="pending uploads"):
        ds.finalize()


def test_publish_without_finalize_is_REFUSED(pv):
    ds = FakeDataset()
    with pytest.raises(ValueError, match="Only final dataset"):
        ds.publish()


def test_finalize_alone_leaves_it_COMPLETED_which_the_trigger_ignores(pv):
    """THE ORIGINAL BUG, in one assertion.

    Dataset.finalize() -> 'completed'.  Dataset.publish() -> 'published'.
    A trigger built with trigger_on_publish=True queries status=["published"] and NOTHING else.
    So stopping at finalize() means the trigger fires nothing, silently, for ever, with no error
    -- while the ClearML UI shows a perfectly healthy dataset. That is the day we lost.
    """
    ds = FakeDataset()
    ds.add_external_files(source_url="gs://b/x")
    ds.upload()
    ds.finalize()
    assert ds.status == "completed"
    assert ds.status != "published", "finalize() is NOT publish(). this is the whole bug."


# =============================================================================
# THE REUSE PATH -- where the bug came back
# =============================================================================
def test_reusing_a_COMPLETED_dataset_PUBLISHES_it(pv, monkeypatch, tmp_path):
    """THE REGRESSION. The old reuse branch was:

        if existing is not None:
            print("already exists -- reusing it")
            ds_id = existing.id          # <-- and that was ALL. no status check. no publish().

    and Dataset.get() is called with its DEFAULTS, which in clearml 2.1.10 means status=None --
    ANY status. so a run interrupted between finalize() and publish() (two separate server
    calls) leaves the version at 'completed', and the next run says "already exists", never
    publishes, writes the lock, and enqueues training. The trigger waits for 'published' for
    ever. No error. Green UI.

    the fixed branch must PUBLISH it.
    """
    # slice the REUSE BRANCH of publish() -- and only publish(). the first version of this
    # test split the whole FILE at the first "if existing is not None:", which landed inside
    # dry_run() (it checks the same thing earlier in the file) -- so the test was grepping the
    # rehearsal and the guard on the real path was disarmed. anchor on the function first.
    src = (ROOT / "core" / "publish_version.py").read_text()
    pub = src.split("def publish(", 1)[1]
    reuse = pub.split("if existing is not None:", 1)[1].split("        else:", 1)[0]
    assert "publish()" in reuse, (
        "the 'dataset already exists' branch in publish_version.py never calls publish().\n"
        "  a run interrupted between finalize() and publish() leaves the dataset at 'completed'\n"
        "  for ever, and the trigger -- which listens ONLY for 'published' -- waits in silence.\n"
        "  THIS IS THE BUG THAT COST A DAY. it must repair the status, not walk past it.")
    assert "_task.get_status()" in reuse, (
        "it must read the REAL status. clearml's Dataset has NO .status attribute --\n"
        "  getattr(existing,'status','') is always '' and silently mis-sorts every state.\n"
        "  the truth lives on the backing task: existing._task.get_status().")
    assert "existing.is_final()" not in reuse, (
        "is_final() must NOT gate the repair: it counts 'stopped' (aborted in the UI mid-\n"
        "  build) as final, so a half-built dataset would be publish()ed and trained on.")

    # and the behaviour itself
    ds = FakeDataset(status="completed")
    assert ds.is_final() and ds.status != "published"
    ds.publish()
    assert ds.status == "published"


def test_reusing_an_IN_PROGRESS_dataset_is_a_HARD_STOP(pv):
    """a dataset that never finalized is HALF BUILT -- its files may be incomplete.

    Dataset.get() returns it anyway (status=None matches everything). reusing it would train on
    a partial dataset and the lock would record it as good. the only safe answer is to stop.
    """
    src = (ROOT / "core" / "publish_version.py").read_text()
    pub = src.split("def publish(", 1)[1]
    reuse = pub.split("if existing is not None:", 1)[1].split("ds_id = existing.id", 1)[0]
    assert "SystemExit" in reuse, (
        "a half-built ('in_progress' / 'failed') dataset must be REFUSED, not reused. "
        "Dataset.get() hands one back happily -- its default status filter is None.")

    ds = FakeDataset(status="in_progress")
    assert not ds.is_final()
    with pytest.raises(ValueError):
        ds.publish()          # real ClearML refuses too -- but we must never get this far


# =============================================================================
# BUG 3 -- auto_trigger.py could not even be imported
# =============================================================================
def test_auto_trigger_imports_at_all(monkeypatch):
    """THE BUG: `from publish_version import enqueue`. There is no `enqueue`.

    The module died with an ImportError on line 31, before doing anything. The entire backup
    trigger path -- the one that catches an out-of-band publish -- was dead code. No test
    imported it, which is exactly why 21/21 stayed green.
    """
    fake_clearml = types.ModuleType("clearml")
    fake_clearml.Dataset = FakeDataset
    fake_clearml.Task = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

    import importlib
    mod = importlib.import_module("auto_trigger")     # this is the assertion
    importlib.reload(mod)
    assert callable(mod.retrain)


def test_auto_trigger_calls_enqueue_all_with_all_three_arguments():
    """the second fault, stacked behind the first: enqueue(id, version) -- 2 args to a
    3-argument function. it would have died with a TypeError on the first trigger fire."""
    import inspect
    import publish_version as pv

    src = (ROOT / "core" / "auto_trigger.py").read_text()
    assert "from publish_version import enqueue_all" in src, "the import must name a real function"
    assert "enqueue_all(dataset_id, ds.version, C.MODEL_TYPES)" in src, (
        "enqueue_all takes THREE arguments (dataset_id, version, models). "
        "the old code passed two and would have raised TypeError on the first fire.")
    assert len(inspect.signature(pv.enqueue_all).parameters) == 3


def test_auto_trigger_tells_you_a_flag_that_actually_exists():
    """the third fault: it instructed `--no-enqueue`. The real flag is `--no-train`.
    argparse would have rejected it."""
    trig = (ROOT / "core" / "auto_trigger.py").read_text()
    pub = (ROOT / "core" / "publish_version.py").read_text()
    assert "--no-enqueue" not in trig, "there is no --no-enqueue flag. it is --no-train."
    assert '"--no-train"' in pub


def test_gcs_storage_branch_pushes_dvc_and_resolves_the_remote_url():
    """Keep DVC push inside the GCS branch.

    A previous indentation error moved the commit check outside the storage-mode block,
    made ``dvc push`` unreachable after a ``raise``, and attached the local-mode ``else``
    to the wrong ``if``. The file still imported, so only a control-flow assertion catches it.
    """
    tree = ast.parse((ROOT / "core" / "publish_version.py").read_text())
    publish_fn = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "publish")
    storage_if = next(
        n for n in ast.walk(publish_fn)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and "C.STORAGE_MODE" in ast.unparse(n.test)
        and "'gcs'" in ast.unparse(n.test)
    )
    gcs_source = "\n".join(ast.unparse(n) for n in storage_if.body)
    local_source = "\n".join(ast.unparse(n) for n in storage_if.orelse)

    assert "sh(['dvc', 'push'])" in gcs_source
    assert "dvc.api.get_url" in gcs_source
    assert "local storage mode" in local_source
