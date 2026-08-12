"""tests for trainer/feature_analysis.py -- the summary workbook step.

WHY THIS TEST EXISTS
    a streak counted on a bundle_random test slice is fiction -- those rows are scattered through
    2015-2024 with training rows in between, so "consecutive" means nothing there. the OOS block is
    contiguous, so a streak in it is real. the two populations are different LENGTHS and different
    ROWS, and the only way to be sure the wiring did not quietly fall back to the test slice is to
    feed two streams that cannot be confused and check which one each sheet came out of.

it uses no model, no parquet and no ClearML -- just arrays -- so it runs in milliseconds offline.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from trainer.feature_analysis import (SHEET_BAD, SHEET_CM, SHEET_DD, SHEET_LS, SHEET_OK,  # noqa
                                      build)

CLASSES = ["ENTRY_SMALL", "ENTRY_SUB", "ENTRY_SUPER", "EXIT_SMALL", "EXIT_SUB", "EXIT_SUPER",
           "NO_TRADE"]
FEATS = ["f_a", "f_b"]


def _streams():
    """two deliberately DIFFERENT streams.

    test: 300 rows, EXIT_SUPER truth, alternating right/wrong -> longest losing streak is 1.
    oos :  40 rows, EXIT_SUPER truth, one unbroken block of 9 wrong -> longest streak is 9.

    9 can only come from the oos stream and 1 can only come from the test stream, so whichever
    number lands in the sheet names the population it was built from.
    """
    t_true = np.array([CLASSES[5]] * 300)
    t_pred = np.array([CLASSES[5] if i % 2 == 0 else CLASSES[6] for i in range(300)])
    t_times = pd.date_range("2015-01-01 09:15", periods=300, freq="1min").to_numpy()

    o_true = np.array([CLASSES[5]] * 40)
    o_pred = np.array([CLASSES[5]] * 40)
    o_pred[10:19] = CLASSES[6]                      # ONE run of exactly 9 wrong
    o_times = pd.date_range("2025-01-01 09:15", periods=40, freq="1min").to_numpy()
    return (t_true, t_pred, t_times), (o_true, o_pred, o_times)


def test_streak_sheets_come_from_oos(tmp_path):
    (t_true, t_pred, t_times), oos = _streams()
    shap = np.zeros((len(t_true), len(FEATS), len(CLASSES)))
    out = build(tmp_path / "wb.xlsx", shap, FEATS, CLASSES, t_true, t_pred, weights={},
                times=t_times, oos=oos)

    ls = pd.read_excel(out, sheet_name=SHEET_LS)
    row = ls[(ls["True Action"] == CLASSES[5]) & (ls["Predicted Action"] == CLASSES[6])].iloc[0]
    assert row["Max Losing Streak1"] == 9, "streak sheet was NOT built from the oos stream"
    # and the period is a 2025 date -- the test stream is 2015, so this pins it twice over.
    assert "2025" in str(row["Losing Streak1 Period"])

    dd = pd.read_excel(out, sheet_name=SHEET_DD)
    drow = dd[(dd["True Action"] == CLASSES[5]) & (dd["Predicted Action"] == CLASSES[6])].iloc[0]
    assert drow["Max DD1 of classifications"] == 9
    assert "2025" in str(drow["DD1 Period"])


def test_shap_and_confusion_stay_on_test(tmp_path):
    """the other three sheets must NOT move to oos -- shap needs the feature rows it ran on."""
    (t_true, t_pred, t_times), oos = _streams()
    shap = np.zeros((len(t_true), len(FEATS), len(CLASSES)))
    out = build(tmp_path / "wb.xlsx", shap, FEATS, CLASSES, t_true, t_pred, weights={},
                times=t_times, oos=oos)

    bad = pd.read_excel(out, sheet_name=SHEET_BAD)
    row = bad[(bad["True Action"] == CLASSES[5]) & (bad["Predicted Action"] == CLASSES[6])].iloc[0]
    assert row["test # of Classifications"] == 150, "shap sheet drifted off the test stream"

    cm = pd.read_excel(out, sheet_name=SHEET_CM)
    assert int(cm[cm.columns[-1]].iloc[-1]) == 300, "confusion matrix drifted off the test stream"
    assert len(pd.read_excel(out, sheet_name=SHEET_OK)) == 7


def test_all_49_rows_survive_the_split(tmp_path):
    """the oos stream carries only 2 of the 7 classes. every sheet must still be 49 rows --
    a pair that did not occur is a 0, not a missing row, or the sheet changes shape run to run."""
    (t_true, t_pred, t_times), oos = _streams()
    shap = np.zeros((len(t_true), len(FEATS), len(CLASSES)))
    out = build(tmp_path / "wb.xlsx", shap, FEATS, CLASSES, t_true, t_pred, weights={},
                times=t_times, oos=oos)
    assert len(pd.read_excel(out, sheet_name=SHEET_DD)) == 49
    assert len(pd.read_excel(out, sheet_name=SHEET_LS)) == 49
    assert len(pd.read_excel(out, sheet_name=SHEET_BAD)) == 42


def test_no_oos_falls_back_but_is_marked(capsys, tmp_path):
    """oos=None is the --allow_test_streaks path. it must still work, and it must SAY so --
    a silent fallback would put fiction in the sheet with nothing to warn the reader."""
    (t_true, t_pred, t_times), _ = _streams()
    shap = np.zeros((len(t_true), len(FEATS), len(CLASSES)))
    out = build(tmp_path / "wb.xlsx", shap, FEATS, CLASSES, t_true, t_pred, weights={},
                times=t_times, oos=None)
    assert "not time-contiguous" in capsys.readouterr().out
    ls = pd.read_excel(out, sheet_name=SHEET_LS)
    row = ls[(ls["True Action"] == CLASSES[5]) & (ls["Predicted Action"] == CLASSES[6])].iloc[0]
    assert row["Max Losing Streak1"] == 1        # the test stream's alternating pattern


def test_patch_rewrites_only_the_two_streak_sheets(tmp_path):
    """patch mode must leave the shap sheets byte-for-byte and fix only the streaks.

    this is what rescues a workbook that already cost a night of shap.
    """
    from trainer.feature_analysis import patch_streaks
    (t_true, t_pred, t_times), oos = _streams()
    shap = np.zeros((len(t_true), len(FEATS), len(CLASSES)))
    book = build(tmp_path / "wb.xlsx", shap, FEATS, CLASSES, t_true, t_pred, weights={},
                 times=t_times, oos=None)                      # built the OLD way -> fiction
    before = pd.read_excel(book, sheet_name=SHEET_BAD)
    assert pd.read_excel(book, sheet_name=SHEET_LS)["Max Losing Streak1"].max() == 1

    patch_streaks(book, {"label_encoder": None}, CLASSES, oos)

    assert pd.read_excel(book, sheet_name=SHEET_LS)["Max Losing Streak1"].max() == 9
    pd.testing.assert_frame_equal(before, pd.read_excel(book, sheet_name=SHEET_BAD))
    assert pd.ExcelFile(book).sheet_names == [SHEET_BAD, SHEET_OK, SHEET_DD, SHEET_LS, SHEET_CM]


def test_the_step_is_wired_into_every_run():
    """the four names that make this run automatically. a rename in any one of them breaks the
    chain SILENTLY -- the trainer just prints 'no base task registered' and moves on, and nobody
    notices the workbook stopped being produced until someone goes looking for it."""
    import config as C
    root = pathlib.Path(__file__).resolve().parent.parent
    for n in ("BASE_FEATURE_ANALYSIS_NAME", "FEATURE_ANALYSIS_QUEUE",
              "FEATURE_ANALYSIS_MAX_ROWS", "FEATURE_ANALYSIS_ARTIFACT"):
        assert hasattr(C, n), f"config lost {n}"
    assert C.FEATURE_ANALYSIS_ARTIFACT == "summary_file"
    assert C.feature_analysis_output_uri()

    train = (root / "trainer" / "train.py").read_text()
    assert "def queue_feature_analysis_for_me" in train
    # CALLED, not just defined. the bug this catches is a function nobody invokes.
    assert train.count("queue_feature_analysis_for_me(") >= 2
    assert "BASE_FEATURE_ANALYSIS_NAME" in (root / "trainer" / "register_base_trainer.py").read_text()


def test_artifact_upload_waits():
    """upload_artifact must wait. a file still going up when the task closes is silently lost,
    and the run looks perfectly successful with no artifact on it."""
    fa = (pathlib.Path(__file__).resolve().parent.parent
          / "trainer" / "feature_analysis.py").read_text()
    assert "wait_on_upload=True" in fa
    # openpyxl is reached through pandas, never imported here, so clearml's import scan cannot
    # see it. it must be DECLARED or the agent builds a venv without it.
    assert 'Task.add_requirements("openpyxl")' in fa


# ---------------------------------------------------------------------------------------
# the waiter. it must stop waiting when the siblings are done, and it must not hang forever
# when one of them never runs.
# ---------------------------------------------------------------------------------------
class _Enum:
    """stands in for clearml's TaskStatusEnum. str() of it is 'TaskStatusEnum.completed',
    NOT 'completed' -- the exact trap that would make a naive waiter wait for a finished task."""
    def __init__(self, v): self.value = v
    def __str__(self): return f"TaskStatusEnum.{self.value}"


def test_status_reads_through_the_enum():
    from trainer.feature_analysis import _status_of
    assert _status_of({"status": _Enum("completed")}) == "completed"
    assert _status_of({"status": "in_progress"}) == "in_progress"
    assert _status_of({"status": _Enum("failed")}) == "failed"


def test_waiter_stops_when_siblings_finish(monkeypatch, capsys):
    """two siblings in progress, then done. it must return -- and only after they finish."""
    import trainer.feature_analysis as fa
    calls = {"n": 0}
    def rows(*_, **__):
        calls["n"] += 1
        st = "in_progress" if calls["n"] < 3 else "completed"
        return [{"name": f"shap_x", "status": _Enum(st),
                 "hyperparams.Args.model_task_id": {"value": "MID"}},
                {"name": "scored_oos_x", "status": _Enum(st),
                 "hyperparams.Args.model_task_id": {"value": "MID"}}]
    monkeypatch.setattr(fa.time, "sleep", lambda *_: None)
    monkeypatch.setitem(sys.modules, "clearml", type("m", (), {"Task": type("T", (), {
        "query_tasks": staticmethod(rows)})})())
    fa.wait_for_siblings("MID", "proj", timeout_min=60, poll_sec=0)
    assert calls["n"] == 3
    assert "all 2 sibling step(s) finished" in capsys.readouterr().out


def test_waiter_ignores_other_models_siblings(monkeypatch, capsys):
    """matched on Args/model_task_id, never on the name -- two runs of a version share a name."""
    import trainer.feature_analysis as fa
    def rows(*_, **__):
        return [{"name": "shap_other", "status": _Enum("in_progress"),
                 "hyperparams.Args.model_task_id": {"value": "SOMEONE_ELSE"}}]
    monkeypatch.setattr(fa.time, "sleep", lambda *_: None)
    monkeypatch.setitem(sys.modules, "clearml", type("m", (), {"Task": type("T", (), {
        "query_tasks": staticmethod(rows)})})())
    fa.wait_for_siblings("MID", "proj", timeout_min=60, poll_sec=0)
    assert "no sibling steps found" in capsys.readouterr().out


def test_waiter_gives_up_into_doing_the_work(monkeypatch, capsys):
    """a sibling that never finishes must not block the workbook forever. the timeout expires
    into RUNNING, not into an error -- a late workbook beats no workbook."""
    import trainer.feature_analysis as fa
    clock = {"t": 0.0}
    monkeypatch.setattr(fa.time, "time", lambda: clock["t"])
    def slp(*_): clock["t"] += 3600
    monkeypatch.setattr(fa.time, "sleep", slp)
    monkeypatch.setitem(sys.modules, "clearml", type("m", (), {"Task": type("T", (), {
        "query_tasks": staticmethod(lambda *_, **__: [
            {"name": "deepchecks_stuck", "status": _Enum("in_progress"),
             "hyperparams.Args.model_task_id": {"value": "MID"}}])})})())
    fa.wait_for_siblings("MID", "proj", timeout_min=60, poll_sec=0)   # returns, does not raise
    assert "going ahead anyway" in capsys.readouterr().out


def test_shap_runs_on_all_test_rows_by_default():
    import config as C
    assert C.FEATURE_ANALYSIS_MAX_ROWS == 0, "0 means every test row -- sampling was rejected"
    train = (pathlib.Path(__file__).resolve().parent.parent / "trainer" / "train.py").read_text()
    assert '"Args/wait_for_siblings": "1"' in train
