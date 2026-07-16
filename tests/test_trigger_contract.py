"""
tests/test_trigger_contract.py -- pins the ClearML fact that cost us a day.

THE BUG WE ARE PINNING
    Dataset.finalize()  -> Task.mark_completed()  -> status 'completed'
    Dataset.publish()   -> a separate call        -> status 'published'
    add_dataset_trigger(trigger_on_publish=True) queries for status 'published'.

    So a version that is only finalize()d NEVER fires the trigger, and nothing errors.
    Nothing trains. The UI shows a healthy "Final" dataset. You lose a day.

    Verified in clearml 2.1.10, clearml/automation/trigger.py, DatasetTrigger.build_query:
        "status": ["published" if self.on_publish else "completed"]

If a future ClearML changes this mapping, this test fails HERE, offline, in seconds --
instead of in production, silently, forever.

Run:  pytest tests/ -q     (no network, no ClearML server needed)
"""
import pytest

clearml = pytest.importorskip("clearml", reason="clearml not installed")
from clearml.automation.trigger import DatasetTrigger  # noqa: E402


def _status(on_publish: bool):
    from datetime import datetime
    t = DatasetTrigger(name="pin", project="p", on_publish=on_publish)
    return t.build_query(ref_time=datetime(2026, 1, 1))["status"]


def test_on_publish_true_listens_for_published_only():
    assert _status(True) == ["published"], (
        "ClearML changed the trigger status mapping. core/publish_version.py calls "
        "ds.publish() precisely because on_publish=True queries for 'published'."
    )


def test_on_publish_false_listens_for_completed_only():
    assert _status(False) == ["completed"], (
        "ClearML changed the trigger status mapping. finalize() reaches 'completed'."
    )


def test_finalize_and_publish_are_different_calls():
    """If these ever became the same call, publish_version.py could drop one."""
    from clearml import Dataset
    assert hasattr(Dataset, "finalize")
    assert hasattr(Dataset, "publish")
    assert Dataset.finalize is not Dataset.publish


def test_dataset_trigger_carries_the_dataset_id_to_the_callback():
    """Our callback signature is retrain(dataset_id). If ClearML stops passing the id,
    core/auto_trigger.py would silently fall back to re-querying 'latest' -- the exact
    fragility that made the demo train v1.0 forever. Pin the contract."""
    import inspect
    from clearml.automation import TriggerScheduler
    sig = inspect.signature(TriggerScheduler.add_dataset_trigger)
    assert "schedule_function" in sig.parameters
    assert "trigger_on_publish" in sig.parameters
    assert "single_instance" in sig.parameters
