"""
tests/test_publish_dry_run.py -- the rehearsal must change NOTHING, and must catch the two
failures that only show up at the very end of a real publish.

WHY
    publish_version writes to four places, in this order: git -> DVC -> your GCS bucket ->
    ClearML -> the queue. so a problem discovered at step 5 has already left git commits and
    pushed bytes behind it. and step 5 is exactly where the two real failures live:

        "no base task 'base_train_xgboost'"  -> the dataset is ALREADY published by then.
                                                you now have a published v2 with nothing
                                                training on it.
        nobody listening on the queue        -> no error at all, ever. the tasks just sit
                                                there. this is the same shape as the
                                                finalize/publish bug that cost a day.

    --dry-run checks those FIRST and touches nothing. these tests prove both halves of that:
    it CATCHES them, and it WRITES nothing.
"""
import sys
import pathlib
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C                       # noqa: E402
import core.publish_version as pv        # noqa: E402


# ---------------------------------------------------------------- a fake ClearML
class FakeTask:
    """stands in for clearml.Task. `known` is the set of base tasks that exist."""
    known: set = set()

    @classmethod
    def get_task(cls, project_name=None, task_name=None, task_id=None):
        return object() if task_name in cls.known else None


class FakeDataset:
    exists = False

    @classmethod
    def get(cls, **kw):
        if cls.exists:
            # the real Dataset carries its status on the BACKING TASK (there is no .status
            # attribute on Dataset itself -- that phantom attribute was a real bug), so the
            # fake must model _task.get_status() or dry_run cannot read the state.
            return types.SimpleNamespace(
                id="abc123",
                _task=types.SimpleNamespace(get_status=lambda: "published"))
        raise ValueError("not found")


class FakeQueue:
    """a queue as the REAL server returns it: it has an id and a name, and NOTHING ELSE.

    THIS FAKE USED TO CARRY A `.workers` LIST, AND THAT IS WHY A BROKEN CHECK SHIPPED.
    the old production code read `client.queues.get_all(name=...)[0].workers`. against this fake
    that worked. against the real clearml 2.1.10 server the Queue entity has no `workers` field at
    all (the server only fills it when explicitly requested), so the real call raised
    AttributeError EVERY time -- and publish_version caught it and downgraded it to
    "WARN could not check for listening agents". the single most important pre-flight check in the
    pipeline never once ran, and the test suite was green the whole time.

    a fake that is kinder than the real thing does not test anything. so this one is now as bare
    as the server's: ask the WORKERS who they are listening to (FakeWorkers below).
    """
    def __init__(self, name):
        self.id, self.name = f"q-{name}", name


class FakeWorkerQueue:
    """the queue entry hanging off a worker -- the code reads `.name` off these."""
    def __init__(self, name):
        self.name = name


class FakeWorker:
    """a registered clearml-agent. `task` is None when idle, and an object with `.name` when it
    is busy -- busy agents cannot pick our training tasks up, which is the whole point of
    counting free ones separately."""
    def __init__(self, spec, queue_name):
        # accept "pc1:1" (idle) or ("pc1:1", "shap_random_forest v4") (busy)
        wid, busy = (spec, None) if isinstance(spec, str) else spec
        self.id = wid
        self.queues = [FakeWorkerQueue(queue_name)]
        self.task = types.SimpleNamespace(name=busy) if busy else None


def fake_clearml(base_tasks, workers, ds_exists=False):
    """build a fake `clearml` package good enough for dry_run to import from."""
    FakeTask.known = set(base_tasks)
    FakeDataset.exists = ds_exists

    m = types.ModuleType("clearml")
    m.Task, m.Dataset = FakeTask, FakeDataset

    client_mod = types.ModuleType("clearml.backend_api.session.client")

    class APIClient:
        class queues:
            @staticmethod
            def get_all(name=None):
                return [FakeQueue(name or C.TRAIN_QUEUE)]

        class workers:
            @staticmethod
            def get_all():
                # every agent registered anywhere. the production code filters these down to the
                # ones polling TRAIN_QUEUE itself -- so hand back one on a DIFFERENT queue too,
                # to prove the filter is real and not just len(get_all()).
                return ([FakeWorker(w, C.TRAIN_QUEUE) for w in workers]
                        + [FakeWorker("someone-elses-box:0", "services")])

        def __init__(self):
            pass
    client_mod.APIClient = APIClient

    backend = types.ModuleType("clearml.backend_api")
    session = types.ModuleType("clearml.backend_api.session")
    return {
        "clearml": m,
        "clearml.backend_api": backend,
        "clearml.backend_api.session": session,
        "clearml.backend_api.session.client": client_mod,
    }


@pytest.fixture
def stub_clearml(monkeypatch):
    def _install(base_tasks=None, workers=None, ds_exists=False):
        if base_tasks is None:
            base_tasks = [C.base_trainer_name(m) for m in C.MODEL_TYPES] + [C.BASE_CHAMPION_NAME]
        if workers is None:
            workers = ["worker-1"]
        for name, mod in fake_clearml(base_tasks, workers, ds_exists).items():
            monkeypatch.setitem(sys.modules, name, mod)
    return _install


@pytest.fixture
def spy_shell(monkeypatch):
    """record every shell command dry_run tries to run, and let none of them do anything."""
    seen = []

    class R:
        returncode = 0
        stdout = "gcs\tgs://bucket/dvc\n"
        stderr = ""

    def fake_run(cmd, *a, **kw):
        seen.append(list(cmd))
        return R()

    monkeypatch.setattr(pv.subprocess, "run", fake_run)
    return seen


# ================================================================== it must change NOTHING
MUTATES = ("add", "commit", "push", "init", "checkout", "rm")


def test_the_rehearsal_never_runs_a_command_that_changes_anything(stub_clearml, spy_shell):
    """THE WHOLE POINT. not one mutating git/dvc command may be executed."""
    stub_clearml()
    try:
        pv.dry_run("v1", list(C.MODEL_TYPES))
    except SystemExit:
        pass                                  # a missing dataset is fine -- we only care what it RAN

    for cmd in spy_shell:
        # BASENAME, not the literal string. dvc is invoked by its ABSOLUTE path
        # (final_venv/bin/dvc) because the bare name "dvc" only resolves when the venv happens to
        # be activated -- unactivated it died at step 2/5 with FileNotFoundError, after step 1 had
        # already printed "certificate OK". what this test actually cares about is that nothing
        # but git and dvc is ever executed, and that is still exactly what it checks.
        assert pathlib.Path(cmd[0]).name in ("dvc", "git"), f"unexpected command {cmd}"
        assert not any(w in cmd for w in MUTATES), \
            f"THE REHEARSAL WROTE SOMETHING: {' '.join(cmd)}"
    # the only thing it is allowed to run is a read
    assert all(cmd[1:3] in (["remote", "list"], ["rev-parse", "HEAD"]) for cmd in spy_shell), \
        f"only read-only commands allowed, got {spy_shell}"


def test_the_rehearsal_writes_no_lock_file(stub_clearml, spy_shell, tmp_path, monkeypatch):
    """it says it WOULD write the lock. it must not actually write it."""
    monkeypatch.setattr(C, "VERSIONS_DIR", tmp_path)
    stub_clearml()
    try:
        pv.dry_run("v9", list(C.MODEL_TYPES))
    except SystemExit:
        pass
    assert list(tmp_path.iterdir()) == [], "the rehearsal created a file"


# ================================================================== it must CATCH the late failures
def test_it_catches_the_missing_base_task_BEFORE_anything_is_published(
        stub_clearml, spy_shell, capsys):
    """the real run finds this at step 5 -- AFTER the dataset is published. by then you have a
    published version in ClearML that nothing will ever train on."""
    stub_clearml(base_tasks=[C.base_trainer_name("random_forest")])   # xgboost + catboost missing
    with pytest.raises(SystemExit):
        pv.dry_run("v1", list(C.MODEL_TYPES))

    out = capsys.readouterr().out
    assert "no base task" in out
    assert "register_base_trainer.py" in out, "it must say how to fix it"


def test_it_catches_that_NOBODY_IS_LISTENING_on_the_queue(stub_clearml, spy_shell, capsys):
    """THE SILENT ONE. every step succeeds, the dataset publishes, three tasks queue --
    and then nothing happens, for ever, with no error anywhere. exactly the shape of the
    finalize/publish bug. the only way to see it is to look for a worker."""
    stub_clearml(workers=[])
    with pytest.raises(SystemExit):
        pv.dry_run("v1", list(C.MODEL_TYPES))

    out = capsys.readouterr().out
    assert "NOBODY IS LISTENING" in out
    assert f"clearml-agent daemon --queue {C.TRAIN_QUEUE}" in out


def test_no_train_means_the_queue_is_not_even_checked(stub_clearml, spy_shell, capsys):
    """--no-train --dry-run: no worker needed, so a missing worker must not be an error."""
    stub_clearml(workers=[], base_tasks=[])
    try:
        pv.dry_run("v1", list(C.MODEL_TYPES), do_train=False)
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "NOBODY IS LISTENING" not in out
    assert "no base task" not in out, "no training means no base task is needed"


def test_it_warns_when_the_clearml_version_already_exists(stub_clearml, spy_shell, capsys):
    """republishing v2 REUSES the existing ClearML dataset. that is a surprise worth printing."""
    stub_clearml(ds_exists=True)
    try:
        pv.dry_run("v2", list(C.MODEL_TYPES))
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "already exists" in out


def test_one_agent_is_a_warning_not_an_error(stub_clearml, spy_shell, capsys):
    """one agent works -- the models just train one after another. it is worth saying, because
    it is also the setup where select_champion could get picked up first."""
    stub_clearml(workers=["only-one"])
    try:
        pv.dry_run("v1", list(C.MODEL_TYPES))
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "only ONE agent" in out
    assert "NOBODY IS LISTENING" not in out, "one agent is a warning, not a stop"


def test_the_agent_check_asks_the_WORKERS_not_the_queue(stub_clearml, spy_shell, capsys):
    """THE REGRESSION PIN. the old code read `queues.get_all(...)[0].workers` -- a field the real
    server does not return -- so it raised AttributeError on every real run and got swallowed into
    'could not check for listening agents'. the check looked fine in the test suite and had never
    actually run in production. if anyone reintroduces it, this fails."""
    stub_clearml(workers=["pc1:1", "pc2:0"])
    try:
        pv.dry_run("v1", list(C.MODEL_TYPES))
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "could not check for listening agents" not in out, (
        "the agent check silently failed -- it is reading a field the server does not send")
    assert "2 agent(s)" in out, "it must count the agents polling OUR queue"
    assert "someone-elses-box" not in out, "an agent on another queue must not be counted"


def test_a_BUSY_agent_is_not_a_free_one(stub_clearml, spy_shell, capsys):
    """an agent already running SHAP cannot pick a trainer up. counting it as available is how
    you conclude '4 agents, all 3 models train at once' and then watch two of them sit queued."""
    stub_clearml(workers=[("pc1:1", "shap_random_forest v4"),
                          ("pc2:0", "shap_xgboost v4"),
                          "pc3:0"])
    try:
        pv.dry_run("v1", list(C.MODEL_TYPES))
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "3 agent(s)" in out and "1 free" in out and "2 busy" in out
    assert "BUSY  shap_random_forest v4" in out, "it must name what is occupying the agent"
    assert "will run in batches" in out, (
        "1 free agent and 3 models to train must be called out, not glossed over")
