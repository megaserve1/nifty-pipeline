"""
core/publish_version.py -- THE ONE COMMAND that turns a built dataset into three trained models.

it does five things, in one script, and each step checks the one before it:

    1. VALIDATE   the certificate must match the recipe and the parquet, or we stop dead
    2. DVC PUSH   the parquet bytes go to YOUR bucket
    3. PUBLISH    ClearML registers the dataset (as a POINTER to those bytes) and publishes it
    4. LOCK       write down exactly what was published, so "what did model X train on?" has
                  an answer for ever
    5. ENQUEUE    launch random_forest + xgboost + catboost -- they train IN PARALLEL, then
                  SHAP explains each one, then a champion is picked

WHY THIS SCRIPT *IS* THE HOOK
    DVC has no "after push" hook -- `dvc install` only writes git hooks. so rather than pretend
    one exists, dvc push and the ClearML publish are two steps of ONE script with exit codes.
    if the push fails, nothing is published, and nothing trains.

THE BUG THAT COST A DAY, AND HOW IT IS PREVENTED HERE
    ClearML has TWO "done" states and they are not the same:
        Dataset.finalize()  -> status 'completed'
        Dataset.publish()   -> status 'published'   <-- the trigger only watches for THIS
    stop at finalize() and nothing fires, nothing errors, and nothing trains. so we call
    finalize() AND THEN publish(). (verified in clearml 2.1.10 source; pinned by
    tests/test_trigger_contract.py so a library upgrade cannot break it silently.)

    also: add_external_files() marks the dataset "dirty", and finalize() REFUSES a dirty
    dataset. upload() clears the flag (it uploads no bytes for an external link). so the order
    is add_external_files -> upload -> finalize -> publish. get it wrong and the first publish
    crashes.

run:
    python core/publish_version.py --version v1 --dry-run       # REHEARSE IT. do this first.
    python core/publish_version.py --version v1
    python core/publish_version.py --version v1 --no-train      # publish only
    python core/publish_version.py --version v1 --models xgboost # just one model

WHY --dry-run MATTERS HERE MORE THAN USUAL
    this script writes to four places in order: git, DVC, your GCS bucket, ClearML. a failure at
    step 5 has already left commits and pushed bytes behind it -- and step 5 is exactly where
    the two real failures live ("no base task", and nobody listening on the queue). --dry-run
    checks the late things FIRST and touches nothing.
"""
import argparse
import subprocess
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import yaml             # noqa: E402
import config as C     # noqa: E402
import contract        # noqa: E402


def sh(cmd: list, check=True) -> str:
    """run a shell command and STOP if it fails. this is what makes the chain safe.

    always runs from the REPO ROOT. without cwd=, `dvc add datasets/v2/...` only worked when
    the operator happened to be standing in final_pipeline/ -- run it from anywhere else and
    git/dvc either fail or, worse, act on a DIFFERENT repo up the directory tree."""
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.ROOT))
    if check and r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stderr.strip()}")
    return r.stdout.strip()


def git_sha() -> str:
    try:
        return sh(["git", "rev-parse", "HEAD"])
    except SystemExit:
        return "unknown"


def semver(version: str) -> str:
    """our version -> ClearML's version. they are the SAME NUMBER, which is the whole point.

        v2    -> "2.0"      a major: a fresh human selection
        v2.1  -> "2.1"      a minor: a variation on v2

    ClearML compares dataset versions NUMERICALLY (PEP440), so a bare 'v3' would not sort
    against 'v10'. Our scheme is already valid PEP440 -- "2.10" correctly sorts after "2.9" --
    so there is no translation layer to get wrong.
    """
    v = version.lstrip("v")
    return v if "." in v else f"{v}.0"


def enqueue_all(dataset_id: str, version: str, models: list) -> dict:
    """clone the base tasks and queue them. the models run in PARALLEL, one per agent."""
    from clearml import Task

    ids = {}
    for mtype in models:
        base = Task.get_task(project_name=C.CLEARML_PROJECT,
                             task_name=C.base_trainer_name(mtype))
        if base is None:
            raise SystemExit(
                f"no base task '{C.base_trainer_name(mtype)}'.\n"
                f"create them once:  python trainer/register_base_trainer.py")
        run = Task.clone(source_task=base, name=f"train_{mtype} v{str(version).lstrip(chr(118))}")
        # 'Args/' is argparse's section in ClearML. the trainer reads Args/dataset_id, so the
        # writer MUST use the same section -- a wrong prefix is a SILENT no-op, and the model
        # would quietly train on its default dataset instead. that exact bug already bit us.
        #
        # THE HYPERPARAMETERS TRAVEL IN THE TASK, NOT IN A FILE. this is the fix for the silent
        # no-op the 2026-07-15 review caught: tuned params live in configs/tuned/<model>.json,
        # which hyperparams.defaults() overlays -- but a clearml AGENT runs a CLONED task from a
        # code snapshot captured at register time, so a tuned file written afterwards is NOT on
        # the agent. Passing only model_type/dataset_id left the agent to read its (stale/absent)
        # tuned file and train on BASELINE params while we announced "tuned". So we resolve the
        # effective params HERE (on the controller, where the fresh tuned file exists) and set
        # every one as Args/<name>. train.py's merge() then reads them from the task -- no file
        # dependency, works on every machine, and the exact params are visible in the ClearML UI.
        from trainer import hyperparams as H
        params = {f"Args/{k}": v for k, v in H.defaults(mtype).items()}
        params.update({
            "Args/model_type": mtype,
            "Args/dataset_id": dataset_id,
            # STRIP THE LEADING 'v'. the task NAME above is "train_{m} v{stripped}" = "train_ v3".
            # select_champion rebuilds that name as f"train_{m} v{dataset_version}" -- so if we
            # shipped "v3" here it would look for "train_ vv3" (double v), find nothing, and die
            # with a false deadlock after 10 min. same value in both places -> the names match.
            "Args/dataset_version": str(version).lstrip("v"),
        })
        run.set_parameters(params)
        Task.enqueue(run, queue_name=C.TRAIN_QUEUE)
        ids[mtype] = run.id
        print(f"      queued  train_{mtype} v{str(version).lstrip('v')}   ({run.id})")

    print(f"\n  {len(ids)} models are queued on '{C.TRAIN_QUEUE}'.")
    print(f"  THEY ONLY RUN IF A WORKER IS LISTENING:")
    print(f"      clearml-agent daemon --queue {C.TRAIN_QUEUE}")
    print(f"  one agent  -> they train ONE AFTER ANOTHER.")
    print(f"  three agents (one per machine) -> they train AT THE SAME TIME.")
    return ids


def enqueue_champion(dataset_id: str, version: str, models: list):
    """pick the best of the three, AFTER they have all finished.

    it is queued now but it WAITS -- see trainer/select_champion.py. it polls until all the
    models are done (or gives up after a timeout and says so). without that wait it would run
    the moment an agent was free, find zero finished models, and crown nothing.
    """
    from clearml import Task

    base = Task.get_task(project_name=C.CLEARML_PROJECT, task_name=C.BASE_CHAMPION_NAME)
    if base is None:
        print("      (no champion base task -- skipping)")
        return None
    run = Task.clone(source_task=base, name=f"select_champion v{str(version).lstrip(chr(118))}")
    run.set_parameters({
        "Args/dataset_id": dataset_id,
        # stripped, same as enqueue_all -- select_champion rebuilds the trainer task names as
        # f"train_{m} v{dataset_version}", so this MUST be "3" not "v3" or it looks for "vv3".
        "Args/dataset_version": str(version).lstrip("v"),
        # WHO to wait for, by name. a bare count made a subset publish (--models xgboost)
        # unable to crown: the champion polled ALL model types and the never-created ones
        # read as 'still running' for ever.
        "Args/expect_models": ",".join(models),
    })
    Task.enqueue(run, queue_name=C.TRAIN_QUEUE)
    print(f"      queued  select_champion v{str(version).lstrip('v')}   (it WAITS for: {', '.join(models)})")
    return run.id


def dry_run(version: str, models: list, do_train: bool = True) -> None:
    """the REHEARSAL. check everything, change nothing.

    WHY IT EXISTS
        the real publish writes to four places -- git, DVC, your GCS bucket, and ClearML --
        and it does them in that order. so a mistake found at step 3 has already left git
        commits and pushed bytes behind it. worse, the two failures that actually happen are
        both at the END:
            "no base task 'base_train_xgboost'"   -> found at step 5, AFTER the dataset is
                                                     published. now v2 exists in ClearML with
                                                     nothing training on it.
            no clearml-agent listening             -> never found at all. it just silently
                                                     never trains, which is the exact bug that
                                                     cost a day.
        so this rehearses the whole thing FIRST, checks the late things EARLY, and touches
        nothing.

    it prints the exact commands it would run, so you can also just read them and decide.
    """
    recipe = C.VERSIONS_DIR / f"dataset_{version}.yaml"
    ds_dir = C.DATASETS_DIR / version
    parquet = ds_dir / f"dataset_{version}.parquet"
    sv = semver(version)
    problems, warnings = [], []

    print(f"===== DRY RUN  {version}  -- nothing will be written =====\n")

    # ---- 1. the certificate (this check is read-only anyway, so it is the REAL one) ----
    print(f"[1/5] the certificate  {ds_dir/'manifest.json'}")
    # a rehearsal must report EVERYTHING that is wrong, not die on the first thing. so every
    # check below is caught and collected. the real publish still stops dead at the first
    # failure -- that is its job; this one's job is to give him the whole list in one go.
    try:
        man = contract.load(ds_dir / "manifest.json")
        contract.validate_manifest(man, recipe, parquet)
        print(f"      OK   {man['rows']:,} rows x {man['n_features']} features   "
              f"no-lookahead rule = {man['no_peek']['rule']}")
        if man.get("zero_weight_classes"):
            warnings.append(f"zero-weight classes {man['zero_weight_classes']} -- "
                            f"no model can learn them")
    except Exception as e:                                    # noqa: BLE001
        problems.append(f"certificate: {e}")
        print(f"      FAIL  {e}")

    # ---- 2. storage: is the ground even there? -------------------------------
    if C.STORAGE_MODE == "local":
        # LOCAL MODE needs no git/dvc/bucket -- the parquet uploads straight to the self-hosted
        # ClearML fileserver. the one thing it DOES need is that the server is not app.clear.ml
        # (that would ship your data to SaaS). we can only warn; the api_server is in clearml.conf.
        print("\n[2/5] storage: LOCAL mode -- data uploads to the self-hosted ClearML fileserver")
        print("      (no git/dvc/bucket needed. make sure clearml.conf points at YOUR server,")
        print("       not app.clear.ml -- else the data would leave your machines.)")
    else:
        print("\n[2/5] git + dvc + the bucket")
        if not (C.ROOT / ".git").exists():
            problems.append("no git repo -- run:  git init")
        if not (C.ROOT / ".dvc").exists():
            problems.append("no dvc repo -- run:  dvc init")
        else:
            remotes = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True,
                                     cwd=C.ROOT)
            if not remotes.stdout.strip():
                problems.append("no dvc remote -- the push has nowhere to go. run:  "
                                "dvc remote add -d gcs gs://<bucket>/final_pipeline/dvc")
            else:
                print(f"      dvc remote: {remotes.stdout.strip().splitlines()[0]}")
    if not parquet.exists():
        problems.append(f"no parquet at {parquet} -- build it first: "
                        f"python bridge/build_dataset.py --version {version}")
    elif C.STORAGE_MODE == "gcs":
        mb = parquet.stat().st_size / 1e6
        print(f"      would push {parquet.name}  ({mb:.1f} MB)")
        print("      WOULD RUN:")
        print(f"        dvc add {parquet}")
        print(f"        git add {parquet}.dvc {ds_dir/'manifest.json'} {recipe}")
        print(f'        git commit -m "dataset {version}"')
        print(f"        dvc push")
    else:
        mb = parquet.stat().st_size / 1e6
        print(f"      LOCAL mode: would upload {parquet.name} ({mb:.1f} MB) INTO the ClearML "
              f"dataset (add_files) -> the self-hosted fileserver. no dvc/git.")

    # ---- 3. ClearML: the version, and the base tasks that step 5 needs --------
    print(f"\n[3/5] ClearML  (project '{C.CLEARML_PROJECT}')")
    print(f"      would create dataset '{C.CLEARML_DATASET}' version {sv}")
    _addstep = "add_external_files" if C.STORAGE_MODE == "gcs" else "add_files"
    print(f"      order: {_addstep} -> upload -> finalize -> PUBLISH")
    print("             (stopping at finalize is the silent bug -- nothing would ever train)")
    try:
        from clearml import Dataset, Task
        existing = None
        try:
            existing = Dataset.get(dataset_project=C.CLEARML_PROJECT,
                                   dataset_name=C.CLEARML_DATASET, dataset_version=sv)
        except Exception:
            pass
        if existing is not None:
            # the REAL status lives on the backing task -- Dataset has no .status attribute
            # (getattr(existing,'status','') is always '' and silently mis-sorts every state).
            # same logic as the real publish path below; keep the two in lock-step.
            status = str(existing._task.get_status() or "").lower()
            print(f"      !! version {sv} already exists: {existing.id}  status={status!r}")
            if status in ("published", "publishing"):
                warnings.append(f"ClearML already has version {sv} ({existing.id}), published -- "
                                f"the real run would REUSE it, not create a new one")
            elif status in ("completed", "closed"):
                warnings.append(f"ClearML has version {sv} ({existing.id}) FINAL but NOT "
                                f"PUBLISHED -- the real run will publish it. left unpublished, "
                                f"the trigger waits for it for ever.")
            else:
                # in_progress / created / failed / stopped(=aborted mid-build).
                # not a warning. a STOP. a half-built dataset must never be reused or trained on.
                problems.append(f"ClearML has version {sv} ({existing.id}) at status {status!r} "
                                f"-- it was never cleanly finalized, so it is HALF BUILT. delete "
                                f"it before publishing:  "
                                f"Dataset.delete(dataset_id='{existing.id}', force=True)")

        # THE LATE FAILURE, CHECKED EARLY. without these, the real run publishes the dataset
        # and only THEN dies -- leaving a published version with nothing training on it.
        if do_train:
            for mtype in models:
                name = C.base_trainer_name(mtype)
                if Task.get_task(project_name=C.CLEARML_PROJECT, task_name=name) is None:
                    problems.append(f"no base task '{name}' -- run:  "
                                    f"python trainer/register_base_trainer.py")
                else:
                    print(f"      base task OK: {name}")
            if Task.get_task(project_name=C.CLEARML_PROJECT,
                             task_name=C.BASE_CHAMPION_NAME) is None:
                warnings.append(f"no base task '{C.BASE_CHAMPION_NAME}' -- "
                                f"the models would train but no champion would be picked")
    except Exception as e:
        problems.append(f"cannot reach ClearML: {type(e).__name__}: {e}")
        print(f"      FAIL  cannot reach ClearML -- {e}")

    # ---- 4. the lock ---------------------------------------------------------
    lock = C.VERSIONS_DIR / f"dataset_{version}.lock.yaml"
    print(f"\n[4/5] would write the lock -> {lock}")
    if lock.exists():
        warnings.append(f"{lock.name} already exists -- the real run would OVERWRITE it")

    # ---- 5. the queue, and whether anyone is listening ------------------------
    print(f"\n[5/5] would queue on '{C.TRAIN_QUEUE}': {models} + select_champion")
    if not do_train:
        print("      (--no-train: nothing would be queued)")
    else:
        try:
            from clearml.backend_api.session.client import APIClient
            client = APIClient()
            q = client.queues.get_all(name=C.TRAIN_QUEUE)
            if not q:
                problems.append(f"queue '{C.TRAIN_QUEUE}' does not exist in ClearML")
            else:
                workers = client.queues.get_all(name=C.TRAIN_QUEUE)[0].workers or []
                if not workers:
                    # THE SILENT ONE. everything succeeds, the dataset publishes, the tasks
                    # queue... and nothing ever runs, with no error anywhere.
                    problems.append(
                        f"NOBODY IS LISTENING on '{C.TRAIN_QUEUE}'. the tasks would queue and "
                        f"then sit there for ever, with no error. start a worker:\n"
                        f"           clearml-agent daemon --queue {C.TRAIN_QUEUE}")
                else:
                    print(f"      {len(workers)} worker(s) listening -- "
                          f"{'they train IN PARALLEL' if len(workers) > 1 else 'they train ONE AFTER ANOTHER'}")
                    if len(workers) == 1:
                        warnings.append(
                            "only ONE agent. the models train one after another, and "
                            "select_champion must not be picked up before them. it detects that "
                            "and exits rather than deadlock -- but two agents is better.")
        except Exception as e:
            warnings.append(f"could not check for listening agents: {e}")

    # ---- the verdict ---------------------------------------------------------
    print("\n" + "=" * 70)
    for w in warnings:
        print(f"  WARN   {w}")
    for p in problems:
        print(f"  STOP   {p}")
    if problems:
        print(f"\n  {len(problems)} thing(s) would break the real run. nothing was changed.")
        raise SystemExit(1)
    print("\n  all checks passed. nothing was changed.")
    print(f"  to do it for real:  python core/publish_version.py --version {version}")


def run_tune(models: list, dataset_id: str, version: str, parquet_sha256, re_hpo: bool = False,
             trials: int = 15) -> None:
    """for each model: search hyperparameters, then promote the winner. THE CACHE LIVES HERE.

    it calls trainer/hpo.py and trainer/apply_hpo.py as SUBPROCESSES -- the exact commands you
    would run by hand. that is deliberate: the manual path and the --tune path run identical
    code, so one can never quietly diverge from the other, and reverting to manual is just
    "don't pass --tune".

    THE CACHE: each promoted tune records the dataset's parquet_sha256. if this dataset's sha
    matches what a model was last tuned on, HPO is skipped -- the data has not changed, so the
    old winner still stands. --re-hpo forces a fresh search anyway.
    """
    from trainer import hyperparams as H

    # WARN BEFORE THE COST. each model that is not cached runs `trials` full training jobs on
    # the queue -- dozens of agent-hours across three models. say so before spending it.
    to_run = [m for m in models if re_hpo or H.tuned_sha(m) != parquet_sha256]
    if to_run:
        print(f"\n  !! --tune will run HPO for {to_run}: up to {trials} training jobs EACH, on "
              f"the '{C.TRAIN_QUEUE}' queue. that is real agent time. (cached models are skipped.)")
    print(f"\n[4b] TUNE: hpo -> promote, per model  (cache key = parquet sha "
          f"{str(parquet_sha256)[:12]}...)")
    for mtype in models:
        cached = H.tuned_sha(mtype)
        if cached and parquet_sha256 and cached == parquet_sha256 and not re_hpo:
            print(f"  {mtype:14s} CACHED -- data unchanged since last tune, skipping HPO. "
                  f"(--re-hpo to force)")
            continue
        why = "re-hpo forced" if re_hpo else ("no cached tune" if not cached else "data changed")
        print(f"  {mtype:14s} tuning ({why}) -- {trials} trials ...")

        winner = C.ROOT / f"best_params_{mtype}.json"
        r = subprocess.run([sys.executable, str(C.ROOT / "trainer" / "hpo.py"),
                            "--dataset_id", dataset_id, "--model_type", mtype,
                            "--dataset_version", version, "--trials", str(trials)],
                           cwd=str(C.ROOT))
        if r.returncode != 0 or not winner.exists():
            raise SystemExit(f"  HPO failed for {mtype} (exit {r.returncode}). "
                             f"nothing promoted; nothing trains on a half-done search.")
        # promote, recording the sha so this exact data is cached. --force so an unattended
        # --tune run does not halt on a range-edge winner (review configs/tuned/*.json after).
        pr = subprocess.run([sys.executable, str(C.ROOT / "trainer" / "apply_hpo.py"),
                             str(winner), "--force", "--sha", str(parquet_sha256 or "")],
                            cwd=str(C.ROOT))
        if pr.returncode != 0:
            raise SystemExit(f"  promoting the {mtype} winner failed (exit {pr.returncode}).")
    print("[4b] tune done -- models below train with the tuned params.\n")


def publish(version: str, models: list, do_train: bool = True,
            tune: bool = False, re_hpo: bool = False, hpo_trials: int = 15) -> str:
    from clearml import Dataset

    recipe = C.VERSIONS_DIR / f"dataset_{version}.yaml"
    ds_dir = C.DATASETS_DIR / version
    parquet = ds_dir / f"dataset_{version}.parquet"

    # ---- 1. the gate ---------------------------------------------------------
    print(f"[1/5] checking the certificate  {ds_dir/'manifest.json'}")
    man = contract.load(ds_dir / "manifest.json")
    contract.validate_manifest(man, recipe, parquet)
    print(f"      OK   {man['rows']:,} rows x {man['n_features']} features   "
          f"no-lookahead rule = {man['no_peek']['rule']}")

    # SAME VERSION, DIFFERENT BYTES = A LIE WAITING TO HAPPEN.
    # if this version was already published once (a lock exists) and the parquet has since been
    # REBUILT, the ClearML dataset still points at the OLD bytes -- the reuse branch below would
    # happily hand its id to the trainers, and the lock would swear everything matched. a
    # version number must mean ONE set of bytes, for ever. rebuilt it? bump the version.
    old_lock = C.VERSIONS_DIR / f"dataset_{version}.lock.yaml"
    if old_lock.exists():
        prev = yaml.safe_load(old_lock.read_text()) or {}
        prev_sha = prev.get("parquet_sha256")
        if prev_sha and prev_sha != man["parquet_sha256"]:
            raise SystemExit(
                f"\n  {version} was already published with DIFFERENT bytes.\n"
                f"    locked sha256 : {prev_sha[:16]}...\n"
                f"    current sha256: {man['parquet_sha256'][:16]}...\n"
                f"  the ClearML dataset for {version} points at the OLD parquet; reusing it\n"
                f"  would train on bytes that no longer match this manifest.\n"
                f"  a version number means ONE set of bytes. make a new version:\n"
                f"      python core/make_version.py --from {version}      # then rebuild + publish")
    if man.get("zero_weight_classes"):
        print(f"      !! ZERO-WEIGHT CLASSES: {man['zero_weight_classes']}")
        print(f"      !! no model can learn these. see the NO_TRADE weight problem.")

    # ---- 2. the bytes go to storage ------------------------------------------
    # gcs_url is the POINTER ClearML will register in "gcs" mode. in "local" mode there is no
    # pointer -- the bytes are uploaded straight into the ClearML dataset (step 3) -- so it
    # stays None. everything downstream (the lock, the reuse branch) already tolerates None.
    gcs_url = None
    if C.STORAGE_MODE == "gcs":
        print("[2/5] dvc add + git commit + dvc push")
        sh(["dvc", "add", str(parquet)])
        sh(["git", "add", f"{parquet}.dvc", str(ds_dir / "manifest.json"), str(recipe)])
        r = subprocess.run(["git", "commit", "-m", f"dataset {version}"],
                           capture_output=True, text=True, cwd=str(C.ROOT))
        # "nothing to commit" has SEVERAL git phrasings, and hitting ANY of them is fine -- it just
        # means these dataset files were already committed (e.g. re-publishing the same version).
        # the old guard only matched "nothing to commit" and so it ERRORED on the equally-harmless
        # "no changes added to commit", turning a no-op into a hard stop on every re-publish.
        _out = (r.stdout + r.stderr)
        _nothing = any(p in _out for p in
                       ("nothing to commit", "no changes added to commit",
                        "nothing added to commit"))
        if r.returncode != 0 and not _nothing:
            # show BOTH streams -- git's "tell me who you are" identity error goes to stdout,
            # not stderr, so printing only stderr left this blank and unhelpable.
            raise SystemExit(f"git commit failed:\n{_out.strip()}\n\n"
                             f"  if it says 'who you are', run once:\n"
                             f"    git config user.email you@example.com && "
                             f"git config user.name you")
        sh(["dvc", "push"])
        import dvc.api
        gcs_url = dvc.api.get_url(str(parquet))
        print(f"      pushed -> {gcs_url}")
    else:
        # LOCAL MODE: no DVC, no bucket. the parquet is uploaded into the ClearML dataset in
        # step 3, and the self-hosted fileserver (on this machine) holds it. data stays local.
        print(f"[2/5] local storage mode -- skipping DVC/GCS. the {parquet.stat().st_size/1e6:.0f} MB "
              f"parquet uploads to the self-hosted ClearML fileserver in step 3.")

    # ---- 3. ClearML: register as a POINTER, then PUBLISH ---------------------
    sv = semver(version)
    existing = None
    try:
        existing = Dataset.get(dataset_project=C.CLEARML_PROJECT,
                               dataset_name=C.CLEARML_DATASET, dataset_version=sv)
    except Exception:
        pass

    if existing is not None:
        # THE REUSE PATH IS WHERE THE finalize()-WITHOUT-publish() BUG CAME BACK.
        #
        # Dataset.get() is called here with its DEFAULTS, and in clearml 2.1.10 the defaults are
        # only_completed=False, only_published=False -- which the query builder turns into
        # status=None, i.e. ANY status. So this branch happily hands back a dataset that is
        # 'in_progress', 'created', 'failed' or 'completed'.
        #
        # The old code took .id and walked on. Picture the sequence:
        #   1. a run is Ctrl-C'd, or the network blips, BETWEEN ds.finalize() and ds.publish()
        #      below -- they are two separate server calls. The version now sits at 'completed'.
        #   2. you re-run publish_version.py.
        #   3. this branch prints "already exists -- reusing it", NEVER PUBLISHES, writes the
        #      lock, and enqueues training.
        #   4. auto_trigger (trigger_on_publish=True) waits for 'published' for ever. No error.
        #      The ClearML UI shows a healthy dataset. This is exactly the day we already lost.
        #
        # So: look at the status, and REPAIR it. Never assume.
        #
        # HOW THE STATUS IS READ, AND WHY IT LOOKS ODD.
        # clearml 2.1.10's Dataset object has NO `.status` attribute -- the first version of
        # this fix read `getattr(existing, "status", "")`, which is ALWAYS the empty string, so
        # the "already published" branch could never fire and an already-published version
        # would have been re-publish()ed -- which raises. The real status lives on the dataset's
        # backing task: `existing._task.get_status()` (this is exactly what Dataset.is_final()
        # calls internally; verified in the installed source).
        #
        # And is_final() itself is NOT the right repair gate: it counts 'stopped' -- a dataset
        # ABORTED from the ClearML UI mid-build -- as final. Repairing a 'stopped' dataset with
        # publish() would bless a half-built artifact and train on it. So the states are handled
        # BY NAME, and everything unrecognised stops.
        status = str(getattr(existing, "_task", None) and existing._task.get_status() or "").lower()
        print(f"[3/5] ClearML dataset v{sv} already exists ({existing.id})  status={status!r}")

        if status in ("published", "publishing"):
            print("      already published -- reusing it")
        elif status in ("completed", "closed"):
            # finalized but not published: the exact half-done state described above.
            print("      it is FINAL but NOT PUBLISHED -- publishing it now.")
            print("      (this is the finalize()-without-publish() state. left alone, the")
            print("       trigger would wait for it for ever, silently.)")
            existing.publish()
            print("      published.")
        else:
            # in_progress / created / failed / STOPPED (= aborted in the UI mid-build).
            # This version is HALF BUILT. Reusing it would train on a dataset whose bytes were
            # never fully uploaded, and the lock would record it as good. Stop.
            raise SystemExit(
                f"\n  ClearML dataset v{sv} exists but its status is {status!r}.\n"
                f"  that means it was never cleanly finalized -- it is half built, and its files\n"
                f"  may be incomplete. it must NOT be reused and it must NOT be trained on.\n\n"
                f"  delete it and publish again:\n"
                f"      from clearml import Dataset\n"
                f"      Dataset.delete(dataset_id='{existing.id}', force=True)\n"
                f"      python core/publish_version.py --version {version}")

        ds_id = existing.id
    else:
        print(f"[3/5] ClearML: create v{sv} -> upload -> finalize -> PUBLISH")
        ds = Dataset.create(dataset_project=C.CLEARML_PROJECT,
                            dataset_name=C.CLEARML_DATASET, dataset_version=sv)
        if C.STORAGE_MODE == "gcs":
            # a LINK to the bytes DVC already pushed. there is ONE copy of the data, it lives in
            # your bucket, and app.clear.ml only ever holds the pointer and the metadata.
            ds.add_external_files(source_url=gcs_url)
        else:
            # LOCAL MODE: put the actual parquet INTO the dataset. upload() then sends it to the
            # self-hosted fileserver on this machine, and the agent on PC2 pulls it from there
            # over the LAN. the bytes never leave your two PCs.
            ds.add_files(path=str(parquet))
        ds.set_metadata(man, metadata_name="manifest")
        # REQUIRED in both modes: add_files/add_external_files mark the dataset "dirty" and
        # finalize() refuses a dirty dataset. in gcs mode upload() sends no bytes (external
        # link); in local mode it uploads the parquet to the fileserver.
        ds.upload()
        ds.finalize()    # -> 'completed'.   NOT enough on its own.
        ds.publish()     # -> 'published'.   this is the state everything else waits for.
        ds_id = ds.id
        print(f"      finalized AND published: {ds_id}")

    # ---- 4. the lock ---------------------------------------------------------
    lock = C.VERSIONS_DIR / f"dataset_{version}.lock.yaml"
    contract.write_lock(lock, man, ds_id, sv, gcs_url, git_sha())
    print(f"[4/5] lock written -> {lock.name}")

    # ---- 4b. TUNE (opt-in) ---------------------------------------------------
    # DEFAULT IS OFF -- plain publish trains with whatever params already exist (baseline, or a
    # tuned file you promoted earlier). that is the MANUAL path, unchanged: run hpo.py and
    # apply_hpo.py yourself whenever you like. --tune is just the one-command shortcut, and it is
    # CACHE-GATED: a model whose data has not changed since its last tune is skipped, so re-
    # publishing the same dataset does NOT burn hours re-searching.
    if tune:
        run_tune(models, ds_id, version, man.get("parquet_sha256"), re_hpo=re_hpo,
                 trials=hpo_trials)

    # ---- 5. train ------------------------------------------------------------
    if not do_train:
        print("[5/5] --no-train: published, nothing queued")
        return ds_id

    print(f"[5/5] queueing {len(models)} models + the champion")
    enqueue_all(ds_id, version, models)
    print()
    # SHAP is NOT queued here. each trainer queues its OWN shap task when it finishes, so the
    # model provably exists by then. queueing them now would let a free agent start explaining
    # a model that is still training.
    enqueue_champion(ds_id, version, models)
    print(f"\n  each model will queue its own SHAP task when it finishes.")
    return ds_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, metavar="vN")
    ap.add_argument("--models", default=",".join(C.MODEL_TYPES),
                    help=f"which models to train. default: all of {C.MODEL_TYPES}")
    ap.add_argument("--no-train", action="store_true",
                    help="publish the dataset but do not queue any training")
    ap.add_argument("--tune", action="store_true",
                    help="search hyperparameters (HPO) for each model, promote the winners, THEN "
                         "train -- one command. cache-gated: skips models whose data is unchanged. "
                         "OFF by default; without it, training uses existing/baseline params (the "
                         "manual path is unchanged).")
    ap.add_argument("--re-hpo", action="store_true",
                    help="with --tune: force a fresh search even if the cache says the data is "
                         "unchanged.")
    ap.add_argument("--hpo-trials", type=int, default=15,
                    help="with --tune: trials per model (default 15).")
    ap.add_argument("--dry-run", action="store_true",
                    help="rehearse it: check everything, write nothing. do this first.")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    bad = [m for m in models if m not in C.MODEL_TYPES]
    if bad:
        raise SystemExit(f"unknown model(s): {bad}. choose from {C.MODEL_TYPES}")
    if a.re_hpo and not a.tune:
        raise SystemExit("--re-hpo only means something with --tune.")
    if a.tune and a.no_train:
        # --tune runs a full, paid HPO (dozens of agent-hours) whose ONLY purpose is to feed the
        # training that --no-train then cancels. that is almost certainly a mistake. to tune
        # without training, run trainer/hpo.py + trainer/apply_hpo.py by hand.
        raise SystemExit("--tune with --no-train is contradictory: --tune runs HPO to feed "
                         "training, and --no-train cancels the training. drop one.")

    if a.dry_run:
        dry_run(a.version, models, do_train=not a.no_train)
        return
    publish(a.version, models, do_train=not a.no_train,
            tune=a.tune, re_hpo=a.re_hpo, hpo_trials=a.hpo_trials)


if __name__ == "__main__":
    main()
