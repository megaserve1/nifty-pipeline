"""
trainer/hpo.py -- search for the best settings for ONE model, on ONE dataset version.

    python trainer/hpo.py --model_type random_forest --dataset_id <id> --dataset_version 3.0


WHAT IT IS FOR (the bug it was written to kill)
--------------------------------------------------------------------------------------
train.py has ONE --max_depth argument and hands it to all three models. its default is 6.
6 is a BOOSTING number. it is the wrong number for a random forest, and not by a little:

    a boosted model builds trees one after another, each one correcting the last. every tree
    only has to be slightly better than nothing, so a shallow tree is exactly right -- and a
    deep one over-fits, because the next tree will amplify whatever it got wrong.

    a random forest builds its trees INDEPENDENTLY and averages them. averaging cancels
    VARIANCE. it does nothing whatsoever to BIAS. so a forest wants trees that are individually
    low-bias -- deep, fully grown, over-fitted if you like -- and it relies on the averaging to
    clean up the variance afterwards. cap them at depth 6 and every tree is biased in the SAME
    direction, the averaging cancels nothing, and you have 300 copies of one stump.

    measured on the installed sklearn 1.9: max_depth=None gives real depths of 10-15 on 400 toy
    rows; on 330k rows with min_samples_leaf=50 it will settle around 25-30. max_depth=6 gives
    exactly 6, always. that is the crippling.

    and the class it cripples hardest is ENTRY_SUB -- 1.2% of the rows. a depth-6 tree has at
    most 64 leaves; a class holding 1.2% of the mass will essentially never be the majority of
    any of them. the forest goes blind to it.

sklearn's own default is max_depth=None. so the fix is to STOP OVERRIDING IT (see the note in
the return-value section of the research answer: --max_depth default 6 -> 0, and build_model
already reads `params["max_depth"] or None` for the forest, so 0 means "fully grown").
this script then searches around that, per model, with the ranges each library actually wants.


HOW CLEARML HPO WORKS -- AND THE TWO WAYS IT FAILS SILENTLY
--------------------------------------------------------------------------------------
it is a CLONE-AND-READ loop, not a callback loop:

    1. take the base task  (train_<model> (base), registered by register_base_trainer.py)
    2. clone it, overwrite some Args/, enqueue it
    3. an agent picks it up and runs train.py exactly as normal
    4. when the task finishes, the optimiser READS ONE SCALAR off it
    5. pick the next set of settings, go to 2

step 2 and step 4 are both string-matching, and BOTH fail by returning nothing rather than by
raising.

    FAILURE 1 -- the parameter name.
        ClearmlJob applies the override with
            task_params = base_task.get_parameters(backwards_compatibility=False)
            task_params.update(parameter_override)
        and get_parameters(backwards_compatibility=False) returns keys like "Args/max_depth" --
        SECTION SLASH NAME. our trainer uses argparse, and clearml files argparse arguments under
        the section "Args". so the parameter MUST be named "Args/max_depth".

        name it "max_depth" and .update() simply ADDS A NEW KEY called "max_depth". the real
        "Args/max_depth" keeps its default. the trial runs. it trains. it reports a score. the
        score is the DEFAULT model's score, every time, for every trial. the optimiser then
        reports that depth 6 is as good as depth 30 -- because it never changed anything.

        SearchStrategy._validate_base_task does notice, and calls logger.warning(). that is all.
        so preflight() below turns that warning into a hard stop.

    FAILURE 2 -- the objective name. see trainer/objective.py. same shape, same silence.

both of these are the "wrong section prefix" family of bug that already cost a day on this
project. they are checked, out loud, before a single agent-hour is spent.


WHICH OPTIMISER
--------------------------------------------------------------------------------------
clearml 2.1.10 ships four. TESTED on this venv:

    OptimizerOptuna   needs the `optuna` package  -- PINNED in requirements.txt (2026-07-20)
    GridSearch        imports OK   -- pure clearml, no extra package
    RandomSearch      imports OK   -- pure clearml, no extra package
    OptimizerBOHB     ImportError  -- "requires 'hpbandster' package, it was not found"

OPTUNA IS NOW THE DEFAULT (was RandomSearch until 2026-07-20).

why it beats random search on the SAME budget: random search samples blindly -- trial 30 knows
nothing about trials 1-29. optuna's TPE builds a model of which regions produced good scores and
spends the remaining trials THERE. with a small budget (15-30 trials) and 4-9 knobs, that focus is
worth real points; with an unlimited budget the two converge.

random search is still the honest fallback, and it is NOT a consolation prize: with 4-6 knobs it
beats grid search for a fixed budget, and it is not close -- grid spends its budget re-testing the
knobs that do not matter. 30 random trials over 5 knobs explore 30 distinct values of EVERY knob;
a 3x3x3x3x3 grid is 243 runs and explores exactly 3 values of each.

    --strategy optuna   (default)  TPE, learns from finished trials
    --strategy random              blind sampling, no extra package, always available
    --strategy grid                the small deliberate sweep you want to explain line by line in
                                   a meeting -- e.g. max_depth {0,16,24,32} x leaf {5,25,100}

NOTE: optuna is only needed WHERE hpo.py RUNS (the controller). the trials themselves are ordinary
training tasks -- the agents run train.py and never import optuna.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C                                            # noqa: E402
from trainer import hyperparams                                    # noqa: E402
from trainer.objective import (OBJECTIVE_TITLE, OBJECTIVE_SERIES,    # noqa: E402
                               OBJECTIVE_SIGN)


# ============================================================ the search spaces
# EVERY NAME HERE STARTS WITH "Args/". that is not decoration -- see FAILURE 1 above.
# the trainer's argparse defines "--max_depth"; clearml files it as "Args/max_depth".
def search_space(model_type: str):
    """the search space, read from configs/hyperparams.yaml.

    IT USED TO BE HARDCODED HERE -- a second copy of numbers that also lived in train.py's
    argparse defaults. two copies of the same idea, in two files, free to disagree. and they did:
    this function searched over `Args/max_features`, `Args/subsample` and four others that
    train.py's parser had never heard of, so hpo.py could not start at all. clearml would only
    WARN about an unknown parameter, so the trial would have trained the DEFAULT model, reported
    the DEFAULT score, and the search would have concluded that nothing you changed made any
    difference. because nothing you changed WAS changed.

    now both sides read the same YAML. they cannot drift apart, because there is one copy.
    to change what gets searched, edit configs/hyperparams.yaml. never this file.
    """
    return hyperparams.search_space(model_type)


# ============================================================ the preflight
def preflight(base_task, space, dataset_id: str) -> None:
    """refuse to start if the optimiser would be shouting into a void.

    clearml checks these two things itself and then calls logger.warning(). a warning inside an
    agent's log is a warning nobody reads, and the run still burns three hours of agent time and
    hands back a confident wrong answer. so we check the same things and STOP.
    """
    have = base_task.get_parameters(backwards_compatibility=False)   # keys are "Section/name"

    missing = [p.name for p in space if p.name not in have]
    if missing:
        raise SystemExit(
            f"\n  the base task '{base_task.name}' has no such parameters:\n"
            f"      {missing}\n\n"
            f"  it has:  {sorted(have)}\n\n"
            f"  clearml would only WARN about this and carry on. every trial would then train\n"
            f"  the DEFAULT model, report the default score, and the search would tell you that\n"
            f"  nothing you changed made any difference -- because nothing you changed WAS\n"
            f"  changed.\n\n"
            f"  two possible causes:\n"
            f"    1. train.py does not have those argparse arguments yet. add them.\n"
            f"    2. you added them but did not re-register:\n"
            f"         python trainer/register_base_trainer.py --force")

    # NO VAL SET -> NOTHING HONEST TO TUNE ON.
    # with VAL_FRACTION = 0 the trainer never reports val/trading_cost, so every trial would
    # score None -- clearml's Objective.get_objective() swallows the miss and returns None, the
    # search keeps going, and four hours later it hands back the parameters it happened to
    # sample first. green all the way, no traceback. refuse instead.
    if getattr(C, "VAL_FRACTION", 0) <= 0:
        raise SystemExit(
            "\n  config.VAL_FRACTION is 0, so there is no validation set.\n"
            "  the trainer reports no 'val/trading_cost', so every trial would score None and\n"
            "  the search would be theatre.\n\n"
            "  set VAL_FRACTION = 0.15 in config.py, re-register the base tasks, then run HPO:\n"
            "      python trainer/register_base_trainer.py --force")

    if "Args/dataset_id" not in have:
        raise SystemExit("the base task has no Args/dataset_id. re-register the base tasks.")

    if not dataset_id:
        raise SystemExit("--dataset_id is required. the optimiser must pin every trial to the "
                         "SAME rows, or it is comparing settings AND data at the same time and "
                         "the winner means nothing.")

    print(f"  preflight OK: all {len(space)} parameters exist on the base task, by their "
          f"exact 'Args/' names")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="hyperparameter search for one model, one dataset")
    ap.add_argument("--model_type", required=True, choices=C.MODEL_TYPES)
    ap.add_argument("--dataset_id", required=True,
                    help="the EXACT ClearML dataset id every trial trains on")
    ap.add_argument("--dataset_version", default="")
    ap.add_argument("--trials", type=int, default=30,
                    help="total number of models to train. 30 is a sane first pass")
    ap.add_argument("--concurrent", type=int, default=1,
                    help="how many trials may run at once. SET THIS TO YOUR NUMBER OF AGENTS "
                         "-- no higher. more just queues them")
    ap.add_argument("--queue", default=C.TRAIN_QUEUE)
    ap.add_argument("--strategy", default="optuna", choices=["optuna", "random", "grid"],
                    help="optuna = TPE, learns from finished trials (default). random = blind "
                         "sampling, no extra package. grid = exhaustive, explainable, expensive.")
    ap.add_argument("--job_minutes", type=float, default=90,
                    help="kill any single trial that runs longer than this. a runaway forest "
                         "with max_depth=None and min_samples_leaf=2 can sit there for hours")
    ap.add_argument("--total_minutes", type=float, default=None,
                    help="give up on the whole search after this long")
    ap.add_argument("--keep_top", type=int, default=0,
                    help="archive every trial outside the top N. 0 = keep them all (default) -- "
                         "the losers are evidence, and 30 tasks is not clutter")
    ap.add_argument("--remote", action="store_true",
                    help="run the CONTROLLER on an agent instead of on this laptop. only do "
                         "this if you have a spare agent -- see the note below")
    a = ap.parse_args()

    from clearml import Task
    from clearml.automation import (HyperParameterOptimizer, RandomSearch, GridSearch,
                                    DiscreteParameterRange)

    if a.strategy == "optuna":
        # OptimizerOptuna lives in a SUBMODULE that imports optuna at import time. guarded so a
        # missing package is a one-line instruction, not an ImportError traceback mid-search.
        try:
            from clearml.automation.optuna import OptimizerOptuna
        except ImportError as e:
            raise SystemExit(
                f"--strategy optuna needs the 'optuna' package on THIS machine.\n"
                f"    final_venv/bin/pip install -r requirements.txt   (optuna is pinned there)\n"
                f"  or fall back with:  --strategy random\n"
                f"  ({e})")
        strategy = OptimizerOptuna
    else:
        strategy = {"random": RandomSearch, "grid": GridSearch}[a.strategy]

    # ---- 1. find the base task ----------------------------------------------
    base_name = C.base_trainer_name(a.model_type)
    base = Task.get_task(project_name=C.CLEARML_PROJECT, task_name=base_name)
    if base is None:
        raise SystemExit(f"no base task '{base_name}'. run: python trainer/register_base_trainer.py")
    print(f"[1/4] base task  {base_name}  ({base.id})")

    space = search_space(a.model_type)
    preflight(base, space, a.dataset_id)

    # PIN THE DATA. the base task's Args/dataset_id is "" -- register_base_trainer.py runs
    # train.py with no dataset, and train.py exits early on purpose. clone that as-is and every
    # trial would exit early too, report NOTHING, and score None. thirty green tasks, no models.
    #
    # a one-value DiscreteParameterRange is how you pin a constant through the HPO: it goes into
    # the same parameter_override dict as everything else, so every trial gets it.
    space = space + [
        DiscreteParameterRange("Args/dataset_id", values=[a.dataset_id]),
        DiscreteParameterRange("Args/dataset_version", values=[a.dataset_version or ""]),
        # every trial uses the same seed. we are measuring the SETTINGS, not the luck of the
        # random state -- if the seed floated, a lucky seed would win and we would ship it.
        DiscreteParameterRange("Args/seed", values=[42]),
    ]

    # ---- 2. the controller ---------------------------------------------------
    task = Task.init(project_name=C.CLEARML_PROJECT,
                     task_name=f"hpo_{a.model_type} v{a.dataset_version or '?'}",
                     task_type=Task.TaskTypes.optimizer,
                     output_uri=C.model_output_uri(),
                     reuse_last_task_id=False)
    task.connect({"model_type": a.model_type, "dataset_id": a.dataset_id,
                  "dataset_version": a.dataset_version, "trials": a.trials,
                  "strategy": a.strategy, "objective": f"{OBJECTIVE_TITLE}/{OBJECTIVE_SERIES}",
                  "sign": OBJECTIVE_SIGN})

    print(f"[2/4] objective  {OBJECTIVE_TITLE}/{OBJECTIVE_SERIES}  ({OBJECTIVE_SIGN}imise)")
    print(f"      this is the VALIDATION trading_cost. the test set is not touched by the")
    print(f"      search -- if it were, the test score would be a number we tuned INTO, and we")
    print(f"      would have nothing honest left to report.")

    opt = HyperParameterOptimizer(
        base_task_id=base.id,
        hyper_parameters=space,
        objective_metric_title=OBJECTIVE_TITLE,      # "Summary"
        objective_metric_series=OBJECTIVE_SERIES,    # "val/trading_cost"
        objective_metric_sign=OBJECTIVE_SIGN,        # "min"
        optimizer_class=strategy,
        execution_queue=a.queue,
        max_number_of_concurrent_tasks=a.concurrent,
        optimization_time_limit=a.total_minutes,
        spawn_project=None,
        save_top_k_tasks_only=a.keep_top or None,
        # ---- everything below is **optimizer_kwargs. it is forwarded, unchecked, to the
        # SearchStrategy constructor, whose signature ENDS IN **_: Any -- so a typo here is
        # swallowed in total silence. these five names are copied from the real signature:
        #   pool_period_min, time_limit_per_job, compute_time_limit,
        #   min_iteration_per_job, max_iteration_per_job, total_max_jobs
        total_max_jobs=a.trials,
        time_limit_per_job=a.job_minutes,
        pool_period_min=1.0,                         # how often to ask clearml "are they done?"
        # max_iteration_per_job MUST BE PASSED, AND MUST BE None. those are two separate facts.
        #
        # WHY None: our objective is a SINGLE value reported once at the end of the run, and
        # report_single_value stamps it with iteration = -(2**31). any iteration-based budget
        # would be reasoning about a nonsense iteration number, so there must not be one. jobs
        # are bounded by time_limit_per_job instead. (verified in clearml 2.1.10
        # automation/optuna/optuna.py: both uses are guarded --
        #     if self.max_iteration_per_job and iteration >= self.max_iteration_per_job
        #     if not self.min_iteration_per_job or iteration >= self.min_iteration_per_job
        # -- so None cleanly means "no budget". it does not mean zero.)
        #
        # WHY IT MUST BE PASSED AT ALL: it was previously omitted, which was correct for
        # RandomSearch -- its __init__ requires only 5 arguments. OptimizerOptuna's signature
        # requires SEVEN; max_iteration_per_job and total_max_jobs are positional-required with
        # no default. so the moment optuna became the default strategy (2026-07-20) every run
        # died before launching a single trial:
        #     TypeError: OptimizerOptuna.__init__() missing 1 required positional argument:
        #                'max_iteration_per_job'
        # RandomSearch and GridSearch both accept the name too, so passing it always is safe.
        max_iteration_per_job=None,
    )

    print(f"[3/4] {a.trials} trials, {a.concurrent} at a time, on queue '{a.queue}'")
    print(f"      a clearml-agent MUST be listening on '{a.queue}' or nothing will ever run:")
    print(f"          clearml-agent daemon --queue {a.queue}")

    t0 = time.time()
    if a.remote:
        # the controller becomes a task on the queue. it then SITS ON AN AGENT for the whole
        # search while waiting for the trials -- which are queued BEHIND it. with one agent that
        # is a deadlock, exactly like the one select_champion.py already guards against.
        # so this is only correct if you have a SEPARATE queue (or a spare agent) for it.
        print("      !! --remote: the controller will occupy an agent for the whole search.")
        print("      !! with only one agent this DEADLOCKS -- the trials queue up behind it.")
        task.execute_remotely(queue_name=a.queue, exit_process=True)
        opt.start()
    else:
        # THE RIGHT CHOICE FOR 1-3 AGENTS.
        # start() runs the controller loop in a thread in THIS process, on the laptop, and the
        # TRIALS go to the queue and the agents. the controller does almost nothing -- it polls
        # clearml every minute and does a little arithmetic -- so it costs no agent.
        # every agent you have is then free to train.
        #
        # (start_locally() is a different thing: it runs the TRIALS locally too, in-process,
        #  ignoring the queue entirely. useful to smoke-test the loop on a laptop with no agent.
        #  useless for a real search -- no parallelism.)
        opt.start()

    opt.wait()
    opt.stop()

    # ---- 4. the scoreboard ---------------------------------------------------
    # read the objective back out of last_metrics -- the SAME field the optimiser read it from
    # (Objective.get_objective queries last_metrics.<md5 title>.<md5 series>.value, and
    # get_last_scalar_metrics unpacks exactly that into {title: {series: {last, min, max}}}).
    # reading it the same way is what makes this scoreboard a real check on the optimiser and
    # not just a second opinion from a different source.
    top = opt.get_top_experiments(top_k=min(10, a.trials))
    rows = []
    for t in top:
        p = t.get_parameters(backwards_compatibility=False)
        scal = t.get_last_scalar_metrics().get(OBJECTIVE_TITLE, {})
        rows.append({
            "task_id": t.id,
            "val_cost": scal.get(OBJECTIVE_SERIES, {}).get("last"),
            # shown side by side ON PURPOSE. if val_cost is much lower than test_cost across the
            # board, the search has over-fitted the validation slice and you should cut the
            # number of trials, not celebrate.
            "test_cost": scal.get("test/trading_cost", {}).get("last"),
            "test_macro_f1": scal.get("test/macro_f1", {}).get("last"),
            **{k.replace("Args/", ""): v for k, v in p.items()
               if k.startswith("Args/") and k not in
               ("Args/dataset_id", "Args/dataset_version", "Args/model_type", "Args/seed")},
        })

    import pandas as pd
    board = pd.DataFrame(rows)
    print(f"\n[4/4] top {len(board)} of {a.trials}, by VALIDATION trading_cost (lower is better)")
    if not board.empty:
        print(board.to_string(index=False))

    if board.empty or board["val_cost"].isna().all():
        raise SystemExit(
            f"\n  EVERY TRIAL SCORED NOTHING.\n"
            f"  the optimiser looked for the scalar '{OBJECTIVE_TITLE}/{OBJECTIVE_SERIES}' and\n"
            f"  found it on none of the {a.trials} tasks.\n\n"
            f"  that means train.py never reported it. it must call, on the VALIDATION rows:\n"
            f"      from trainer.objective import trading_cost, report_trading_cost\n"
            f"      report_trading_cost(logger, 'val', trading_cost(y_val, pred_val, classes))\n\n"
            f"  this is the #1 silent failure of clearml HPO and it is why it is checked here\n"
            f"  rather than left for you to notice in the UI.")

    best = board.iloc[0]
    print(f"\n  BEST: task {best['task_id']}")
    print(f"    val_trading_cost  {best['val_cost']}   <- what the search optimised")
    print(f"    test_trading_cost {best['test_cost']}   <- THE HONEST NUMBER. quote this one.")
    print(f"\n  the val number is the LOWEST OF {a.trials} DRAWS, so it is biased low by")
    print(f"  construction -- that is the winner's curse, and it is not a bug, it is what")
    print(f"  selection does. the test slice was never looked at during the search, so its")
    print(f"  number is the one that survives a manager's question.")

    # hand the winner's settings back as a file, so publish_version.py can train the next
    # dataset version with them instead of the argparse defaults.
    drop = ("task_id", "val_cost", "test_cost", "test_macro_f1")
    winner = {k: v for k, v in best.items() if k not in drop}
    out = pathlib.Path(f"best_params_{a.model_type}.json")
    out.write_text(json.dumps({"model_type": a.model_type,
                               "dataset_version": a.dataset_version,
                               "val_trading_cost": best["val_cost"],
                               "test_trading_cost": best["test_cost"],
                               "params": winner}, indent=2))
    task.upload_artifact("best_params", str(out))
    print(f"\n  written: {out}  ({time.time() - t0:.0f}s)")
    task.close()


if __name__ == "__main__":
    main()
