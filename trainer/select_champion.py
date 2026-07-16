"""
trainer/select_champion.py -- pick the best of the three models, and say WHY.

runs after random_forest, xgboost and catboost have all trained on the same dataset version.

HOW WE PICK -- AND WHY ACCURACY IS BANNED
--------------------------------------------------------------------------------------
    accuracy is a lie here. NO_TRADE is 53% of the data, so a model that says NO_TRADE and
    nothing else scores 53% and is worth nothing. a model that scores 60% by never finding a
    single ENTRY signal makes no money at all.

    what actually makes money is FINDING THE RARE SIGNALS. so we pick on:

        PRIMARY   trading_cost  = sum over all mistakes of (how often x how much it costs)
                                  using configs/severity_7class.json.
                                  LOWER IS BETTER. this is the closest thing to "how much
                                  money did this model's mistakes cost", and it is the only
                                  metric that knows a reversal is worse than a wrong size.

        REPORTED ALONGSIDE (so you can see WHY the winner won):
            macro_f1     the average F1 across all 7 classes, treating the 1.2% ENTRY_SUB
                         exactly as seriously as the 53% NO_TRADE. that is what we want.
            mean_pr_auc  how well the model RANKS each class, ignoring the threshold. for a
                         rare class this says more than precision at one arbitrary cut-off.
            never_predicted   classes the model is completely blind to. a model blind to a
                              class it must trade is disqualified, whatever it scores.

    ALL THREE MODELS ARE SCORED ON THE SAME TEST ROWS, so the comparison is fair.

A HARD GUARD
    if a model never predicts a class that carries real weight, it is UNUSABLE no matter how
    pretty its numbers are. it is flagged and cannot be champion. (right now every model is
    blind to NO_TRADE, because NO_TRADE has weight 0 in the labels file -- that is a label
    policy problem, not a model problem, and this script says so out loud.)
"""
import argparse
import time
import json
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C          # noqa: E402
from trainer.shap_logic import rank_mistakes   # noqa: E402


def load_severity() -> tuple[dict, float]:
    if not C.SEVERITY_FILE.exists():
        return {}, 1.0
    cfg = json.loads(C.SEVERITY_FILE.read_text())
    sev = {k: v for k, v in cfg.get("severity", {}).items() if not k.startswith("_")}
    return sev, float(cfg.get("default", 1))


def trading_cost(y_true, y_pred, classes, sev, default_sev) -> float:
    """the total expected cost of this model's mistakes. LOWER IS BETTER.

    it is just the sum of (rate x severity) over every kind of mistake -- the same number the
    SHAP ranking uses, added up. a model that makes many cheap mistakes can beat one that makes
    a few catastrophic ones, which is exactly the trade we want the metric to make.
    """
    rank = rank_mistakes(np.asarray(y_true), np.asarray(y_pred), classes, sev, default_sev)
    return float(rank["importance"].sum()) if not rank.empty else 0.0


class Deadlock(SystemExit):
    """this task is sitting on the only agent, waiting for tasks that need that agent."""


# the statuses ClearML can hand back. named here so a typo cannot silently become "still running".
DEAD = ("failed", "stopped", "closed")
QUEUED = ("queued", "created")               # not started -- NOBODY is working on it yet


def clearml_probe(version: str, models: list | None = None):
    """ask ClearML how each trainer is doing. returns {model: (status, has_model_artifact)}.

    kept as a separate function purely so the waiting logic below can be TESTED without
    ClearML, a network, or four hours of real time.

    `models` is WHICH trainers to poll. it used to be hardwired to ALL of C.MODEL_TYPES --
    so a subset publish (--models xgboost, a documented flag) could NEVER crown a champion:
    the two never-created tasks polled as 'missing' for ever, landed in `running`, and main()
    correctly refused to crown while anything was 'still running'. the race had one runner and
    the judge waited for three.
    """
    from clearml import Task

    out = {}
    for mtype in (models or C.MODEL_TYPES):
        t = Task.get_task(project_name=C.CLEARML_PROJECT,
                          task_name=f"train_{mtype} v{version}")
        if t is None:
            out[mtype] = ("missing", False, None)
        else:
            out[mtype] = (t.get_status(), "model" in t.artifacts, t)
    return out


def wait_for_models(version: str, expect, timeout_min: int = 240,
                    poll: int = 30, stall_min: int = 10,
                    probe=None, sleep=time.sleep, now=time.time) -> tuple[dict, list, list]:
    """WAIT until the models have finished. returns (finished, still_running, dead).

    the champion task is queued at the same time as the models, so without this it would start
    the moment an agent was free, find nothing finished, and crown a winner out of an empty set.
    so it polls: every 30 seconds, ask ClearML which of the trainers are done.

    THREE OUTCOMES, AND THEY ARE NOT THE SAME THING -- which is why all three are returned:

        finished       trained, saved a model. these are the candidates.
        dead           failed / aborted / finished without saving a model. these are a VERDICT:
                       that model is out, and we can fairly choose between the survivors.
        still_running  we ran out of patience. that is NOT a verdict. the missing model might
                       have been the best one, so crowning a champion without it would be
                       naming a winner while a runner is still on the track. main() refuses.

    THE DEADLOCK (the real bug, worse than the timeout)
        this task waits by OCCUPYING AN AGENT. if there is only one agent and it picked THIS
        task up, the three trainers are stuck behind us in the queue and can never start.
        we would then poll happily for four hours and give up -- having caused the very thing
        we were waiting for. so: if NOTHING has even STARTED after stall_min minutes, we are
        almost certainly the blocker. we say so and get out of the way immediately.
    """
    # `expect` is a LIST OF MODEL NAMES (or, for backward compatibility, a bare count --
    # in which case all of C.MODEL_TYPES are polled, the old behavior).
    models = list(expect) if isinstance(expect, (list, tuple)) else list(C.MODEL_TYPES)
    expect = len(models) if isinstance(expect, (list, tuple)) else int(expect)
    probe = probe or (lambda: clearml_probe(version, models))
    t0 = now()
    deadline = t0 + timeout_min * 60
    last_note = ""

    while True:
        status = probe()
        found, running, dead = {}, [], []
        for mtype, (st, has_model, task) in status.items():
            if st == "completed":
                if has_model:
                    found[mtype] = task
                else:
                    dead.append(f"{mtype}(completed but saved NO model)")
            elif st in DEAD:
                dead.append(f"{mtype}({st})")
            else:
                running.append(f"{mtype}({st})")

        if len(found) + len(dead) >= expect or not running:
            return found, running, dead

        # deadlock: nothing finished, nothing failed, and nothing has even STARTED.
        not_started = all(st in QUEUED or st == "missing" for st, _, _ in status.values())
        if not_started and not found and (now() - t0) > stall_min * 60:
            raise Deadlock(
                f"\n  DEADLOCK: after {stall_min} min, not one of the {expect} models has even "
                f"STARTED.\n"
                f"  they are all still sitting in the '{C.TRAIN_QUEUE}' queue -- and this task is "
                f"waiting\n"
                f"  for them WHILE OCCUPYING AN AGENT. if that is the only agent, they are stuck "
                f"behind us\n"
                f"  and can never start. so we are the reason they are not running.\n\n"
                f"  exiting now so the agent is freed and they can train.\n\n"
                f"  fix it either way:\n"
                f"    - start another worker:   clearml-agent daemon --queue {C.TRAIN_QUEUE}\n"
                f"    - or re-run this task from the ClearML UI once the models have finished.")

        note = f"waiting: {len(found)} done, {len(dead)} dead, still going: {running}"
        if note != last_note:                       # only print when something changes
            print(f"      {note}")
            last_note = note

        if now() > deadline:
            print(f"      !! gave up waiting after {timeout_min} min. still going: {running}")
            return found, running, dead
        sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="")
    ap.add_argument("--dataset_version", default="")
    # THE MODEL NAMES to wait for, comma-separated (e.g. "random_forest,xgboost,catboost").
    # This was type=int -- but enqueue_champion sends the NAMES (so a --models subset can crown),
    # and argparse crashed with "invalid int value: 'random_forest,xgboost,catboost'" before the
    # code below (which already expects names) ever ran. A half-applied fix. It is a string now.
    ap.add_argument("--expect_models", default=",".join(C.MODEL_TYPES),
                    help="comma-separated model NAMES to wait for before choosing")
    ap.add_argument("--wait_minutes", type=int, default=240)
    ap.add_argument("--crown_partial", action="store_true",
                    help="crown a champion even if a model was STILL TRAINING when we gave up. "
                         "off by default: the missing model might have been the best one.")
    from clearml import Dataset, Task   # BEFORE parse_args -- clearml patches argparse at import;
    # parse first and every Args/ override from the clone is silently lost. see train.py.
    a = ap.parse_args()

    import joblib

    task = Task.init(project_name=C.CLEARML_PROJECT,
                     task_name=C.BASE_CHAMPION_NAME,
                     task_type=Task.TaskTypes.qc,
                     output_uri=C.model_output_uri())
    logger = task.get_logger()

    if not a.dataset_id:
        print("no --dataset_id: base-task registration run. exiting cleanly.")
        task.close()
        return

    version = a.dataset_version

    # ---- 1. WAIT for the models, then collect them --------------------------
    # expect_models arrives as "random_forest,xgboost,catboost" (names -- the enqueuer says WHO
    # to wait for). a bare "3" from an old clone means "the first 3 known types" (count fallback).
    raw = str(a.expect_models).strip()
    if raw.isdigit():
        expected = list(C.MODEL_TYPES)[:int(raw)] or list(C.MODEL_TYPES)
    else:
        expected = [m.strip() for m in raw.split(",") if m.strip()]
    print(f"[1/4] waiting for {len(expected)} model(s) trained on v{version}: {expected}")
    print(f"      (this task was queued alongside them, so it must wait -- otherwise it would")
    print(f"       start on the first free agent and find nothing finished)")
    found, running, dead = wait_for_models(version, expected, a.wait_minutes)
    for mtype in found:
        print(f"      OK  {mtype}")
    for d in dead:
        print(f"      DEAD  {d}")

    if not found:
        raise SystemExit("no finished models to choose between. is a clearml-agent running "
                         f"on the '{C.TRAIN_QUEUE}' queue?")

    # a model that DIED is a verdict -- it is out, and the survivors can be judged fairly.
    # a model that is STILL TRAINING is not a verdict. crowning a champion without it would be
    # naming a winner while someone is still running the race, and the champion tag is what
    # goes to production. so we refuse, unless told otherwise.
    if running and not a.crown_partial:
        raise SystemExit(
            f"\n  NOT crowning a champion.\n"
            f"  {len(found)} model(s) finished, but these were STILL TRAINING when we gave up "
            f"after {a.wait_minutes} min:\n"
            f"      {running}\n"
            f"  the unfinished one might be the best -- picking now could ship the wrong model.\n\n"
            f"  do one of these:\n"
            f"    - wait for them, then re-run this task from the ClearML UI (nothing is lost)\n"
            f"    - re-run with a longer wait:  --wait_minutes {a.wait_minutes * 2}\n"
            f"    - or accept a partial answer:  --crown_partial")

    if running and a.crown_partial:
        print(f"\n      !! --crown_partial: choosing WITHOUT {running} -- still training.")
    if len(found) < len(C.MODEL_TYPES):
        missing = [m for m in C.MODEL_TYPES if m not in found]
        print(f"\n      !! choosing from only {len(found)} of {len(C.MODEL_TYPES)} models.")
        print(f"      !! missing: {missing}. the champion may not be the true best.")

    # ---- 2. score them all on the SAME rows ---------------------------------
    print(f"[2/4] scoring {len(found)} models on the same test rows")
    sev, default_sev = load_severity()
    rows = []
    for mtype, t in found.items():
        bundle = joblib.load(t.artifacts["model"].get_local_copy())
        m = bundle["metrics"]
        classes = list(bundle["label_encoder"].classes_)

        # rebuild the cost from the confusion matrix the trainer already saved, so we do not
        # have to download and re-score the whole dataset three times.
        cm = pd.DataFrame(m["confusion"])
        y_t, y_p = [], []
        for i, tc in enumerate(classes):
            for j, pc in enumerate(classes):
                n = int(cm.loc[f"true_{tc}", f"pred_{pc}"])
                y_t.extend([i] * n)
                y_p.extend([j] * n)
        cost = trading_cost(y_t, y_p, classes, sev, default_sev)

        blind = m.get("never_predicted", [])
        rows.append({
            "model": mtype,
            "trading_cost": round(cost, 4),          # PRIMARY -- lower is better
            "macro_f1": round(m["macro_f1"], 4),
            "mean_pr_auc": round(m["mean_pr_auc"], 4),
            "blind_to": ", ".join(blind) if blind else "-",
            "usable": "NO" if blind else "yes",
            "task_id": t.id,
        })

    board = pd.DataFrame(rows).sort_values("trading_cost").reset_index(drop=True)
    print("\n--- THE SCOREBOARD (trading_cost: LOWER is better. accuracy is not here on purpose) ---")
    print(board.drop(columns=["task_id"]).to_string(index=False))
    logger.report_table("Model comparison", f"v{version}", table_plot=board)

    # ---- 3. the guard --------------------------------------------------------
    blind_all = board[board["usable"] == "NO"]["blind_to"].tolist()
    if len(board[board["usable"] == "yes"]) == 0:
        print("\n  !! EVERY model is blind to at least one class.")
        print(f"  !! blind to: {sorted(set(', '.join(blind_all).split(', ')))}")
        print("  !! with NO_TRADE at weight 0 in the labels, no model can learn to stay out,")
        print("  !! so all three will want to trade every minute. this is a LABEL POLICY")
        print("  !! problem. the champion below is the best of a bad set -- do not deploy it.")
        logger.report_text("ALL MODELS BLIND TO A CLASS -- see the NO_TRADE weight=0 problem")
        task.add_tags(["ALL_MODELS_BLIND"])

    # ---- 4. crown the winner -------------------------------------------------
    #
    # THE GUARD ABOVE USED TO BE INERT, AND THAT WAS WORSE THAN HAVING NO GUARD.
    # the old code was:
    #     pool = usable if len(usable) else board      # if none are usable, still name the best
    #     win = pool.iloc[0]
    #     ... champ_task.add_tags(["champion", ...])
    # so it printed "do not deploy it" and then, four lines later, tagged it `champion` anyway.
    # the tag is what production reads. nothing enforced the refusal -- and by this script's own
    # docstring, "no usable model" is the EXPECTED case today, because NO_TRADE has weight 0 and
    # every model is blind to it. so the guard fired every single time and did nothing every
    # single time.
    #
    # now: the best model is still named, still scored, still recorded -- you need to see the
    # scoreboard. but it is tagged `champion-BLIND`, not `champion`. nothing downstream that
    # looks for `champion` will pick it up. to deploy, fix the NO_TRADE weight upstream in the
    # label policy and re-run. that is the actual fix, and this makes you do it.
    usable = board[board["usable"] == "yes"]
    crownable = len(usable) > 0
    pool = usable if crownable else board
    win = pool.iloc[0]

    print(f"\n[4/4] BEST MODEL: {win['model']}")
    print(f"      trading_cost {win['trading_cost']}  (lowest = its mistakes cost the least)")
    print(f"      macro_f1 {win['macro_f1']}   mean_pr_auc {win['mean_pr_auc']}")

    champ_task = Task.get_task(task_id=win["task_id"])
    if crownable:
        champ_task.add_tags(["champion", f"v{version}"])
        print(f"      tagged: champion")
    else:
        champ_task.add_tags(["champion-BLIND", f"v{version}"])
        task.add_tags(["NO_CHAMPION"])
        print(f"      !! it is BLIND to: {win['blind_to']}")
        print(f"      !! tagged 'champion-BLIND', NOT 'champion'. nothing will deploy it.")
        print(f"      !! there is NO champion for v{version}. that is the correct answer:")
        print(f"      !!   a model that cannot predict {win['blind_to']} is not a model,")
        print(f"      !!   it is a model that has learned to ignore 53% of the market.")
        print(f"      !! the fix is upstream: give NO_TRADE a weight of ~0.1-0.2 in the LABEL")
        print(f"      !! POLICY, rebuild the dataset, and run this again.")
        logger.report_text(
            f"NO CHAMPION for v{version}. best model ({win['model']}) is blind to "
            f"{win['blind_to']}. tagged champion-BLIND. fix the NO_TRADE weight upstream.")

    for _, r in board.iterrows():                    # a stale champion tag would be a lie
        if r["task_id"] != win["task_id"]:
            Task.get_task(task_id=r["task_id"]).add_tags([f"challenger-v{version}"])

    logger.report_single_value("champion_trading_cost", float(win["trading_cost"]))
    logger.report_single_value("champion_crowned", int(crownable))
    logger.report_text(f"best model for v{version}: {win['model']} "
                       f"({'champion' if crownable else 'BLIND -- not crowned'})")
    task.upload_artifact("scoreboard", board)
    task.add_tags(["champion-selection", f"v{version}"])
    task.close()


if __name__ == "__main__":
    main()
