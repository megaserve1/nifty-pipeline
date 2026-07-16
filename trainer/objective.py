"""
trainer/objective.py -- the ONE definition of the thing HPO is allowed to optimise.

read this before hpo.py. it is short, and it is the whole ball game.


WHY THIS FILE EXISTS AT ALL
--------------------------------------------------------------------------------------
ClearML's HyperParameterOptimizer does NOT call a score function of yours. it clones the
trainer, lets it run to completion, and then goes and READS ONE SCALAR off the finished task.
it finds that scalar by two strings -- a TITLE and a SERIES:

    clearml/automation/optimization.py, Objective.get_objective:
        task = Task._query_tasks(task_ids=[task_id],
                                 only_fields=["last_metrics.<md5(title)>.<md5(series)>"])
        values = metrics[md5(title)][md5(series)]
        return values["value"]
        ...
        except Exception:
            return None            <-- THIS IS THE WHOLE PROBLEM

if the trainer reports "Summary"/"val_cost" and the optimiser asks for "Summary"/"val/trading_cost",
every lookup misses and get_objective returns None. None is NOT an error. the optimiser keeps
going, every trial scores the same nothing, and four hours later it hands you a "best" set of
parameters which are really just the first ones it happened to sample. green all the way. no
traceback. no warning in the run.

(SearchStrategy._validate_base_task does check for the metric -- but it only calls
logger.warning(). a warning in an agent's log is a warning nobody reads.)

this is the same SHAPE of bug as finalize()-without-publish(). so the two strings are defined
HERE, once, and train.py and hpo.py both import them. they cannot drift apart because there is
only one copy of them.


HOW THE SCALAR ACTUALLY GETS THERE
--------------------------------------------------------------------------------------
verified in clearml 2.1.10, clearml/logger.py line 190:

    def report_single_value(self, name, value):
        return self.report_scalar(title="Summary", series=name,
                                  value=value, iteration=-(2**31))

so report_single_value("val/trading_cost", 41.2) IS a scalar, under title "Summary", series
"val/trading_cost". it lands in last_metrics, which is exactly where the optimiser looks.
that is why OBJECTIVE_TITLE below is the literal string "Summary" -- it is not a name we chose,
it is the title ClearML hard-codes.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C                              # noqa: E402
from trainer.shap_logic import rank_mistakes    # noqa: E402


# ---------------------------------------------------------------- the contract
# the names every reported number uses. they carry their split on purpose -- see report_split()
# for the silent collision this prevents.
SPLITS = ("val", "test")


def series_for(split: str, metric: str) -> str:
    """the one place a reported metric name is built. 'val' + 'trading_cost' -> 'val/trading_cost'."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}. one of {SPLITS}")
    return f"{split}/{metric}"


# the two strings the optimiser searches by. SERIES is DERIVED from series_for(), not typed out
# again -- so it is impossible for the optimiser to be looking for a name the trainer never says.
OBJECTIVE_TITLE = "Summary"          # fixed by clearml: report_single_value -> title="Summary"
OBJECTIVE_SERIES = series_for("val", "trading_cost")
OBJECTIVE_SIGN = "min"               # trading_cost is a COST. lower is better.


# ---------------------------------------------------------------- the split
def three_way_split(ts: pd.Series, val_fraction: float, test_fraction: float,
                    embargo_sessions: int):
    """cut the data into  TRAIN | embargo | VALIDATION | embargo | TEST.

    WHY A THIRD SLICE. THIS IS THE POINT OF THE WHOLE EXERCISE.
        train.py today cuts train | embargo | test, and reports its score on TEST. that is
        honest for ONE model with fixed settings.

        the moment we run 50 trials and KEEP THE ONE WITH THE BEST TEST SCORE, the test set has
        entered the training loop. we did not fit the model on it, but we fitted the SETTINGS on
        it -- and settings are parameters too. the winning number is then the best of 50 draws
        from a noisy distribution, so it is biased high by construction, and there is nothing
        left to measure the real thing with.

        worked example, and it is not a small effect. suppose every one of 50 candidate models
        is genuinely identical, and the only thing separating their test trading_cost is noise
        with a spread of 1.5. pick the lowest of 50 draws and you will land roughly 2 standard
        deviations below the true mean -- about 3 points of trading_cost that DO NOT EXIST. you
        would report 38 to the manager, deploy it, and live-trade a 41.

        so: the optimiser tunes on VALIDATION. TEST is opened once, at the end, on the winner.
        that number is the one you are allowed to say out loud.

    WHY TWO EMBARGOES, NOT ONE
        the same argument that puts a gap between train and test puts one between train and
        validation. a validation row a minute after the train cut shares nearly all of its
        20-day rolling window with the training rows -- it is a near-duplicate. tune on those
        and the optimiser will simply pick whichever settings memorise the training period best.
        the embargo before VAL is what stops that.

        the embargo before TEST stays too, for the original reason.

    WHY THE EMBARGO IS COUNTED IN SESSIONS, NOT CALENDAR DAYS
        this used to take embargo_days and add pd.Timedelta(days=21). 21 CALENDAR days, to cover
        a feature lookback of "20 days". but "20 days" means 20 TRADING SESSIONS, and the market
        is shut at weekends and on holidays -- 21 calendar days spans about 14 sessions, never
        20. so the gap was ~30% too short. we now count sessions, off the data's own calendar.
        see trainer/purged_cv.embargo_end.

    the layout, by time:

        |<---------- train ---------->|xxx|<--- val --->|xxx|<---- test ---->|
        ^                             ^   ^             ^   ^                ^
        oldest                    val_cut |         test_cut                 newest
                                    +25 sessions      +25 sessions
                                    (both xxx blocks are thrown away)

    returns (train_mask, val_mask, test_mask, info) -- three BOOLEAN masks that never overlap.
    """
    from trainer.purged_cv import embargo_end

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be 0 (off) or between 0 and 1")
    if val_fraction + test_fraction >= 0.9:
        raise ValueError(f"val {val_fraction} + test {test_fraction} leaves almost nothing to "
                         f"train on. refusing.")

    ts = pd.to_datetime(ts)

    # quantiles of TIME, so the cuts are the same rows whatever order the frame happens to be in
    test_cut = ts.quantile(1 - test_fraction)
    test_emb_end = embargo_end(ts, test_cut, embargo_sessions)
    test = ts > test_emb_end

    # VAL_FRACTION = 0 -> no tuning set. a plain TRAIN | embargo | TEST. this is the smoke-run
    # shape: honest for ONE model with fixed settings, and NOT enough to tune on. hpo.py refuses
    # to run in this mode, deliberately -- see its preflight.
    if val_fraction == 0:
        train = ts <= test_cut
        val = pd.Series(False, index=ts.index)
        val_cut = test_cut
    else:
        val_cut = ts.quantile(1 - test_fraction - val_fraction)
        val_emb_end = embargo_end(ts, val_cut, embargo_sessions)
        train = ts <= val_cut
        val = (ts > val_emb_end) & (ts <= test_cut)

    # the guard that matters. if the val slice is shorter than the embargo, val comes back EMPTY
    # and the optimiser would score every trial on zero rows -- which does not crash, it just
    # makes every trial tie. so we say so, loudly, before a single agent-hour is spent.
    required = ("train", "test") if val_fraction == 0 else ("train", "val", "test")
    for name in required:
        m = {"train": train, "val": val, "test": test}[name]
        if int(m.sum()) == 0:
            raise ValueError(
                f"the {name} slice is EMPTY.\n"
                f"  val_fraction={val_fraction} test_fraction={test_fraction} "
                f"embargo_sessions={embargo_sessions}\n"
                f"  the {embargo_sessions}-session embargo is eating a whole slice. either widen\n"
                f"  the fraction or shorten the embargo -- but the embargo must stay >= the\n"
                f"  longest feature lookback (20 sessions), so widen the fraction.")

    # they must not overlap. a row in two slices is a leak, and it would never announce itself.
    overlap = int((train & val).sum() + (val & test).sum() + (train & test).sum())
    if overlap:
        raise AssertionError(f"{overlap} rows are in two slices at once -- that is a leak")

    info = {
        "val_cut": str(val_cut),
        "test_cut": str(test_cut),
        "embargo_sessions": embargo_sessions,
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "n_embargoed": int((~train & ~val & ~test).sum()),
        "train_end": str(ts[train].max()),
        "val_start": str(ts[val].min()) if int(val.sum()) else None,
        "val_end": str(ts[val].max()) if int(val.sum()) else None,
        "test_start": str(ts[test].min()),
        "val_enabled": bool(int(val.sum())),
    }
    return train, val, test, info


# ---------------------------------------------------------------- the cost
def load_severity() -> tuple[dict, float]:
    """the same severity matrix select_champion.py uses. one file, one truth."""
    if not C.SEVERITY_FILE.exists():
        return {}, 1.0
    cfg = json.loads(C.SEVERITY_FILE.read_text())
    sev = {k: v for k, v in cfg.get("severity", {}).items() if not k.startswith("_")}
    return sev, float(cfg.get("default", 1))


def trading_cost(y_true, y_pred, classes: list, severity: dict | None = None,
                 default_sev: float | None = None) -> float:
    """total expected cost of this model's mistakes. LOWER IS BETTER.

    identical maths to select_champion.trading_cost -- sum of (rate x severity) over every kind
    of mistake. it is repeated here only so the TRAINER can compute it on the validation rows
    without importing the champion script (which drags ClearML in).

    one property worth knowing before you optimise it: rate = count / n_true is computed PER TRUE
    CLASS, so the 1.2% ENTRY_SUB contributes on exactly the same footing as the 53% NO_TRADE.
    that is a macro metric. it is precisely why it is safe to optimise here and accuracy is not:
    a model that ignores the rare classes cannot hide from it.
    """
    if severity is None or default_sev is None:
        severity, default_sev = load_severity()
    rank = rank_mistakes(np.asarray(y_true), np.asarray(y_pred), classes, severity, default_sev)
    return float(rank["importance"].sum()) if not rank.empty else 0.0


# ---------------------------------------------------------------- the report
def report_trading_cost(logger, split: str, cost: float) -> None:
    """report one split's trading_cost, with the split IN THE NAME.

    calling this with split="val" is what makes a trial VISIBLE to the optimiser: it writes
    "Summary"/"val/trading_cost", which is exactly OBJECTIVE_TITLE / OBJECTIVE_SERIES above.
    if the trainer never calls it, every trial scores None and the whole search is theatre.

    THE COLLISION THE SPLIT PREFIX EXISTS TO PREVENT
        train.py's report_metrics() currently calls report_single_value("macro_f1", ...) with no
        prefix at all. once we score TWO splits, calling it twice puts both on the SAME
        title/series -- "Summary"/"macro_f1". last_metrics keeps only the LAST value written, so
        the two splits silently overwrite each other and you can no longer tell which number you
        are looking at.

        worse: point the optimiser at "Summary"/"macro_f1" and it would read whichever split was
        reported last -- which, in the order train.py runs them, is TEST. you would be tuning on
        the test set, and absolutely nothing about the run would look wrong.

        so every number carries its split. "val/macro_f1", "test/macro_f1". see the note on
        report_metrics in the research answer -- it needs the same prefix.
    """
    if logger is None:
        return
    logger.report_single_value(series_for(split, "trading_cost"), round(float(cost), 4))
