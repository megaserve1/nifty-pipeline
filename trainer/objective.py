"""trainer/objective.py -- the one definition of what HPO optimises.

ClearML does not call a score function of yours. it reads ONE scalar off the finished task, found
by two strings (title, series). get the strings wrong and get_objective returns None, silently --
every trial scores the same nothing and the "best" params are just the first ones sampled. green
all the way, no traceback. so the two strings live here once and train.py and hpo.py both import
them; they cannot drift apart.

report_single_value() is a scalar under the title "Summary" (clearml/logger.py:190), which is why
OBJECTIVE_TITLE is that literal -- ClearML hard-codes it, we did not choose it.
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
def bundle_random_split(ts: pd.Series, val_fraction: float, test_fraction: float,
                        bundle_minutes: int = 15, seed: int = 42):
    """cut by whole 15-min candles, assigned at RANDOM. no minute is split across slices.

    NO EMBARGO, and none is possible: random assignment puts training bundles minutes away from
    every test bundle on both sides, while the features look back ~500 bundles. purging that would
    delete the dataset. that is the open risk -- scripts/forward_holdout_test.py measures it.
    the seed is recorded; every consumer re-uses it or it would mislabel which rows were test.
    """
    import numpy as np

    ts = pd.to_datetime(ts)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be 0 (off) or between 0 and 1")

    bundle = ts.dt.floor(f"{bundle_minutes}min")          # every minute -> its candle
    keys = pd.Index(bundle.unique()).sort_values()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(keys))

    n_test = int(round(len(keys) * test_fraction))
    n_val = int(round(len(keys) * val_fraction))
    test_keys = set(keys[shuffled[:n_test]])
    val_keys = set(keys[shuffled[n_test:n_test + n_val]])

    test = bundle.isin(test_keys)
    val = bundle.isin(val_keys)
    train = ~test & ~val

    for name, m in (("train", train), ("test", test)) + ((("val", val),) if val_fraction else ()):
        if int(m.sum()) == 0:
            raise ValueError(f"the {name} slice is EMPTY -- {len(keys):,} bundles, "
                             f"val_fraction={val_fraction} test_fraction={test_fraction}")
    overlap = int((train & val).sum() + (val & test).sum() + (train & test).sum())
    if overlap:
        raise AssertionError(f"{overlap} rows are in two slices at once -- that is a leak")

    info = {
        "strategy": "bundle_random",
        "bundle_minutes": bundle_minutes,
        "seed": seed,
        "n_bundles": int(len(keys)),
        "embargo_sessions": 0,
        "embargo_note": "NONE -- impossible under random assignment, see bundle_random_split()",
        "n_train": int(train.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()),
        "n_embargoed": 0,
        "train_end": str(ts[train].max()),
        "val_start": str(ts[val].min()) if int(val.sum()) else None,
        "val_end": str(ts[val].max()) if int(val.sum()) else None,
        # NOT a time cut -- test rows are scattered. read `strategy` first: treating this as
        # "everything after is test" selects the whole dataset (it did, in shap_explain, until 08-05).
        "test_start": str(ts[test].min()),
        "val_enabled": bool(int(val.sum())),
    }
    return train, val, test, info


def bundle_time_split(ts: pd.Series, val_fraction: float, test_fraction: float,
                      bundle_minutes: int, embargo_sessions: int):
    """BUNDLES, BUT IN TIME ORDER -- the middle ground between the other two.

    bundle_random keeps whole candles together but scatters them, so a test minute can sit two
    minutes from a training minute and no embargo is possible. the plain time split has a clean
    forward cut but slices through candles at the boundary.

    this takes bundle_random's candles and assigns them CHRONOLOGICALLY: oldest bundles train,
    then val, then the newest test. no candle is split, the cut is forward-only, and an embargo
    becomes possible again -- which it is not under random assignment.

    THE EMBARGO IS IN TRADING SESSIONS, NOT CALENDAR DAYS. it drops the last N sessions before
    each boundary, because the features look back 20 sessions and a row inside that window has
    already seen part of the next slice.
    """
    import numpy as np

    ts = pd.to_datetime(ts)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be 0 (off) or between 0 and 1")

    bundle = ts.dt.floor(f"{bundle_minutes}min")
    keys = pd.Index(bundle.unique()).sort_values()          # CHRONOLOGICAL, not shuffled
    n = len(keys)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    if n_test == 0 or (val_fraction and n_val == 0):
        raise ValueError(f"only {n} bundles -- val_fraction={val_fraction} "
                         f"test_fraction={test_fraction} rounds a slice to zero")

    test_keys = set(keys[n - n_test:])
    val_keys = set(keys[n - n_test - n_val: n - n_test]) if n_val else set()
    test = bundle.isin(test_keys)
    val = bundle.isin(val_keys)
    train = ~test & ~val

    day = ts.dt.normalize()

    def embargo_before(mask, boundary):
        """drop the last `embargo_sessions` trading sessions of `mask` before `boundary`."""
        if embargo_sessions <= 0 or boundary is None:
            return mask, 0
        days = sorted(day[mask & (ts < boundary)].unique())
        if len(days) <= embargo_sessions:
            return mask, 0
        cut = days[-embargo_sessions]
        keep = mask & (day < cut)
        return keep, int((mask & ~keep).sum())

    n_emb = 0
    if n_val:
        val_start = ts[val].min()
        train, d1 = embargo_before(train, val_start)
        test_start = ts[test].min()
        val, d2 = embargo_before(val, test_start)
        n_emb = d1 + d2
    else:
        test_start = ts[test].min()
        train, n_emb = embargo_before(train, test_start)

    for name, m in (("train", train), ("test", test)) + ((("val", val),) if val_fraction else ()):
        if int(m.sum()) == 0:
            raise ValueError(f"the {name} slice is EMPTY -- {n:,} bundles, "
                             f"val_fraction={val_fraction} test_fraction={test_fraction}, "
                             f"embargo {embargo_sessions} sessions")
    overlap = int((train & val).sum() + (val & test).sum() + (train & test).sum())
    if overlap:
        raise AssertionError(f"{overlap} rows are in two slices at once -- that is a leak")

    info = {
        "strategy": "bundle_time",
        "bundle_minutes": bundle_minutes,
        "seed": None,
        "n_bundles": int(n),
        "embargo_sessions": int(embargo_sessions),
        "embargo_note": "sessions dropped before each boundary; possible here, unlike bundle_random",
        "n_train": int(train.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()),
        "n_embargoed": int(n_emb),
        "train_end": str(ts[train].max()),
        "val_start": str(ts[val].min()) if int(val.sum()) else None,
        "val_end": str(ts[val].max()) if int(val.sum()) else None,
        "test_start": str(ts[test].min()),
        "val_enabled": bool(int(val.sum())),
    }
    return train, val, test, info


def three_way_split(ts: pd.Series, val_fraction: float, test_fraction: float,
                    embargo_sessions: int, strategy: str = None,
                    bundle_minutes: int = None, seed: int = None):
    """cut the data into TRAIN | embargo | VAL | embargo | TEST.

    the third slice exists because HPO fits the SETTINGS on whatever it scores. tune on VAL, open
    TEST once at the end on the winner -- otherwise the reported number is the best of 50 draws.
    """
    from trainer.purged_cv import embargo_end
    import config as _C

    # THE SWITCH. default comes from config, so one setting changes every caller -- but a caller
    # that KNOWS its strategy (a consumer rebuilding a saved split from a model bundle) passes it
    # explicitly, because rebuilding a random split with today's config instead of the recorded
    # one would silently mislabel which rows were test.
    strategy = strategy or getattr(_C, "SPLIT_STRATEGY", "time")
    if strategy == "bundle_random":
        return bundle_random_split(
            ts, val_fraction, test_fraction,
            bundle_minutes=bundle_minutes or getattr(_C, "BUNDLE_MINUTES", 15),
            seed=seed if seed is not None else getattr(_C, "SPLIT_SEED", 42))
    if strategy == "bundle_time":
        return bundle_time_split(
            ts, val_fraction, test_fraction,
            bundle_minutes=bundle_minutes or getattr(_C, "BUNDLE_MINUTES", 15),
            embargo_sessions=embargo_sessions)
    if strategy != "time":
        raise ValueError(f"unknown SPLIT_STRATEGY {strategy!r} -- use 'time', "
                         f"'bundle_random' or 'bundle_time'")

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
        "strategy": "time",
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
