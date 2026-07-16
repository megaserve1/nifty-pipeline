"""
tests/test_hpo_contract.py -- pins the four ClearML HPO facts that fail SILENTLY.

every test here is an argument, not an assertion. read the docstrings.

the theme: ClearML's hyperparameter optimiser finds things BY STRING MATCH, and when a string
does not match it returns None and carries on. nothing raises. nothing goes red. you get a
"best" answer that is really the first thing it tried. the finalize()/publish() bug was the same
shape and it cost a day. so every string in that chain is nailed down here, offline, in seconds.

run:  final_venv/bin/python -m pytest tests/test_hpo_contract.py -q
"""
import inspect

import numpy as np
import pandas as pd
import pytest

clearml = pytest.importorskip("clearml", reason="clearml not installed")

import config as C                                                                # noqa: E402
from trainer.objective import (OBJECTIVE_TITLE, OBJECTIVE_SERIES, OBJECTIVE_SIGN,  # noqa: E402
                               three_way_split, trading_cost, series_for)
from trainer.hpo import search_space                                               # noqa: E402


# =====================================================================================
# 1. WHICH OPTIMISERS WE ACTUALLY HAVE
# =====================================================================================
def test_random_and_grid_search_need_no_extra_package():
    """these two are pure clearml. they are the only optimisers we are allowed to use.

    optuna is not installed and we are not installing it, so if a future clearml made
    RandomSearch depend on it, hpo.py would die on the agent, four hours in. catch it here.
    """
    from clearml.automation import RandomSearch, GridSearch, HyperParameterOptimizer
    assert RandomSearch and GridSearch and HyperParameterOptimizer


def test_optuna_and_bohb_are_NOT_available_and_fail_loudly():
    """the good news buried in this test: these two fail with a real ImportError.

    that is the ONE failure in ClearML HPO that is not silent. so if someone copies an
    optimizer_class=OptimizerOptuna line off the internet into hpo.py, it stops immediately
    instead of quietly degrading. we pin that it still stops immediately.

    if this test ever FAILS, it means someone installed optuna. that is a decision, not an
    accident -- go and make it deliberately.
    """
    with pytest.raises(ImportError):
        from clearml.automation.optuna import OptimizerOptuna     # noqa: F401
    with pytest.raises(ImportError):
        from clearml.automation.hpbandster import OptimizerBOHB   # noqa: F401


# =====================================================================================
# 2. THE SECTION PREFIX -- the silent no-op that already cost this project a day once
# =====================================================================================
def test_hyperparameter_names_must_carry_the_Args_slash_prefix():
    """THE BUG: a parameter named 'max_depth' instead of 'Args/max_depth' changes NOTHING.

    the mechanism, from clearml 2.1.10, automation/job.py:
        task_params = base_temp_task.get_parameters(backwards_compatibility=False)
        task_params.update(parameter_override)

    and get_parameters(backwards_compatibility=False) returns keys of the form 'Section/name'
    (backend_interface/task/task.py: parameters["{}/{}".format(section, key)] = ...).
    argparse arguments live in the section 'Args'. so the real key is 'Args/max_depth'.

    dict.update() with the key 'max_depth' does not overwrite 'Args/max_depth'. it ADDS a
    second, unrelated key. the trial then trains with the DEFAULT depth, reports a score, and
    goes green. do that thirty times and the optimiser tells you, with total confidence, that
    max_depth makes no difference to anything.

    clearml does notice -- SearchStrategy._validate_base_task calls logger.warning("Could not
    find requested hyper-parameters ..."). a warning in an agent log is a warning nobody reads.

    so: every name in every search space starts with 'Args/'. checked here for all three models.
    """
    for model_type in ("random_forest", "xgboost", "catboost"):
        for p in search_space(model_type):
            assert p.name.startswith("Args/"), (
                f"{model_type}: parameter {p.name!r} has no 'Args/' prefix. clearml would "
                f"silently ignore it and every trial would train the default model."
            )


def test_a_parameter_object_uses_its_name_verbatim_as_the_override_key():
    """the prefix only matters because the name goes into the override dict UNCHANGED.

    pinning it: Parameter.get_value() returns {self.name: value}. nothing strips, adds or
    normalises a prefix anywhere. so what you type is what clearml looks up.
    """
    from clearml.automation import UniformIntegerParameterRange
    p = UniformIntegerParameterRange("Args/max_depth", min_value=3, max_value=10)
    assert list(p.get_value().keys()) == ["Args/max_depth"]
    assert list(p.to_list()[0].keys()) == ["Args/max_depth"]


def test_clearml_only_WARNS_about_a_bad_parameter_name():
    """the reason hpo.preflight() exists at all.

    _validate_base_task builds `missing_params` and then calls logger.warning() -- it does not
    raise. this test reads the source and pins that there is no raise on that path, so nobody
    later deletes preflight() thinking clearml has it covered.
    """
    from clearml.automation.optimization import SearchStrategy
    src = inspect.getsource(SearchStrategy._validate_base_task)
    assert "missing_params" in src
    assert "logger.warning" in src
    # the ONLY raise in there is for a base task that does not exist at all
    raises = [ln.strip() for ln in src.splitlines() if "raise" in ln]
    assert all("Could not find base task id" in ln for ln in raises), (
        "clearml now raises on a missing hyper-parameter. good -- but check preflight() still "
        "agrees with it before relaxing anything."
    )


# =====================================================================================
# 3. THE OBJECTIVE -- the #1 silent failure of ClearML HPO
# =====================================================================================
def test_report_single_value_really_is_a_scalar_under_the_title_Summary():
    """THE FACT THE WHOLE OBJECTIVE HANGS ON.

    HyperParameterOptimizer does not call a score function. it reads ONE scalar off the finished
    task, out of last_metrics, keyed by md5(title) and md5(series):

        Objective.get_objective:
            only_fields=[f"last_metrics.{md5(title)}.{md5(series)}"]
            ...
            except Exception:
                return None          <-- a miss is not an error

    so the trainer has to put the number somewhere the optimiser will look. clearml/logger.py:

        def report_single_value(self, name, value):
            return self.report_scalar(title="Summary", series=name, value=value,
                                      iteration=-(2**31))

    the title is HARD-CODED to "Summary". that is why OBJECTIVE_TITLE is "Summary" -- it is not
    a name we picked, it is the one clearml uses. if a future clearml renames it, every trial
    silently scores None and the search becomes theatre. this test breaks first.
    """
    src = inspect.getsource(clearml.Logger.report_single_value)
    assert 'title="Summary"' in src, "clearml changed where report_single_value files its value"
    assert "series=name" in src
    assert OBJECTIVE_TITLE == "Summary"


def test_the_optimiser_reads_last_metrics_which_is_where_a_single_value_lands():
    """pins the OTHER end of the same wire: the optimiser really does read last_metrics.

    if clearml ever switched Objective to the events API instead, a single value stamped with
    iteration=-(2**31) might not be found the same way -- so pin that it still reads
    last_metrics["value"].
    """
    from clearml.automation.optimization import Objective
    src = inspect.getsource(Objective.get_objective)
    assert "last_metrics" in src
    assert 'values["value"]' in src


def test_the_trainer_and_the_optimiser_cannot_disagree_about_the_name():
    """the structural fix, pinned.

    OBJECTIVE_SERIES is not typed out twice. it is DERIVED from series_for('val',
    'trading_cost') -- the same function the trainer calls when it reports. so there is no way
    to change one and forget the other, which is exactly how this bug happens in the wild.
    """
    assert OBJECTIVE_SERIES == series_for("val", "trading_cost")
    assert OBJECTIVE_SIGN == "min", "trading_cost is a COST. minimising it is the whole point."


def test_the_objective_is_the_VALIDATION_cost_never_the_test_cost():
    """the leakage guard, stated as code.

    if OBJECTIVE_SERIES were 'test/trading_cost', the optimiser would pick the settings that do
    best on the test set. the test set would then be part of the training loop -- we would not
    have fitted the MODEL on it, but we would have fitted the SETTINGS on it, and settings are
    parameters too. the reported test score would be the best of N noisy draws, biased low, and
    there would be nothing honest left to measure with.

    this is the single most valuable line in the file.
    """
    assert OBJECTIVE_SERIES.startswith("val/"), (
        "the optimiser is pointed at a non-validation metric. if that is the test set, every "
        "number this project reports from now on is a lie."
    )


def test_a_float_range_on_an_int_argument_would_crash_every_trial():
    """a trap worth one line of test.

    clearml round-trips every hyper-parameter through the task's Args as a STRING, and argparse
    parses it back with type=int. UniformParameterRange samples FLOATS, so
    UniformParameterRange('Args/max_depth', 3, 10) sends "7.431..." and argparse dies with
        invalid int value: '7.4312'
    thirty trials, thirty crashes.

    UniformIntegerParameterRange samples real python ints. pin that, and pin that every
    int-valued argparse arg in our spaces uses an int-safe range.
    """
    from clearml.automation import (UniformIntegerParameterRange, DiscreteParameterRange)
    p = UniformIntegerParameterRange("Args/max_depth", min_value=3, max_value=10)
    for _ in range(20):
        v = p.get_value()["Args/max_depth"]
        assert isinstance(v, int) and not isinstance(v, bool)

    with pytest.raises(ValueError):
        int("7.4312")           # what argparse would do with a float sample

    int_args = {"Args/max_depth", "Args/n_estimators", "Args/min_samples_leaf"}
    for model_type in ("random_forest", "xgboost", "catboost"):
        for p in search_space(model_type):
            if p.name in int_args:
                assert isinstance(p, (UniformIntegerParameterRange, DiscreteParameterRange)), (
                    f"{model_type}: {p.name} is an argparse int, but the search space samples "
                    f"floats for it. every trial would crash on argparse."
                )
                for d in p.to_list():
                    assert float(d[p.name]).is_integer(), f"{p.name} sampled a non-integer"


def test_log_uniform_takes_EXPONENTS_not_values():
    """the learning-rate trap. LogUniformParameterRange(min_value, max_value) are POWERS OF 10.

        get_value -> {name: base ** v}      base defaults to 10

    so (min_value=-2, max_value=-0.7) samples 0.01 .. 0.20 -- correct.
    write (0.01, 0.2) "because that's the range I want" and you sample
        10 ** 0.01 .. 10 ** 0.2   =   1.02 .. 1.58
    a learning rate of 1.5. it does not error. it just trains rubbish, for hours.
    """
    from clearml.automation import LogUniformParameterRange
    good = LogUniformParameterRange("Args/learning_rate", min_value=-2, max_value=-0.7, base=10)
    for _ in range(50):
        v = good.get_value()["Args/learning_rate"]
        assert 0.005 < v < 0.25, f"{v} is not a sane learning rate"

    trap = LogUniformParameterRange("Args/learning_rate", min_value=0.01, max_value=0.2, base=10)
    assert trap.get_value()["Args/learning_rate"] > 1.0, (
        "if this ever fails, clearml changed LogUniformParameterRange to take values instead of "
        "exponents -- and our spaces in hpo.py, which pass exponents, are now all wrong."
    )


# =====================================================================================
# 4. THE SPLIT -- train | embargo | VAL | embargo | TEST
# =====================================================================================
def _minute_index(start="2020-01-01", days=1400):
    """375 minutes a day, weekdays only. close enough to a nifty session for a split test."""
    days_idx = pd.bdate_range(start, periods=days)
    return pd.Series([d + pd.Timedelta(minutes=555 + m) for d in days_idx for m in range(375)])


def test_three_way_split_never_puts_a_row_in_two_slices():
    ts = _minute_index()
    tr, va, te, info = three_way_split(ts, val_fraction=0.15, test_fraction=0.20,
                                       embargo_sessions=C.EMBARGO_SESSIONS)
    assert not (tr & va).any()
    assert not (va & te).any()
    assert not (tr & te).any()
    assert info["n_train"] > 0 and info["n_val"] > 0 and info["n_test"] > 0


def test_three_way_split_is_in_time_order_and_leaves_a_real_gap():
    """THE BUG THIS PINS: an embargo that is shorter than the longest feature lookback.

    the source data has 20-day features (ret_20d, hurst_200, drawdown_from_20d_high). a
    validation row one minute after the training cut shares nineteen and a half days of its
    rolling window with the training rows -- it is a near-duplicate, not an independent sample.
    tune on rows like that and the optimiser will simply pick whatever memorises the training
    period best, and it will look brilliant doing it.

    so BOTH gaps -- train->val and val->test -- must be at least the embargo. not one of them.
    both. the train->val one is the new one, and it is the one people forget.
    """
    ts = _minute_index()
    tr, va, te, info = three_way_split(ts, val_fraction=0.15, test_fraction=0.20,
                                       embargo_sessions=C.EMBARGO_SESSIONS)
    train_end = ts[tr].max()
    val_start, val_end = ts[va].min(), ts[va].max()
    test_start = ts[te].min()

    assert train_end < val_start < val_end < test_start, "the slices are out of time order"
    # 25 TRADING SESSIONS, which because of the weekends spans MORE than 25 calendar days.
    # this used to assert >= 21 CALENDAR days, and 21 calendar days is only ~14 sessions --
    # the bug this whole change exists to kill. see purged_cv.embargo_end.
    assert (val_start - train_end) > pd.Timedelta(days=C.EMBARGO_SESSIONS), (
        "no embargo between TRAIN and VALIDATION. the optimiser would be tuning on rows that "
        "share a 20-SESSION window with the training set."
    )
    assert (test_start - val_end) > pd.Timedelta(days=C.EMBARGO_SESSIONS), (
        "no embargo between VALIDATION and TEST."
    )


def test_three_way_split_refuses_rather_than_returning_an_empty_validation_set():
    """an empty val set does not crash -- it makes every trial TIE.

    trading_cost of an empty slice is 0.0 for everybody. the optimiser would then rank thirty
    identical zeros and hand back whichever it saw first. so the split raises instead.
    """
    ts = _minute_index(days=200)                    # ~10 months
    with pytest.raises(ValueError, match="EMPTY|empty"):
        # a 1% val slice of 10 months is ~3 days, and the 25-session embargo eats all of it
        three_way_split(ts, val_fraction=0.01, test_fraction=0.20, embargo_sessions=C.EMBARGO_SESSIONS)


def test_the_split_costs_about_two_embargoes_and_no_more():
    """the price of honesty, measured. two 25-session embargoes out of 5.5 years is ~4% of the rows.

    if this ever balloons, someone has widened EMBARGO_SESSIONS without thinking about the fact that
    there are now TWO of them.
    """
    ts = _minute_index()
    tr, va, te, info = three_way_split(ts, 0.15, 0.20, C.EMBARGO_SESSIONS)
    lost = info["n_embargoed"] / len(ts)
    assert 0.01 < lost < 0.06, f"the embargoes are eating {lost:.1%} of the data"


# =====================================================================================
# 5. THE OBJECTIVE VALUE ITSELF
# =====================================================================================
CLASSES = ["ENTRY_SMALL", "ENTRY_SUB", "ENTRY_SUPER", "EXIT_SMALL", "EXIT_SUB",
           "EXIT_SUPER", "NO_TRADE"]


def test_trading_cost_is_zero_for_a_perfect_model_and_positive_otherwise():
    y = np.array([0, 1, 2, 3, 4, 5, 6] * 10)
    assert trading_cost(y, y, CLASSES) == 0.0
    wrong = np.roll(y, 1)
    assert trading_cost(y, wrong, CLASSES) > 0.0


def test_trading_cost_punishes_a_full_reversal_far_harder_than_a_wrong_size():
    """the metric has to know the difference, or optimising it is pointless.

    ENTRY_SUPER -> EXIT_SUPER is severity 100: the biggest possible position in exactly the
    wrong direction. ENTRY_SUPER -> ENTRY_SMALL is severity 3: right call, too small.
    same number of mistakes, same rate. the cost must not be the same.
    """
    y = np.full(100, CLASSES.index("ENTRY_SUPER"))
    reversal = np.full(100, CLASSES.index("EXIT_SUPER"))
    undersize = np.full(100, CLASSES.index("ENTRY_SMALL"))
    assert trading_cost(y, reversal, CLASSES) > 20 * trading_cost(y, undersize, CLASSES)


def test_trading_cost_cannot_be_gamed_by_ignoring_the_rare_class():
    """WHY IT IS SAFE TO OPTIMISE THIS AND NOT ACCURACY.

    rank_mistakes computes rate = count / n_true, PER TRUE CLASS. so the 1.2% ENTRY_SUB carries
    exactly the same weight in the sum as the 53% NO_TRADE. a model that abandons the rare class
    to buy accuracy elsewhere gets no discount at all.

    worked example below: 1000 rows, 990 NO_TRADE and 10 ENTRY_SUPER. model A gets every
    NO_TRADE right and every ENTRY_SUPER wrong -- 99% "accurate", and useless. it must score
    WORSE than model B, which trades a handful of NO_TRADEs by mistake but finds the entries.
    """
    nt, es = CLASSES.index("NO_TRADE"), CLASSES.index("ENTRY_SUPER")
    y = np.array([nt] * 990 + [es] * 10)

    a = np.array([nt] * 990 + [nt] * 10)                 # 99.0% accurate, blind to the signal
    b = np.array([nt] * 950 + [es] * 40 + [es] * 10)     # 96.0% accurate, finds every signal

    cost_a = trading_cost(y, a, CLASSES)
    cost_b = trading_cost(y, b, CLASSES)
    assert cost_a > cost_b, (
        f"the blind-but-accurate model scored better ({cost_a} vs {cost_b}). if that is ever "
        f"true, trading_cost is not safe to optimise and the HPO will breed a model that never "
        f"trades."
    )


# =====================================================================================
# 6. THE RANDOM FOREST DEPTH BUG -- the thing that started all this
# =====================================================================================
def test_max_depth_zero_means_fully_grown_for_the_forest_and_six_for_the_boosters():
    """THE BUG: train.py's argparse has ONE --max_depth, default 6, for all three models.

    6 is a boosting default. a boosted model adds trees that correct each other, so each tree
    only has to be a little better than nothing -- shallow is right. a random forest averages
    INDEPENDENT trees, and averaging cancels variance but does nothing to bias. so it wants
    deep, low-bias trees. cap them at 6 and every tree is biased the same way, the averaging
    cancels nothing, and 300 trees are 300 copies of one stump.

    sklearn's own default is max_depth=None. build_model already reads
        random_forest: params["max_depth"] or None
        xgboost:       params["max_depth"] or 6
        catboost:      params["max_depth"] or 6
    so 0 already means "fully grown" for the forest and "6" for the boosters. the fix is one
    character: the argparse DEFAULT goes from 6 to 0.

    this test pins that reading of build_model, so nobody "tidies up" the `or None`.
    """
    from trainer.train import build_model
    p = {"n_estimators": 5, "max_depth": 0, "min_samples_leaf": 5, "learning_rate": 0.1,
         "seed": 1}
    rf = build_model("random_forest", 7, p)
    assert rf.max_depth is None, "max_depth=0 must mean FULLY GROWN for a random forest"

    xgb = build_model("xgboost", 7, p)
    assert xgb.max_depth == 6, "max_depth=0 must fall back to 6 for a booster, not to unlimited"

    rf6 = build_model("random_forest", 7, {**p, "max_depth": 6})
    assert rf6.max_depth == 6, "an explicit depth must still be honoured"


def test_the_forest_search_space_actually_searches_deep_trees():
    """if the forest's space topped out at 10 we would have moved the bug, not fixed it."""
    depths = next(p for p in search_space("random_forest") if p.name == "Args/max_depth")
    values = [d["Args/max_depth"] for d in depths.to_list()]
    assert 0 in values, "the forest must be allowed to grow fully (0 = no limit)"
    assert max(values) >= 24, f"the deepest forest the search can try is {max(values)}. too shallow."


def test_catboost_depth_stays_under_the_librarys_hard_cap():
    """catboost REFUSES depth > 16: 'Maximum tree depth is 16'. a trial at 20 would just die.

    (and its trees are OBLIVIOUS -- the same split on every node of a level -- so its depth is
    not comparable to xgboost's anyway. 4-10 is the useful band.)
    """
    depths = next(p for p in search_space("catboost") if p.name == "Args/max_depth")
    values = [d["Args/max_depth"] for d in depths.to_list()]
    assert max(values) <= 16, f"catboost cannot build a tree of depth {max(values)}"


def test_every_search_space_pins_the_seed_free_knobs_only():
    """the seed must NOT be searched.

    if the seed were a hyper-parameter, the search would find the LUCKY seed -- the one whose
    random state happens to suit the validation slice -- and we would ship it. that is not a
    better model, it is a better coin flip, and it will not repeat live.
    """
    for model_type in ("random_forest", "xgboost", "catboost"):
        names = [p.name for p in search_space(model_type)]
        assert "Args/seed" not in names, f"{model_type} is searching the random seed"
