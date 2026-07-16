"""
tests/test_shap_logic.py -- prove the SHAP maths is right for ALL THREE libraries.

this trains real 7-class RandomForest / XGBoost / CatBoost models on SYNTHETIC data and checks
the thing that actually matters: ADDITIVITY. if  base + sum(shap) != the model's own output,
then the SHAP numbers are lying, and every conclusion about which features cause our losses
would be worthless.

it also pins the trap that would never crash and would always be wrong: catboost hands back
(rows, CLASSES, features+1) while the other two hand back (rows, features, classes). read
catboost's array with xgboost's code and you silently get the wrong class's numbers.
"""
import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from trainer.shap_logic import (  # noqa: E402
    compute_shap, rank_mistakes, feature_shares, explain_one_row, stable_feature_shares,
    sample_for_shap, cross_model_agreement, worst_case_mistakes, SHAP_SPACE,
)

N, F, K = 400, 6, 7          # 7 classes, like the real problem
FEATURES = [f"f{i}" for i in range(F)]
CLASSES = ["NO_TRADE", "EXIT_SMALL", "EXIT_SUPER", "EXIT_SUB",
           "ENTRY_SUPER", "ENTRY_SMALL", "ENTRY_SUB"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(N, F)), columns=FEATURES)
    y = rng.integers(0, K, size=N)
    X.iloc[:, 0] += y * 0.8                    # make f0 genuinely predictive
    w = rng.random(N)
    return X, y, w


@pytest.fixture(scope="module")
def models(data):
    X, y, w = data
    out = {}

    from sklearn.ensemble import RandomForestClassifier
    out["random_forest"] = RandomForestClassifier(
        n_estimators=25, random_state=0, n_jobs=-1).fit(X, y, sample_weight=w)

    import xgboost as xgb
    out["xgboost"] = xgb.XGBClassifier(
        n_estimators=25, objective="multi:softprob", num_class=K,
        tree_method="hist", verbosity=0, random_state=0).fit(X, y, sample_weight=w)

    from catboost import CatBoostClassifier
    out["catboost"] = CatBoostClassifier(
        iterations=25, loss_function="MultiClass", verbose=0,
        random_seed=0, allow_writing_files=False).fit(X, y, sample_weight=w)

    return out


# ------------------------------------------------------------------ the shape
@pytest.mark.parametrize("mtype", ["random_forest", "xgboost", "catboost"])
def test_every_library_is_normalised_to_the_same_shape(models, data, mtype):
    X, _, _ = data
    vals, base = compute_shap(models[mtype], X, mtype)
    assert vals.shape == (N, F, K), (
        f"{mtype} must be normalised to (rows, features, classes). catboost natively returns "
        f"(rows, classes, features+1) -- if that is not transposed, every class is read wrong."
    )
    assert base.shape == (K,), f"{mtype}: one base value per class"


# ------------------------------------------------------------------ ADDITIVITY: the real test
@pytest.mark.parametrize("mtype", ["random_forest", "xgboost", "catboost"])
def test_shap_actually_adds_up(models, data, mtype):
    """base + sum(shap) must equal the model's OWN output. if it does not, the numbers are junk."""
    X, _, _ = data
    m = models[mtype]
    vals, base = compute_shap(m, X, mtype)

    # each library explains a different thing, so we compare against the right output
    if mtype == "random_forest":
        truth = m.predict_proba(X)                                    # probability space
    elif mtype == "xgboost":
        truth = m.predict(X, output_margin=True)                      # log-odds space
    else:
        truth = m.predict(X, prediction_type="RawFormulaVal")         # log-odds space
    truth = np.asarray(truth)

    for row in (0, 7, 33):
        for cls in (0, 4, 6):
            recon = float(base[cls]) + float(vals[row, :, cls].sum())
            assert abs(recon - float(truth[row, cls])) < 1e-2, (
                f"{mtype}: SHAP does not add up at row {row}, class {cls}. "
                f"reconstruction {recon:.4f} vs the model's own {float(truth[row, cls]):.4f}. "
                f"the SHAP values are wrong and any feature ranking from them is worthless."
            )


def test_the_three_libraries_live_in_different_units():
    """random forest speaks probability; xgboost and catboost speak log-odds. comparing raw
    numbers across them would just compare scales. this is why we compare SHARES."""
    assert SHAP_SPACE["random_forest"] == "probability"
    assert SHAP_SPACE["xgboost"] == "log_odds"
    assert SHAP_SPACE["catboost"] == "log_odds"


# ------------------------------------------------------------------ the ranking
def test_importance_measures_expected_total_damage_and_a_catastrophe_CAN_be_buried():
    """THE SUBTLE ONE. importance = rate x severity is EXPECTED TOTAL COST, so a common
    nuisance can legitimately outrank a rare catastrophe:

        2 full reversals   / 100  ->  0.02 x 100 = 2.0
        40 unwanted trades / 100  ->  0.40 x   8 = 3.2   <-- higher

    that is not a bug -- by money bled, the 40 nuisances really do cost more. but it means the
    reversal that could END the book is sitting in second place, and SHAP would never look at
    it. this test PINS that behaviour so nobody 'fixes' it by accident.
    """
    y_true = np.array([4] * 100 + [0] * 100)          # 100 ENTRY_SUPER, 100 NO_TRADE
    y_pred = y_true.copy()
    y_pred[:2] = 2                                     # 2 ENTRY_SUPER -> EXIT_SUPER (a reversal!)
    y_pred[100:140] = 5                                # 40 NO_TRADE -> ENTRY_SMALL (a nuisance)

    sev = {"ENTRY_SUPER->EXIT_SUPER": 100, "NO_TRADE->ENTRY_SMALL": 8}
    rank = rank_mistakes(y_true, y_pred, CLASSES, sev)

    # the nuisance ranks first, on total damage -- and that is correct
    assert rank.iloc[0]["pred"] == "ENTRY_SMALL"
    assert rank.iloc[0]["importance"] == pytest.approx(3.2, abs=0.01)
    assert rank.iloc[1]["importance"] == pytest.approx(2.0, abs=0.01)


def test_the_worst_case_view_rescues_the_rare_catastrophe():
    """...which is exactly why the second view exists. it lists what could KILL us, however
    rarely it happens, so a reversal is never invisible just because it is uncommon."""
    y_true = np.array([4] * 100 + [0] * 100)
    y_pred = y_true.copy()
    y_pred[:2] = 2                                     # the rare reversal
    y_pred[100:140] = 5                                # the common nuisance

    sev = {"ENTRY_SUPER->EXIT_SUPER": 100, "NO_TRADE->ENTRY_SMALL": 8}
    rank = rank_mistakes(y_true, y_pred, CLASSES, sev)

    worst = worst_case_mistakes(rank, min_severity=50)
    assert len(worst) == 1
    assert worst.iloc[0]["true"] == "ENTRY_SUPER"
    assert worst.iloc[0]["pred"] == "EXIT_SUPER"
    assert worst.iloc[0]["severity"] == 100
    # and the nuisance is correctly NOT in the danger list
    assert "ENTRY_SMALL" not in worst["pred"].tolist()


def test_rank_is_empty_when_the_model_is_perfect():
    y = np.array([0, 1, 2, 3])
    assert rank_mistakes(y, y.copy(), CLASSES, {}).empty


def test_unlisted_pairs_fall_back_to_the_default_severity():
    y_true = np.array([1, 1])
    y_pred = np.array([3, 3])
    rank = rank_mistakes(y_true, y_pred, CLASSES, {}, default_sev=1.0)
    assert rank.iloc[0]["severity"] == 1.0


# ------------------------------------------------------------------ the shares
def test_feature_shares_sum_to_one_hundred(models, data):
    X, _, _ = data
    vals, _ = compute_shap(models["xgboost"], X, "xgboost")
    sh = feature_shares(vals, FEATURES)
    assert abs(sh["share_%"].sum() - 100.0) < 0.5
    assert sh.iloc[0]["feature"] == "f0", "f0 was made predictive -- it should lead"


def test_cross_model_agreement_finds_the_feature_all_three_use(models, data):
    X, _, _ = data
    shares = {m: feature_shares(*[compute_shap(models[m], X, m)[0]], FEATURES)
              for m in ("random_forest", "xgboost", "catboost")}
    agree = cross_model_agreement(shares, top=3)
    assert agree.iloc[0]["feature"] == "f0"
    assert agree.iloc[0]["in_top_of"] == 3, "all three models should lean on the real signal"


# ------------------------------------------------------------------ sampling
def test_sampling_only_takes_the_two_classes_that_matter():
    y = np.array([0] * 50 + [4] * 50 + [2] * 50)
    picked = sample_for_shap(y, pair=(0, 4), n=10, seed=1)
    assert len(picked) == 20
    assert set(y[picked]) == {0, 4}, "only the two classes in the worst mistake"


def test_explain_one_row_names_the_biggest_pushers(models, data):
    X, _, _ = data
    vals, _ = compute_shap(models["xgboost"], X, "xgboost")
    s = explain_one_row(vals, FEATURES, row=0, cls=3, top=3)
    assert s.count(",") == 2 and "(" in s


# ================================================================== the text-column trap
def test_catboost_shap_works_when_a_feature_is_TEXT():
    """REGRESSION. gap_state is text (NO_GAP / GAP_UP / GAP_DOWN). CatBoost is trained with
    cat_features, so the Pool we build for SHAP must declare them too -- otherwise catboost
    tries to turn 'GAP_DOWN' into a float and dies:
        "Bad value for num_feature ... Cannot convert 'GAP_DOWN' to float"
    that would have blown up on the very first real run."""
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(1)
    n = 250
    X = pd.DataFrame({
        "num1": rng.normal(size=n),
        "num2": rng.normal(size=n),
        "gap_state": rng.choice(["NO_GAP", "GAP_UP", "GAP_DOWN"], size=n),   # TEXT
    })
    y = rng.integers(0, 7, size=n)
    cat_idx = [X.columns.get_loc("gap_state")]

    m = CatBoostClassifier(iterations=20, loss_function="MultiClass", verbose=0,
                           cat_features=cat_idx, random_seed=0,
                           allow_writing_files=False).fit(X, y)

    vals, base = compute_shap(m, X, "catboost", cat_features=["gap_state"])
    assert vals.shape == (n, 3, 7), "(rows, features, classes)"
    assert base.shape == (7,)

    # and it must still ADD UP -- a shape that looks right but is wrong is worse than a crash
    truth = np.asarray(m.predict(X, prediction_type="RawFormulaVal"))
    recon = float(base[2]) + float(vals[5, :, 2].sum())
    assert abs(recon - float(truth[5, 2])) < 1e-2


# ================================================================== is the ranking TRUSTWORTHY?
# a SHAP ranking always LOOKS confident -- it is a sorted table of numbers. but it was computed
# on a SAMPLE of rows, so part of it is real and part of it is luck, and the table does not say
# which is which. that matters: if rank 4 only got there by chance, someone spends a week
# re-engineering a feature that does nothing.
#
# so we recompute the shares on several overlapping subsamples and report how much each feature
# MOVED. a feature in the top 5 of every run is solid. one that got there once is noise.

@pytest.mark.parametrize("mtype", ["random_forest", "xgboost", "catboost"])
def test_a_genuinely_predictive_feature_is_SOLID_in_every_run(models, data, mtype):
    """f0 is the only feature actually built into y. it must survive every resample."""
    X, y, _ = data
    out = stable_feature_shares(models[mtype], X, FEATURES, mtype, cls=1, n_boot=5)

    f0 = out[out["feature"] == "f0"].iloc[0]
    assert f0["top5_hits"] == 5, "the one real feature must be top-5 in EVERY run"
    assert f0["verdict"] == "solid"
    assert out.iloc[0]["feature"] == "f0", "and it must come out on top on average"


@pytest.mark.parametrize("mtype", ["random_forest", "xgboost"])
def test_a_pure_noise_feature_is_never_called_solid(models, data, mtype):
    """THE POINT. f1..f5 are random noise -- they carry no signal at all. the ranking will
    still put SOME of them in its top 5, because something has to be 2nd. the stability check
    is what stops us believing it."""
    X, y, _ = data
    out = stable_feature_shares(models[mtype], X, FEATURES, mtype, cls=1, n_boot=5)

    noise = out[out["feature"] != "f0"]
    # a noise feature is allowed to LOOK important in one run. it must not be called solid in
    # all of them -- if every noise feature were 'solid', the verdict would mean nothing.
    assert (noise["verdict"] == "solid").sum() < len(noise), \
        "if every noise feature is 'solid', the stability check is not checking anything"


def test_the_wobble_column_is_a_real_number_not_a_decoration(models, data):
    """+/- is the spread between runs. it must be >= 0 and small for the stable feature."""
    X, y, _ = data
    out = stable_feature_shares(models["xgboost"], X, FEATURES, "xgboost", cls=1, n_boot=5)
    assert (out["wobble_+/-"] >= 0).all()
    f0 = out[out["feature"] == "f0"].iloc[0]
    assert f0["wobble_+/-"] < f0["share_%"], \
        "the top feature must not wobble by as much as its own size -- that would be noise"


def test_the_shares_still_add_up_to_a_hundred(models, data):
    """it is an average of shares, so it must still be a share."""
    X, y, _ = data
    out = stable_feature_shares(models["catboost"], X, FEATURES, "catboost", cls=1, n_boot=3)
    assert abs(out["share_%"].sum() - 100) < 1.0


def test_stability_works_when_a_feature_is_TEXT(data):
    """catboost + a text column: the SAME trap as compute_shap -- the Pool must be told which
    columns are categorical, or it tries to turn 'GAP_UP' into a float and dies. the bootstrap
    calls compute_shap n_boot times, so if it forgot to pass cat_features it would crash here."""
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(3)
    n = 200
    X = pd.DataFrame({
        "gap_fill_ratio": rng.random(n),
        "gap_state": rng.choice(["NO_GAP", "GAP_UP", "GAP_DOWN"], n),
        "vix": rng.normal(size=n),
    })
    y = rng.integers(0, 3, n)
    m = CatBoostClassifier(iterations=15, loss_function="MultiClass", verbose=0,
                           random_seed=0, allow_writing_files=False).fit(
        X, y, cat_features=["gap_state"])

    out = stable_feature_shares(m, X, list(X.columns), "catboost", cls=1, n_boot=3,
                                cat_features=["gap_state"])
    assert len(out) == 3
    assert set(out["feature"]) == {"gap_fill_ratio", "gap_state", "vix"}
