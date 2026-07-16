"""
trainer/shap_logic.py -- the SHAP maths, kept separate so it can be TESTED without ClearML.

TWO THINGS THIS FILE EXISTS TO GET RIGHT
--------------------------------------------------------------------------------------
1. EACH LIBRARY RETURNS A DIFFERENT SHAPE. measured, not guessed:

       random forest   shap.TreeExplainer(m)(X)          -> (rows, features, classes)   OK
       xgboost         shap.TreeExplainer(m)(X)          -> (rows, features, classes)   OK
       catboost        get_feature_importance(ShapValues)-> (rows, CLASSES, FEATURES+1)  <-- DIFFERENT!
                       the axes are SWAPPED and the base value is hidden in the last column.

   feed catboost's raw output into code written for xgboost's shape and it will not crash.
   it will quietly read the WRONG CLASS's numbers and hand you a confident, wrong answer about
   which features drive your losses. so we normalise all three to (rows, features, classes).

2. THEY DO NOT SPEAK THE SAME LANGUAGE.

       random forest   SHAP values are in PROBABILITY  (0 .. 1)
       xgboost         SHAP values are in LOG-ODDS     (-3.86, +0.22, ...)
       catboost        SHAP values are in LOG-ODDS

   so a raw SHAP number from random forest and one from xgboost are in different units.
   comparing them directly would just tell you which model happens to sit on a bigger scale.
   when we compare feature importance ACROSS models we therefore normalise each model to its
   own total first, and compare SHARES, not raw sizes.

THE RANKING IDEA (the user's own, and it is a good one)
--------------------------------------------------------------------------------------
       importance = error_rate x severity

   how OFTEN the mistake happens, times how much it COSTS. a catastrophic mistake that happens
   rarely still outranks a harmless one that happens constantly -- which is the whole point,
   because the rare catastrophic one is the one that empties the account.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# which space each library's SHAP values live in. needed when comparing models.
SHAP_SPACE = {
    "random_forest": "probability",
    "xgboost": "log_odds",
    "catboost": "log_odds",
}


def compute_shap(model, X: pd.DataFrame, model_type: str,
                 cat_features=None) -> tuple[np.ndarray, np.ndarray]:
    """return (values, base) with values ALWAYS shaped (rows, features, classes).

    every library is bent into the same shape here, so no code downstream has to remember
    which one it is talking to.

    cat_features: the TEXT columns, needed only by catboost. if the model was trained with
    cat_features then the Pool we build for SHAP MUST declare them too, or catboost tries to
    turn 'GAP_UP' into a float and dies with
        "Bad value for num_feature ... Cannot convert 'GAP_DOWN' to float"
    it does not fail quietly, but it fails on the very first real run, so it is passed here.
    """
    if model_type == "catboost":
        from catboost import Pool
        # ask the fitted model itself which columns it treated as categorical. that cannot
        # drift out of step with the model, whereas a list passed in by a caller can.
        idx = list(getattr(model, "get_cat_feature_indices", lambda: [])() or [])
        if not idx and cat_features:
            idx = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

        pool = Pool(X, cat_features=idx or None)
        # catboost's own SHAP is faster and better tested than TreeExplainer on catboost.
        raw = np.asarray(model.get_feature_importance(pool, type="ShapValues"))
        # raw is (rows, classes, features+1). the last column is the base value.
        vals = raw[:, :, :-1].transpose(0, 2, 1)     # -> (rows, features, classes)
        base = raw[0, :, -1]                          # one base value per class
        return vals, np.asarray(base)

    import shap
    expl = shap.TreeExplainer(model)(X)
    vals = np.asarray(expl.values)
    base = np.asarray(expl.base_values)
    if base.ndim > 1:
        base = base[0]
    if vals.ndim != 3:
        raise ValueError(f"expected a 3D shap array (rows, features, classes) for {model_type}, "
                         f"got {vals.shape}. refusing to guess -- the numbers would be wrong.")
    return vals, base


def rank_mistakes(y_true: np.ndarray, y_pred: np.ndarray, classes: list,
                  severity: dict, default_sev: float = 1.0) -> pd.DataFrame:
    """rank every kind of mistake by  rate x severity.

    rate     = of all the minutes that were REALLY class A, what fraction did the model call B?
    severity = what that particular confusion COSTS, from configs/severity_7class.json
    """
    rows = []
    for ti, tname in enumerate(classes):
        n_true = int((y_true == ti).sum())
        if n_true == 0:
            continue                                  # this class never appears -- skip it
        for pi, pname in enumerate(classes):
            if pi == ti:
                continue                              # a correct call is not a mistake
            cnt = int(((y_true == ti) & (y_pred == pi)).sum())
            if cnt == 0:
                continue                              # this mistake never happened
            rate = cnt / n_true
            sev = float(severity.get(f"{tname}->{pname}", default_sev))
            rows.append({
                "true": tname, "pred": pname, "count": cnt,
                "of_true_%": round(100 * rate, 2),
                "severity": sev,
                "importance": round(rate * sev, 4),   # how often x how much it hurts
            })
    if not rows:
        return pd.DataFrame(columns=["true", "pred", "count", "of_true_%",
                                     "severity", "importance"])
    return (pd.DataFrame(rows)
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def worst_case_mistakes(rank: pd.DataFrame, min_severity: float = 50.0) -> pd.DataFrame:
    """the mistakes that could BLOW UP THE ACCOUNT, however rarely they happen.

    WHY THIS EXISTS, AND IT IS NOT OBVIOUS
        importance = rate x severity measures EXPECTED TOTAL DAMAGE, and that is the right
        thing to minimise on average. but it can bury a catastrophe. worked example:

            2 full reversals out of 100   ->  0.02 x 100 = 2.0
            40 unwanted small trades /100 ->  0.40 x   8 = 3.2   <-- ranks HIGHER

        by total money bled, the 40 nuisances really are worse, so the ranking is not lying.
        but a full reversal at maximum size is the kind of mistake that ends a trading book,
        and it must never be invisible just because it is rare.

    so we always ALSO list every high-severity mistake that happened at all, no matter how
    seldom. the first ranking says "where is the money going". this one says "what could kill us".
    """
    if rank.empty:
        return rank
    out = rank[rank["severity"] >= min_severity].copy()
    return out.sort_values(["severity", "count"], ascending=False).reset_index(drop=True)


def feature_shares(vals: np.ndarray, features: list, cls: int | None = None) -> pd.DataFrame:
    """which features carry the weight -- as a SHARE of the total, not a raw size.

    using shares is what lets us compare a random forest (probability space) against an
    xgboost (log-odds space) without the units making a nonsense of it.
    """
    v = np.abs(vals[:, :, cls]) if cls is not None else np.abs(vals).sum(axis=2)
    mean_abs = v.mean(axis=0)
    total = mean_abs.sum()
    share = mean_abs / total if total > 0 else mean_abs
    return (pd.DataFrame({"feature": features,
                          "mean_abs_shap": mean_abs,
                          "share_%": (share * 100).round(2)})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


def stable_feature_shares(model, X: pd.DataFrame, features: list, model_type: str,
                          cls: int | None = None, n_boot: int = 5, frac: float = 0.7,
                          seed: int = 42, cat_features=None) -> pd.DataFrame:
    """which features matter -- AND HOW SURE ARE WE?

    WHY THIS EXISTS
        one SHAP run gives you one ranking, and a ranking always LOOKS confident. but it was
        computed on a sample, so some of it is luck. measured on synthetic data: the TOP feature
        is 100% stable across different samples, but ranks 4-5 move around ~5% of the time.
        without a stability number you cannot tell which is which, and you might go and re-engineer
        a feature that only ranked 4th by chance.

    HOW
        recompute the shares on several overlapping subsamples, and report:
            share_%   the average importance
            +/-       how much it wobbled between runs (std). small = trust it.
            top5_hits how many of the runs put it in the top 5. 5/5 = solid. 2/5 = noise.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    k = max(30, int(n * frac))
    runs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=min(k, n), replace=False)
        vals, _ = compute_shap(model, X.iloc[idx], model_type, cat_features=cat_features)
        runs.append(feature_shares(vals, features, cls).set_index("feature")["share_%"])

    mat = pd.concat(runs, axis=1)
    top5 = [set(r.nlargest(5).index) for r in runs]
    out = pd.DataFrame({
        "feature": mat.index,
        "share_%": mat.mean(axis=1).round(2).values,
        "wobble_+/-": mat.std(axis=1).round(2).values,      # small = the number is trustworthy
        "top5_hits": [sum(f in s for s in top5) for f in mat.index],
        "runs": n_boot,
    })
    # a feature is TRUSTED if it landed in the top 5 in every single run
    out["verdict"] = np.where(out["top5_hits"] == n_boot, "solid",
                     np.where(out["top5_hits"] >= n_boot * 0.6, "probable", "noise"))
    return out.sort_values("share_%", ascending=False).reset_index(drop=True)


def explain_one_row(vals: np.ndarray, features: list, row: int, cls: int, top: int = 3) -> str:
    """the few features that pushed one particular prediction, biggest push first."""
    sv = vals[row, :, cls]
    order = np.argsort(-np.abs(sv))[:top]
    return ", ".join(f"{features[j]}({sv[j]:+.3f})" for j in order)


def sample_for_shap(y_true: np.ndarray, pair: tuple[int, int], n: int,
                    seed: int = 42) -> np.ndarray:
    """pick rows to explain.

    SHAP is slow -- computing it for 500,000 rows x 7 classes would take hours and tell us
    nothing extra. we only need the rows of the TWO classes involved in the worst mistake.
    a few hundred of each is plenty to see the pattern.
    """
    rng = np.random.default_rng(seed)
    picks = []
    for cls in pair:
        idx = np.where(y_true == cls)[0]
        if len(idx):
            picks.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return np.concatenate(picks) if picks else np.array([], dtype=int)


def cross_model_agreement(shares: dict[str, pd.DataFrame], top: int = 15) -> pd.DataFrame:
    """do the three models agree on WHICH features matter?

    this is worth more than any single model's ranking. a feature all three lean on is
    probably real. a feature only one of them likes is probably that model chasing noise --
    and it is the first thing to be suspicious of.
    """
    all_feats = sorted({f for d in shares.values() for f in d["feature"]})
    out = {"feature": all_feats}
    for m, d in shares.items():
        lookup = dict(zip(d["feature"], d["share_%"]))
        out[f"{m}_share_%"] = [lookup.get(f, 0.0) for f in all_feats]
    df = pd.DataFrame(out)
    share_cols = [c for c in df.columns if c.endswith("_share_%")]
    df["mean_share_%"] = df[share_cols].mean(axis=1).round(2)
    # how many models put this feature in their own top-N
    tops = {m: set(d.nlargest(top, "share_%")["feature"]) for m, d in shares.items()}
    df["in_top_of"] = [sum(f in s for s in tops.values()) for f in all_feats]
    return df.sort_values("mean_share_%", ascending=False).reset_index(drop=True)
