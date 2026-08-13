"""scripts/tree_scales.py -- read the REAL gain, cover and gradient scale out of a trained xgboost
model, so gamma / reg_lambda / reg_alpha are chosen against measured numbers instead of guessed.

THE THREE KNOBS AND WHAT EACH ONE IS COMPARED AGAINST

    gamma       compared against GAIN.   a split is kept only if gain > gamma.
    reg_lambda  compared against COVER.  it sits in the denominator: G/(H + lambda).
    reg_alpha   compared against |G|.    it soft-thresholds the numerator: |G| - alpha.

    all three are ABSOLUTE numbers in the model's own units, and those units come from the
    sample weights. that is why a value copied from a tutorial means nothing here: our weights
    run from 0.10 (NO_TRADE) to 55.03 (ENTRY_SUB), so our gains and covers are on a scale that
    tutorial has never seen.

WHERE THE NUMBERS COME FROM
    booster.get_dump(with_stats=True) prints, for every node xgboost actually built:
        split node   gain=<loss reduction>   cover=<sum of hessians>
        leaf node    leaf=<value>            cover=<sum of hessians>

    cover IS the hessian sum -- that is H, measured, not derived.
    the gradient comes back out of the leaf value, because xgboost solved for it:

        leaf = -eta * G / (H + lambda)      ->      G = -leaf * (H + lambda) / eta

    so a model that has already been trained tells us its own G and H distribution. nothing is
    simulated and nothing is assumed about the data.

    NOTE ON eta: the leaf value in the dump is ALREADY multiplied by the learning rate, so it has
    to be divided back out or every gradient comes back 50x too small at eta=0.02.

RUN
    final_venv/bin/python scripts/tree_scales.py --bundle <model_xgboost.joblib>
    final_venv/bin/python scripts/tree_scales.py --model_task_id <training task id>
"""
import argparse
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C                                       # noqa: E402

SPLIT = re.compile(r"gain=([-\d.e+]+),cover=([-\d.e+]+)")
LEAF = re.compile(r"leaf=([-\d.e+]+),cover=([-\d.e+]+)")
PCTS = [1, 5, 25, 50, 75, 95, 99]


def pct_line(name: str, arr: np.ndarray, unit: str = "") -> str:
    if not len(arr):
        return f"  {name:<12} (none)"
    q = np.percentile(arr, PCTS)
    return (f"  {name:<12}" + "".join(f"{v:>12,.3f}" for v in q)
            + f"{arr.mean():>13,.3f}{unit}")


def collect(booster, max_trees: int, eta: float, lam: float):
    """parse a SAMPLE of trees. sampled by SLICING the booster, not by dumping all 42,000 at once
    -- a full dump of a 6000-round 7-class model is millions of nodes and several GB of strings.

    THE SLICE IS BY ROUND, NOT BY TREE. booster[i:i+1] takes one boosting LAYER, and a 7-class
    model puts SEVEN trees in every layer (one per class). so 6,000 rounds is 42,000 trees, and
    indexing past 6,000 raises 'Layer index out of range'.
    """
    rounds = booster.num_boosted_rounds()
    per_round = max(1, len(booster.get_dump()) // max(rounds, 1))
    want_rounds = max(1, (max_trees // per_round)) if max_trees else rounds
    step = max(1, rounds // want_rounds)
    idx = list(range(0, rounds, step))[:want_rounds]
    gains, covers, leaves, leaf_cov = [], [], [], []
    for i in idx:
        for dump in booster[i:i + 1].get_dump(with_stats=True):
          for line in dump.splitlines():
            m = SPLIT.search(line)
            if m:
                gains.append(float(m.group(1)))
                covers.append(float(m.group(2)))
                continue
            m = LEAF.search(line)
            if m:
                leaves.append(float(m.group(1)))
                leaf_cov.append(float(m.group(2)))
    g, c = np.array(gains), np.array(covers)
    lv, lc = np.array(leaves), np.array(leaf_cov)
    total = len(booster.get_dump())
    idx = idx * per_round        # report TREES sampled, not rounds
    # THE GRADIENT, RECOVERED. xgboost solved leaf = -eta*G/(H+lambda) when it built the tree,
    # so inverting it gives the G that was actually there. eta divided back out (see the header).
    grad = np.abs(lv) * (lc + lam) / max(eta, 1e-12)
    return dict(n_trees=total, sampled=len(idx), gain=g, cover=c,
                leaf=lv, leaf_cover=lc, grad=grad)


def report(d: dict, eta: float, lam: float, alpha: float, gamma: float, mcw: float):
    head = "  " + " " * 12 + "".join(f"{f'p{p}':>12}" for p in PCTS) + f"{'mean':>13}"
    print(f"\n  {d['n_trees']:,} trees in the model, {d['sampled']:,} sampled, "
          f"{len(d['gain']):,} splits + {len(d['leaf']):,} leaves read")
    print(f"  the model was built with  eta={eta}  lambda={lam}  alpha={alpha}  "
          f"gamma={gamma}  min_child_weight={mcw}")

    print(f"\n  ---- THE THREE SCALES ----------------------------------------------------")
    print(head)
    print(pct_line("GAIN", d["gain"]))
    print(pct_line("COVER (=H)", d["cover"]))
    print(pct_line("|G|", d["grad"]))
    print(pct_line("leaf value", d["leaf"]))

    print(f"\n  ---- GAMMA: a split needs gain > gamma to survive -------------------------")
    print(f"     gamma     splits it would have pruned")
    for gval in (0, 0.1, 0.3, 1, 3, 10, 30, 100, 300):
        pruned = float((d["gain"] < gval).mean()) * 100
        mark = "   <- current" if abs(gval - gamma) < 1e-9 else ""
        print(f"     {gval:>7}     {pruned:>5.1f}%{mark}")

    print(f"\n  ---- REG_LAMBDA: it shrinks the leaf by H/(H+lambda) ----------------------")
    print(f"     lambda" + "".join(f"{f'p{p}':>10}" for p in PCTS) + "     (% of the leaf kept)")
    for lval in (0.1, 1, 5, 10, 50, 200):
        keep = [np.percentile(d["leaf_cover"], p) for p in PCTS]
        row = "".join(f"{h/(h+lval)*100:>9.1f}%" for h in keep)
        mark = "  <- current" if abs(lval - lam) < 1e-9 else ""
        print(f"     {lval:>6}{row}{mark}")

    print(f"\n  ---- REG_ALPHA: it zeroes any leaf whose |G| < alpha ----------------------")
    print(f"     alpha     leaves it would zero")
    for aval in (0, 0.2, 1, 5, 20, 100, 500):
        killed = float((d["grad"] < aval).mean()) * 100
        mark = "   <- current" if abs(aval - alpha) < 1e-9 else ""
        print(f"     {aval:>7}     {killed:>5.1f}%{mark}")

    print(f"\n  ---- MIN_CHILD_WEIGHT is a floor on COVER, not on rows --------------------")
    W = getattr(C, "CLASS_WEIGHTS", {}) or {}
    if W:
        p = 1 / max(len(W), 1)
        h1 = 2 * p * (1 - p)          # hessian of ONE row of weight 1, at the softmax start
        print(f"     one row of weight 1 carries h = 2p(1-p) = {h1:.4f}   (p = 1/{len(W)})")
        print(f"     so min_child_weight={mcw} needs:")
        for k, w in sorted(W.items(), key=lambda kv: -kv[1]):
            print(f"        {k:<14}{mcw / (h1 * w):>10.1f} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="", help="a model_xgboost.joblib on disk")
    ap.add_argument("--model_task_id", default="", help="or fetch it from a ClearML training task")
    ap.add_argument("--eta", type=float, default=None,
                    help="the learning rate the model was TRAINED with. xgboost does not store it in\n"
                         "the model file, and |G| is divided by it. gain and cover do not need it.")
    ap.add_argument("--lam", type=float, default=None, help="the reg_lambda used at training")
    ap.add_argument("--max_trees", type=int, default=400,
                    help="trees to sample, spread evenly. 0 = all (slow, and GBs of strings)")
    a = ap.parse_args()

    from trainer.train import load_model_bundle
    src, task_params = a.bundle, {}
    if not src:
        if not a.model_task_id:
            raise SystemExit("give --bundle or --model_task_id")
        from clearml import Task
        t = Task.get_task(task_id=a.model_task_id)
        if t is None or "model" not in t.artifacts:
            raise SystemExit(f"task {a.model_task_id} has no 'model' artifact")
        # THE REAL TRAINING PARAMS LIVE ON THE TASK, NOT IN THE MODEL FILE.
        # xgboost's save_model keeps only what PREDICTION needs -- eta, lambda, alpha and gamma
        # are dropped, so a loaded booster reports library defaults for all four however it was
        # actually trained. the task recorded them at launch; that is the only honest source.
        flat = t.get_parameters() or {}
        for k, v in flat.items():
            key = str(k).split("/")[-1]
            try:
                task_params[key] = float(v)
            except (TypeError, ValueError):
                pass
        src = t.artifacts["model"].get_local_copy()

    b = load_model_bundle(src)
    if str(b.get("model_type", "")) != "xgboost":
        raise SystemExit(f"this reads xgboost trees; that bundle is "
                         f"{b.get('model_type')!r}. gamma/alpha/lambda are xgboost knobs.")
    print(f"  xgboost  v{b.get('dataset_version','?')}  labels {b.get('labels_name','?')}  "
          f"{len(b.get('features',[]))} features")

    bo = b["model"].get_booster()
    # FROM THE BOOSTER'S OWN CONFIG, NOT FROM get_params().
    # load_model_bundle rebuilds the model as a BARE XGBClassifier() and calls load_model() on it.
    # that restores the booster, but the sklearn wrapper's python-side attributes stay at their
    # constructor defaults -- so get_params() reports eta=0.3, lambda=1, alpha=0, gamma=0 for
    # EVERY bundle, whatever it was really trained with. save_config() is the model's own record
    # of what it actually used, and eta divides the gradient, so getting it wrong rescales |G|.
    import json
    tp = json.loads(bo.save_config())["learner"]["gradient_booster"]["tree_train_param"]

    def pick(names, default):
        """task params first, then the booster config, then the default.
        the booster's own numbers are library DEFAULTS for anything training-only, so they are
        the last resort, not the first -- see the note where task_params is filled."""
        for n in names:
            if n in task_params:
                return float(task_params[n]), "task"
            if n in overrides and overrides[n] is not None:
                return float(overrides[n]), "flag"
        for n in names:
            if n in tp:
                return float(tp[n]), "booster(default!)"
        return default, "fallback"

    overrides = {"eta": a.eta, "learning_rate": a.eta, "lambda": a.lam, "reg_lambda": a.lam}
    for n in ("eta", "learning_rate", "lambda", "reg_lambda"):
        if overrides.get(n) is not None:
            task_params.pop(n, None)                  # an explicit flag beats the task record
            task_params[n] = float(overrides[n])
    eta, eta_src = pick(("learning_rate", "eta"), 0.3)
    lam, lam_src = pick(("reg_lambda", "lambda"), 1.0)
    alpha, _ = pick(("reg_alpha", "alpha"), 0.0)
    gamma, _ = pick(("gamma", "min_split_loss"), 0.0)
    mcw, _ = pick(("min_child_weight",), 1.0)
    print(f"  eta={eta} (from {eta_src})   lambda={lam} (from {lam_src})")
    if "default" in eta_src:
        print(f"  !! eta could not be recovered -- |G| below is scaled by the WRONG eta.\n"
              f"     pass the real one:  --eta 0.05     (GAIN and COVER are unaffected)")

    report(collect(bo, a.max_trees, eta, lam), eta, lam, alpha, gamma, mcw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
