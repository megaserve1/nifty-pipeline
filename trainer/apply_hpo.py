"""
trainer/apply_hpo.py -- PROMOTE an hpo winner to the tuned defaults.

    python trainer/hpo.py --dataset_id <id> --model xgboost   ->  best_params_xgboost.json
    python trainer/apply_hpo.py best_params_xgboost.json       ->  configs/tuned/xgboost.json

After this, EVERY train / publish uses the tuned numbers automatically -- hyperparams.defaults()
overlays configs/tuned/<model>.json on top of the hand-authored baseline in hyperparams.yaml.
No editing yaml by hand. No forgetting which run the numbers came from.

WHY IT IS A SEPARATE, DELIBERATE STEP (and not automatic at the end of hpo.py)
    the hpo winner is the LOWEST of N noisy draws -- the winner's curse. its validation score is
    biased low by construction. before those numbers become what production trains on, a human
    should LOOK: is the val score wildly better than the test score (over-fitted search)? did a
    regulariser slam to its extreme (the search escaping, not learning)? apply_hpo is the pause
    where you look, and the tuned file records exactly what you promoted and from which run.

    it is the same shape as clock_override and allow_columns: the machine proposes, a human
    promotes, and the promotion is on the record in a file with a name against it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C                        # noqa: E402
from trainer import hyperparams as H      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("winner", help="the best_params_<model>.json that hpo.py wrote")
    ap.add_argument("--force", action="store_true",
                    help="promote even if a value pins a search-space extreme (normally a warning)")
    ap.add_argument("--sha", default=None,
                    help="the dataset's parquet_sha256. recorded in the tuned file so publish "
                         "--tune can tell whether the data has changed and skip re-tuning.")
    a = ap.parse_args()

    src = pathlib.Path(a.winner)
    if not src.exists():
        raise SystemExit(f"no such file: {src}\n  run hpo.py first -- it writes best_params_<model>.json")

    doc = json.loads(src.read_text())
    model = doc.get("model_type")
    params = doc.get("params") or {}
    if model not in C.MODEL_TYPES:
        raise SystemExit(f"this winner is for model {model!r}, which is not one of {C.MODEL_TYPES}")

    # keep ONLY the real knobs for this model -- drop any bookkeeping columns hpo left in.
    known = set(H.defaults(model))
    tuned = {}
    for k, v in params.items():
        if k not in known:
            continue
        # coerce to the baseline's type, exactly as training will (merge() does the same).
        base = H.defaults(model)[k]
        try:
            if isinstance(base, bool):
                tuned[k] = str(v).strip().lower() in ("1", "true", "yes")
            elif isinstance(base, int):
                tuned[k] = int(float(v))
            elif isinstance(base, float):
                tuned[k] = float(v)
            else:
                tuned[k] = str(v)
        except (TypeError, ValueError):
            raise SystemExit(f"winner value {k}={v!r} is not a valid {type(base).__name__}")

    if not tuned:
        raise SystemExit(f"the winner file had no recognisable knobs for {model}. "
                         f"known knobs: {sorted(known)}")

    # LOOK-BEFORE-YOU-LEAP warnings. a value sitting at the very edge of its search range usually
    # means the search wanted to go further and could not -- the range was the binding constraint,
    # not the data. promoting it bakes in an artefact of the range you happened to pick.
    warnings = []
    space = {p.name.split("/", 1)[1]: p for p in H.search_space(model)}
    for k, v in tuned.items():
        sp = space.get(k)
        vals = getattr(sp, "values", None)
        lo = getattr(sp, "min_value", None)
        hi = getattr(sp, "max_value", None)
        if vals and v in (vals[0], vals[-1]):
            warnings.append(f"  {k}={v} sits at a discrete extreme {vals}")
        elif lo is not None and hi is not None and (abs(float(v) - lo) < 1e-9 or abs(float(v) - hi) < 1e-9):
            warnings.append(f"  {k}={v} pins the range edge [{lo}, {hi}]")

    vcost, tcost = doc.get("val_trading_cost"), doc.get("test_trading_cost")
    print(f"promoting hpo winner for {model} (from {doc.get('dataset_version','?')})")
    print(f"  val_trading_cost={vcost}   test_trading_cost={tcost}")
    print(f"  tuned params: {tuned}")
    if vcost is not None and tcost is not None and tcost > 0 and vcost < 0.7 * tcost:
        print(f"  !! the val cost is much lower than the test cost -- the search may have "
              f"over-fitted the validation slice. quote the TEST number, not the val one.")
    if warnings:
        print("  !! some winners sit at a search-range edge -- widen the range and re-search "
              "instead of promoting an artefact:")
        for w in warnings:
            print(w)
        if not a.force:
            raise SystemExit("  refusing to promote a range-edge winner. re-run with --force if "
                             "you have looked and you mean it.")

    H.TUNED_DIR.mkdir(parents=True, exist_ok=True)
    dst = H.TUNED_DIR / f"{model}.json"
    dst.write_text(json.dumps({
        "model_type": model,
        "from_dataset": doc.get("dataset_version"),
        # the checksum of the exact parquet these params were tuned on. publish --tune compares
        # it to the current dataset: same sha -> reuse (skip HPO); different -> the data changed,
        # so re-tune. this is the whole cache -- the data decides, not a human.
        "dataset_sha256": a.sha or doc.get("dataset_sha256"),
        "val_trading_cost": vcost,
        "test_trading_cost": tcost,
        "params": tuned,
    }, indent=2))
    print(f"\n  wrote {dst}")
    print(f"  from now on, every train / publish of {model} uses these values.")
    print(f"  to revert: delete that file (the hyperparams.yaml baseline takes over again).")


if __name__ == "__main__":
    main()
