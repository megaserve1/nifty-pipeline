"""ui/_hparams.py -- upload hyperparameters for the next training run.

WHERE THEY GO. configs/tuned/<model>.json -- the SAME file apply_hpo.py writes when you promote an
HPO winner. trainer/hyperparams.defaults() layers it over the yaml baseline:

    1. configs/hyperparams.yaml  `default:`   the hand-authored set, in git
    2. configs/tuned/<model>.json             THIS -- an overlay a human chose
    3. a CLI / ClearML override               applied later, in merge()

so "use these numbers" and "if none given, use the last saved ones" both fall out of the existing
mechanism. nothing new is invented, and the manual path (apply_hpo.py) still works unchanged.

THE TRAP THIS GUARDS. defaults() keeps only keys that already exist in the model's `default:`
block -- `tuned = {k: v for k, v in _tuned(m).items() if k in d}`. a key that is not there is
DROPPED IN SILENCE. so `max_dept: 8` (typo) reads as accepted and trains with the old value. every
upload is checked against the real key set and unknown keys are named on screen.

FORMATS ACCEPTED -- json or yaml, any of these shapes:

    {"learning_rate": 0.05, "n_estimators": 400}            one model (pick it in the UI)
    {"xgboost": {...}, "catboost": {...}}                   several models
    {"xgboost": {"default": {...}}}                         same shape as hyperparams.yaml
    {"model_type": "xgboost", "params": {...}}              what apply_hpo.py / hpo.py write
"""
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C                       # noqa: E402
from trainer import hyperparams as H     # noqa: E402


def parse(text: str, filename: str, only_model: str = "") -> dict:
    """file contents -> {model_type: {param: value}}. raises ValueError with a plain message."""
    try:
        doc = json.loads(text) if filename.lower().endswith(".json") else yaml.safe_load(text)
    except Exception as exc:
        raise ValueError(f"could not read {filename}: {exc}")
    if not isinstance(doc, dict):
        raise ValueError("the file must contain a mapping (key: value), not a list or a scalar.")

    # shape 4: what hpo.py / apply_hpo.py write
    if "params" in doc and isinstance(doc["params"], dict):
        m = doc.get("model_type") or only_model
        if not m:
            raise ValueError("this file has 'params' but no 'model_type' -- pick a model above.")
        return {m: dict(doc["params"])}

    # shapes 2 and 3: keyed by model name
    known = set(C.MODEL_TYPES) | {"random_forest"}
    if set(doc) & known:
        out = {}
        for m, block in doc.items():
            if m not in known or not isinstance(block, dict):
                continue
            out[m] = dict(block.get("default") or block)      # tolerate the yaml's nesting
        if out:
            return out

    # shape 1: a flat block of params for the model chosen in the UI
    if not only_model:
        raise ValueError("this looks like one model's parameters -- pick which model above.")
    return {only_model: dict(doc)}


def check(model: str, params: dict) -> dict:
    """compare against what the model actually accepts.

    -> {"known": {...}, "unknown": [...], "changes": {k: (baseline, new)}, "same": [...]}
    """
    base = H.defaults(model)                     # yaml baseline + any overlay already saved
    raw = yaml.safe_load(H.HP_FILE.read_text())[model]["default"]   # the yaml alone
    known = {k: v for k, v in params.items() if k in raw}
    unknown = [k for k in params if k not in raw]
    changes, same = {}, []
    for k, v in known.items():
        if str(base.get(k)) != str(v):
            changes[k] = (base.get(k), v)
        else:
            same.append(k)
    return {"known": known, "unknown": unknown, "changes": changes, "same": same}


def saved(model: str) -> dict:
    """what is on disk right now for this model -- the numbers the next run WILL use."""
    p = H.TUNED_DIR / f"{model}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def save(model: str, params: dict, source: str) -> pathlib.Path:
    """write the overlay. NO dataset_sha256 on purpose -- that field is publish --tune's cache
    key ("these numbers were found on this exact data"). a hand-supplied set was not found on any
    dataset, and claiming one would make --tune skip a search it should run."""
    H.TUNED_DIR.mkdir(parents=True, exist_ok=True)
    p = H.TUNED_DIR / f"{model}.json"
    p.write_text(json.dumps({"model_type": model, "source": source,
                             "applied_by": "ui", "params": params}, indent=2))
    return p


def clear(model: str) -> bool:
    """drop the overlay -> the model goes back to the yaml baseline."""
    p = H.TUNED_DIR / f"{model}.json"
    if p.exists():
        p.unlink()
        return True
    return False
