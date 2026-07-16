"""
trainer/hyperparams.py -- read configs/hyperparams.yaml. the ONE source of every model setting.

BEFORE THIS FILE
    the defaults were argparse literals inside train.py, and the search space was a separate
    hardcoded list inside hpo.py. two copies, in two files, of the same idea. they could
    disagree, and they did:

        train.py:  ap.add_argument("--max_depth", type=int, default=6)   # ONE default,
                                                                         # THREE models
        hpo.py:    DiscreteParameterRange("Args/max_features", ...)      # a parameter train.py
                                                                         # had never heard of

    the first cost a day (a depth-6 RandomForest -- a boosting number on a forest, silently
    crippling it). the second meant hpo.py could not start at all.

    changing a number meant editing Python. that is not a config, that is a hardcode with extra
    steps.

AFTER
    one YAML. defaults and search space side by side, per model. train.py reads `default:`,
    hpo.py reads `search:`. neither contains a number.

    argparse still EXISTS -- it has to, because ClearML's optimiser overrides task parameters BY
    NAME (Args/max_depth). But the VALUES come from the YAML. You never edit code to change a
    setting again.
"""
from __future__ import annotations

import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C      # noqa: E402

HP_FILE = C.CONFIGS_DIR / "hyperparams.yaml"
# where an APPLIED hpo winner lives. one json per model, written by trainer/apply_hpo.py.
# it is a SEPARATE file on purpose: hyperparams.yaml is the hand-authored baseline that lives
# in git; the tuned file is a machine-found overlay a human chose to promote. keeping them apart
# means you can always see, and diff, exactly what the search changed.
TUNED_DIR = C.CONFIGS_DIR / "tuned"


def _load() -> dict:
    if not HP_FILE.exists():
        raise SystemExit(f"no hyperparameter file at {HP_FILE}\n"
                         f"  every model setting lives in there. it is not optional.")
    return yaml.safe_load(HP_FILE.read_text()) or {}


def _tuned(model_type: str) -> dict:
    """the applied hpo winner for this model, if a human has promoted one. {} otherwise."""
    import json
    p = TUNED_DIR / f"{model_type}.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    return dict(doc.get("params") or {})


def tuned_sha(model_type: str) -> str | None:
    """the parquet_sha256 the tuned params for this model were found on. None if not tuned.

    publish --tune uses this as the cache key: if it equals the dataset being published, the
    data has not changed and HPO is skipped. if it differs (features changed) HPO re-runs.
    """
    import json
    p = TUNED_DIR / f"{model_type}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("dataset_sha256")


def defaults(model_type: str, quiet: bool = True) -> dict:
    """the settings this model trains with unless something overrides them.

    THREE LAYERS, LOWEST TO HIGHEST:
        1. hyperparams.yaml  `default:`   the hand-authored baseline (in git)
        2. configs/tuned/<model>.json     an hpo winner a human PROMOTED (apply_hpo.py)
        3. a CLI / ClearML override        applied later, in merge()

    the tuned layer is why HPO is not a dead end: you run the search, LOOK at the winner, run
    `python trainer/apply_hpo.py best_params_<model>.json` to promote it, and from then on every
    train / publish uses those numbers -- no editing yaml by hand, no forgetting to.
    """
    hp = _load()
    if model_type not in hp:
        raise SystemExit(f"'{model_type}' is not in {HP_FILE.name}. "
                         f"it has: {sorted(hp)}")
    d = dict(hp[model_type].get("default") or {})
    if not d:
        raise SystemExit(f"'{model_type}' has no `default:` block in {HP_FILE.name}")
    tuned = {k: v for k, v in _tuned(model_type).items() if k in d}   # only known knobs
    if tuned and not quiet:
        print(f"      hyperparams[{model_type}]: using {len(tuned)} TUNED value(s) from "
              f"configs/tuned/{model_type}.json  ({tuned})")
    d.update(tuned)
    return d


def merge(model_type: str, overrides: dict) -> dict:
    """the YAML defaults, with anything the CLI (or ClearML) actually set on top.

    THE TYPE PROBLEM, AND WHY THE YAML SOLVES IT.
        every override arrives as a STRING. argparse hands us a string; ClearML's optimiser sets
        task parameters as strings too. so "500" arrives where 500 is meant, and sklearn would
        raise -- or worse, "0" is truthy as a string, so `params["max_depth"] or None` would give
        the forest a max_depth of "0" instead of None. a silent, wrong model.

        the YAML default tells us the type. 500 is an int, so the override is cast to int. 0.05
        is a float, so it is cast to float. "sqrt" is a string, so it stays one. no type= tables
        to keep in sync, no guessing.
    """
    params = defaults(model_type, quiet=False)   # announce tuned values when a model trains
    for k, v in (overrides or {}).items():
        # None AND the empty string both mean NOT SET. this matters because of how the values
        # travel: argparse defaults are None, but ClearML stores every task parameter as a
        # STRING -- so a cloned task hands the un-overridden knobs back as ''. treating '' as a
        # value meant int('') -> SystemExit on EVERY agent run the moment the base task was
        # registered with generated defaults. '' is silence, not a zero.
        if v is None or (isinstance(v, str) and v.strip() == "") or k not in params:
            continue
        want = type(params[k])
        try:
            if want is bool:
                params[k] = str(v).strip().lower() in ("1", "true", "yes")
            elif want is int:
                f = float(v)
                if not f.is_integer():
                    # int(float('2.5')) would silently truncate to 2 -- the model would train
                    # with a setting nobody chose, and the ClearML record would say '2.5'.
                    raise SystemExit(
                        f"--{k}={v!r} is not a whole number, and this knob is an integer "
                        f"(configs/hyperparams.yaml: {model_type}.default.{k} = {params[k]!r}). "
                        f"refusing to round it silently.")
                params[k] = int(f)
            elif want is float:
                params[k] = float(v)
            else:
                params[k] = str(v)
        except (TypeError, ValueError):
            raise SystemExit(f"--{k}={v!r} is not a valid {want.__name__} "
                             f"(configs/hyperparams.yaml has {model_type}.default.{k} = "
                             f"{params[k]!r})")
    return params


def all_param_names() -> list:
    """every setting name any model uses. train.py's argparse must accept ALL of them.

    WHY ALL, AND NOT JUST THIS MODEL'S. one script trains all three, and ClearML clones ONE base
    task per model. if the parser only knew about xgboost's knobs, an HPO run against the forest
    would set Args/max_features -- a name argparse does not have -- and clearml would only WARN.
    the trial would train the DEFAULT model, report the default score, and the search would
    conclude that nothing you changed made any difference. because nothing you changed WAS
    changed. so the parser accepts every name, and each model ignores the ones that are not its
    own.
    """
    hp = _load()
    names = set()
    for m in hp.values():
        names |= set((m.get("default") or {}).keys())
        names |= set((m.get("search") or {}).keys())
    return sorted(names)


def search_space(model_type: str) -> list:
    """the ClearML search space for this model, built from the YAML.

    the three shapes the YAML allows:
        [a, b, c]              -> DiscreteParameterRange     (try exactly these)
        {min, max, step}       -> UniformParameterRange      (a plain range)
        {min, max, log: true}  -> LogUniformParameterRange   (orders of magnitude)

    THE LOG TRAP, VERIFIED IN THE CLEARML SOURCE:
        LogUniformParameterRange.get_value() returns  {name: base ** v},  base=10.
        so its min_value/max_value are EXPONENTS, not values.

        learning_rate 0.01 .. 0.20  ->  you must pass  min=-2, max=-0.7
        pass min=0.01, max=0.2 by mistake and you get  10^0.01 .. 10^0.2  =  1.02 .. 1.58
        -- a learning rate of 1.5. it does not error. it just trains garbage.

        so the YAML holds the REAL values (0.01, 0.2) and we do the log10 here, once, where it
        can be read and tested. nobody has to remember the trap.
    """
    import math
    from clearml.automation import (UniformIntegerParameterRange, LogUniformParameterRange,
                                    UniformParameterRange, DiscreteParameterRange)

    hp = _load()
    if model_type not in hp:
        raise SystemExit(f"'{model_type}' is not in {HP_FILE.name}")
    space_cfg = hp[model_type].get("search") or {}
    if not space_cfg:
        raise SystemExit(f"'{model_type}' has no `search:` block in {HP_FILE.name} -- "
                         f"there is nothing to tune.")

    space = []
    for name, spec in space_cfg.items():
        key = f"Args/{name}"

        if isinstance(spec, list):
            # everything travels down argparse as a string anyway; keep mixed types working
            space.append(DiscreteParameterRange(key, values=spec))
            continue

        if not isinstance(spec, dict) or "min" not in spec or "max" not in spec:
            raise SystemExit(f"{HP_FILE.name}: {model_type}.search.{name} must be a list, "
                             f"or a dict with min and max. got {spec!r}")

        lo, hi = spec["min"], spec["max"]

        if spec.get("log"):
            # the YAML holds real values; clearml wants exponents. convert HERE, once.
            space.append(LogUniformParameterRange(key, min_value=math.log10(lo),
                                                  max_value=math.log10(hi), base=10))
        elif isinstance(lo, int) and isinstance(hi, int):
            space.append(UniformIntegerParameterRange(key, min_value=lo, max_value=hi,
                                                      step_size=spec.get("step", 1)))
        else:
            space.append(UniformParameterRange(key, min_value=float(lo), max_value=float(hi),
                                               step_size=float(spec.get("step", 0.1))))
    return space
