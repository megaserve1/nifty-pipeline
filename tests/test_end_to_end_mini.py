"""
tests/test_end_to_end_mini.py -- the WHOLE chain, on a synthetic mini-world, offline.

register -> freeze a version -> build_dataset -> validate the certificate -> load it the way
train.py does. Every stage runs the REAL code; only config's paths are pointed at tmp_path.

WHY THIS FILE EXISTS. The 2026-07-15 audit found that test_build_dataset.py never imported
bridge/build_dataset.py -- the single busiest file of the rebuild had NO test driving it, and
three separate bugs (a bool-column crash, a dtype round-trip that invalidated the certificate,
a 60-minute clock floor on 181 good columns) all shipped in one day because every test checked
a PIECE and nothing checked the PATH. This is the path.

It is also where the per-stage pins live that have no cheaper home:
  - planted leak columns (a fwd_ret, a date column, an id counter) must die at registration
  - measured clocks: 1-min, 5-min smeared, 5-min one-row-per-bar, and a bool column
  - the manifest schema must equal the parquet READ BACK, dtype for dtype
  - na_policy='drop' must drop the label minutes the bad bar was SERVED at, not its stamp
  - the trailing-space labels must arrive clean, with their weights
  - contract.validate_manifest and contract.assert_schema must both pass on what was built

Everything under tmp_path. No network, no ClearML, no real dirs touched.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C                                   # noqa: E402
import contract                                      # noqa: E402
from bridge import register as regmod                # noqa: E402
from bridge import build_dataset as bd               # noqa: E402
from core import make_version as mkv                 # noqa: E402


CLASSES = ["NO_TRADE", "EXIT_SMALL", "EXIT_SUPER", "EXIT_SUB",
           "ENTRY_SUPER", "ENTRY_SMALL", "ENTRY_SUB"]
WEIGHTS = {"NO_TRADE": 0.0638, "EXIT_SMALL": 0.18, "EXIT_SUPER": 0.91, "EXIT_SUB": 0.46,
           "ENTRY_SUPER": 0.91, "ENTRY_SMALL": 0.18, "ENTRY_SUB": 0.46}


def _grid(days: int) -> pd.DatetimeIndex:
    """the NSE minute grid: 09:15..15:29, weekdays."""
    out = []
    for d in pd.bdate_range("2024-01-01", periods=days):
        out.append(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=375, freq="1min"))
    return pd.DatetimeIndex(np.concatenate([m.values for m in out]))


@pytest.fixture
def world(tmp_path, monkeypatch):
    """a tiny but honest copy of the production layout, with config pointed at it."""
    rng = np.random.default_rng(7)
    grid = _grid(60)
    n = len(grid)

    feat = tmp_path / "data" / "features"
    labd = tmp_path / "data" / "labels"
    for d in (feat, labd, tmp_path / "versions", tmp_path / "datasets"):
        d.mkdir(parents=True)

    # ---- labels, with the REAL file's trailing-space disease ----
    lab = pd.Series(rng.choice(CLASSES, n, p=[.532, .189, .146, .059, .038, .024, .012]),
                    index=grid)
    pd.DataFrame({
        "timestamp": grid.strftime("%d-%m-%Y %H:%M"),
        "primary_label": [c + " " if c != "NO_TRADE" else c for c in lab],   # 6 of 7 dirty
        "weight": lab.map(WEIGHTS).values,
        "weight_raw": (lab.map(WEIGHTS) * 110000).round().values,
    }).to_csv(labd / "labels_mini.csv", index=False)

    # ---- feature 1: 1-min numerics + a bool + PLANTED LEAKS ----
    close = 20000 + np.cumsum(rng.normal(0, 3, n))
    pd.DataFrame({
        "close": close,
        "mom_10": pd.Series(close).pct_change(10).values,
        "is_bull": pd.Series(close).diff().gt(0).fillna(False).values.astype(bool),
        "fwd_ret_5": pd.Series(close).shift(-5).values / close - 1,        # THE FUTURE
        "session": grid.strftime("%Y-%m-%d"),                              # THE CALENDAR
        "event_id": np.repeat(np.arange(n // 50 + 1), 50)[:n].astype(float),  # ID COUNTER
    }, index=grid).to_parquet(feat / "fast_feature.parquet")

    # ---- feature 2: a 5-min bar smeared over 1-min rows, plus a categorical ----
    bar5 = grid.floor("5min")
    v5 = pd.Series(rng.normal(size=len(bar5.unique())), index=bar5.unique())
    pd.DataFrame({
        "sig5": v5.reindex(bar5).values,
        "state": pd.Series(rng.choice(["CALM", "STRESS"], len(bar5.unique())),
                           index=bar5.unique()).reindex(bar5).values,
    }, index=grid).to_parquet(feat / "smeared5.parquet")

    # ---- feature 3: 5-min, ONE ROW PER BAR (nothing to repeat) ----
    sparse_idx = grid[::5]
    pd.DataFrame({"x": np.arange(len(sparse_idx), dtype=float) % 7},
                 index=sparse_idx).to_parquet(feat / "sparse5.parquet")

    # ---- feature 4: na_policy='drop' -- one poisoned bar, to pin WHERE the drop lands ----
    # NOT named "y" -- a column called y IS the answer, and the guard rightly bans it
    dropf = pd.DataFrame({"dval": rng.normal(size=n)}, index=grid)
    poison_stamp = grid[375 * 10 + 100]              # day 11, 10:55
    dropf.loc[poison_stamp, "dval"] = np.nan
    dropf.to_parquet(feat / "droppy.parquet")

    monkeypatch.setattr(C, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(C, "FEATURES_DIR", feat)
    monkeypatch.setattr(C, "LABELS_DIR", labd)
    monkeypatch.setattr(C, "LABELS_FILE", "labels_mini.csv")
    monkeypatch.setattr(C, "REGISTRY", tmp_path / "registry.yaml")
    monkeypatch.setattr(C, "SELECTION_SHEET", tmp_path / "selection_sheet.yaml")
    monkeypatch.setattr(C, "VERSIONS_DIR", tmp_path / "versions")
    monkeypatch.setattr(C, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(C, "EMBARGO_SESSIONS", 5)    # 60 synthetic sessions, not 5 years
    return tmp_path, grid, poison_stamp


def _register_all(world_root):
    """stage 1: the real register.inspect on every parquet, into the real registry format."""
    import yaml
    reg = {}
    for p in sorted(C.FEATURES_DIR.glob("*.parquet")):
        name = regmod.feature_name(p)
        info = regmod.inspect(p)
        reg[name] = {"file": p.name, "clock": info["clock"],
                     "clock_measured": info["clock_measured"],
                     "na_policy": "drop" if name == "droppy" else C.DEFAULT_NA_POLICY,
                     "desc": "", "columns": info["columns"],
                     "banned": info["banned"] or None,
                     "categorical": info["categorical"], "rows": info["rows"],
                     "span": info["span"], "na_counts": info["na_counts"],
                     "added": "2024-01-01"}
    C.REGISTRY.write_text(yaml.safe_dump(reg, sort_keys=True))
    return reg


def test_the_whole_chain_register_freeze_build_certify_load(world, monkeypatch):
    tmp, grid, poison_stamp = world

    # ---- 1. REGISTER: the planted leaks must die here --------------------------------
    reg = _register_all(tmp)
    fast = reg["fast_feature"]
    banned = set(fast["banned"] or {})
    assert "fwd_ret_5" in banned, "a forward return survived registration"
    assert "session" in banned, "a date column survived registration"
    assert "event_id" in banned, "a running id counter survived registration"
    assert "is_bull" in fast["categorical"], "a bool column must register as categorical"
    assert "close" in fast["columns"] and "mom_10" in fast["columns"]

    # measured clocks: the three shapes
    assert reg["smeared5"]["clock"] == "5min", "smeared 5-min bar mis-measured"
    assert reg["sparse5"]["clock"] == "5min", "one-row-per-bar 5-min table mis-measured"

    # ---- 2. FREEZE: machine clocks must NOT be frozen as human declarations -----------
    v = mkv.freeze(sorted(reg), source="--all")
    recipe = mkv.load_version(v)
    assert recipe["feature_clocks"] == {}, (
        "clock == clock_measured means the MACHINE wrote both -- freezing it would floor "
        "every column of a mixed file at its slowest column's clock")

    # ---- 3. BUILD: the real build_dataset.main ---------------------------------------
    monkeypatch.setattr(sys, "argv", ["build_dataset.py", "--version", v])
    bd.main()
    ds_dir = C.DATASETS_DIR / v
    parquet = ds_dir / f"dataset_{v}.parquet"
    man = json.loads((ds_dir / "manifest.json").read_text())

    # ---- 4. THE CERTIFICATE: gate + schema truth --------------------------------------
    contract.validate_manifest(man, C.VERSIONS_DIR / f"dataset_{v}.yaml", parquet)
    df = pd.read_parquet(parquet)
    contract.assert_schema(df, man)          # the read-back dtype fix, end to end

    # labels arrived clean, with weights
    assert not df[C.LABEL_COL].str.endswith(" ").any(), "trailing spaces survived"
    assert df.groupby(C.LABEL_COL)[C.WEIGHT_COL].mean().loc["NO_TRADE"] == pytest.approx(0.0638)

    # the banned columns are NOT in the artifact
    cols = set(df.columns)
    for bad in ("fast_feature__fwd_ret_5", "fast_feature__session", "fast_feature__event_id"):
        assert bad not in cols, f"{bad} leaked into the built dataset"

    # ---- 5. NO-PEEK, row-level, for the smeared 5-min feature -------------------------
    # each value of sig5 belongs to a 5-min bar stamped floor(t,5): a label minute may only
    # ever hold the value of a bar whose CLOSE (stamp+5) is <= the minute.
    src = pd.read_parquet(C.FEATURES_DIR / "smeared5.parquet")
    bar_of = {}
    for stamp, val in src.groupby(src.index.floor("5min"))["sig5"].first().items():
        bar_of[round(float(val), 9)] = stamp
    ts = pd.to_datetime(df[C.LABEL_TS_COL])
    served = df["smeared5__sig5"]
    checked = 0
    for i in range(0, len(df), 89):                 # a spread sample, cheap but real
        if pd.isna(served.iloc[i]):
            continue
        stamp = bar_of[round(float(served.iloc[i]), 9)]
        assert stamp + pd.Timedelta(minutes=5) <= ts.iloc[i], (
            f"minute {ts.iloc[i]} was served the 5-min bar stamped {stamp}, "
            f"which had not closed. LOOKAHEAD.")
        checked += 1
    assert checked > 100, "the no-peek sample checked almost nothing"

    # ---- 6. THE DROP LANDS ON THE SERVED WINDOW, NOT THE STAMP ------------------------
    # droppy's poisoned bar is STAMPED at poison_stamp (1-min clock => served the minute
    # after close). the label minute equal to the stamp itself was fed the PREVIOUS, good
    # bar and must SURVIVE; the minutes in the served window must be gone.
    survived = set(ts)
    assert poison_stamp in survived, (
        "na_policy='drop' deleted the minute of the bad bar's STAMP -- that minute was served "
        "the previous, perfectly good bar. off by one bar, the exact bug the audit found.")
    close = poison_stamp + pd.Timedelta(minutes=1)   # a 1-min bar closes one minute later
    assert close not in survived, (
        "the minute the unusable bar was actually SERVED at is still in the dataset")

    # ---- 7. LOAD THE WAY TRAIN.PY DOES -------------------------------------------------
    from trainer.objective import three_way_split
    tr, va, te, info = three_way_split(ts, 0.0, 0.3, C.EMBARGO_SESSIONS)
    assert int(va.sum()) == 0 and int(tr.sum()) > 0 and int(te.sum()) > 0
    tr_days = set(pd.DatetimeIndex(ts[tr]).normalize())
    te_days = set(pd.DatetimeIndex(ts[te]).normalize())
    assert not (tr_days & te_days), "a session appears in both train and test"
    assert len(tr_days.union(te_days)) < 60, "the embargo removed no sessions at all"


def test_merge_treats_empty_string_as_not_set():
    """ClearML stores task parameters as STRINGS: un-overridden knobs come back as ''.
    int('') crashed every agent run. '' is silence, not a zero."""
    from trainer.hyperparams import merge, defaults
    d = defaults("xgboost")
    m = merge("xgboost", {"max_depth": "", "learning_rate": None, "subsample": "0.8"})
    assert m["max_depth"] == d["max_depth"], "'' must fall back to the YAML default"
    assert m["learning_rate"] == d["learning_rate"]
    assert m["subsample"] == 0.8


def test_merge_refuses_to_silently_truncate_a_fraction_to_an_int():
    from trainer.hyperparams import merge
    with pytest.raises(SystemExit, match="whole number"):
        merge("xgboost", {"max_depth": "2.5"})


def test_full_proba_realigns_when_a_class_is_missing_from_train():
    """sklearn's RF returns probability columns only for the classes it SAW. on a small slice
    that silently shifts every column one class over -- the worst kind of wrong."""
    from trainer.train import full_proba

    class FakeRF:
        classes_ = np.array([0, 2, 5])               # classes 1,3,4,6 never seen

    p = np.array([[0.7, 0.2, 0.1]])
    out = full_proba(FakeRF(), p, 7)
    assert out.shape == (1, 7)
    assert out[0, 0] == 0.7 and out[0, 2] == 0.2 and out[0, 5] == 0.1
    assert out[0, 1] == 0 and out[0, 6] == 0


def test_dataset_diff_survives_lock_files_and_minor_versions(tmp_path, monkeypatch):
    """--all crashed with int('1.lock') / int('2.1') the moment the pipeline was actually
    used -- locks appear on the first publish, minors are the point of the ablation design."""
    from core import dataset_diff as dd
    monkeypatch.setattr(C, "VERSIONS_DIR", tmp_path)
    for name in ("dataset_v1.yaml", "dataset_v2.yaml", "dataset_v2.1.yaml",
                 "dataset_v1.lock.yaml"):
        (tmp_path / name).write_text("features: [a]\nname: x\n")
    files = dd._recipe_files()
    assert all(not f.name.endswith(".lock.yaml") for f in files)
    vs = sorted((p.stem.replace("dataset_", "") for p in files), key=dd._sort_key)
    assert vs == ["v1", "v2", "v2.1"]


def test_freeze_ignores_a_clock_with_no_measurement_behind_it(world):
    """a registry entry with `clock` but NO `clock_measured` proves nothing about WHO wrote
    the number. treating it as a human declaration would floor every column of the file."""
    import yaml
    tmp, _, _ = world
    _register_all(tmp)
    reg = yaml.safe_load(C.REGISTRY.read_text())
    reg["smeared5"].pop("clock_measured")            # an old-format / hand-written entry
    reg["smeared5"]["clock"] = "60min"               # looks scary, provenance unknown
    C.REGISTRY.write_text(yaml.safe_dump(reg))

    v = mkv.freeze(sorted(reg), source="--all")
    assert mkv.load_version(v)["feature_clocks"] == {}, (
        "no clock_measured = no evidence of a human = freeze nothing")


def test_a_variation_inherits_its_parents_frozen_clocks(world):
    """v2 vs v2.1 must differ in ONE thing -- the dropped feature. if the derive re-reads
    clocks from today's registry, the comparison silently stops meaning anything."""
    import yaml
    tmp, _, _ = world
    _register_all(tmp)
    reg = yaml.safe_load(C.REGISTRY.read_text())
    reg["smeared5"]["clock"] = "15min"               # a HUMAN declaration (differs from measured)
    C.REGISTRY.write_text(yaml.safe_dump(reg))

    v1 = mkv.freeze(sorted(reg), source="--all")
    assert mkv.load_version(v1)["feature_clocks"] == {"smeared5": "15min"}

    # the registry then MOVES (rescan wrote a new measurement, human edit, whatever)
    reg["smeared5"]["clock"] = "30min"
    C.REGISTRY.write_text(yaml.safe_dump(reg))

    v11 = mkv.freeze([f for f in sorted(reg) if f != "droppy"], source=f"derived from {v1}",
                     parent=v1)
    assert mkv.load_version(v11)["feature_clocks"]["smeared5"] == "15min", (
        "the variation took TODAY'S registry clock instead of its parent's frozen one -- "
        "now v1 vs v1.1 differ in selection AND alignment, and the ablation teaches nothing")


def test_hpo_winner_promotes_into_the_training_defaults(tmp_path, monkeypatch):
    """the wiring that closes the HPO loop: a winner json -> apply_hpo -> tuned overlay ->
    hyperparams.defaults() -> what every train/publish actually uses. before this, HPO found
    the best params and NOTHING read them back."""
    import json, subprocess
    from trainer import hyperparams as H

    monkeypatch.setattr(H, "TUNED_DIR", tmp_path / "tuned")

    base = H.defaults("xgboost")
    baseline_depth = base["max_depth"]

    # no tuned file yet -> baseline unchanged
    assert H.defaults("xgboost")["max_depth"] == baseline_depth

    # write the tuned overlay directly (apply_hpo's output), and confirm defaults() picks it up
    (tmp_path / "tuned").mkdir()
    (tmp_path / "tuned" / "xgboost.json").write_text(json.dumps({
        "model_type": "xgboost", "params": {"max_depth": 7, "learning_rate": 0.083}}))
    d = H.defaults("xgboost")
    assert d["max_depth"] == 7, "the promoted winner did not reach the training defaults"
    assert abs(d["learning_rate"] - 0.083) < 1e-9
    assert d["n_estimators"] == base["n_estimators"], "un-tuned knobs must keep the baseline"

    # a CLI/ClearML override still wins over the tuned value (the 3-layer order)
    m = H.merge("xgboost", {"max_depth": "8"})
    assert m["max_depth"] == 8, "an explicit override must beat the tuned default"


def test_apply_hpo_refuses_a_range_edge_winner(tmp_path, monkeypatch):
    """a winner sitting at the extreme of its search range is usually the range binding, not the
    data speaking -- apply_hpo makes you look before baking in that artefact."""
    import json, subprocess
    from trainer import hyperparams as H

    # max_depth range for xgboost is min 3..max 10; a winner AT 10 pins the edge
    winner = tmp_path / "best_params_xgboost.json"
    winner.write_text(json.dumps({"model_type": "xgboost", "dataset_version": "v3",
                                  "val_trading_cost": 30, "test_trading_cost": 33,
                                  "params": {"max_depth": 10}}))
    r = subprocess.run([str(ROOT / "final_venv/bin/python"), "trainer/apply_hpo.py", str(winner)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode != 0, "a range-edge winner must not promote without --force"
    assert "range" in (r.stdout + r.stderr).lower()
