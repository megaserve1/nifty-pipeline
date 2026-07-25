"""scripts/trace_path.py -- watch ONE data point travel down a tree to a decision.

for a boosted model the final score is the SUM of the leaf a point lands in, across ALL trees.
this script shows that journey concretely:

    --row N --tree K   trace how row N walks tree K: at each node "feature < threshold? yes/no",
                       step by step, until it lands in a leaf (a number it adds to the score).
    --deepest          find the tree with the LONGEST branch (biggest max_depth) and print that
                       longest root-to-leaf path -- this is what leaf-wise (loss-based) growth
                       produces: one very deep, lopsided branch.

why this matters for your loss-based (lossguide) model: it grows by splitting the worst leaf over
and over, so instead of a balanced tree it builds a few VERY long branches. this shows how long.

run:
    final_venv/bin/python scripts/trace_path.py --bundle ~/Downloads/model_xgboost.joblib \
        --data datasets/v4/dataset_v4.parquet --row 9761 --tree 0
    final_venv/bin/python scripts/trace_path.py --bundle ~/Downloads/model_xgboost.joblib --deepest
"""
import sys, re, argparse, pathlib
_here = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent))   # repo root (config, trainer, na_policy)
sys.path.insert(0, str(_here.parent))          # scripts/ (predict) -- works when imported too
import numpy as np, pandas as pd
from trainer.train import load_model_bundle
from predict import prepare                    # reuse the exact training preprocessing

SPLIT = re.compile(r"^(\d+):\[(.+)<([\-\d.eE+]+)\]\s+yes=(\d+),no=(\d+),missing=(\d+)")
LEAF  = re.compile(r"^(\d+):leaf=([\-\d.eE+]+)")


def parse_tree(dump: str) -> dict:
    """text dump -> {node_id: (feature, thr, yes, no, missing)} and {node_id: leaf_value}."""
    nodes, leaves = {}, {}
    for line in dump.splitlines():
        s = line.strip()
        m = SPLIT.match(s)
        if m:
            nid, feat, thr, yes, no, miss = m.groups()
            nodes[int(nid)] = (feat, float(thr), int(yes), int(no), int(miss))
            continue
        m = LEAF.match(s)
        if m:
            leaves[int(m.group(1))] = float(m.group(2))
    return nodes, leaves


def depth_of(nodes, leaves, nid=0, d=0):
    if nid in leaves:
        return d
    _, _, yes, no, _ = nodes[nid]
    return max(depth_of(nodes, leaves, yes, d + 1), depth_of(nodes, leaves, no, d + 1))


def longest_path(nodes, leaves, nid=0):
    """the deepest root-to-leaf path as a list of (node_id, description)."""
    if nid in leaves:
        return [(nid, f"LEAF = {leaves[nid]:+.4f}")]
    feat, thr, yes, no, _ = nodes[nid]
    left = longest_path(nodes, leaves, yes)
    right = longest_path(nodes, leaves, no)
    deeper, dirn = (left, "yes") if len(left) >= len(right) else (right, "no")
    return [(nid, f"{feat} < {thr:.4f} ? -> take '{dirn}'")] + deeper


def trace_point(nodes, leaves, row: pd.Series):
    """follow the ACTUAL path this row takes through the tree."""
    steps, nid = [], 0
    while nid not in leaves:
        feat, thr, yes, no, miss = nodes[nid]
        val = row.get(feat, np.nan)
        if pd.isna(val):
            nxt, why = miss, f"{feat} is MISSING -> missing branch"
        elif float(val) < thr:
            nxt, why = yes, f"{feat}={float(val):.4f}  <  {thr:.4f}  -> YES (left)"
        else:
            nxt, why = no, f"{feat}={float(val):.4f}  >=  {thr:.4f}  -> no (right)"
        steps.append((nid, why))
        nid = nxt
    steps.append((nid, f"LEAF = {leaves[nid]:+.4f}   <- this tree's contribution to the score"))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--data", default="", help="parquet/csv (needed for --row)")
    ap.add_argument("--row", type=int, default=None, help="row index to trace")
    ap.add_argument("--tree", type=int, default=0, help="which tree (default 0)")
    ap.add_argument("--deepest", action="store_true", help="show the model's longest branch")
    a = ap.parse_args()

    b = load_model_bundle(a.bundle)
    booster = b["model"].get_booster()
    if b.get("features"):
        booster.feature_names = list(b["features"])
    dumps = booster.get_dump()
    print(f"{b.get('model_type')}  {len(dumps)} trees  (multiclass: n_estimators x n_classes)")

    if a.deepest:
        depths = []
        for k, d in enumerate(dumps):
            nodes, leaves = parse_tree(d)
            depths.append((depth_of(nodes, leaves), k))
        depth, k = max(depths)
        print(f"\nDEEPEST tree = #{k}, longest branch = depth {depth}")
        nodes, leaves = parse_tree(dumps[k])
        for i, (nid, desc) in enumerate(longest_path(nodes, leaves)):
            print(f"  depth {i:>3}  node {nid:<5} {desc}")
        print("\n^ leaf-wise (loss-based) growth: one very long branch, not a balanced tree.")
        return

    if a.row is None or not a.data:
        raise SystemExit("give --row N and --data <parquet> to trace a point, or use --deepest")
    p = pathlib.Path(a.data)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    X = prepare(df, b)
    row = X.iloc[a.row]
    nodes, leaves = parse_tree(dumps[a.tree])
    print(f"\nrow {a.row} walking tree #{a.tree}  (this tree's max depth = {depth_of(nodes,leaves)}):")
    for i, (nid, why) in enumerate(trace_point(nodes, leaves, row)):
        print(f"  step {i:>3}  node {nid:<5} {why}")
    print("\nthe model adds THIS leaf value to the leaves from every other tree; the sum (through "
          "softmax) is the class probability. one tree is one small vote.")


if __name__ == "__main__":
    main()
