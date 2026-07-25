"""scripts/visualize_model.py -- look inside a trained xgboost model: the trees, and how it grew.

it loads the model bundle you downloaded (model_xgboost.joblib), rebuilds the xgboost model from
its portable UBJ, and writes several views into an output folder:

    growth.png        leaves-per-tree and depth-per-tree across ALL boosting rounds -- "how it grew"
    tree_stats.csv    the same numbers, per tree
    feature_splits.csv how often each feature is used as a split (tree-level importance)
    tree_<N>.png      a RENDERED picture of one tree   (needs system graphviz: apt install graphviz)
    tree_<N>.txt      the same tree as text            (always written, no graphviz needed)

WHY NOT ONE BIG PICTURE OF THE WHOLE MODEL. a gradient booster is 6000 trees added together; there
is no single tree to look at. and with your max-overfit config (unlimited depth/leaves) each tree
is huge. so the honest views are the GROWTH curve (how big the trees got, round by round) and a
readable EARLY tree.

run:
    final_venv/bin/python scripts/visualize_model.py --bundle ~/Downloads/model_xgboost.joblib
    final_venv/bin/python scripts/visualize_model.py --bundle <path> --tree 0 --out tree_out
"""
import sys, re, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SPLIT = re.compile(r"^(\d+):\[(.+)<([\-\d.eE+]+)\]\s+yes=(\d+),no=(\d+),missing=(\d+)")
LEAF  = re.compile(r"^(\d+):leaf=([\-\d.eE+]+)")
from trainer.train import load_model_bundle


def per_tree_stats(dumps):
    """for each tree's text dump: leaf count and max depth (from tab indentation)."""
    leaves, depths = [], []
    for d in dumps:
        leaves.append(d.count("leaf="))
        md = 0
        for line in d.splitlines():
            md = max(md, len(line) - len(line.lstrip("\t")))
        depths.append(md)
    return leaves, depths


def feature_split_counts(dumps, feature_names):
    """how many times each feature is used as a split across every tree."""
    counts = {}
    # dump lines look like:  3:[f12<0.5] yes=.. no=..   OR   3:[featurename<0.5] ...
    pat = re.compile(r"\[([^<>\]]+)[<>]")
    for d in dumps:
        for m in pat.findall(d):
            name = m.strip()
            if name.startswith("f") and name[1:].isdigit() and feature_names:
                idx = int(name[1:])
                if idx < len(feature_names):
                    name = feature_names[idx]
            counts[name] = counts.get(name, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="the downloaded model_xgboost.joblib")
    ap.add_argument("--out", default="model_view", help="output folder")
    ap.add_argument("--tree", type=int, default=0, help="which tree to render/dump (default 0)")
    ap.add_argument("--levels", type=int, default=3,
                    help="how many TOP levels of the tree to draw (default 3). a full loss-based "
                         "tree is unreadable; the top few levels are legible and show the trunk.")
    ap.add_argument("--format", default="svg", choices=["svg", "pdf", "png"],
                    help="image format. DEFAULT svg -- a VECTOR format that stays sharp at any "
                         "zoom (png is pixels and 'bursts' when you zoom in). open .svg in a browser.")
    ap.add_argument("--full", action="store_true",
                    help="draw the WHOLE tree, root to every leaf (no '... nodes below' stubs). "
                         "the canvas gets huge -- but as SVG every node stays crisp; open in a "
                         "browser and pan/zoom. use --tree to pick which tree.")
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    b = load_model_bundle(a.bundle)
    if b.get("model_type") != "xgboost":
        raise SystemExit(f"this bundle is {b.get('model_type')!r}, not xgboost.")
    model = b["model"]
    feats = b.get("features") or []
    booster = model.get_booster()
    if feats:
        booster.feature_names = list(feats)          # real names instead of f0, f1, ...

    print("reading every tree (this can take a moment on 6000 trees) ...")
    dumps = booster.get_dump(with_stats=False)
    n = len(dumps)
    leaves, depths = per_tree_stats(dumps)
    print(f"  {n} trees.  leaves/tree: min {min(leaves)}, max {max(leaves)}, "
          f"avg {sum(leaves)/n:.0f}.  depth/tree: max {max(depths)}")

    # ---- 1. GROWTH: leaves and depth per tree, round by round --------------------
    import csv
    with open(out / "tree_stats.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tree", "leaves", "depth"])
        for i, (lv, dp) in enumerate(zip(leaves, depths)):
            w.writerow([i, lv, dp])
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(range(n), leaves, lw=0.8); ax[0].set_ylabel("leaves in the tree")
    ax[0].set_title("how the model grew -- tree SIZE over boosting rounds")
    ax[1].plot(range(n), depths, lw=0.8, color="tab:red"); ax[1].set_ylabel("tree depth")
    ax[1].set_xlabel("tree number (boosting round)")
    fig.tight_layout(); fig.savefig(out / f"growth.{a.format}", dpi=150); plt.close(fig)
    print(f"  wrote {out/('growth.'+a.format)}  and  {out/'tree_stats.csv'}")

    # ---- 2. which features the trees actually split on ---------------------------
    counts = feature_split_counts(dumps, feats)
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    with open(out / "feature_splits.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature", "times_used_as_split"])
        for name, c in ranked:
            w.writerow([name, c])
    top = ranked[:25][::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([t[0] for t in top], [t[1] for t in top])
    ax.set_title("top 25 features by how often a tree splits on them")
    fig.tight_layout(); fig.savefig(out / f"feature_splits.{a.format}", dpi=150); plt.close(fig)
    print(f"  wrote {out/('feature_splits.'+a.format)}  and  {out/'feature_splits.csv'}")

    # ---- 3. ONE tree: text always, plus a READABLE top-N-levels picture ----------
    t = a.tree
    (out / f"tree_{t}.txt").write_text(dumps[t] if t < n else "(no such tree)")
    print(f"  wrote {out/f'tree_{t}.txt'}   (leaves={leaves[t]}, depth={depths[t]})")

    # a full loss-based tree is thousands of nodes -- unreadable. so render ONLY the top
    # `--levels` levels: the main splits, legible. everything below is shown as a "... (N more)" stub.
    try:
        if a.full:
            stem = out / f"tree_{t}_full"
            render_top_levels(dumps[t], stem, a.levels, a.format, full=True)
            print(f"  wrote {out/(f'tree_{t}_full.'+a.format)}   (WHOLE tree, {leaves[t]} leaves, "
                  f"depth {depths[t]} -- big canvas, open the .svg in a browser and zoom)")
        else:
            stem = out / f"tree_{t}_top{a.levels}"
            render_top_levels(dumps[t], stem, a.levels, a.format)
            print(f"  wrote {out/(f'tree_{t}_top{a.levels}.'+a.format)}   (READABLE, vector -- top {a.levels} levels)")
    except Exception as e:
        print(f"  (could not render the tree picture: {type(e).__name__}: {str(e)[:120]})")

    print(f"\ndone -> {out}/   (open growth.png and feature_splits.png first;")
    print(f"                    tree_{t}_top{a.levels}.{a.format} is the readable tree, sharp at any zoom)")


def render_top_levels(dump: str, out_path, levels: int, fmt: str = "svg", full: bool = False):
    """draw ONLY the top `levels` levels of a tree with graphviz -- the readable part.

    a full loss-based tree has thousands of nodes; drawing it all is the unreadable blob. the top
    few levels are the trunk (the biggest, most important splits) and they FIT on a page. nodes at
    the cutoff become '... (N more below)' stubs so you know the tree continues.
    """
    import graphviz
    nodes, leaves = {}, {}
    for line in dump.splitlines():
        s = line.strip()
        m = SPLIT.match(s)
        if m:
            nid, feat, thr, yes, no, miss = m.groups()
            nodes[int(nid)] = (feat, float(thr), int(yes), int(no))
        else:
            m = LEAF.match(s)
            if m:
                leaves[int(m.group(1))] = float(m.group(2))

    def count_below(nid):
        if nid in leaves:
            return 1
        _, _, y, no = nodes[nid]
        return 1 + count_below(y) + count_below(no)

    g = graphviz.Digraph(graph_attr={"rankdir": "TB"}, node_attr={"shape": "box", "fontsize": "11"})
    def add(nid, depth):
        if nid in leaves:
            g.node(str(nid), f"leaf = {leaves[nid]:+.3f}", style="filled", fillcolor="#e8f0e8")
            return
        feat, thr, yes, no = nodes[nid]
        g.node(str(nid), f"{feat}\n< {thr:.4f} ?")
        if not full and depth >= levels:             # cutoff: stub out what's below (skipped if full)
            for child, lab in ((yes, "yes"), (no, "no")):
                stub = f"stub_{child}"
                g.node(stub, f"... ({count_below(child)} nodes below)",
                       shape="ellipse", style="dashed")
                g.edge(str(nid), stub, label=lab)
            return
        for child, lab in ((yes, "yes"), (no, "no")):
            g.edge(str(nid), str(child), label=lab)
            add(child, depth + 1)
    add(0, 0)
    g.render(str(out_path), format=fmt, cleanup=True)


if __name__ == "__main__":
    main()
