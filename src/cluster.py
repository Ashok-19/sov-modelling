"""Per-topic clustering: PCA -> UMAP -> HDBSCAN, with settings picked by a sweep.

Nothing here takes a number of clusters. HDBSCAN finds them from density, and
we try a range of density settings and keep the one with the best mix of
DBCV (hdbscan's relative_validity_) and stability across UMAP seeds.

DBCV on its own tends to reward the coarsest split (two or three big blobs),
which isn't much use for share of voice. So the split is applied recursively:
each cluster gets its own UMAP and sweep, and is split again only if that
split is valid and holds up across seeds. Where it doesn't, we stop. The
number of themes falls out of that, it's never set.

This tends to give fairly fine clusters. That's deliberate: the LLM merge step
afterwards joins the ones that are really the same theme, which is much easier
to get right than splitting a cluster that's too broad.
"""

import json
import logging
import warnings

import hdbscan
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from . import progress

log = logging.getLogger(__name__)

SEED = 42
STABILITY_SEEDS = (7, 13, 101)
# A setting that throws away most posts as noise can score well on DBCV
# (a couple of tight blobs) while telling us little, so we prefer candidates
# that keep at least half the posts when any exist.
MAX_NOISE = 0.5
# When to stop splitting: a split must agree with itself across UMAP seeds at
# least this well (adjusted Rand index), and a group needs this many posts
# before we try to split it (the smallest cluster HDBSCAN may form is 5).
MIN_STABILITY = 0.8
MIN_SPLIT_SIZE = 30


def reduce(vecs, seed):
    n = len(vecs)
    x = PCA(n_components=min(50, n - 1), random_state=SEED).fit_transform(vecs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_neighbors=min(15, n - 1), n_components=5, min_dist=0.0,
                            metric="cosine", random_state=seed)
        return reducer.fit_transform(x)


def candidate_grid(n):
    sizes = np.unique(np.geomspace(5, max(6, n // 5), 8).astype(int))
    for mcs in sizes:
        for ms in (1, 3, 5, 10):
            if ms <= mcs:
                yield int(mcs), ms


def fit(x, mcs, ms):
    model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                            gen_min_span_tree=True, cluster_selection_method="eom")
    labels = model.fit_predict(x)
    n_clusters = labels.max() + 1
    dbcv = float(model.relative_validity_) if n_clusters >= 2 else float("nan")
    return labels, dbcv


def agreement(a, b):
    """Adjusted Rand index between two labellings, on posts both put in a cluster.

    Counting noise as its own label made a split look unstable just because a
    few borderline posts flipped in and out of noise between seeds.
    """
    both = (a >= 0) & (b >= 0)
    if both.sum() < 10:
        return 0.0
    return adjusted_rand_score(a[both], b[both])


def best_split(vecs):
    """Sweep HDBSCAN settings on one group of posts and return the best split.

    Returns (labels, chosen, sweep). labels is None when no setting gives a
    valid, stable split, meaning the group should stay as it is.
    """
    n = len(vecs)
    x = reduce(vecs, SEED)
    sweep = []
    for mcs, ms in candidate_grid(n):
        labels, dbcv = fit(x, mcs, ms)
        sweep.append({"min_cluster_size": mcs, "min_samples": ms, "dbcv": dbcv,
                      "n_clusters": int(labels.max() + 1),
                      "noise_frac": float((labels < 0).mean()), "labels": labels})
    valid = [s for s in sweep if not np.isnan(s["dbcv"]) and s["dbcv"] > 0]
    if not valid:
        return None, None, _strip(sweep)

    pool = [s for s in valid if s["noise_frac"] <= MAX_NOISE] or valid

    # Stability: refit the same settings on UMAP runs with other seeds and see
    # how well the labels agree with the main run. Every candidate is checked;
    # looking only at the best few by DBCV sometimes missed every stable one,
    # and a whole topic collapsed into a single theme.
    others = [reduce(vecs, s) for s in STABILITY_SEEDS]
    for s in pool:
        s["stability"] = float(np.mean(
            [agreement(s["labels"], fit(xo, s["min_cluster_size"], s["min_samples"])[0])
             for xo in others]))
        s["score"] = s["dbcv"] * s["stability"]

    stable = [s for s in pool if s["stability"] >= MIN_STABILITY]
    if not stable:
        return None, None, _strip(sweep)
    best = max(stable, key=lambda s: s["score"])
    return best["labels"], {k: v for k, v in best.items() if k != "labels"}, _strip(sweep)


def cluster_topic(vecs):
    """Cluster one topic's embeddings. Returns (labels, chosen, sweep).

    chosen describes every split that was made (where, with which settings and
    scores); sweep is the full sweep at the top level.
    """
    n = len(vecs)
    labels = np.full(n, -1)
    splits, top_sweep = [], None
    next_label = 0
    # Work list of (path, post indices). Path is just a readable name like "1.3".
    todo = [("all", np.arange(n))]
    while todo:
        path, idx = todo.pop(0)
        progress.ui.advance(description=f"Clustering: trying to split group {path} ({len(idx)} posts)")
        sub, chosen, sweep = best_split(vecs[idx]) if len(idx) >= MIN_SPLIT_SIZE else (None, None, None)
        if top_sweep is None:
            top_sweep = sweep
        if sub is None:
            labels[idx] = next_label
            next_label += 1
            continue
        splits.append({"group": path, "posts": int(len(idx)), **chosen})
        log.info("Split %s (%d posts): min_cluster_size=%d min_samples=%d -> %d clusters, "
                 "DBCV %.3f, stability %.3f, noise %.0f%%", path, len(idx),
                 chosen["min_cluster_size"], chosen["min_samples"], chosen["n_clusters"],
                 chosen["dbcv"], chosen["stability"], 100 * chosen["noise_frac"])
        for c in range(sub.max() + 1):
            child = f"{c + 1}" if path == "all" else f"{path}.{c + 1}"
            todo.append((child, idx[sub == c]))
        # posts HDBSCAN called noise at this level stay at -1

    if not splits:
        log.warning("No stable split found; treating the topic as one theme")
    top = splits[0] if splits else {}
    summary = {"min_cluster_size": top.get("min_cluster_size"), "min_samples": top.get("min_samples"),
               "dbcv": top.get("dbcv"), "stability": top.get("stability"),
               "n_clusters": int(labels.max() + 1), "noise_frac": float((labels < 0).mean()),
               "splits": json.dumps(splits, default=float)}
    log.info("Topic ends with %d clusters, %.0f%% noise, after %d splits",
             summary["n_clusters"], 100 * summary["noise_frac"], len(splits))
    return labels, summary, top_sweep


def _strip(sweep):
    return [{k: v for k, v in s.items() if k != "labels"} for s in sweep]


def leiden_check(vecs, hdb_labels, k=15):
    """Optional cross-check: Leiden on a kNN graph, resolution picked by seed stability.

    Returns a small dict with the chosen resolution, cluster count and the ARI
    against HDBSCAN on the posts HDBSCAN didn't call noise.
    """
    import igraph as ig
    import leidenalg

    n = len(vecs)
    k = min(k, n - 1)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1)
    nbrs = np.argsort(-sims, axis=1)[:, :k]
    edges = {}
    for i in range(n):
        for j in nbrs[i]:
            a, b = sorted((i, int(j)))
            edges[(a, b)] = max(float(sims[i, j]), 0.0)
    g = ig.Graph(n=n, edges=list(edges.keys()))
    g.es["weight"] = list(edges.values())

    best = None
    for res in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        runs = [np.array(leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=res, seed=s).membership) for s in range(5)]
        stab = np.mean([adjusted_rand_score(runs[0], r) for r in runs[1:]])
        # Lots of tiny clusters are trivially "stable", so skip a partition
        # where most posts sit in clusters below HDBSCAN's smallest size (5).
        sizes = np.bincount(runs[0])
        if (sizes[runs[0]] < 5).mean() > 0.5:
            continue
        if best is None or stab >= best["stability"]:  # ties go to the finer split
            best = {"resolution": res, "stability": float(stab), "labels": runs[0]}
    if best is None:
        return {"resolution": None, "stability": None, "n_clusters": 1, "ari_vs_hdbscan": None,
                "note": "every resolution gave mostly tiny clusters"}
    mask = hdb_labels >= 0
    return {"resolution": best["resolution"], "stability": best["stability"],
            "n_clusters": int(best["labels"].max() + 1),
            "ari_vs_hdbscan": float(adjusted_rand_score(hdb_labels[mask], best["labels"][mask]))}
