"""Per-topic clustering: PCA -> UMAP -> HDBSCAN, with settings picked by a sweep.

Nothing here takes a number of clusters. HDBSCAN finds them from density, and
we try a range of density settings and keep the one with the best mix of
DBCV (hdbscan's relative_validity_) and stability across UMAP seeds.
"""

import logging
import warnings

import hdbscan
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

log = logging.getLogger(__name__)

SEED = 42
STABILITY_SEEDS = (7, 13, 101)
TOP_FOR_STABILITY = 6
# A setting that throws away most posts as noise can score well on DBCV
# (a couple of tight blobs) while telling us little, so we prefer candidates
# that keep at least half the posts when any exist.
MAX_NOISE = 0.5


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


def cluster_topic(vecs):
    """Cluster one topic's embeddings. Returns (labels, chosen, sweep)."""
    n = len(vecs)
    x = reduce(vecs, SEED)

    sweep = []
    for mcs, ms in candidate_grid(n):
        labels, dbcv = fit(x, mcs, ms)
        sweep.append({"min_cluster_size": mcs, "min_samples": ms, "dbcv": dbcv,
                      "n_clusters": int(labels.max() + 1),
                      "noise_frac": float((labels < 0).mean()), "labels": labels})
    valid = [s for s in sweep if not np.isnan(s["dbcv"])]
    if not valid:
        log.warning("No setting found 2+ clusters; treating the topic as one theme")
        return np.zeros(n, dtype=int), {"note": "single cluster"}, _strip(sweep)

    pool = [s for s in valid if s["noise_frac"] <= MAX_NOISE] or valid
    pool.sort(key=lambda s: s["dbcv"], reverse=True)

    # Stability: refit the same settings on UMAP runs with other seeds and see
    # how well the labels agree with the main run.
    others = [reduce(vecs, s) for s in STABILITY_SEEDS]
    for s in pool[:TOP_FOR_STABILITY]:
        aris = [adjusted_rand_score(s["labels"], fit(xo, s["min_cluster_size"], s["min_samples"])[0])
                for xo in others]
        s["stability"] = float(np.mean(aris))
        s["score"] = max(s["dbcv"], 0) * s["stability"]

    best = max(pool[:TOP_FOR_STABILITY], key=lambda s: s["score"])
    chosen = {k: v for k, v in best.items() if k != "labels"}
    log.info("Chose min_cluster_size=%d min_samples=%d: %d clusters, DBCV %.3f, "
             "stability %.3f, noise %.0f%%", best["min_cluster_size"], best["min_samples"],
             best["n_clusters"], best["dbcv"], best["stability"], 100 * best["noise_frac"])
    return best["labels"], chosen, _strip(sweep)


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
        if best is None or stab >= best["stability"]:  # ties go to the finer split
            best = {"resolution": res, "stability": float(stab), "labels": runs[0]}
    mask = hdb_labels >= 0
    return {"resolution": best["resolution"], "stability": best["stability"],
            "n_clusters": int(best["labels"].max() + 1),
            "ari_vs_hdbscan": float(adjusted_rand_score(hdb_labels[mask], best["labels"][mask]))}
