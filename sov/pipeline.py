"""Runs the stages in order and writes the outputs."""

import json
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import cluster, config, data, embed, llm, themes
from .keywords import ctfidf_keywords

log = logging.getLogger(__name__)


class Timer:
    def __init__(self):
        self.times = {}

    def __call__(self, name):
        timer = self

        class _Block:
            def __enter__(self):
                self.t = time.time()

            def __exit__(self, *exc):
                timer.times[name] = timer.times.get(name, 0) + time.time() - self.t
                log.info("[%s] %.1fs", name, time.time() - self.t)

        return _Block()


def run(topic_numbers=None, leiden=False, ctx=config.LLM_CTX, out_name=None):
    config.CACHE_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    timer = Timer()
    run_log = {"started": datetime.now().isoformat(timespec="seconds"),
               "embed_model": config.EMBED_FILE, "llm_model": config.LLM_FILE,
               "llm_context": ctx, "topics": {}}

    with timer("load"):
        df = data.load()
        available = data.list_topics(df)
        wanted = topic_numbers or list(available)
        missing = [t for t in wanted if t not in available]
        if missing:
            raise SystemExit(f"Unknown topic number(s) {missing}. Available: {available}")
        kept = df[df["topic_num"].isin(wanted)].copy()
        log.info("%d posts in selected topics", len(kept))

    with timer("embed"):
        vecs = embed.embed_posts(kept[config.TEXT_COL].fillna("").astype(str).tolist())

    # Clustering and keywords need no model, so do all topics before loading the LLM.
    per_topic = {}
    for num in wanted:
        name = available[num]
        mask = (kept["topic_num"] == num).to_numpy()
        texts = kept.loc[mask, config.TEXT_COL].fillna("").astype(str).tolist()
        tv = vecs[mask]
        log.info("=== Topic %s: %d posts ===", name, len(texts))
        with timer("cluster"):
            labels, chosen, sweep = cluster.cluster_topic(tv)
        info = {"posts": len(texts), "hdbscan": chosen, "sweep": sweep}
        if leiden:
            with timer("leiden check"):
                info["leiden"] = cluster.leiden_check(tv, labels)
                log.info("Leiden cross-check: %s", info["leiden"])
        with timer("keywords"):
            kw = ctfidf_keywords(texts, labels)
        per_topic[num] = (mask, texts, tv, labels, kw)
        run_log["topics"][name] = info

    kept["Cluster"] = -1
    kept["Consolidated Theme"] = None
    kept["Theme Description"] = None
    kept["Theme Assigned By"] = None
    kept["Theme Truncated"] = False
    with llm.start(ctx) as url:
        client = llm.Client(url, ctx)
        for num, (mask, texts, tv, labels, kw) in per_topic.items():
            name = available[num]
            info = run_log["topics"][name]
            with timer("llm naming"):
                named = themes.name_clusters(client, name, texts, tv, labels, kw)
            with timer("llm merge"):
                final = themes.merge_themes(client, name, named, texts, tv, labels)
            theme_of, floor, reassigned = themes.assign_noise(tv, labels, final)
            desc = {t: d for t, d in final.values()}
            desc["Other"] = "Did not fit any theme in this topic closely enough."
            truncated_themes = {final[c][0] for c, n in named.items() if n["truncated"]}

            rows = kept.index[mask]
            kept.loc[rows, "Cluster"] = labels
            kept.loc[rows, "Consolidated Theme"] = theme_of
            kept.loc[rows, "Theme Description"] = [desc[t] for t in theme_of]
            kept.loc[rows, "Theme Assigned By"] = np.where(
                labels >= 0, "cluster", np.where(reassigned, "nearest theme", "below floor"))
            kept.loc[rows, "Theme Truncated"] = [t in truncated_themes for t in theme_of]

            info["clusters"] = {int(c): {k: v for k, v in n.items()} | {"final_theme": final[c][0]}
                                for c, n in named.items()}
            info["noise_floor"] = floor
            info["largest_cluster_tokens"] = max((n["prompt_tokens"] for n in named.values()), default=0)
            info["truncated_clusters"] = [int(c) for c, n in named.items() if n["truncated"]]

    run_log["timings_s"] = {k: round(v, 1) for k, v in timer.times.items()}
    run_log["finished"] = datetime.now().isoformat(timespec="seconds")
    out_name = out_name or f"sov_themes_{datetime.now():%Y%m%d_%H%M}"
    write_outputs(kept, run_log, out_name)
    return kept, run_log


def summary_table(kept):
    rows = []
    for (topic, theme), g in kept.groupby(["topic_clean", "Consolidated Theme"], sort=True):
        clustered = g[g["Theme Assigned By"] == "cluster"]
        examples = (clustered if len(clustered) else g)[config.TEXT_COL].astype(str).head(5)
        row = {"Topic": topic, "Consolidated Theme": theme,
               "Description": g["Theme Description"].iloc[0],
               "Posts": len(g),
               "From noise": int((g["Theme Assigned By"] == "nearest theme").sum()),
               "Truncated in LLM step": bool(g["Theme Truncated"].any())}
        for i, ex in enumerate(examples, 1):
            row[f"Example {i}"] = ex[:1000]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["Topic", "Posts"], ascending=[True, False])


def write_outputs(kept, run_log, out_name):
    xlsx = config.OUTPUT_DIR / f"{out_name}.xlsx"
    original_cols = [c for c in kept.columns if c not in ("topic_clean", "topic_num")]
    with pd.ExcelWriter(xlsx) as w:
        kept.rename(columns={"topic_clean": "Topic"})[["Topic"] + original_cols].to_excel(
            w, sheet_name="Posts", index=False)
        summary_table(kept).to_excel(w, sheet_name="Theme Summary", index=False)
        params = [{"Topic": t, "Posts": i["posts"], **{k: v for k, v in i["hdbscan"].items()},
                   "Leiden": json.dumps(i.get("leiden")) if i.get("leiden") else "",
                   "Noise floor": i.get("noise_floor"),
                   "Largest cluster prompt tokens": i.get("largest_cluster_tokens"),
                   "Truncated clusters": ", ".join(map(str, i.get("truncated_clusters", [])))}
                  for t, i in run_log["topics"].items()]
        pd.DataFrame(params).to_excel(w, sheet_name="Run Log", index=False)
        pd.DataFrame([run_log["timings_s"]]).to_excel(w, sheet_name="Run Log", index=False,
                                                       startrow=len(params) + 3)
    with open(config.OUTPUT_DIR / f"{out_name}_runlog.json", "w") as f:
        json.dump(run_log, f, indent=2, default=float)
    log.info("Wrote %s (+ _runlog.json)", xlsx)
