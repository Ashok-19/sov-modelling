"""Runs the stages in order and writes the outputs."""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import cluster, config, data, embed, llm, progress, themes
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


def run(input_path=config.DATA_FILE, sheet=config.SHEET, output_dir=config.OUTPUT_DIR,
        topic_numbers=None, leiden=False, out_name=None):
    ui = progress.ui
    output_dir = Path(output_dir)
    config.CACHE_DIR.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    run_log = {"started": datetime.now().isoformat(timespec="seconds"),
               "embed_model": config.EMBED_FILE, "llm_model": config.LLM_FILE,
               "llm_thinking": config.LLM_THINKING,
               "input": str(input_path), "topics": {}}

    ui.set_stage("Loading posts")
    with timer("load"):
        df = data.load(input_path, sheet)
        available = data.list_topics(df)
        wanted = sorted(set(topic_numbers)) if topic_numbers else list(available)
        missing = [t for t in wanted if t not in available]
        if missing:
            raise SystemExit(f"Unknown topic number(s) {missing}. Available: {available}")
        kept = df[df["topic_num"].isin(wanted)].copy()
        log.info("%d posts in selected topics", len(kept))
        for num in wanted:
            ui.add_topic(available[num], int((kept["topic_num"] == num).sum()))

    ui.set_stage("Embedding posts (Qwen3-Embedding-4B)")
    with timer("embed"):
        vecs = embed.embed_posts(kept[config.TEXT_COL].fillna("").astype(str).tolist())

    # Clustering and keywords need no model, so do all topics before loading the LLM.
    per_topic = {}
    ui.set_stage("Clustering each topic")
    for num in wanted:
        name = available[num]
        mask = (kept["topic_num"] == num).to_numpy()
        texts = kept.loc[mask, config.TEXT_COL].fillna("").astype(str).tolist()
        tv = vecs[mask]
        log.info("=== Topic %s: %d posts ===", name, len(texts))
        ui.topic(name, status="running", step="clustering")
        ui.task(f"Clustering: {name}", None)
        t0 = time.time()
        words = np.array([len(re.findall(r"[^\W\d_]{2,}", t)) for t in texts])
        substantive = words >= config.MIN_WORDS_TO_CLUSTER
        with timer("cluster"):
            labels = np.full(len(texts), -1)
            sub_labels, chosen, sweep = cluster.cluster_topic(tv[substantive])
            labels[substantive] = sub_labels
        log.info("%d short posts (under %d words) held back from clustering",
                 int((~substantive).sum()), config.MIN_WORDS_TO_CLUSTER)
        ui.topic(name, status="waiting", step="clustered, waiting for LLM",
                 clusters=labels.max() + 1, noise=f"{(labels < 0).mean():.0%}")
        info = {"posts": len(texts), "hdbscan": chosen, "sweep": sweep}
        if leiden:
            with timer("leiden check"):
                info["leiden"] = cluster.leiden_check(tv, labels)
                log.info("Leiden cross-check: %s", info["leiden"])
        with timer("keywords"):
            kw = ctfidf_keywords(texts, labels)
        per_topic[num] = (mask, texts, tv, labels, kw, substantive, time.time() - t0)
        run_log["topics"][name] = info

    kept["Cluster"] = -1
    kept["Consolidated Theme"] = None
    kept["Theme Description"] = None
    kept["Theme Assigned By"] = None
    kept["Theme Truncated"] = False
    kept["Theme Similarity"] = np.nan
    server = LLMServer(config.LLM_CTX, timer)
    try:
        ui.set_stage("Naming and merging themes (Gemma 4 E4B" +
                     (", thinking on)" if config.LLM_THINKING else ")"))
        for num, (mask, texts, tv, labels, kw, substantive, cluster_time) in per_topic.items():
            name = available[num]
            info = run_log["topics"][name]
            t0 = time.time() - cluster_time
            server.ensure(estimate_prompt_tokens(texts, tv, labels))
            ui.set_stage(f"Naming and merging themes (Gemma 4 E4B, {server.ctx // 1024}k context" +
                         (", thinking on)" if config.LLM_THINKING else ")"))
            while True:
                try:
                    ui.topic(name, status="running", step="naming clusters")
                    with timer("llm naming"):
                        named = themes.name_clusters(server.client, name, texts, tv, labels, kw)
                    ui.topic(name, step="merging themes")
                    with timer("llm merge"):
                        final = themes.merge_themes(server.client, name, named, texts, tv, labels)
                    break
                except llm.ContextTooSmall as e:
                    # Earlier answers are cached, so redoing the topic is quick.
                    log.info("A prompt needs %d tokens; restarting Gemma with a bigger context",
                             e.prompt_tokens)
                    server.ensure(e.prompt_tokens, grow_only=True)
            theme_of, floor, reassigned, closeness = themes.assign_noise(tv, labels, final)
            desc = {t: d for t, d in final.values()}
            desc["Other"] = "Did not fit any theme in this topic closely enough."
            truncated_themes = {final[c][0] for c, n in named.items() if n["truncated"]}

            rows = kept.index[mask]
            kept.loc[rows, "Cluster"] = labels
            kept.loc[rows, "Consolidated Theme"] = theme_of
            kept.loc[rows, "Theme Description"] = [desc[t] for t in theme_of]
            how = np.where(labels >= 0, "cluster",
                           np.where(reassigned, "nearest theme", "below floor"))
            how = np.where(~substantive, np.char.add("short post, ", how.astype(str)), how)
            kept.loc[rows, "Theme Assigned By"] = how
            kept.loc[rows, "Theme Truncated"] = [t in truncated_themes for t in theme_of]
            kept.loc[rows, "Theme Similarity"] = np.round(closeness, 3)

            info["clusters"] = {int(c): {k: v for k, v in n.items()} | {"final_theme": final[c][0]}
                                for c, n in named.items()}
            info["noise_floor"] = floor
            info["largest_cluster_tokens"] = max((n["prompt_tokens"] for n in named.values()), default=0)
            info["truncated_clusters"] = [int(c) for c, n in named.items() if n["truncated"]]
            n_themes = len({t for t in theme_of if t != "Other"})
            note = f"done, {len(info['truncated_clusters'])} truncated!" if info["truncated_clusters"] else "done"
            ui.topic(name, status="done", step=note, themes=n_themes, time=f"{time.time() - t0:.0f}s")
        run_log["llm_context"] = server.history
    finally:
        server.stop()

    run_log["timings_s"] = {k: round(v, 1) for k, v in timer.times.items()}
    run_log["finished"] = datetime.now().isoformat(timespec="seconds")
    out_name = out_name or f"sov_themes_{datetime.now():%Y%m%d_%H%M}"
    # Rows we never cluster (the Non-MS / Noise topic and posts with no topic)
    # still go in the output, untouched, so it has every input row.
    skipped = df[~df["topic_num"].isin(list(available))].copy()
    skipped["Theme Assigned By"] = np.where(skipped["topic_clean"].isna() | skipped["topic_num"].isna(),
                                            "skipped (no topic)", "skipped (noise topic)")
    skipped["Cluster"] = -1
    skipped["Theme Truncated"] = False
    out = pd.concat([kept, skipped]).sort_index()
    log.info("%d rows not clustered (noise topic or no topic) kept in the output as they are",
             len(skipped))

    ui.set_stage("Writing output")
    write_outputs(out, run_log, output_dir, out_name)
    ui.set_stage("Finished")
    return kept, run_log


def estimate_prompt_tokens(texts, vecs, labels):
    """A cautious guess at the biggest prompt a topic will need, in tokens.

    Builds the same post lists the naming and merge steps will send (every
    distinct post of a cluster; a theme's examples plus its nearest
    neighbours') and counts 3 characters per token, which overshoots for
    English, so the window chosen is usually big enough. If it's still short,
    the server grows mid-topic.
    """
    ids = sorted(set(labels) - {-1})
    if not ids:
        return 2000
    cents = themes.centroids(vecs, labels)
    naming = max(sum(len(t) for t in set(t for t, l in zip(texts, labels) if l == c)) for c in ids)
    examples = {c: sum(len(t) for t in themes._examples(texts, vecs, labels, c, cents)) for c in ids}
    cmat = np.stack([cents[c] for c in ids])
    merge = 0
    for i, c in enumerate(ids):
        near = [ids[j] for j in np.argsort(-(cmat @ cmat[i])) if ids[j] != c][:themes.MERGE_NEIGHBOURS]
        merge = max(merge, examples[c] + sum(examples[o] for o in near))
    return max(naming, merge) // 3 + 1500


class LLMServer:
    """The Gemma server, restarted with a different context size when needed."""

    def __init__(self, max_ctx, timer):
        self.max_ctx = max_ctx
        self.timer = timer
        self.cm = None
        self.client = None
        self.ctx = 0
        self.history = []

    def ensure(self, prompt_tokens, grow_only=False):
        """Make sure the running server fits a prompt of this size.

        Picks the smallest window that fits. Between topics it may also shrink,
        since a smaller window is much faster. Mid-topic (grow_only) it only grows.
        """
        ctx, cpu_layers = llm.pick_context(prompt_tokens)
        ctx = min(ctx, self.max_ctx)
        if grow_only and ctx <= self.ctx:
            bigger = [t for t in config.CONTEXT_TIERS if self.ctx < t[0] <= self.max_ctx]
            if not bigger:
                raise RuntimeError(f"a prompt of {prompt_tokens} tokens doesn't fit even the "
                                   f"largest context ({self.ctx})")
            ctx, cpu_layers = bigger[0]
        if ctx == self.ctx:
            return
        self.stop()
        # If the GPU has less room than the table expects (another program using
        # VRAM, say), move a few more layers to the CPU and try again.
        for attempt in range(4):
            progress.ui.set_stage(f"Loading Gemma 4 E4B ({ctx // 1024}k context)")
            try:
                with self.timer("llm load"):
                    self.cm = llm.start(ctx, cpu_layers)
                    url = self.cm.__enter__()
                break
            except RuntimeError:
                if attempt == 3:
                    raise
                log.warning("Gemma didn't fit with %d layers on the CPU; trying %d",
                            cpu_layers, cpu_layers + 6)
                cpu_layers += 6
        self.client = llm.Client(url, ctx)
        self.ctx = ctx
        self.history.append({"context": ctx, "cpu_layers": cpu_layers})

    def stop(self):
        if self.cm is not None:
            self.cm.__exit__(None, None, None)
            self.cm = None


def summary_table(kept):
    rows = []
    themed = kept[kept["Consolidated Theme"].notna()]
    for (topic, theme), g in themed.groupby(["topic_clean", "Consolidated Theme"], sort=True):
        # The most typical posts, skipping exact repeats so the examples differ.
        typical = g.sort_values("Theme Similarity", ascending=False).drop_duplicates(config.TEXT_COL)
        examples = typical[config.TEXT_COL].astype(str).head(5)
        row = {"Topic": topic, "Consolidated Theme": theme,
               "Description": g["Theme Description"].iloc[0],
               "Posts": len(g),
               "From noise": int(g["Theme Assigned By"].str.endswith("nearest theme").sum()),
               "Truncated in LLM step": bool(g["Theme Truncated"].any())}
        for i, ex in enumerate(examples, 1):
            row[f"Example {i}"] = ex[:1000]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["Topic", "Posts"], ascending=[True, False])


def write_outputs(kept, run_log, output_dir, out_name):
    xlsx = output_dir / f"{out_name}.xlsx"
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
    with open(output_dir / f"{out_name}_runlog.json", "w") as f:
        json.dump(run_log, f, indent=2, default=float)
    log.info("Wrote %s (+ _runlog.json)", xlsx)
