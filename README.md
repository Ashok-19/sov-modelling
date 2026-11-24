# Share of voice: sub-themes within topics

Takes the classified social listening file and, inside each topic, groups the
posts into sub-themes and gives each group a plain-English name and
description. Everything runs on the laptop: no cloud APIs.

The plan behind it is in [docs/pipeline-plan.md](docs/pipeline-plan.md).

## How it works

1. **Load.** Read `data/ocrevus_ms_social_listening_classified.xlsx` (sheet
   "Master Data") and fold topic spelling variants into one name. The
   "Non-MS / Noise" topic is skipped. Post text is never edited.
2. **Dedupe.** Exact copies of a post within a topic are set aside in a
   separate file, each pointing at the Post_ID we kept. They get the same theme
   as that post.
3. **Embed.** Qwen3-Embedding-4B (GGUF, Q5_K_M) in llama.cpp with last-token
   pooling. Every post gets the same English instruction in front of it.
   Vectors are L2-normalised and cached in `cache/`, so reruns are quick.
4. **Cluster each topic.** PCA, then UMAP to 5 dimensions, then HDBSCAN. We
   never set a number of clusters. Instead we try a range of HDBSCAN density
   settings and keep the one with the best DBCV score that also holds up when
   UMAP is rerun with other seeds. `--leiden` adds a Leiden cross-check.
5. **Keywords.** c-TF-IDF per cluster with English and Spanish stopwords.
6. **Name.** Gemma 4 E4B reads the *whole* cluster in one call, plus its
   keywords and ~25 posts from neighbouring clusters for contrast, and returns
   `{theme, description, confidence}` as JSON. There is no chunking. If a
   cluster is too big for the context window, the log says so loudly and the
   theme is marked `Theme Truncated` in the output.
7. **Merge.** Gemma looks at all theme names in a topic and merges the ones
   that mean the same thing. There's no target count.
8. **Noise posts.** Posts HDBSCAN left out go to the closest theme if they are
   similar enough (the bar is set from the data), otherwise to "Other".
9. **Output** in `output/`:
   - `<name>.xlsx`: the posts with Topic, Cluster, Consolidated Theme, Theme
     Description and how the theme was assigned; a Theme Summary sheet with
     post counts and examples; a Run Log sheet with the chosen settings and
     timings.
   - `<name>_duplicates.xlsx`: the duplicates with the Post_ID they copy.
   - `<name>_runlog.json`: the full sweep, scores and timings.

The embedder and the LLM never run at the same time; each gets its own
llama-server that is shut down when its stage ends.

## Setup

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

You also need a CUDA build of `llama-server` on your PATH (or set
`LLAMA_SERVER`). The two GGUF models download into `~/models` on first use
(set `SOV_MODELS_DIR` to change that).

## Running

```bash
.venv/bin/python -m sov --list-topics      # what topics are in the file
.venv/bin/python -m sov --topics 9         # one topic
.venv/bin/python -m sov --topics 5 9 11    # a few
.venv/bin/python -m sov                    # all of them
```

Useful options: `--leiden` for the cross-check, `--ctx 65536` to use a
smaller LLM context if memory is short, `--out name` to pick the output name.
`SOV_LLM_NGL` sets how many Gemma layers go on the GPU.
