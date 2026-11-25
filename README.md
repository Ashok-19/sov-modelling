# Share of voice: sub-themes within topics

Takes a social listening file where every post already has a topic, and inside
each topic groups the posts into sub-themes. Each theme gets a short name and
a description. Everything runs on your own machine; nothing is sent to a cloud
service.

## Quick start

```bash
# 1. once: create a virtual environment and install the packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. see what you can pass in
python run.py --help

# 3. run it
python run.py --list-topics              # topics found in the input file
python run.py -t 9                       # just topic 9
python run.py -t 5 9 11                  # a few topics
python run.py                            # every topic
python run.py -i my_posts.csv -o results -t 9   # your own file and output folder
```

The first run downloads the two models (about 7.5 GB in total) into the repo's
`model/` folder.
After that it works offline.

## What you need

- **Python 3.10 or newer.**
- **An NVIDIA GPU** with CUDA. 4 GB of VRAM is enough; 16 GB of RAM is
  recommended.
- **llama.cpp's `llama-server`, built with CUDA,** on your `PATH`. If you don't
  have it:

  ```bash
  git clone https://github.com/ggml-org/llama.cpp
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
  # then put build/bin on your PATH, or point LLAMA_SERVER at build/bin/llama-server
  ```

## Input file

An Excel file (`.xlsx`) or a CSV with the same columns as the original sheet.
Three columns are used:

| Column | What it holds |
| --- | --- |
| `Post_ID` | an id for each post |
| `Cleaned Post Content` | the post text (used exactly as it is) |
| `topics` | the topic, numbered like `9. Diagnosis & Healthcare Navigation` |

All other columns are carried through to the output untouched. For Excel, the
sheet is `Master Data` unless you pass `--sheet`. Topics called "Non-MS /
Noise" are skipped, and a few spelling variants of topic names are folded
together (see `TOPIC_ALIASES` in `sov/config.py`).

## Output

In the output folder (`output/` unless you pass `-o`):

- `<name>.xlsx`
  - **Posts:** the rows of the topics you ran plus Topic, Cluster,
    Consolidated Theme, Theme Description, how the theme was assigned, and how
    close the post is to its theme. Rows from the Non-MS / Noise topic (and
    any without a topic) are included too, untouched, with blank theme
    columns and "skipped" in Theme Assigned By.
  - **Theme Summary:** one row per theme with its post count, description
    and the five most typical posts.
  - **Run Log:** the clustering settings and scores per topic, and how long
    each stage took.
- `<name>_runlog.json`: the same, with every setting that was tried.

`<name>` is `sov_themes_<date>_<time>` unless you pass `--output-name`. A full log of
every run is kept in `cache/run.log`.

## What you see while it runs

A live display with:

- the topics and where each one is (clustering, naming, merging, done),
  with its cluster count, theme count and noise share;
- a progress bar with a time estimate for the current step;
- GPU memory, GPU load and RAM, so you can see if the machine is struggling;
- the latest events: settings chosen, each theme as it's named, merge
  decisions, and any warnings in yellow or errors in red.

If the output isn't a terminal (piped, or redirected to a file), you get plain
log lines instead.

## How it works

1. **Load** the file and tidy the topic names.
2. **Embed** every post with Qwen3-Embedding-4B (in llama.cpp, fully on the
   GPU). Vectors are cached in `cache/`, so a rerun only embeds new posts.
3. **Cluster each topic.** UMAP, then HDBSCAN. No number of clusters is ever
   set. A sweep of HDBSCAN settings is scored on DBCV and on how stable the
   result is across random seeds. Each cluster is then split again in the
   same way, until a split stops being stable.
4. **Keywords** per cluster (c-TF-IDF), used as hints for the next step.
5. **Name** each cluster with Gemma 4 E4B. It reads the whole cluster in one
   call (no chunking), and only that cluster's posts. Gemma thinks before it
   answers (up to 8k tokens). The context window is sized per topic to the
   biggest prompt plus the thinking budget, from 32k (all on the GPU, fastest)
   up to 128k, and grows mid-run if a prompt needs more. If a cluster is ever too big
   to fit, the log says so and the theme is flagged in the output.
6. **Merge** themes that are really the same. Each theme is compared with
   its nearest neighbours, using example posts from both, and merged only
   when they make the same point. There's no target count.
7. **Leftover posts** that HDBSCAN called noise go to the nearest theme if
   they're close enough, otherwise to "Other".

The embedding model and the LLM never run at the same time.

## Options worth knowing

| Option | What it does |
| --- | --- |
| `-t 5 9 11` | only these topic numbers (default: all) |
| `-i file.xlsx` / `-i file.csv` | input file |
| `--sheet "Master Data"` | Excel sheet to read |
| `-o folder` | output folder |
| `--output-name topic11` | base name of the result files (`topic11.xlsx`, `topic11_runlog.json`); reusing a name overwrites |
| `--leiden` | also run Leiden clustering as a cross-check |

Environment variables for unusual setups: `SOV_MODELS_DIR` (where models are
kept), `LLAMA_SERVER` (path to llama-server), `SOV_PORT` (port it uses), and
`SOV_LLM_CPU_FFN_LAYERS` (force how many Gemma layers keep their feed-forward
weights in RAM; by default it depends on the context size, see `sov/config.py`,
and 0 suits a GPU bigger than 4 GB). The LLM settings can be changed the same
way: `SOV_LLM_THINKING_BUDGET` (default 8192 tokens), `SOV_LLM_THINKING=0` to
switch thinking off, and `SOV_LLM_CTX` (the largest context allowed, default
131072).

