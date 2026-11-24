# Share of voice clustering pipeline: plan

**Goal:** within each pre-assigned topic, group posts into sub-themes and give each group a consolidated theme name.
**Rules:** fully local (16 GB RAM, 4 GB GPU); quality first, overnight runs OK; no cluster count (min or max) anywhere.

## Stages (run one at a time; free the GPU between stages)

1. **Prepare.** Load the Excel file and clean the topic names (merge variants such as "Care gaps" and "Care Gaps"). Keep the post text as-is: no stripping of RT, URLs or mentions. Remove exact duplicate posts and store them separately (with the ID of the kept original) for the later N2 table. Skip the "Non-MS / Noise" topic.
2. **Embed.** `Qwen/Qwen3-Embedding-4B` from `Qwen/Qwen3-Embedding-4B-GGUF`: Q5_K_M (2.89 GB), or Q6_K if it fits, or Q4_K_M. Run it with llama.cpp `--embedding --pooling last` and append `<|endoftext|>`. Put the same English instruction on every post: `Instruct: Identify the main topic or theme of the given social media post about multiple sclerosis treatments\nQuery: {raw post}`. L2-normalize the vectors and cache them to disk. Fallback: `microsoft/harrier-oss-v1-0.6b`, fp16, with the same prefix. Optional test: `codefuse-ai/F2LLM-v2-1.7B`. Before committing, compare the candidates on 300–500 hand-labelled English and Spanish posts (V-measure/ARI).
3. **Cluster per topic (count found automatically).** Reduce with PCA (about 50 dims), then UMAP (5-d, cosine, fixed seed). Sweep HDBSCAN density settings (min_cluster_size, min_samples). Keep the run with the best DBCV (`relative_validity_`) and stability across 3–5 seeds (ARI). Cross-check with Leiden on a kNN graph, with resolution chosen the same way.
4. **Keywords.** c-TF-IDF per cluster with English and Spanish stopwords. Used as hints in the prompt.
5. **Name clusters with Gemma 4 E4B** (llama.cpp, Q4_K_M or higher, thinking off, flash attention, q8 KV cache, JSON schema).
   - Cluster ≤ 32K tokens: send the whole cluster.
   - Larger cluster: split it into 16–24K chunks, summarize each chunk with counts, then name the cluster from the summaries.
   - Add 20–30 posts from neighbouring clusters as contrast, plus the keywords and the parent topic.
   - Output: `{theme, description, confidence}`.
6. **Merge themes.** Gemma compares the theme names and descriptions within each topic, grouped by embedding similarity, and merges any that mean the same thing. It never works to a target count.
7. **Outliers.** Assign each HDBSCAN noise post to the nearest final theme if its similarity clears a floor; otherwise mark it "Other".
8. **Output.** Write the original rows back with Topic, Consolidated theme and Theme description, plus a summary sheet listing each theme with its post count and 5 example posts.

## Quality checks
- LLM judge: choose the right theme for a post from its theme plus 3 sibling themes, and spot the intruder post among 5 posts from one cluster.
- Flag near-duplicate theme names (similarity > 0.85).
- Human spot-check of about 20 posts × 30 themes, covering both English and Spanish.

## Rollout
1. Pilot on the 2,716-post Ocrevus file and review the themes.
2. Run `llama-bench` on the laptop to get real speeds.
3. Full 70k overnight run (estimated 3–8 h).
