"""Paths, model files and the few settings shared across stages."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "ocrevus_ms_social_listening_classified.xlsx"
SHEET = "Master Data"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"

ID_COL = "Post_ID"
TEXT_COL = "Cleaned Post Content"
TOPIC_COL = "topics"

# Topic spellings that mean the same thing as an existing topic.
TOPIC_ALIASES = {
    "5. Unmet Needs & Care gaps": "5. Unmet Needs & Care Gaps",
    "10. Community & Events": "10. Awareness, Community & Events",
    "9. Healthcare Navigation": "9. Diagnosis & Healthcare Navigation",
}
# Posts the classifier marked as unrelated to MS. We don't cluster these.
SKIP_TOPIC_WORDS = ("non-ms", "noise")

MODELS_DIR = Path(os.environ.get("SOV_MODELS_DIR", Path.home() / "models"))
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "llama-server")

EMBED_REPO = "Qwen/Qwen3-Embedding-4B-GGUF"
EMBED_FILE = "Qwen3-Embedding-4B-Q5_K_M.gguf"
EMBED_INSTRUCTION = (
    "Instruct: Identify the main claim, topic or theme of the given social media post "
    "about multiple sclerosis (MS)\nQuery: "
)
# Posts with fewer real words than this ("This is interesting", "OMG!") say too
# little to shape a cluster. They're left out of clustering and naming, then
# placed in their nearest theme afterwards, the same way as noise posts.
MIN_WORDS_TO_CLUSTER = 5

LLM_REPO = "unsloth/gemma-4-E4B-it-GGUF"
LLM_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"
# Context window for the naming model. Gemma 4 E4B supports 128k; we use as
# much as the machine allows. Override with SOV_LLM_CTX if memory is tight.
LLM_CTX = int(os.environ.get("SOV_LLM_CTX", 131072))
# How Gemma is split between the 4 GB GPU and RAM at 128k context. Attention
# and the KV cache stay on the GPU; the per-layer embedding table (1.8 GB, only
# used for lookups) and the feed-forward weights of the first N layers go to RAM.
# 30 was the smallest N that ran a long prompt without running out of VRAM.
# Lower it if you use a smaller context or have a bigger GPU.
LLM_CPU_FFN_LAYERS = int(os.environ.get("SOV_LLM_CPU_FFN_LAYERS", 30))
LLM_MAX_OUTPUT = 1024
# Let Gemma think before it answers. Better answers, but each call takes longer.
# Set SOV_LLM_THINKING=0 to switch it off. The budget is a ceiling, not a target:
# it's set high so the model is never cut off mid-thought (in practice it uses
# under 1k tokens per answer). It comes out of the same 128k window as the posts.
LLM_THINKING = os.environ.get("SOV_LLM_THINKING", "1") != "0"
LLM_THINKING_BUDGET = int(os.environ.get("SOV_LLM_THINKING_BUDGET", 32768))

PORT = int(os.environ.get("SOV_PORT", 8091))
