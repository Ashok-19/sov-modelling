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
# Largest context window the naming model may use. Gemma 4 E4B supports 128k.
LLM_CTX = int(os.environ.get("SOV_LLM_CTX", 131072))
# The context is sized to the job: the run measures its biggest prompt and picks
# the smallest window below that fits it plus the thinking budget. A bigger
# window needs more VRAM for the KV cache, so more of Gemma's feed-forward
# weights have to move to the CPU, which is much slower:
#   window   layers on CPU   speed measured on the 4 GB RTX 3050
#    32k          0            42 tokens/s
#    48k          6            31 tokens/s
#    64k         10            22 tokens/s
#    96k         20            12 tokens/s
#   128k         30            13 tokens/s
# A cluster that doesn't fit restarts the server one size up, so nothing is
# ever chunked. The per-layer embedding table always stays in RAM (it's only
# used for lookups). On a bigger GPU, set SOV_LLM_CPU_FFN_LAYERS=0.
CONTEXT_TIERS = [(32768, 0), (49152, 6), (65536, 10), (98304, 20), (131072, 30)]
LLM_CPU_FFN_LAYERS = os.environ.get("SOV_LLM_CPU_FFN_LAYERS")
LLM_MAX_OUTPUT = 1024
# Let Gemma think before it answers. Better answers, but each call takes longer.
# Set SOV_LLM_THINKING=0 to switch it off. The budget is a ceiling, not a target:
# it's set high so the model is never cut off mid-thought (in practice it uses
# under 1.5k tokens per answer). It's reserved in the context window next to the
# posts, so a bigger budget means a bigger, slower window (see above).
LLM_THINKING = os.environ.get("SOV_LLM_THINKING", "1") != "0"
LLM_THINKING_BUDGET = int(os.environ.get("SOV_LLM_THINKING_BUDGET", 16384))

PORT = int(os.environ.get("SOV_PORT", 8091))
