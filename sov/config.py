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
    "Instruct: Identify the main topic or theme of the given social media post "
    "about multiple sclerosis treatments\nQuery: "
)

LLM_REPO = "unsloth/gemma-4-E4B-it-GGUF"
LLM_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"
# Context window for the naming model. Gemma 4 E4B supports 128k; we use as
# much as the machine allows. Override with SOV_LLM_CTX if memory is tight.
LLM_CTX = int(os.environ.get("SOV_LLM_CTX", 131072))
# Layers to put on the 4 GB GPU. Unset means let llama.cpp's --fit work it out,
# which keeps whatever doesn't fit (layers or KV cache) in system RAM.
LLM_GPU_LAYERS = os.environ.get("SOV_LLM_NGL")
LLM_MAX_OUTPUT = 1024

PORT = int(os.environ.get("SOV_PORT", 8091))
