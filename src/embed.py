"""Post embeddings with Qwen3-Embedding-4B running in llama.cpp.

Vectors are cached on disk by a hash of the exact text sent to the model, so a
rerun (or a run on another topic) only embeds posts it hasn't seen before.
"""

import hashlib
import logging
import time

import numpy as np
import requests

from . import config, progress
from .server import llama_server, model_path

log = logging.getLogger(__name__)

BATCH = 32


def _key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_file():
    return config.CACHE_DIR / f"embeddings-{config.EMBED_FILE.removesuffix('.gguf')}.npz"


def _load_cache():
    path = _cache_file()
    if not path.exists():
        return {}
    z = np.load(path)
    return dict(zip(z["keys"].tolist(), z["vecs"]))


def _save_cache(cache):
    if not cache:
        return
    keys = np.array(list(cache.keys()))
    vecs = np.stack(list(cache.values())).astype(np.float32)
    np.savez(_cache_file(), keys=keys, vecs=vecs)


# Ways to start the embedder, best first. All on the GPU is fastest; if
# another program is holding some GPU memory, a smaller micro-batch frees a
# little, and after that a few layers move to the CPU (slower, but it runs).
GPU_SETUPS = [(99, 512), (99, 256), (34, 256), (32, 256), (28, 256), (20, 256), (0, 512)]


def server_args(gpu_layers=99, ubatch=512):
    # A small micro-batch (-ub) is what lets the model sit fully on a 4 GB GPU:
    # the model is causal, so a long post can be fed through in several steps
    # and the last-token vector comes out the same (we checked: cosine 0.998
    # against a single 8k step on CPU). The 8k context and batch cover the
    # longest post, about 5.8k tokens.
    return ["--embedding", "--pooling", "last", "-ngl", str(gpu_layers), "-fa", "on",
            "-c", "8192", "-np", "1", "-b", "8192", "-ub", str(ubatch),
            "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "0"]


def _start(model):
    """Start the embedder with the best setup that fits in free GPU memory."""
    for gpu_layers, ubatch in GPU_SETUPS:
        cm = llama_server(model, server_args(gpu_layers, ubatch))
        try:
            url = cm.__enter__()
        except RuntimeError:
            log.warning("Embedder didn't fit on the GPU with %s layers and micro-batch %d "
                        "(another program may be using GPU memory); trying a lighter setup",
                        "all" if gpu_layers == 99 else gpu_layers, ubatch)
            continue
        if gpu_layers != 99:
            log.warning("Embedder running with only %d layers on the GPU, so it will be slower. "
                        "Free up GPU memory (check nvidia-smi) for full speed.", gpu_layers)
        return cm, url
    raise RuntimeError("the embedder couldn't start even on the CPU; see cache/llama-server-*.log")


def embed_posts(texts):
    """Return an (n, dim) array of L2-normalised vectors, one per post."""
    # Qwen3-Embedding pools the final end-of-text token. llama.cpp appends it
    # for us (the GGUF sets add_eos), so we don't add another one here.
    prompts = [config.EMBED_INSTRUCTION + t for t in texts]
    keys = [_key(p) for p in prompts]
    cache = _load_cache()
    distinct = dict(zip(keys, prompts))
    todo = sorted((k, p) for k, p in distinct.items() if k not in cache)
    log.info("%d posts (%d distinct texts): %d already cached, %d to embed",
             len(texts), len(distinct), len(distinct) - len(todo), len(todo))

    if todo:
        model = model_path(config.EMBED_REPO, config.EMBED_FILE)
        progress.ui.embed_start(len(distinct) - len(todo))
        cm, url = _start(model)
        try:
            progress.ui.task("Embedding posts", len(todo))
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                t0 = time.time()
                r = requests.post(url + "/v1/embeddings",
                                  json={"input": [p for _, p in chunk]}, timeout=600)
                r.raise_for_status()
                tokens = r.json().get("usage", {}).get("prompt_tokens", 0)
                progress.ui.embed_done(len(chunk), tokens, time.time() - t0)
                for (k, _), item in zip(chunk, sorted(r.json()["data"], key=lambda d: d["index"])):
                    cache[k] = np.asarray(item["embedding"], dtype=np.float32)
                progress.ui.advance(len(chunk))
                if (i // BATCH) % 20 == 0:
                    log.info("  embedded %d / %d (%.0f tokens/s)", min(i + BATCH, len(todo)),
                             len(todo), tokens / max(time.time() - t0, 1e-6))
                    _save_cache(cache)  # so an interrupted run keeps what it has done
        finally:
            cm.__exit__(None, None, None)
            _save_cache(cache)

    vecs = np.stack([cache[k] for k in keys])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
