"""Post embeddings with Qwen3-Embedding-4B running in llama.cpp.

Vectors are cached on disk by a hash of the exact text sent to the model, so a
rerun (or a run on another topic) only embeds posts it hasn't seen before.
"""

import hashlib
import logging

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
    keys = np.array(list(cache.keys()))
    vecs = np.stack(list(cache.values())).astype(np.float32)
    np.savez(_cache_file(), keys=keys, vecs=vecs)


def server_args():
    # Every layer goes on the GPU (-ngl 99), so if it doesn't fit the server
    # fails loudly instead of quietly moving layers to the CPU. The trick is a
    # small micro-batch (-ub 512): the model is causal, so a long post can be
    # fed through in several 512-token steps and the last-token vector comes out
    # the same (we checked: cosine 0.998 against a single 8k step on CPU).
    # The 8k context and batch cover the longest post, about 5.8k tokens.
    return ["--embedding", "--pooling", "last", "-ngl", "99", "-fa", "on",
            "-c", "8192", "-np", "1", "-b", "8192", "-ub", "512",
            "-ctk", "q8_0", "-ctv", "q8_0"]


def embed_posts(texts):
    """Return an (n, dim) array of L2-normalised vectors, one per post."""
    # Qwen3-Embedding pools the final end-of-text token. llama.cpp appends it
    # for us (the GGUF sets add_eos), so we don't add another one here.
    prompts = [config.EMBED_INSTRUCTION + t for t in texts]
    keys = [_key(p) for p in prompts]
    cache = _load_cache()
    todo = sorted({k: p for k, p in zip(keys, prompts) if k not in cache}.items())
    log.info("%d posts, %d already cached, %d to embed", len(texts), len(texts) - len(todo), len(todo))

    if todo:
        model = model_path(config.EMBED_REPO, config.EMBED_FILE)
        with llama_server(model, server_args()) as url:
            progress.ui.task("Embedding posts", len(todo))
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                r = requests.post(url + "/v1/embeddings",
                                  json={"input": [p for _, p in chunk]}, timeout=600)
                r.raise_for_status()
                for (k, _), item in zip(chunk, sorted(r.json()["data"], key=lambda d: d["index"])):
                    cache[k] = np.asarray(item["embedding"], dtype=np.float32)
                progress.ui.advance(len(chunk))
                if (i // BATCH) % 20 == 0:
                    log.info("  embedded %d / %d", min(i + BATCH, len(todo)), len(todo))
        _save_cache(cache)

    vecs = np.stack([cache[k] for k in keys])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
