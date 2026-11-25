"""Start and stop a llama.cpp server, one model at a time.

The GPU only has 4 GB, so the embedder and the LLM never run together: each
stage opens a server in a `with` block and it is shut down when the block ends.
"""

import logging
import subprocess
import time
from contextlib import contextmanager

import requests
from huggingface_hub import hf_hub_download

from . import config

log = logging.getLogger(__name__)


def model_path(repo, filename):
    """Path to a GGUF file, downloading it the first time."""
    local = config.MODELS_DIR / filename
    if local.exists():
        return local
    log.info("Downloading %s from %s", filename, repo)
    return hf_hub_download(repo, filename, local_dir=config.MODELS_DIR)


@contextmanager
def llama_server(model, args, port=config.PORT, timeout=600):
    # A server left over from an earlier, interrupted run would answer our
    # health check and we'd quietly talk to the wrong model.
    try:
        requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        raise RuntimeError(f"Something is already listening on port {port}. Stop it "
                           "(it's probably an old llama-server) or set SOV_PORT.")
    except requests.ConnectionError:
        pass
    cmd = [config.LLAMA_SERVER, "-m", str(model), "--port", str(port),
           "--host", "127.0.0.1", *args]
    log.info("Starting: %s", " ".join(cmd))
    logfile = open(config.CACHE_DIR / f"llama-server-{port}.log", "w")
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        start = time.time()
        time.sleep(1)
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early, see {logfile.name}")
            try:
                if requests.get(url + "/health", timeout=2).status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            if time.time() - start > timeout:
                raise TimeoutError("llama-server did not become ready")
            time.sleep(1)
        log.info("Server ready after %.0fs", time.time() - start)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        logfile.close()
