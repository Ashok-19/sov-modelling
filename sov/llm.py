"""Thin client for the Gemma naming model served by llama.cpp."""

import json
import logging

import requests

from . import config
from .server import llama_server, model_path

log = logging.getLogger(__name__)


def server_args(ctx=config.LLM_CTX):
    # Flash attention plus a q8_0 KV cache is what lets a 128k window fit, and
    # -ub 256 keeps the compute buffer small. One slot, so each request gets
    # the whole window. See config.py for why some weights live in RAM.
    to_cpu = "per_layer_token_embd=CPU"
    n = config.LLM_CPU_FFN_LAYERS
    if n:
        layers = "|".join(str(i) for i in range(n))
        to_cpu += rf",blk\.({layers})\.ffn_(up|gate|down)=CPU"
    return ["-c", str(ctx), "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
            "-ngl", "99", "-ot", to_cpu, "-b", "2048", "-ub", "256",
            "--jinja", "--reasoning", "off"]


def start(ctx=config.LLM_CTX):
    return llama_server(model_path(config.LLM_REPO, config.LLM_FILE), server_args(ctx))


# Gemma 4 starts a hidden "thought" channel on its own even when the template
# and server say thinking is off, which made short answers take 40-50s. An
# empty thought block at the start of its turn makes it answer straight away.
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


class Client:
    def __init__(self, url, ctx=config.LLM_CTX):
        self.url = url
        self.ctx = ctx

    def _prompt(self, messages):
        r = requests.post(self.url + "/apply-template", json={"messages": messages}, timeout=120)
        r.raise_for_status()
        return r.json()["prompt"] + EMPTY_THOUGHT

    def count_tokens(self, messages):
        r = requests.post(self.url + "/tokenize",
                          json={"content": self._prompt(messages), "parse_special": True}, timeout=120)
        r.raise_for_status()
        return len(r.json()["tokens"])

    def prompt_budget(self):
        return self.ctx - config.LLM_MAX_OUTPUT - 256

    def chat_json(self, messages, schema, name="answer"):
        """Send a chat and get back JSON that matches `schema`."""
        body = {"prompt": self._prompt(messages), "json_schema": schema,
                "temperature": 0.2, "n_predict": config.LLM_MAX_OUTPUT}
        r = requests.post(self.url + "/completion", json=body, timeout=3600)
        r.raise_for_status()
        out = r.json()
        t = out.get("timings", {})
        log.info("  LLM %s: %s prompt tokens, %s generated, %.0fs", name,
                 out["tokens_evaluated"], out["tokens_predicted"],
                 (t.get("prompt_ms", 0) + t.get("predicted_ms", 0)) / 1000)
        usage = {"prompt_tokens": out["tokens_evaluated"], "completion_tokens": out["tokens_predicted"]}
        return json.loads(out["content"]), usage
