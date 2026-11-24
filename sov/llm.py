"""Thin client for the Gemma naming model served by llama.cpp."""

import json
import logging

import requests

from . import config
from .server import llama_server, model_path

log = logging.getLogger(__name__)


def server_args(ctx=config.LLM_CTX):
    # Flash attention plus a q8_0 KV cache is what lets a 128k window fit.
    # One slot, so the whole window is available to each request.
    return ["-c", str(ctx), "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
            "-ngl", config.LLM_GPU_LAYERS, "--jinja", "--reasoning-budget", "0",
            "-b", "2048", "-ub", "512"]


def start(ctx=config.LLM_CTX):
    return llama_server(model_path(config.LLM_REPO, config.LLM_FILE), server_args(ctx))


class Client:
    def __init__(self, url, ctx=config.LLM_CTX):
        self.url = url
        self.ctx = ctx

    def count_tokens(self, messages):
        r = requests.post(self.url + "/apply-template", json={"messages": messages}, timeout=120)
        r.raise_for_status()
        prompt = r.json()["prompt"]
        r = requests.post(self.url + "/tokenize", json={"content": prompt}, timeout=120)
        r.raise_for_status()
        return len(r.json()["tokens"])

    def prompt_budget(self):
        return self.ctx - config.LLM_MAX_OUTPUT - 256

    def chat_json(self, messages, schema, name="answer"):
        body = {
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": config.LLM_MAX_OUTPUT,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": name, "schema": schema, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = requests.post(self.url + "/v1/chat/completions", json=body, timeout=3600)
        r.raise_for_status()
        out = r.json()
        timings = out.get("timings", {})
        log.info("  LLM: %s prompt tokens, %s generated, %.0fs",
                 out["usage"]["prompt_tokens"], out["usage"]["completion_tokens"],
                 (timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)) / 1000)
        return json.loads(out["choices"][0]["message"]["content"]), out["usage"]
