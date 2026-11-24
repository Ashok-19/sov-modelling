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
            "--jinja", "--reasoning", "on" if config.LLM_THINKING else "off",
            "--reasoning-budget", str(config.LLM_THINKING_BUDGET if config.LLM_THINKING else 0)]


def start(ctx=config.LLM_CTX):
    return llama_server(model_path(config.LLM_REPO, config.LLM_FILE), server_args(ctx))


# With thinking switched off, Gemma 4 still opens a "thought" channel on its
# own. An empty thought block at the start of its turn makes it answer straight
# away. Only used when config.LLM_THINKING is off.
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


class Client:
    def __init__(self, url, ctx=config.LLM_CTX):
        self.url = url
        self.ctx = ctx

    def _prompt(self, messages):
        r = requests.post(self.url + "/apply-template", json={"messages": messages}, timeout=120)
        r.raise_for_status()
        prompt = r.json()["prompt"]
        return prompt if config.LLM_THINKING else prompt + EMPTY_THOUGHT

    def count_tokens(self, messages):
        r = requests.post(self.url + "/tokenize",
                          json={"content": self._prompt(messages), "parse_special": True}, timeout=120)
        r.raise_for_status()
        return len(r.json()["tokens"])

    def prompt_budget(self):
        return self.ctx - config.LLM_MAX_OUTPUT - config.LLM_THINKING_BUDGET - 256

    def chat_json(self, messages, schema, name="answer"):
        """Send a chat and get back JSON that matches `schema`.

        With thinking on, the model reasons first and the server keeps that
        apart from the answer, so the JSON schema only applies to the answer.
        """
        if config.LLM_THINKING:
            body = {"messages": messages, "temperature": 0.2,
                    "max_tokens": config.LLM_THINKING_BUDGET + config.LLM_MAX_OUTPUT,
                    "response_format": {"type": "json_schema",
                                        "json_schema": {"name": name, "schema": schema, "strict": True}},
                    "chat_template_kwargs": {"enable_thinking": True}}
            r = requests.post(self.url + "/v1/chat/completions", json=body, timeout=3600)
            r.raise_for_status()
            out = r.json()
            usage = out["usage"]
            content = out["choices"][0]["message"]["content"]
        else:
            body = {"prompt": self._prompt(messages), "json_schema": schema,
                    "temperature": 0.2, "n_predict": config.LLM_MAX_OUTPUT}
            r = requests.post(self.url + "/completion", json=body, timeout=3600)
            r.raise_for_status()
            out = r.json()
            usage = {"prompt_tokens": out["tokens_evaluated"], "completion_tokens": out["tokens_predicted"]}
            content = out["content"]
        t = out.get("timings", {})
        log.info("  LLM %s: %s prompt tokens, %s generated (incl. thinking), %.0fs", name,
                 usage["prompt_tokens"], usage["completion_tokens"],
                 (t.get("prompt_ms", 0) + t.get("predicted_ms", 0)) / 1000)
        return json.loads(content), usage
