"""Thin client for the Gemma naming model served by llama.cpp."""

import hashlib
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
# A fixed seed so the same input gives the same names and merge decisions.
SEED = 42

EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


class Client:
    def __init__(self, url, ctx=config.LLM_CTX):
        self.url = url
        self.ctx = ctx
        # Answers are saved by a hash of the exact request, so rerunning the
        # same input (same posts, prompts, model and settings) is instant and
        # gives the same result. Delete the file to start fresh.
        self.cache_file = config.CACHE_DIR / "llm_answers.jsonl"
        self.cache = {}
        if self.cache_file.exists():
            for line in open(self.cache_file):
                item = json.loads(line)
                self.cache[item["key"]] = item["answer"]

    def _key(self, messages, schema, name):
        blob = json.dumps([config.LLM_FILE, config.LLM_THINKING, config.LLM_THINKING_BUDGET,
                           SEED, name, schema, messages], sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

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

    def chat_json(self, messages, schema, name="answer", tries=2):
        """Send a chat and get back JSON that matches `schema`.

        Tries twice. If both fail it raises ValueError, and the caller decides
        what to do, so one bad answer can't sink a long run.
        """
        key = self._key(messages, schema, name)
        if key in self.cache:
            return self.cache[key], {"prompt_tokens": 0, "completion_tokens": 0, "cached": True}
        for attempt in range(1, tries + 1):
            try:
                answer, usage = self._chat_json_once(messages, schema, name)
                empty = [k for k, v in answer.items() if isinstance(v, str) and not v.strip()]
                if empty:
                    raise ValueError(f"empty field(s) {empty}")
                self.cache[key] = answer
                with open(self.cache_file, "a") as f:
                    f.write(json.dumps({"key": key, "answer": answer}) + "\n")
                return answer, usage
            # A lost connection means the server itself has gone, so we let that
            # stop the run rather than fill the output with placeholders.
            except (ValueError, KeyError, requests.HTTPError, requests.Timeout) as e:
                log.warning("LLM %s answer unusable (try %d of %d): %s", name, attempt, tries, e)
        raise ValueError(f"no usable answer from the LLM for {name}")

    def chat_json_no_thinking(self, messages, schema, name="answer"):
        """Same as chat_json but with thinking off for this one call.

        A last resort when the thinking model keeps failing on a prompt: a
        plain answer is better than no name at all.
        """
        saved = config.LLM_THINKING
        config.LLM_THINKING = False
        try:
            return self.chat_json(messages, schema, name, tries=1)
        finally:
            config.LLM_THINKING = saved

    def _chat_json_once(self, messages, schema, name):
        """One request.

        With thinking on, the model reasons first and the server keeps that
        apart from the answer, so the JSON schema only applies to the answer.
        """
        if config.LLM_THINKING:
            body = {"messages": messages, "temperature": 0.2, "seed": SEED,
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
                    "temperature": 0.2, "seed": SEED, "n_predict": config.LLM_MAX_OUTPUT}
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
