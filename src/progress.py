"""Live progress display for the terminal, built on rich.

The rest of the code talks to the module-level `ui` object: start a task with a
total, advance it, mark topic status, and log events. When the display is off
(output isn't a terminal) the same calls just do nothing, and
plain log lines still go to the console and to cache/run.log.
"""

import logging
import subprocess
import time
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TextColumn, TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from rich.text import Text

STATUS_STYLE = {"waiting": "dim", "running": "bold yellow", "done": "bold green",
                "failed": "bold red"}


def _gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2).stdout
        used, total, util, temp = [x.strip() for x in out.split(",")]
        return f"GPU {used}/{total} MiB  {util}% busy  {temp}°C"
    except Exception:
        return "GPU: not available"


def _ram_line():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k] = int(v.split()[0]) // 1024
        used = info["MemTotal"] - info["MemAvailable"]
        swap = info["SwapTotal"] - info["SwapFree"]
        return f"RAM {used}/{info['MemTotal']} MiB  swap {swap} MiB"
    except Exception:
        return ""


class _LogToPanel(logging.Handler):
    def __init__(self, dash):
        super().__init__(logging.INFO)
        self.dash = dash

    def emit(self, record):
        self.dash.event(record.getMessage(), level=record.levelno)


class Dashboard:
    def __init__(self):
        self.live = None
        self.topics = {}          # name -> dict of columns
        self.events = deque(maxlen=14)
        self.stage = "starting"
        self.started = time.time()
        self._sys = ("", "")
        self._sys_at = 0
        self.progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                                 MofNCompleteColumn(), TimeElapsedColumn(),
                                 TextColumn("eta"), TimeRemainingColumn())
        self.task_id = None
        self.embed = None
        self.llm = {"calls": 0, "cached": 0, "gen_tokens": 0, "gen_seconds": 0.0,
                    "last": None, "context": None, "busy_since": None, "busy_name": ""}

    # --- lifecycle -------------------------------------------------------
    def start(self, console=None):
        self.console = console or Console()
        self.live = Live(self, console=self.console, refresh_per_second=4, transient=False)
        self.live.start()
        logging.getLogger().addHandler(_LogToPanel(self))

    def stop(self):
        if self.live:
            self.live.stop()
            self.live = None

    # --- things the pipeline calls --------------------------------------
    def set_stage(self, text):
        self.stage = text

    def add_topic(self, name, posts):
        self.topics[name] = {"posts": posts, "status": "waiting", "step": "", "clusters": "",
                             "themes": "", "noise": "", "time": "", "_t": None}

    def topic(self, name, **fields):
        t = self.topics.get(name)
        if t is None:
            return
        if fields.get("status") == "running" and t["_t"] is None:
            t["_t"] = time.time()
        t.update({k: str(v) for k, v in fields.items()})

    def task(self, description, total):
        if self.task_id is not None:
            self.progress.remove_task(self.task_id)
        self.task_id = self.progress.add_task(description, total=total)

    def advance(self, n=1, description=None):
        if self.task_id is not None:
            self.progress.update(self.task_id, advance=n,
                                 **({"description": description} if description else {}))

    def embed_start(self, cached):
        self.embed = {"posts": 0, "tokens": 0, "seconds": 0.0, "cached": cached, "last": None}

    def embed_done(self, posts, tokens, seconds):
        e = self.embed
        if e is None:
            return
        e["posts"] += posts
        e["tokens"] += tokens
        e["seconds"] += seconds
        e["last"] = (posts, tokens, seconds)

    def _embed_line(self):
        e = self.embed
        if not e or not e["last"]:
            return None
        posts, tokens, secs = e["last"]
        line = Text("Embedder  ", style="bold")
        line.append(f"last batch {tokens / secs:.0f} tok/s", style="bold green")
        line.append(f" ({posts} posts, {tokens} tokens, {secs:.1f}s)   ")
        line.append(f"average {e['tokens'] / e['seconds']:.0f} tok/s, "
                    f"{e['posts'] / e['seconds']:.1f} posts/s   ", style="green")
        line.append(f"embedded {e['posts']}")
        if e["cached"]:
            line.append(f" (+{e['cached']} from cache)", style="dim")
        return line

    def llm_context(self, ctx, cpu_layers):
        self.llm["context"] = (ctx, cpu_layers)

    def llm_busy(self, name):
        self.llm["busy_since"] = time.time()
        self.llm["busy_name"] = name

    def llm_done(self, prompt_tokens, gen_tokens, prompt_per_s, gen_per_s, seconds):
        s = self.llm
        s["busy_since"] = None
        s["calls"] += 1
        s["gen_tokens"] += gen_tokens
        s["gen_seconds"] += gen_tokens / gen_per_s if gen_per_s else 0
        s["last"] = (prompt_tokens, gen_tokens, prompt_per_s, gen_per_s, seconds)

    def llm_cached(self):
        self.llm["cached"] += 1

    def _llm_line(self):
        s = self.llm
        if s["context"] is None and not s["calls"] and not s["cached"]:
            return None
        line = Text("LLM  ", style="bold")
        if s["context"]:
            ctx, layers = s["context"]
            line.append(f"{ctx // 1024}k context, {layers} layers on CPU   ", style="dim")
        if s["last"]:
            p_tok, g_tok, p_rate, g_rate, secs = s["last"]
            line.append(f"last call {g_rate:.1f} tok/s", style="bold green")
            line.append(f" ({g_tok} generated incl. thinking, prompt {p_tok} tokens at "
                        f"{p_rate:.0f} tok/s, {secs:.0f}s)   ")
        if s["gen_seconds"]:
            line.append(f"average {s['gen_tokens'] / s['gen_seconds']:.1f} tok/s   ", style="green")
        line.append(f"calls {s['calls']}")
        if s["cached"]:
            line.append(f" (+{s['cached']} from cache)", style="dim")
        if s["busy_since"]:
            line.append(f"   waiting on {s['busy_name']} for {time.time() - s['busy_since']:.0f}s",
                        style="yellow")
        return line

    def event(self, text, level=logging.INFO):
        stamp = time.strftime("%H:%M:%S")
        style = "red" if level >= logging.ERROR else "yellow" if level >= logging.WARNING else ""
        self.events.append((stamp, text, style))

    # --- drawing ---------------------------------------------------------
    def _topic_table(self):
        tbl = Table(expand=True, header_style="bold", box=None, pad_edge=False)
        for col, kw in [("Topic", {"ratio": 4}), ("Posts", {"justify": "right"}),
                        ("Status", {}), ("Step", {"ratio": 2}), ("Clusters", {"justify": "right"}),
                        ("Themes", {"justify": "right"}), ("Noise", {"justify": "right"}),
                        ("Time", {"justify": "right"})]:
            tbl.add_column(col, **kw)
        for name, t in self.topics.items():
            took = t["time"]
            if t["_t"] and t["status"] == "running":
                took = f"{time.time() - t['_t']:.0f}s"
            tbl.add_row(name, str(t["posts"]), Text(t["status"], style=STATUS_STYLE.get(t["status"], "")),
                        t["step"], t["clusters"], t["themes"], t["noise"], took)
        return tbl

    def _events(self):
        lines = Text()
        for stamp, text, style in self.events:
            lines.append(f"{stamp} ", style="dim")
            lines.append(text[:220] + "\n", style=style)
        return lines

    def __rich__(self):
        if time.time() - self._sys_at > 2:
            self._sys = (_gpu_line(), _ram_line())
            self._sys_at = time.time()
        elapsed = time.time() - self.started
        header = Text.assemble(("Share of voice themes  ", "bold cyan"),
                               (f"{self.stage}", "bold"),
                               (f"   elapsed {elapsed / 60:.1f} min   ", "dim"),
                               (f"{self._sys[0]}   {self._sys[1]}", "magenta"))
        return Group(
            Panel(header, padding=(0, 1)),
            Panel(self._topic_table(), title="Topics", padding=(0, 1)),
            Panel(Group(self.progress, *[x for x in [self._embed_line(), self._llm_line()] if x]),
                  title="Current step", padding=(0, 1)),
            Panel(self._events(), title="What's happening", padding=(0, 1)),
        )


class _Quiet:
    """Stand-in used when the live display is off: every call does nothing."""

    def __getattr__(self, _):
        return lambda *a, **k: None


ui = _Quiet()


def enable():
    global ui
    ui = Dashboard()
    ui.start()
    return ui


def disable():
    global ui
    if isinstance(ui, Dashboard):
        ui.stop()
    ui = _Quiet()
