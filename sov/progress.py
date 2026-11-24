"""Live progress display for the terminal, built on rich.

The rest of the code talks to the module-level `ui` object: start a task with a
total, advance it, mark topic status, and log events. When the display is off
(--no-ui, or output isn't a terminal) the same calls just do nothing, and
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
            Panel(self.progress, title="Current step", padding=(0, 1)),
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
