#!/usr/bin/env python3
"""
Remote SiamRPN++ training monitor.

Run on the training machine (from the repo root):
    python remote_monitor_server.py

Then open from any browser on any device:
    http://<training-machine-ip>:8765

JSON API:
    http://<training-machine-ip>:8765/status
    http://<training-machine-ip>:8765/log
"""

import glob
import html
import json
import os
import re
import socket
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG_DIR = "pysot/logs/all_datasets"
EVAL_DIR = "eval_results"
DEFAULT_PORT = 8765
UPDATE_INTERVAL = 10  # seconds between background state refreshes

# Matches completed-epoch summary lines, e.g.:
#   Epoch [  92/500]  train=0.8401  val=0.8123  lr=5.00e-03  best_val=0.7654  ES=2/50
_EPOCH_RE = re.compile(
    r"Epoch\s+\[\s*(\d+)/(\d+)\]\s+"
    r"train=([\d.]+)\s+val=([\d.]+)\s+"
    r"lr=([\d.e+\-]+)\s+best_val=([\d.]+)\s+"
    r"ES=(\d+)/(\d+)"
)
# Matches intra-epoch step lines, e.g.:
#   Epoch[92] step[150/312] loss=0.8421
_STEP_RE = re.compile(
    r"Epoch\[(\d+)\]\s+step\[(\d+)/(\d+)\]\s+loss=([\d.]+)"
)


class TrainingState:
    def __init__(self):
        self._lock = threading.Lock()
        self._d = {
            "status": "waiting",
            "epoch": None,
            "total_epochs": None,
            "progress_pct": None,
            "step": None,
            "total_steps": None,
            "step_loss": None,
            "train_loss": None,
            "val_loss": None,
            "best_val": None,
            "lr": None,
            "es_counter": None,
            "es_patience": None,
            "log_file": None,
            "last_update": None,
            "eval_results": {},
        }
        self._lines = []

    @property
    def recent_lines(self):
        with self._lock:
            return list(self._lines)

    def to_dict(self):
        with self._lock:
            return dict(self._d)

    def update(self):
        logs = sorted(glob.glob(f"{LOG_DIR}/train_ddp_*.log"))
        if not logs:
            with self._lock:
                self._d["status"] = "waiting"
                self._d["last_update"] = datetime.now().isoformat()
            return

        log_path = logs[-1]
        try:
            with open(log_path) as f:
                lines = f.readlines()
        except IOError:
            return

        epoch_d = {}
        step_d = {}
        for line in reversed(lines):
            if not epoch_d:
                m = _EPOCH_RE.search(line)
                if m:
                    epoch_d = dict(
                        epoch=int(m.group(1)),
                        total_epochs=int(m.group(2)),
                        train_loss=float(m.group(3)),
                        val_loss=float(m.group(4)),
                        lr=float(m.group(5)),
                        best_val=float(m.group(6)),
                        es_counter=int(m.group(7)),
                        es_patience=int(m.group(8)),
                    )
            if not step_d:
                m = _STEP_RE.search(line)
                if m:
                    step_d = dict(
                        step_epoch=int(m.group(1)),
                        step=int(m.group(2)),
                        total_steps=int(m.group(3)),
                        step_loss=float(m.group(4)),
                    )
            if epoch_d and step_d:
                break

        age = time.time() - os.path.getmtime(log_path)
        tail = lines[-30:]

        if any("EARLY STOPPING TRIGGERED" in l for l in tail):
            status = "early_stopped"
        elif any("Training complete" in l or "Finished" in l for l in lines[-10:]):
            status = "completed"
        elif epoch_d:
            status = "running" if age < 300 else "stalled"
        else:
            status = "starting"

        # Only show intra-epoch step data when it's for the current epoch
        step = step_epoch = total_steps = step_loss = None
        if step_d:
            current_epoch = epoch_d.get("epoch")
            if current_epoch is None or step_d["step_epoch"] >= current_epoch:
                step = step_d["step"]
                step_epoch = step_d["step_epoch"]
                total_steps = step_d["total_steps"]
                step_loss = step_d["step_loss"]

        epoch = epoch_d.get("epoch")
        total_epochs = epoch_d.get("total_epochs")
        progress_pct = round(100 * epoch / total_epochs, 1) if epoch and total_epochs else None

        eval_results = {}
        for path in sorted(glob.glob(f"{EVAL_DIR}/epoch_*.json"), reverse=True):
            try:
                with open(path) as f:
                    eval_results = json.load(f)
                break
            except (IOError, json.JSONDecodeError):
                continue

        with self._lock:
            self._d.update(
                status=status,
                epoch=epoch,
                total_epochs=total_epochs,
                progress_pct=progress_pct,
                step=step,
                total_steps=total_steps,
                step_loss=step_loss,
                train_loss=epoch_d.get("train_loss"),
                val_loss=epoch_d.get("val_loss"),
                best_val=epoch_d.get("best_val"),
                lr=epoch_d.get("lr"),
                es_counter=epoch_d.get("es_counter"),
                es_patience=epoch_d.get("es_patience"),
                log_file=log_path,
                last_update=datetime.now().isoformat(),
                eval_results=eval_results,
            )
            self._lines = lines[-60:]


def get_gpu_stats():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        gpus = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 6:
                gpus.append(
                    dict(
                        index=p[0],
                        name=p[1],
                        util_pct=p[2],
                        mem_used_mb=p[3],
                        mem_total_mb=p[4],
                        temp_c=p[5],
                    )
                )
        return gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"


# ── HTML template ──────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0 }
body { font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; padding: 16px }
h1 { color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 8px; margin-bottom: 4px; font-size: 1.15em }
h3 { color: #8b949e; font-size: .78em; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 10px }
.note { color: #484f58; font-size: .75em; text-align: right; margin-bottom: 14px }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-bottom: 10px }
.card { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 14px }
.metric { margin: 6px 0 }
.lbl { color: #8b949e; font-size: .78em }
.val { color: #58a6ff; font-size: 1.05em; font-weight: 700 }
.green { color: #3fb950 }
.yellow { color: #d29922 }
.red { color: #f85149 }
.status-label { font-size: .95em; font-weight: 700; margin-bottom: 10px }
.waiting, .starting { color: #8b949e }
.running { color: #3fb950 }
.stalled { color: #d29922 }
.completed { color: #3fb950 }
.early_stopped { color: #f85149 }
.bar-wrap { background: #0d1117; border-radius: 3px; height: 6px; margin: 4px 0 10px }
.bar { height: 6px; border-radius: 3px; background: #58a6ff }
.gpu-row { font-size: .82em; color: #8b949e; margin: 4px 0 }
.gpu-val { color: #c9d1d9 }
.log { background: #0d1117; border: 1px solid #21262d; border-radius: 4px; padding: 10px;
       font-size: .72em; max-height: 260px; overflow-y: auto; white-space: pre; color: #8b949e; margin-top: 10px }
@media (max-width: 500px) { .grid { grid-template-columns: 1fr } }
"""

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SiamRPN Monitor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15">
<style>{css}</style>
</head>
<body>
<h1>SiamRPN++ Training Monitor</h1>
<p class="note">Auto-refresh every 15 s &mdash; {ts}</p>

<div class="grid">
  <div class="card">
    <h3>Status</h3>
    <div class="status-label {status}">{status_label}</div>
    <div class="metric"><span class="lbl">Epoch</span><br>
      <span class="val">{epoch} / {total_epochs}</span></div>
    <div class="bar-wrap"><div class="bar" style="width:{pct}%"></div></div>
    <div class="metric"><span class="lbl">Overall progress</span>
      <span class="val"> {pct}%</span></div>
    {step_block}
  </div>

  <div class="card">
    <h3>Losses</h3>
    <div class="metric"><span class="lbl">Train</span><br>
      <span class="val">{train_loss}</span></div>
    <div class="metric"><span class="lbl">Val</span><br>
      <span class="val">{val_loss}</span></div>
    <div class="metric"><span class="lbl">Best val (all time)</span><br>
      <span class="val green">{best_val}</span></div>
  </div>

  <div class="card">
    <h3>Schedule</h3>
    <div class="metric"><span class="lbl">Learning rate</span><br>
      <span class="val">{lr}</span></div>
    <div class="metric"><span class="lbl">Early-stop counter</span><br>
      <span class="val {es_color}">{es_counter} / {es_patience}</span></div>
  </div>
</div>

{gpu_block}
{eval_block}

<div class="card">
  <h3>Recent log</h3>
  <div class="log">{log_text}</div>
</div>
</body>
</html>"""


def _f(v, n=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return str(v)


def build_html(d, gpus, lines):
    status = d.get("status", "waiting")
    labels = dict(
        waiting="Waiting — training not started",
        starting="Starting up…",
        running="Running",
        stalled="Stalled (no update >5 min)",
        completed="Completed",
        early_stopped="Early Stopped",
    )

    epoch = d.get("epoch") or "—"
    total_epochs = d.get("total_epochs") or "—"
    pct = d.get("progress_pct") or 0

    step_block = ""
    if d.get("step") and d.get("total_steps"):
        sp = round(100 * d["step"] / d["total_steps"])
        step_block = (
            f'<div class="metric"><span class="lbl">Current epoch step</span><br>'
            f'<span class="val">{d["step"]} / {d["total_steps"]} ({sp}%)'
            f' &mdash; loss {_f(d.get("step_loss"), 4)}</span></div>'
        )

    es_c = d.get("es_counter")
    es_p = d.get("es_patience")
    es_color = "green"
    if es_c is not None and es_p:
        ratio = es_c / es_p
        es_color = "red" if ratio > 0.7 else ("yellow" if ratio > 0.4 else "green")

    # GPU block
    if gpus:
        rows = []
        for g in gpus:
            try:
                mem_pct = round(100 * int(g["mem_used_mb"]) / int(g["mem_total_mb"]))
            except (ValueError, ZeroDivisionError):
                mem_pct = "?"
            rows.append(
                f'<div class="gpu-row">GPU {g["index"]} — {g["name"]}: '
                f'<span class="gpu-val">{g["util_pct"]}%</span> util &nbsp;|&nbsp; '
                f'<span class="gpu-val">{g["mem_used_mb"]}/{g["mem_total_mb"]} MB</span>'
                f' ({mem_pct}%) &nbsp;|&nbsp; '
                f'<span class="gpu-val">{g["temp_c"]}°C</span></div>'
            )
        gpu_block = f'<div class="card" style="margin-bottom:10px"><h3>GPUs</h3>{"".join(rows)}</div>'
    else:
        gpu_block = ""

    # Eval block
    er = d.get("eval_results", {})
    eval_block = ""
    if er and "overall" in er:
        ov = er["overall"]
        eval_block = (
            f'<div class="card" style="margin-bottom:10px">'
            f'<h3>Latest evaluation &mdash; epoch {er.get("epoch", "?")}</h3>'
            f'<div class="grid" style="margin-bottom:0">'
            f'<div class="metric"><span class="lbl">Mean IoU</span><br>'
            f'<span class="val green">{_f(ov.get("mean_iou"))}</span></div>'
            f'<div class="metric"><span class="lbl">Success@0.5</span><br>'
            f'<span class="val">{_f(ov.get("success_rate@0.5"))}</span></div>'
            f'<div class="metric"><span class="lbl">AUC</span><br>'
            f'<span class="val">{_f(ov.get("auc"))}</span></div>'
            f'</div></div>'
        )

    return _HTML.format(
        css=_CSS,
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status=status,
        status_label=labels.get(status, status),
        epoch=epoch,
        total_epochs=total_epochs,
        pct=pct,
        step_block=step_block,
        train_loss=_f(d.get("train_loss")),
        val_loss=_f(d.get("val_loss")),
        best_val=_f(d.get("best_val")),
        lr=f'{d["lr"]:.2e}' if d.get("lr") else "—",
        es_counter=es_c if es_c is not None else "—",
        es_patience=es_p if es_p is not None else "—",
        es_color=es_color,
        gpu_block=gpu_block,
        eval_block=eval_block,
        log_text=html.escape("".join(lines)) if lines else "(no log data yet)",
    ).encode("utf-8")


# ── Server ─────────────────────────────────────────────────────────────────────

_state = TrainingState()


def _bg_update():
    while True:
        _state.update()
        time.sleep(UPDATE_INTERVAL)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/status", "/status.json"):
            d = _state.to_dict()
            d["gpus"] = get_gpu_stats()
            body = json.dumps(d, indent=2).encode()
            self._send(200, "application/json", body)
        elif self.path.startswith("/log"):
            body = "".join(_state.recent_lines).encode()
            self._send(200, "text/plain; charset=utf-8", body)
        else:
            body = build_html(_state.to_dict(), get_gpu_stats(), _state.recent_lines)
            self._send(200, "text/html; charset=utf-8", body)

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass  # suppress per-request logs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SiamRPN++ remote training monitor")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    _state.update()
    threading.Thread(target=_bg_update, daemon=True).start()

    ip = _local_ip()
    print(f"Dashboard : http://{ip}:{args.port}")
    print(f"JSON API  : http://{ip}:{args.port}/status")
    print(f"Log tail  : http://{ip}:{args.port}/log")
    print("Ctrl+C to stop\n")

    server = HTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Monitor stopped.")
