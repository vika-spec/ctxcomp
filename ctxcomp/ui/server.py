"""Context Inspector server and static export.

- `python -m ctxcomp ui` runs the scenarios, then serves the inspector locally.
  With --backend ollama/openai it also enables the Live chat tab.
- `python -m ctxcomp ui --export inspector.html` writes a self-contained page.

Only the Python standard library is used for serving.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from ..context import REGISTRY, make_context
from ..eval.scenario import Scenario
from ..llm import LLMBackend
from .trace import TraceRecorder, scenario_meta, trace_scenario

INDEX = Path(__file__).with_name("index.html")
PLACEHOLDER = "<!--CTXCOMP_DATA-->"


def _default_make(strategy: str, max_ctx: int):
    return make_context(strategy, max_context_tokens=max_ctx)


def build_payload(scenarios: list[Scenario], runs: list[tuple[str, int]],
                  make: Callable[[str, int], Any] | None = None,
                  trace_backend: LLMBackend | None = None, progress: bool = False) -> dict[str, Any]:
    """Trace every scenario for every (strategy, context window) run."""
    make = make or _default_make
    out = []
    for sc in scenarios:
        traced = []
        for strategy, ctx in runs:
            t0 = time.perf_counter()
            traced.append(trace_scenario(sc, partial(make, strategy, ctx), backend=trace_backend))
            if progress:
                r = traced[-1]
                print(f"  {sc.id:28s} {r['label']:26s} {'PASS' if r['summary']['task_pass'] else 'FAIL'}"
                      f"  ({time.perf_counter() - t0:.1f}s)")
        out.append({**scenario_meta(sc), "runs": traced})
    return {"generated": time.strftime("%Y-%m-%d %H:%M"), "scenarios": out}


def render_page(payload: dict[str, Any], standalone: bool = True) -> str:
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = INDEX.read_text(encoding="utf-8").replace(PLACEHOLDER, f"<script>window.__CTXCOMP__={data};</script>")
    if not standalone:  # the Artifact publisher adds its own document skeleton
        return html
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body style="margin:0">\n' + html + "\n</body>\n</html>\n")


class _Sessions:
    def __init__(self, backend: LLMBackend, make: Callable[[str, int], Any], max_steps: int) -> None:
        self.backend, self.make, self.max_steps = backend, make, max_steps
        self.items: dict[str, tuple[TraceRecorder, threading.Lock]] = {}

    def create(self, strategy: str, max_ctx: int) -> tuple[str, TraceRecorder]:
        if strategy not in REGISTRY:
            raise ValueError(f"unknown strategy '{strategy}'")
        rec = TraceRecorder(self.make(strategy, int(max_ctx)), self.backend, max_steps=self.max_steps)
        sid = uuid.uuid4().hex[:12]
        self.items[sid] = (rec, threading.Lock())
        return sid, rec


def serve(payload: dict[str, Any], host: str = "127.0.0.1", port: int = 8765,
          backend: LLMBackend | None = None, live_info: dict[str, Any] | None = None,
          make_context: Callable[[str, int], Any] | None = None,
          max_steps: int = 5, open_browser: bool = True) -> None:
    payload = dict(payload, mode="server",
                   live={"available": backend is not None, **(live_info or {})})
    page = render_page(payload).encode("utf-8")
    sessions = _Sessions(backend, make_context or _default_make, max_steps) if backend else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter console
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if sessions is None:
                return self._json(400, {"error": "live chat needs --backend ollama or openai"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "invalid JSON"})
            try:
                if self.path == "/api/session":
                    sid, rec = sessions.create(body.get("strategy", "naive"), body.get("max_ctx", 8192))
                    return self._json(200, {"id": sid, "run": rec.export()})
                if self.path == "/api/send":
                    item = sessions.items.get(body.get("id", ""))
                    if not item:
                        return self._json(404, {"error": "session expired, start a new one"})
                    rec, lock = item
                    text = str(body.get("text", "")).strip()
                    if not text:
                        return self._json(400, {"error": "empty message"})
                    with lock:
                        rec.step(text)
                    return self._json(200, {"run": rec.export()})
                return self._json(404, {"error": "not found"})
            except Exception as e:  # report model/server failures to the page
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"\nContext Inspector running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
