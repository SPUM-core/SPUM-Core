"""SPUM Web 实时可视化服务器 (标准库 HTTP + SSE, 零第三方依赖).

"分布式交互" = 在浏览器中实时观察 universe 演化、鼠标拖拽视角、
控制播放/暂停/重置/调帧率。前端用 Three.js (CDN) 渲染粒子球与边。

架构:
    ┌──────────┐   push   ┌──────────┐  SSE   ┌─────────┐
    │ SPUMEngine│ ──────► │Broadcaster│ ─────► │浏览器     │
    │  后台线程  │         └──────────┘        │ Three.js │
    └──────────┘                               └─────────┘
                     controls (POST /api/control) ▲
                          │                        │
                          └────────────────────────┘

GPU 部署: 传入 backend='auto' 时引擎自动探测 CUDA (见 gpu_backend)。
        CPU 与 GPU 帧语义严格等价 (test_gpu_backend.py 验证)。

运行:
    python webviz/server.py                          # CPU/auto 探测
    python webviz/server.py --backend cuda           # 强制 GPU
    python webviz/server.py --seed_geometry star --port 8080
    浏览器打开 http://localhost:8888 (或你指定的 --port)
"""

from __future__ import annotations

import argparse
import json
import os
import queue as _queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Set
from urllib.parse import urlparse, parse_qs

# 让 `from Phase_0...` 可导入 (脚本在 webviz/ 下运行)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from Phase_0.gpu_engine import SPUMEngine, EngineConfig

MAX_RENDER_NODES = 4000      # 渲染节点上限 (超出则抽样, 保护浏览器)
DEADLINE = 4096              # 每个 SSE 客户端队列上限


class Broadcaster:
    """fan-out: 引擎推快照 → 所有 SSE 订阅者队列。"""

    def __init__(self):
        self._subs: Set[_queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> _queue.Queue:
        q = _queue.Queue(maxsize=DEADLINE)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    def broadcast(self, obj: dict):
        payload = ("data: " + json.dumps(obj, separators=(",", ":")) + "\n\n")
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                # 满则丢最旧 (慢客户端), 保证实时性
                if q.full():
                    try:
                        q.get_nowait()
                    except _queue.Empty:
                        pass
                q.put_nowait(payload)
            except _queue.Full:
                pass


class Runtime:
    """持有引擎 + 控制状态 + 演化线程。"""

    def __init__(self, seed_geometry: str, backend: str,
                 n_surface: int = 42, max_fps: float = 15.0):
        cfg = EngineConfig(seed_geometry=seed_geometry, n_surface=n_surface,
                           backend=backend, gap_enabled=True, verbose=False)
        self.engine = SPUMEngine(cfg)
        self.backend = self.engine.backend
        self.running = True
        self.max_fps = max_fps
        self.lock = threading.Lock()
        self.broadcaster = Broadcaster()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _extract(self) -> dict:
        p = self.engine.particles
        act = np.where(p.active)[0]
        n = len(act)
        # 抽样以保护浏览器
        if n > MAX_RENDER_NODES:
            keep = np.linspace(0, n - 1, MAX_RENDER_NODES).astype(int)
            keep = np.unique(keep)
        else:
            keep = np.arange(n)
        idx = act[keep]
        pos = p.pos[idx]
        rad = p.radius[idx]
        deg = p.degree[idx]
        uid = p.uid[idx]
        index_of_raw = {int(raw): k for k, raw in enumerate(idx)}

        edges = []
        for a, b in p.connections:
            ia = index_of_raw.get(a)
            ib = index_of_raw.get(b)
            if ia is not None and ib is not None:
                edges.append([ia, ib])

        particles = [
            {"p": [float(pos[k, 0]), float(pos[k, 1]), float(pos[k, 2])],
             "r": float(rad[k]),
             "d": int(deg[k]),
             "u": str(uid[k])}
            for k in range(len(idx))
        ]
        return {"particles": particles, "edges": edges}

    def _metrics(self, frame) -> dict:
        total_deg = int(np.sum(self.engine.particles.degree))
        edge = total_deg // 2
        return {
            "frame": frame.frame_number,
            "V": frame.active_count,
            "latent": frame.latent_count,
            "E": edge,
            "invariant": frame.spum_invariant,
            "crystallites": frame.crystallite_count,
            "dangling": frame.dangling_count,
        }

    def _loop(self):
        fps_t = 1.0 / max(self.max_fps, 1.0)
        while True:
            if self.running:
                with self.lock:
                    snap = self.engine.run_frame()
                payload = {"type": "frame", "metrics": self._metrics(snap),
                           "geom": self._extract(), "backend": self.backend}
                self.broadcaster.broadcast(payload)
            time.sleep(fps_t)

    def control(self, action: str, **kw):
        with self.lock:
            if action == "pause":
                self.running = False
            elif action == "play":
                self.running = True
            elif action == "step":
                snap = self.engine.run_frame()
                payload = {"type": "frame", "metrics": self._metrics(snap),
                           "geom": self._extract(), "backend": self.backend}
                self.broadcaster.broadcast(payload)
            elif action == "reset":
                seed = kw.get("seed") or self.engine.config.seed_geometry
                if seed not in ("star", "sequential", "multi_center"):
                    seed = "star"
                cfg = EngineConfig(seed_geometry=seed,
                                   n_surface=self.engine.config.n_surface,
                                   backend=self.engine.config.backend,
                                   gap_enabled=True, verbose=False)
                self.engine = SPUMEngine(cfg)
                self.backend = self.engine.backend
            elif action == "fps":
                self.max_fps = float(kw.get("value", self.max_fps))
        return {"ok": True, "running": self.running, "backend": self.backend,
                "fps": self.max_fps}


def _build_index_html() -> bytes:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "index.html"), "r", encoding="utf-8") as f:
        return f.read().encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    runtime: Runtime = None
    _index = None

    def _send_headers(self, ctype, extra=None, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(0))
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            if self._index is None:
                self._index = _build_index_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self._index)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(self._index)
        elif path == "/events":
            self._stream_events()
        elif path == "/api/state":
            body = json.dumps({"backend": self.runtime.backend,
                               "running": self.runtime.running,
                               "fps": self.runtime.max_fps}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Not Found")

    def _stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self.runtime.broadcaster.subscribe()
        try:
            heartbeat = 0.0
            while True:
                try:
                    payload = q.get(timeout=10.0)
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                except _queue.Empty:
                    now = time.time()
                    if now - heartbeat > 15.0:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        heartbeat = now
        except (BrokenPipeError, ConnectionResetError, OSError, KeyboardInterrupt):
            pass
        finally:
            self.runtime.broadcaster.unsubscribe(q)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/control":
            try:
                ln = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(min(ln, 65536)) if ln > 0 else b"{}"
                data = json.loads(raw)
            except Exception:
                data = {}
            action = data.pop("action", "")
            res = self.runtime.control(action, **data)
            body = json.dumps(res).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        # 精简日志
        return


def main():
    ap = argparse.ArgumentParser(description="SPUM Web 实时可视化服务器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--seed_geometry", default="star",
                    choices=("star", "sequential", "multi_center"))
    ap.add_argument("--n_surface", type=int, default=42)
    ap.add_argument("--backend", default="auto",
                    help="auto|cuda|cpu (auto 探测 CUDA→GPU/降级)")
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    rt = Runtime(args.seed_geometry, args.backend,
                 n_surface=args.n_surface, max_fps=args.fps)
    Handler.runtime = rt
    rt.start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SPUM Web viz   →  http://{args.host}:{args.port}")
    print(f"  后端          =  {rt.backend}  (seed={args.seed_geometry}, "
          f"fps={args.fps})")
    print("  控制接口      =  POST /api/control  {action: play|pause|step|reset|fps}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()