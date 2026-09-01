"""
青檬引擎 · REST API — 引擎的 HTTP 外壳
========================================

FastAPI 封装（零业务逻辑——全部委托给 QingmengEngine）。
每个端点返回 结构数据 + consensus 块（11 条不变量实时校验）。

端点：
    GET  /health           存活探针
    GET  /state            当前帧快照（结构摘要 + 五形 + 对偶账本 + 共识）
    POST /evolve           演化 N 帧          {frames}
    POST /touch            触觉桥输入         {text}
    POST /reason           L2 帧推理循环      {prompt, max_frames?, domain?, backend?, preset?}

运行：
    python -m qingmeng.api.rest --host 0.0.0.0 --port 8000

SPUM 校准：
    - 引擎是唯一状态源，REST 层只做序列化（不引入独立实体，不变量 I）
    - 连续量（σ/置信度）是认知投影，已在 payload 注明（不变量 VIII/X）
    - 共识违规不抛 HTTP 错误——在 payload 中报告（不完美是常态）
"""

from __future__ import annotations

import argparse
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import QingmengEngine, __version__

# 供 python -m qingmeng.api.rest 使用的全局应用
app: Optional[FastAPI] = None
_app_lock = threading.Lock()


# ════════════════════════════════════════════════════════════════════
# 请求模型
# ════════════════════════════════════════════════════════════════════

class EvolveRequest(BaseModel):
    frames: int = Field(default=1, ge=1, le=1000)


class TouchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ReasonRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_frames: int = Field(default=5, ge=1, le=20)
    domain: Optional[str] = None
    backend: str = Field(default="deterministic", pattern="^(deterministic|http)$")
    preset: str = Field(default="deepseek", pattern="^(deepseek|openai)$")


# ════════════════════════════════════════════════════════════════════
# 应用工厂
# ════════════════════════════════════════════════════════════════════

def create_app(engine: Optional[QingmengEngine] = None) -> FastAPI:
    """创建 REST 应用。engine 可注入（测试用）；默认新建独立引擎。"""
    eng = engine or QingmengEngine()
    lock = threading.Lock()
    http_backend_cache: Dict[str, Any] = {}

    def _consensus_block() -> Dict[str, Any]:
        report = eng.check()
        return {
            "passed": report.passed,
            "checks": len(report.checks),
            "failures": [f.detail for f in report.failures],
        }

    def _http_backend(preset: str):
        # 按 preset 缓存——重复请求不重建（失败自动回退确定性后端）
        if preset not in http_backend_cache:
            from ..api.llm_bridge import HttpLLMBackend
            http_backend_cache[preset] = HttpLLMBackend(preset=preset)
        return http_backend_cache[preset]

    app_ = FastAPI(
        title="青檬引擎 REST API",
        version=__version__,
        description="基于 SPUM 内化的复杂系统推理引擎——HTTP 外壳，引擎为唯一状态源",
    )
    app_.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 存活探针 ─────────────────────────────────────────────────

    @app_.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "engine": eng.describe(),
            "consensus": _consensus_block(),
        }

    # ── 当前帧快照 ────────────────────────────────────────────────

    @app_.get("/state")
    def state() -> Dict[str, Any]:
        with lock:
            summary = eng.summarize()
            wuxing = eng.emergence()
            ledger = dict(eng.state.ledger)
        return {
            "frame": eng.state.frame,
            "graph": {
                **summary,
                "sigma": round(summary.get("sigma", 0.0), 4),  # 认知投影（不变量 X）
            },
            "wuxing": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in wuxing.get("vector", {}).items()
            },
            "ledger": ledger,
            "consensus": _consensus_block(),
        }

    # ── 演化 ─────────────────────────────────────────────────────

    @app_.post("/evolve")
    def evolve(req: EvolveRequest) -> Dict[str, Any]:
        with lock:
            snap = eng.evolve_many(req.frames)
            snap_dict = snap.to_dict()
        snap_dict["consensus"] = _consensus_block()
        return snap_dict

    # ── 触觉桥 ───────────────────────────────────────────────────

    @app_.post("/touch")
    def touch(req: TouchRequest) -> Dict[str, Any]:
        with lock:
            feeling = eng.touch(req.text)   # 走引擎入口 → 自动记录推理轨迹
        feeling["consensus"] = _consensus_block()
        return feeling

    # ── L2 帧推理循环 ────────────────────────────────────────────

    @app_.post("/reason")
    def reason(req: ReasonRequest) -> Dict[str, Any]:
        with lock:
            if req.backend == "http":
                eng.reasoner.backend = _http_backend(req.preset)
            else:
                from ..api.llm_bridge import DeterministicBackend
                eng.reasoner.backend = DeterministicBackend()
            trace = eng.reason(
                req.prompt, max_frames=req.max_frames, domain=req.domain
            )
            out = trace.to_dict()
        out["consensus"] = _consensus_block()
        return out

    # ── 观测性：推理轨迹 ────────────────────────────────────────

    @app_.get("/debug/traces")
    def debug_traces(limit: int = 20) -> Dict[str, Any]:
        """最近 N 条推理轨迹的元数据（trace_id/kind/frame/共识裁决）。"""
        with lock:
            recs = eng.trajectory.recent(max(1, min(limit, 100)))
        return {
            "count": len(recs),
            "traces": [
                {
                    "trace_id": r.trace_id,
                    "kind": r.kind,
                    "input": r.input,
                    "frame": r.frame,
                    "consensus_passed": r.consensus.get("passed"),
                    "axiom_path": r.axiom_path,
                }
                for r in recs
            ],
        }

    @app_.get("/debug/trace/{trace_id}")
    def debug_trace(trace_id: str) -> Dict[str, Any]:
        """按 trace_id 取完整轨迹（axiom_path + 状态快照 + 共识 + 回滚点）。"""
        with lock:
            rec = eng.trajectory.get(trace_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"trace 不存在: {trace_id}")
        return rec.to_dict()

    @app_.post("/debug/rollback/{trace_id}")
    def debug_rollback(trace_id: str) -> Dict[str, Any]:
        """版本回滚——恢复到指定轨迹发生前的状态（观测面保留）。"""
        with lock:
            rec = eng.trajectory.rollback(trace_id, eng)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"trace 不存在或无回滚点: {trace_id}")
        return {
            "rolled_back_to": rec.trace_id,
            "frame": eng.state.frame,
            "nodes": eng.graph.number_of_nodes(),
            "edges": eng.graph.number_of_edges(),
            "consensus": _consensus_block(),
        }

    # ── 错误兜底 ─────────────────────────────────────────────────

    @app_.exception_handler(Exception)
    async def _unhandled(_request, exc):
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc)[:500]},
        )

    return app_


# ════════════════════════════════════════════════════════════════════
# 运行入口
# ════════════════════════════════════════════════════════════════════

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="青檬引擎 REST API")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发热重载")
    args = parser.parse_args(argv)

    global app
    with _app_lock:
        if app is None:
            app = create_app()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


# 全局应用（uvicorn 直接加载 / 测试导入时使用；python -m 时由 main() 创建）
if __name__ != "__main__":
    app = create_app()


if __name__ == "__main__":
    import sys
    sys.exit(main())
