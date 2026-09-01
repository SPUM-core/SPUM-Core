"""
青檬引擎 · CLI 调试工具
========================

快速验证内核：构建 12 晶子骨架 → 演化 N 帧 → 共识校验 → 五形向量。

用法:
    python -m qingmeng.api.cli --frames 10
    python -m qingmeng.api.cli --frames 10 --check
    python -m qingmeng.api.cli --reason "水为什么往低处流"
"""

from __future__ import annotations

import argparse
import sys

from ..core import Core, StateEvolutionEngine, ConsensusValidator
from ..graph import Topology, EmergenceDetector
from ..domain import DomainRegistry, PhysicsAdapter
from .. import QingmengEngine


def run_cli(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="青檬引擎 CLI")
    parser.add_argument("--frames", type=int, default=10, help="演化帧数")
    parser.add_argument("--check", action="store_true", help="演化后运行共识校验")
    parser.add_argument("--reason", type=str, default=None,
                        help="L2 帧推理循环：输入一句话（LLM 提议 → 引擎执行 → 共识校验）")
    parser.add_argument("--backend", type=str, default="deterministic",
                        choices=["deterministic", "http"],
                        help="推理后端（http 需设置 QINGMENG_LLM_API_KEY 等，见 README）")
    parser.add_argument("--preset", type=str, default="deepseek",
                        choices=["deepseek", "openai"],
                        help="HTTP 后端预设：deepseek（默认）或 openai")
    args = parser.parse_args(argv)

    # 组装四层
    core = Core()
    G = core.build_icosahedron()
    engine = StateEvolutionEngine(core)
    topo = Topology(core)
    emer = EmergenceDetector()
    validator = ConsensusValidator()
    reg = DomainRegistry()
    reg.register(PhysicsAdapter())

    print("═" * 56)
    print("青檬引擎 · 基础框架自检")
    print(f"公理内核: {core.describe()}")
    print(f"初始骨架: {G.number_of_nodes()} 节点 / {G.number_of_edges()} 边")
    print(f"Σ(6−deg): {core.angle_deficit(G)['sum_6_minus_deg']}")
    print("═" * 56)

    # 演化
    for _ in range(args.frames):
        engine.evolve()
    snap = engine.snapshot()
    print(f"演化 {args.frames} 帧 → 帧 #{snap.frame_id}: "
          f"{snap.node_count} 节点 / {snap.edge_count} 边 / "
          f"δ={snap.dangling_density:.4f} / σ={snap.sigma:.3f}")
    print(f"对偶账本: {engine.ledger}")

    # 结构摘要
    summary = topo.summarize(G)
    print(f"结构摘要: {summary}")

    # 五形向量
    emergence = emer.detect(G)
    vec = emergence["vector"]
    print(f"五形向量: 水={vec['水']:.2f} 木={vec['木']:.2f} "
          f"土={vec['土']:.2f} 金={vec['金']:.2f} 火={vec['火']:.2f}")

    # 领域解释
    interp = reg.interpret("physics", summary)
    print(f"物理适配: {interp['reduction']}")

    # 共识校验
    if args.check:
        report = validator.validate(G, ledger=engine.ledger)
        print(f"共识校验: {'PASS' if report.passed else 'FAIL'} "
              f"({len(report.checks)} 项检查, {len(report.failures)} 项违规)")
        for f in report.failures:
            print(f"  ✗ [{f.invariant}] {f.detail}")
        return 0 if report.passed else 1

    # L2 帧推理循环
    if args.reason:
        print("═" * 56)
        print("L2 帧推理循环（LLM 提议 → 引擎执行 → 共识校验）")
        eng = QingmengEngine()
        if args.backend == "http":
            from ..api.llm_bridge import HttpLLMBackend
            eng.reasoner.backend = HttpLLMBackend(preset=args.preset)
        trace = eng.reason(args.reason)
        print(trace.summary_text)
        for f in trace.frames:
            print(f"  [帧{f.frame_id}] {f.frame_type} | {f.claim} | "
                  f"ops={f.operations} | 悬挂={f.dangling_count} | "
                  f"闭合={f.is_closed} | 共识={'PASS' if f.consensus_passed else 'FAIL'}")
        status = "已收束" if trace.concluded else \
                 ("回退到最近闭合帧" if trace.rolled_back else
                  "栈溢出保护(无闭合帧)" if trace.stack_overflow else "未收束")
        print(f"循环状态: {status} | 共 {trace.total_frames} 帧")
        report = eng.check()
        print(f"循环后共识校验: {'PASS' if report.passed else 'FAIL'} "
              f"({len(report.checks)} 项检查, {len(report.failures)} 项违规)")
        for f in report.failures:
            print(f"  ✗ [{f.invariant}] {f.detail}")
        return 0 if report.passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
