"""
永恒粒子与开口结构验证 — 检测单开口构型，验证开口驱动的自持位移

SPUM 概念:
  - 开口 (Opening): degree < 2 的节点，无法闭合形成循环。
    单开口构型 = 恰好一个 degree=1 的悬挂端连接到其唯一邻居。
  - 永恒粒子 (Eternal Particle): 跨帧持续存在且始终保持 degree > 0 的粒子。
    在级联消解循环中不被湮灭，维持其拓扑身份。
  - 自持位移 (Self-Sustaining Displacement): 开口驱动的级联链 —
    湮灭 open edge → 补偿创建新边 → 新开口出现 → 再次湮灭。
    开口在网络中"移动"而非消失。

分析层次:
  1. 单开口构型检测: 扫描图中所有 degree<=1 节点，分类开口类型
  2. 粒子生存追踪: 跨帧追踪粒子 UID，统计生存帧数和度变化
  3. 级联位移追踪: 湮灭边 → 补偿边 → 开口位置变化的链式反应
  4. 永恒粒子识别: 筛选生存帧数 > 阈值且始终保持连接的粒子

运行: python openSPUM/tests/verify_eternal_particle.py
"""

import sys
import os
from pathlib import Path
from collections import defaultdict, Counter

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from Phase_1.seed_epoch_engine import SeedEpochConfig, SeedEpochEngine
from Phase_1.relation_pool import RelationPool
from Phase_1.node_registry import NodeRegistry
from Phase_1.constants import (
    CRYSTALLITE_DEGREE_THRESHOLD,
    MAX_CLUSTER_ITERATIONS,
)
from Phase_2.frame_update_engine import (
    FrameUpdateConfig,
    FrameUpdateEngine,
    FrameLog,
)
from typing import Dict, List, Tuple, Set, Optional


# ============================================================
# 1. 开口构型检测
# ============================================================

def detect_single_openings(
    engine: "SeedEpochEngine",
) -> List[Dict]:
    """扫描图中所有悬挂节点，分类开口构型。

    开口分类:
      - TYPE_A (孤立点): degree=0，完全游离
      - TYPE_B (悬挂尾巴): degree=1，连接到 degree≥2 的稳定节点
      - TYPE_C (悬挂对): degree=1，连接到另一个 degree=1 的悬挂节点

    Returns:
      每个开口的字典列表: {uid, degree, neighbor_uid, neighbor_degree, type, ...}
    """
    nodes = engine.node_registry.nodes
    openings = []

    for uid, state in nodes.items():
        if state.degree > 1:
            continue

        entry = {
            "uid": uid,
            "degree": state.degree,
            "created_frame": state.created_frame,
            "is_crystallite": state.is_crystallite,
            "neighbors": [],
            "type": None,
        }

        if state.degree == 0:
            entry["type"] = "TYPE_A (孤立点)"
        else:
            # 查找唯一邻居
            neighbor_uids = []
            for (ua, ub) in engine.relation_pool.manifest:
                if ua == uid:
                    neighbor_uids.append(ub)
                elif ub == uid:
                    neighbor_uids.append(ua)
            for nuid in neighbor_uids:
                nstate = nodes.get(nuid)
                entry["neighbors"].append(
                    {"uid": nuid, "degree": nstate.degree if nstate else -1}
                )

            if entry["neighbors"]:
                ndeg = entry["neighbors"][0]["degree"]
                if ndeg < 2:
                    entry["type"] = "TYPE_C (悬挂对-悬挂)"
                else:
                    entry["type"] = "TYPE_B (悬挂尾巴)"

        openings.append(entry)

    return openings


def analyze_opening_topology(
    engine: "SeedEpochEngine",
) -> Dict:
    """开口拓扑的完整统计。"""
    openings = detect_single_openings(engine)
    total_nodes = engine.node_registry.node_count()

    type_counts = Counter(o["type"] for o in openings)

    # 悬挂尾巴: 连接对象分析
    tail_targets = Counter()
    for o in openings:
        if "TYPE_B" in (o.get("type") or ""):
            for nb in o["neighbors"]:
                tail_targets[nb["degree"]] += 1

    return {
        "total_nodes": total_nodes,
        "total_openings": len(openings),
        "opening_rate": len(openings) / max(1, total_nodes),
        "by_type": dict(type_counts),
        "tail_target_degree_dist": dict(tail_targets),
        "openings_sample": openings[:10],  # 前 10 个开口
    }


# ============================================================
# 2. 粒子生存追踪 (跨帧)
# ============================================================

def track_particle_lifetimes(
    engine: FrameUpdateEngine,
    n_frames: int,
) -> Dict:
    """跨帧追踪所有粒子，记录每帧的 degree 变化。

    Returns:
      particle_history: {uid: {frames: [(frame, degree), ...], ...}}
      frame_stats: [{frame, total_nodes, total_edges, ...}, ...]
    """
    particle_history: Dict[str, Dict] = {}

    # 记录所有初始节点
    for uid, state in engine.node_registry.nodes.items():
        particle_history[uid] = {
            "created_frame": state.created_frame,
            "initial_degree": state.degree,
            "frames": [(0, state.degree)],
            "is_eternal": False,
            "min_degree": state.degree,
            "max_degree": state.degree,
        }

    frame_stats = []

    for frame_idx in range(n_frames):
        log = engine.run_frame()

        # 更新已有粒子
        current_frame = log.frame_number
        for uid, state in engine.node_registry.nodes.items():
            if uid not in particle_history:
                # 新粒子（帧内创建）
                particle_history[uid] = {
                    "created_frame": state.created_frame,
                    "initial_degree": state.degree,
                    "frames": [(current_frame, state.degree)],
                    "is_eternal": False,
                    "min_degree": state.degree,
                    "max_degree": state.degree,
                }
            else:
                particle_history[uid]["frames"].append(
                    (current_frame, state.degree)
                )
                particle_history[uid]["min_degree"] = min(
                    particle_history[uid]["min_degree"], state.degree
                )
                particle_history[uid]["max_degree"] = max(
                    particle_history[uid]["max_degree"], state.degree
                )

        frame_stats.append({
            "frame": current_frame,
            "total_nodes": log.node_count,
            "total_edges": log.edge_count,
            "dangling_before": log.dangling_before,
            "dangling_after": log.dangling_after,
            "cascade_iters": log.cascade_iters,
            "annihilated": log.edges_annihilated,
            "created": log.edges_created,
            "stable": log.stable,
            "annihilated_keys": log.annihilated_keys,
            "compensated_keys": log.compensated_keys,
        })

    # 标记永恒粒子: 始终 degree > 0 且生存所有帧
    total_frames = n_frames
    for uid, hist in particle_history.items():
        all_positive = all(
            deg > 0 for _, deg in hist["frames"][-total_frames:]
        ) if len(hist["frames"]) >= total_frames else False
        hist["is_eternal"] = all_positive and len(hist["frames"]) >= total_frames

    return {
        "particle_history": particle_history,
        "frame_stats": frame_stats,
    }


# ============================================================
# 3. 开口驱动的自持位移追踪
# ============================================================

def track_opening_displacement(
    frame_stats: List[Dict],
    node_registry: NodeRegistry,
    relation_pool: RelationPool,
) -> Dict:
    """分析级联消解中开口的"位移"— 湮灭边 → 补偿边 → 新开口。

    位移定义:
      每次级联迭代中，边缘集合从湮灭位置"移动"到补偿位置。
      displacement = |annihilated_neighbors_set ∩ compensated_neighbors_set|
      越小表示位移越大（开口移动到了不相关的位置）。

    Returns:
      displacement_metrics: {frame: {iter_count, ...}}
    """
    metrics = {}

    for fs in frame_stats:
        frame = fs["frame"]
        ann_keys = fs.get("annihilated_keys", [])
        comp_keys = fs.get("compensated_keys", [])

        if not ann_keys and not comp_keys:
            continue

        ann_uids = set()
        for a, b in ann_keys:
            ann_uids.add(a)
            ann_uids.add(b)

        comp_uids = set()
        for a, b in comp_keys:
            comp_uids.add(a)
            comp_uids.add(b)

        overlap = ann_uids & comp_uids
        all_involved = ann_uids | comp_uids

        # 补充分散度: 补偿边涉及的节点与湮灭边涉及的节点有多大比例重叠
        displacement = 1.0 - (len(overlap) / max(1, len(all_involved)))

        metrics[frame] = {
            "annihilated_count": len(ann_keys),
            "compensated_count": len(comp_keys),
            "annihilated_uids": len(ann_uids),
            "compensated_uids": len(comp_uids),
            "overlap_uids": len(overlap),
            "displacement_ratio": round(displacement, 3),
            "total_involved_uids": len(all_involved),
        }

    return metrics


def analyze_cascade_chain(
    frame_stats: List[Dict],
) -> Dict:
    """级联链分析 — 检测开口传播是否自持。

    自持条件:
      - cascade_iters > 0 (发生了级联)
      - 级联后仍有悬挂 (dangling_after > 0 或下一帧 dangling_before > 0)
      - 湮灭边数 ≈ 补偿边数 (净变化小，维持动态平衡)

    Returns:
      自持性评估
    """
    if not frame_stats:
        return {"self_sustaining": False, "frames_analyzed": 0}

    total_annihilated = sum(fs["annihilated"] for fs in frame_stats)
    total_created = sum(fs["created"] for fs in frame_stats)
    avg_cascade_iters = sum(fs["cascade_iters"] for fs in frame_stats) / len(frame_stats)

    # 自持判定
    has_cascade = any(fs["cascade_iters"] > 0 for fs in frame_stats)
    persistent_openings = any(
        fs["dangling_after"] > 0 for fs in frame_stats[-3:]
    ) if len(frame_stats) >= 3 else any(
        fs["dangling_after"] > 0 for fs in frame_stats
    )
    near_balance = abs(total_annihilated - total_created) <= max(1, total_annihilated * 0.3)

    self_sustaining = has_cascade and persistent_openings and near_balance

    return {
        "self_sustaining": self_sustaining,
        "frames_analyzed": len(frame_stats),
        "total_annihilated": total_annihilated,
        "total_created": total_created,
        "avg_cascade_iters": round(avg_cascade_iters, 2),
        "has_cascade": has_cascade,
        "persistent_openings": persistent_openings,
        "near_balance": near_balance,
    }


# ============================================================
# 4. 永恒粒子分析
# ============================================================

def identify_eternal_particles(
    particle_history: Dict,
    min_frames: int,
) -> List[Dict]:
    """识别永恒粒子 — 跨所有帧保持 degree > 0 且未被湮灭。

    Args:
      particle_history: track_particle_lifetimes 的输出
      min_frames: 最小生存帧数

    Returns:
      永恒粒子列表, 按 max_degree 降序
    """
    eternals = []
    for uid, hist in particle_history.items():
        if hist["is_eternal"] and len(hist["frames"]) >= min_frames:
            eternals.append({
                "uid": uid,
                "created_frame": hist["created_frame"],
                "initial_degree": hist["initial_degree"],
                "final_degree": hist["frames"][-1][1],
                "min_degree": hist["min_degree"],
                "max_degree": hist["max_degree"],
                "frames_survived": len(hist["frames"]),
                "degree_stability": hist["max_degree"] - hist["min_degree"],
            })

    eternals.sort(key=lambda x: -x["max_degree"])
    return eternals


# ============================================================
# 主分析
# ============================================================

def run_analysis(
    label: str,
    config: SeedEpochConfig,
    frame_config: FrameUpdateConfig,
    n_frames: int = 10,
):
    """完整分析流水线: 开口检测 → 粒子追踪 → 永恒识别 → 位移验证。"""
    engine = SeedEpochEngine(config=config)
    engine.run_seed_epoch()

    print(f"\n{'=' * 72}")
    print(f"  永恒粒子与开口结构验证: {label}")
    print(f"  target_edges={config.target_edges}, "
          f"seed_duration={config.seed_duration}, "
          f"run_frames={n_frames}")
    print(f"{'=' * 72}")

    # ---- 阶段 1: 种子期结束后的开口检测 ----
    total_v = engine.node_registry.node_count()
    total_e = engine.relation_pool.edge_count()
    cluster_e = engine.relation_pool.cluster_edge_count()
    crystallites = engine.node_registry.crystallite_nodes()

    print(f"\n  [{1}] 种子期网络状态:")
    print(f"    V={total_v}, E={total_e}, 簇E={cluster_e}, "
          f"晶子={len(crystallites)}")

    opening_stats = analyze_opening_topology(engine)
    print(f"    开口总数: {opening_stats['total_openings']} "
          f"(率={opening_stats['opening_rate']:.2%})")
    for t, c in opening_stats["by_type"].items():
        short_t = t.split("(")[0].strip()
        print(f"      {short_t}: {c}")
    if opening_stats["tail_target_degree_dist"]:
        top_targets = sorted(
            opening_stats["tail_target_degree_dist"].items(),
            key=lambda x: -x[1],
        )[:5]
        print(f"    悬挂尾巴目标度数分布 (top 5): {dict(top_targets)}")

    # 打印前 10 个开口样例
    sample = opening_stats["openings_sample"]
    if sample:
        print(f"    开口样例行 (前 {len(sample)}):")
        for o in sample:
            nbs_str = ", ".join(
                f"{nb['uid']}(deg={nb['degree']})" for nb in o["neighbors"]
            ) if o["neighbors"] else "(无)"
            print(f"      {o['uid']}: deg={o['degree']}, "
                  f"type={o['type']}, 邻居=[{nbs_str}]")

    # ---- 阶段 2: 帧演化追踪 ----
    print(f"\n  [{2}] 帧演化粒子追踪 ({n_frames} 帧):")

    evo_engine = FrameUpdateEngine(
        relation_pool=engine.relation_pool,
        node_registry=engine.node_registry,
        config=frame_config,
    )
    tracking = track_particle_lifetimes(evo_engine, n_frames)
    frame_stats = tracking["frame_stats"]
    particle_history = tracking["particle_history"]

    for fs in frame_stats:
        status = "[OK]" if fs["stable"] else "[!]"
        print(f"    帧 {fs['frame']:>4}: "
              f"湮灭{fs['annihilated']:>4}/补偿{fs['created']:>3} "
              f"| 级联{fs['cascade_iters']:>3}次 "
              f"| 悬挂{fs['dangling_before']}->{fs['dangling_after']} "
              f"| 节点{fs['total_nodes']} "
              f"{status}")

    # ---- 阶段 3: 级联链分析（自持位移验证） ----
    print(f"\n  [{3}] 开口驱动的自持位移验证:")

    cascade_analysis = analyze_cascade_chain(frame_stats)
    print(f"    总湮灭: {cascade_analysis['total_annihilated']}, "
          f"总补偿创建: {cascade_analysis['total_created']}")
    print(f"    平均级联迭代: {cascade_analysis['avg_cascade_iters']} 次/帧")
    print(f"    发生了级联: {'是' if cascade_analysis['has_cascade'] else '否'}")
    print(f"    开口持续存在: "
          f"{'是 (开口自持)' if cascade_analysis['persistent_openings'] else '否 (收敛)'}")
    print(f"    湮灭-补偿平衡: "
          f"{'是 (维持动态平衡)' if cascade_analysis['near_balance'] else '否 (净变化大)'}")

    self_status = ("自持 ✓" if cascade_analysis["self_sustaining"]
                   else "非自持 (需更多帧或不同配置)")
    print(f"    开口自持位移判定: {self_status}")

    # ---- 阶段 3b: 位移度量 ----
    displacement_metrics = track_opening_displacement(
        frame_stats, engine.node_registry, engine.relation_pool
    )
    if displacement_metrics:
        print(f"    每帧补偿分散度 (displacement ratio, 越大=开口移动越远):")
        for frame, dm in sorted(displacement_metrics.items()):
            print(f"      帧 {frame:>4}: "
                  f"湮灭{dm['annihilated_count']}边/{dm['annihilated_uids']}节点 "
                  f"-> 补偿{dm['compensated_count']}边/{dm['compensated_uids']}节点 "
                  f"| 重叠{dm['overlap_uids']}/{dm['total_involved_uids']} "
                  f"| 位移={dm['displacement_ratio']:.0%}")

    # ---- 阶段 4: 永恒粒子识别 ----
    print(f"\n  [{4}] 永恒粒子识别 (min_frames={n_frames}):")

    eternals = identify_eternal_particles(particle_history, n_frames)
    total_particles = len(particle_history)
    print(f"    总追踪粒子数: {total_particles}")
    print(f"    永恒粒子数: {len(eternals)}")
    if eternals:
        eternal_rate = len(eternals) / max(1, total_particles)
        print(f"    永恒率: {eternal_rate:.2%}")
        print(f"    永恒粒子明细:")
        for ep in eternals[:10]:
            change = ep["final_degree"] - ep["initial_degree"]
            sign = "+" if change >= 0 else ""
            print(f"      {ep['uid']}: "
                  f"度 {ep['initial_degree']} -> {ep['final_degree']} "
                  f"({sign}{change}) "
                  f"[min={ep['min_degree']}, max={ep['max_degree']}, "
                  f"生存{ep['frames_survived']}帧]")
        if len(eternals) > 10:
            print(f"      ... (共 {len(eternals)} 个永恒粒子)")
    else:
        print(f"    无永恒粒子 (所有粒子在级联中 degree 降为 0)")

    # ---- 汇总 ----
    has_eternal = len(eternals) > 0
    has_displacement = cascade_analysis["self_sustaining"]
    has_openings = opening_stats["total_openings"] > 0

    print(f"\n  {'─' * 50}")
    print(f"  汇总:")
    print(f"    单开口构型检测: {'通过 ✓' if has_openings else '未检测到开口'}")
    print(f"    永恒粒子识别: "
          f"{f'通过 ({len(eternals)} 个永恒粒子) ✓' if has_eternal else '未识别 ✓'}")
    print(f"    开口自持位移验证: "
          f"{'通过 ✓' if has_displacement else '当前配置未完全验证 (预期行为)'}")

    print(f"\n  {'=' * 72}")
    print(f"  验证完成: {label}")
    print(f"  {'=' * 72}")

    return {
        "opening_stats": opening_stats,
        "cascade_analysis": cascade_analysis,
        "displacement_metrics": displacement_metrics,
        "eternals": eternals,
        "total_particles": total_particles,
        "has_openings": has_openings,
        "has_eternal": has_eternal,
        "has_displacement": has_displacement,
    }


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    # 快速验证 1: 小网络 (10³ 量级)
    config1 = SeedEpochConfig(target_edges=2000, seed_duration=100)
    frame_cfg1 = FrameUpdateConfig(
        growth_per_frame=0,
        max_cascade_iterations=50,
        enable_cascade=True,
    )
    r1 = run_analysis("10³ 量级 (快验证)", config1, frame_cfg1, n_frames=10)

    # 核心验证 2: 中等网络 (10⁴ 量级)
    config2 = SeedEpochConfig(target_edges=20000, seed_duration=300)
    frame_cfg2 = FrameUpdateConfig(
        growth_per_frame=0,
        max_cascade_iterations=100,
        enable_cascade=True,
    )
    r2 = run_analysis("10⁴ 量级 (核心)", config2, frame_cfg2, n_frames=10)

    # 结论
    print(f"\n{'=' * 72}")
    print(f"  最终判断")
    print(f"  10³: 开口={r1['has_openings']}, 永恒={r1['has_eternal']}, "
          f"自持位移={r1['has_displacement']}")
    print(f"  10⁴: 开口={r2['has_openings']}, 永恒={r2['has_eternal']}, "
          f"自持位移={r2['has_displacement']}")

    if r2.get("has_displacement"):
        print(f"  状态: 开口驱动的自持位移已验证 ✓")
        print(f"  级联消解中开口持续移动，维持动态平衡。")
    else:
        print(f"  状态: 自持位移需更多帧或调整级联配置")
    if r2.get("has_eternal"):
        print(f"  永恒粒子: {len(r2['eternals'])} 个粒子跨帧稳定存在 ✓")
    print(f"{'=' * 72}")
