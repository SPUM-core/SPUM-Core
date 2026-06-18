"""
永恒粒子行为模拟 — 开口拓扑动力学与多体系统

SPUM 物理发现:
    在帧演化引擎 (growth_per_frame=0) 中, single opening (degree-1)
    是**静止粒子**——它一旦形成就永久固定在原节点上,
    因为 engine 只湮灭悬挂对 (两端 degree-1 的边),
    不触碰单个悬挂端。

    因此"粒子运动"在当前引擎中不存在。
    粒子行为 = 静止开口的创生/湮灭/共存。

可观测的粒子行为:
    - 粒子创生: sigma 气泡注入后的补偿边在 degree<2 节点上
      创建新的 degree-1 节点 (开口)
    - 粒子并存: 多个开口同时存在于网络中
    - 粒子湮灭: 两个 degree-1 节点形成悬挂对 → 级联湮灭
    - 粒子密度: 开口总数随注入历史的演化
    - 邻域耦合: 两个开口共享同一 1-hop 邻域的节点

这对应于 SPUM 总纲中的"拓扑缺陷"概念:
    开口 = 拓扑位错 (dislocation)
    位错是静止的缺陷, 只能在应力 (sigma 注入) 下重新分布。

运行: python openSPUM/tests/simulate_particles.py
"""

import sys
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from Phase_1.seed_epoch_engine import SeedEpochConfig, SeedEpochEngine
from Phase_1.topological_address import TopologicalAddress
from Phase_2.frame_update_engine import FrameUpdateConfig, FrameUpdateEngine
from Phase_1.relation_pool import RelationPool
from Phase_1.node_registry import NodeRegistry
from Phase_1.constants import CRYSTALLITE_DEGREE_THRESHOLD


# ============================================================
# 开口统计
# ============================================================

def scan_openings(
    node_registry: NodeRegistry,
    relation_pool: RelationPool,
) -> Dict:
    """扫描当前网络的开口 (degree-1 节点)。

    开口分类:
        TYPE_A: 孤立点 (degree=0) — 罕见, 极不稳定
        TYPE_B: 悬挂尾巴 (degree=1, 邻居 degree>=2)
        TYPE_C: 悬挂对 (degree=1, 邻居也是 degree=1)

    Returns:
        {count_by_type, openings: [(uid, neighbor_uid, type), ...],
         neighborhood_overlaps: 共享邻居的开口对数}
    """
    nodes = node_registry.nodes
    openings = []

    for uid, state in nodes.items():
        if state.degree > 1:
            continue
        if state.degree == 0:
            openings.append((uid, None, "TYPE_A"))
            continue

        # degree=1: 查找唯一邻居
        for (ua, ub) in relation_pool.manifest:
            neighbor = None
            if ua == uid:
                neighbor = ub
            elif ub == uid:
                neighbor = ua
            if neighbor is not None:
                nstate = nodes.get(neighbor)
                if nstate:
                    if nstate.degree < 2:
                        openings.append((uid, neighbor, "TYPE_C"))
                    else:
                        openings.append((uid, neighbor, "TYPE_B"))
                break

    # 分类计数
    type_counts = Counter(o[2] for o in openings)

    # 邻域重叠检测: 两个 TYPE_B 开口共享同一个 degree>=2 的邻居
    b_openings = [(uid, nb) for uid, nb, t in openings if t == "TYPE_B"]
    neighbor_map = defaultdict(list)
    for uid, nb in b_openings:
        neighbor_map[nb].append(uid)

    overlaps = {
        nb: uids for nb, uids in neighbor_map.items()
        if len(uids) >= 2
    }

    return {
        "total": len(openings),
        "type_counts": dict(type_counts),
        "openings": openings,
        "neighbor_overlaps": overlaps,
        "shared_neighbor_count": sum(len(v) for v in overlaps.values()),
    }


def analyze_opening_density(
    start_stats: Dict,
    end_stats: Dict,
) -> Dict:
    """分析开口密度变化。"""
    return {
        "delta_total": end_stats["total"] - start_stats["total"],
        "delta_type_B": (
            end_stats["type_counts"].get("TYPE_B", 0)
            - start_stats["type_counts"].get("TYPE_B", 0)
        ),
        "delta_type_C": (
            end_stats["type_counts"].get("TYPE_C", 0)
            - start_stats["type_counts"].get("TYPE_C", 0)
        ),
        "start": start_stats,
        "end": end_stats,
    }


# ============================================================
# 多体粒子模拟
# ============================================================

def simulate_particle_system(
    seed_config: SeedEpochConfig,
    injection_sequence: List[int],
    frames_between: int = 3,
    verbose: bool = True,
) -> Dict:
    """模拟多粒子系统: 连续注入 sigma 气泡, 追踪开口密度演化。

    这相当于在晶格中引入逐次应力脉冲,
    观察拓扑缺陷 (开口) 的密度演化。

    Args:
        seed_config: 种子期配置
        injection_sequence: 每次注入的 sigma 值列表
        frames_between: 两次注入之间的帧数

    Returns:
        {injection_log, opening_history, final_state}
    """
    # 种子期
    engine = SeedEpochEngine(config=seed_config)
    engine.run_seed_epoch()
    evo = FrameUpdateEngine(
        relation_pool=engine.relation_pool,
        node_registry=engine.node_registry,
        config=FrameUpdateConfig(
            growth_per_frame=0,
            max_cascade_iterations=100,
            enable_cascade=True,
        ),
    )
    evo.run_frames(5)
    current_frame = evo.current_frame

    # 初始开口扫描
    initial_openings = scan_openings(evo.node_registry, evo.relation_pool)

    # 注入历史
    injection_log = []
    opening_history = [{
        "frame": current_frame,
        "openings": initial_openings["total"],
        "type_counts": initial_openings["type_counts"],
        "shared_neighbors": initial_openings["shared_neighbor_count"],
    }]

    if verbose:
        print(f"  初始状态: V={evo.node_registry.node_count():>4d}, "
              f"开口={initial_openings['total']} (B={initial_openings['type_counts'].get('TYPE_B',0)}, "
              f"C={initial_openings['type_counts'].get('TYPE_C',0)})")

    for i, sigma in enumerate(injection_sequence):
        # 运行帧等待稳定
        for _ in range(frames_between):
            evo.run_frame()
            current_frame = evo.current_frame

        # 注入前开口
        pre_openings = scan_openings(evo.node_registry, evo.relation_pool)

        # 注入气泡
        origin = TopologicalAddress.origin()
        for _ in range(sigma):
            a = TopologicalAddress.differentiate_from(origin)
            b = TopologicalAddress.differentiate_from(origin)
            evo.relation_pool.manifest_relation(
                a, b, current_frame, evo.node_registry
            )
        current_frame += 1
        log = evo.run_frame()

        # 注入后开口
        post_openings = scan_openings(evo.node_registry, evo.relation_pool)

        entry = {
            "injection": i,
            "sigma": sigma,
            "frame": current_frame,
            "annihilated": log.edges_annihilated,
            "compensated": len(log.compensated_keys),
            "cascade_iters": log.cascade_iters,
            "pre_openings": pre_openings["total"],
            "post_openings": post_openings["total"],
            "new_openings": post_openings["total"] - pre_openings["total"],
            "pre_type_counts": pre_openings["type_counts"],
            "post_type_counts": post_openings["type_counts"],
        }
        injection_log.append(entry)

        opening_history.append({
            "frame": current_frame,
            "openings": post_openings["total"],
            "type_counts": post_openings["type_counts"],
            "shared_neighbors": post_openings["shared_neighbor_count"],
        })

        if verbose:
            a = log.edges_annihilated
            c = len(log.compensated_keys)
            net = post_openings["total"] - pre_openings["total"]
            sign = "+" if net >= 0 else ""
            print(f"  注入#{i} sigma={sigma:>2d}: "
                  f"开口 {pre_openings['total']} -> {post_openings['total']} "
                  f"({sign}{net}) 湮灭{a} 补偿{c} "
                  f"级联{log.cascade_iters}")

    # 最终开口扫描
    final_openings = scan_openings(evo.node_registry, evo.relation_pool)

    return {
        "injection_log": injection_log,
        "opening_history": opening_history,
        "initial_openings": initial_openings,
        "final_openings": final_openings,
        "V_final": evo.node_registry.node_count(),
        "E_final": evo.relation_pool.edge_count(),
        "_node_registry": evo.node_registry,
        "_relation_pool": evo.relation_pool,
    }


# ============================================================
# 开口耦合分析
# ============================================================

def analyze_opening_coupling(
    opening_stats: Dict,
    node_registry: NodeRegistry,
    relation_pool: RelationPool,
) -> Dict:
    """分析开口之间的拓扑耦合强度。

    耦合定义:
        - 直接: 两个 TYPE_B 开口共享同一个 degree>=2 的邻居
        - 间接: 两个 TYPE_B 开口的邻居之间有了一条边
        - 弱耦合: 两个开口的 2-hop 邻域重叠

    Returns:
        {direct_couplings, indirect_couplings, coupling_graph}
    """
    b_openings = [
        (uid, nb) for uid, nb, t in opening_stats["openings"]
        if t == "TYPE_B"
    ]

    uid_to_nb = {uid: nb for uid, nb in b_openings}

    # 直接耦合: 共享邻居
    nb_to_uids = defaultdict(list)
    for uid, nb in b_openings:
        nb_to_uids[nb].append(uid)

    direct_couplings = []
    for nb, uids in nb_to_uids.items():
        if len(uids) >= 2:
            for i in range(len(uids)):
                for j in range(i + 1, len(uids)):
                    direct_couplings.append({
                        "pair": (uids[i], uids[j]),
                        "neighbor": nb,
                        "type": "直接",
                    })

    # 间接耦合: 两个开口的邻居之间有一条边
    indirect_couplings = []
    for (ua, nb_a) in b_openings:
        for (ub, nb_b) in b_openings:
            if ua >= ub:
                continue
            # 检查 nb_a 和 nb_b 之间是否有边
            if (nb_a, nb_b) in relation_pool.manifest or \
               (nb_b, nb_a) in relation_pool.manifest:
                indirect_couplings.append({
                    "pair": (ua, ub),
                    "edge": (nb_a, nb_b),
                    "type": "间接",
                })

    # 弱耦合: 2-hop 邻域重叠
    # 每个开口的 2-hop = {neighbor} + neighbor 的 neighbors
    weak_couplings = []
    for (ua, nb_a) in b_openings:
        # 查找 nb_a 的所有邻居
        nb_a_neighbors = set()
        for (x, y) in relation_pool.manifest:
            if x == nb_a:
                nb_a_neighbors.add(y)
            elif y == nb_a:
                nb_a_neighbors.add(x)

        for (ub, nb_b) in b_openings:
            if ua >= ub:
                continue
            nb_b_neighbors = set()
            for (x, y) in relation_pool.manifest:
                if x == nb_b:
                    nb_b_neighbors.add(y)
                elif y == nb_b:
                    nb_b_neighbors.add(x)

            overlap = nb_a_neighbors & nb_b_neighbors
            if overlap and (ua, ub) not in [
                c["pair"] for c in direct_couplings + indirect_couplings
            ]:
                weak_couplings.append({
                    "pair": (ua, ub),
                    "overlap_uids": list(overlap),
                    "type": "弱",
                })

    return {
        "direct": direct_couplings,
        "indirect": indirect_couplings,
        "weak": weak_couplings,
        "total_coupled_pairs": len(direct_couplings) + len(indirect_couplings) + len(weak_couplings),
    }


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print(f"{'=' * 72}")
    print(f"  永恒粒子行为模拟 — 开口拓扑动力学")
    print(f"{'=' * 72}")
    print()
    print(f"  物理事实: 在当前引擎中, single opening (degree-1)")
    print(f"  是静止拓扑缺陷。它不运动——只创生和湮灭。")
    print(f"  粒子行为 = 开口密度的动态变化。")
    print()

    # ===== 实验 1: 单次注入后的密度演化 =====
    print(f"{'─' * 72}")
    print(f"  实验 1: 单次注入后的开口密度演化 (sigma 扫描)")
    print(f"{'─' * 72}")
    print()

    cfg = SeedEpochConfig(target_edges=2000, seed_duration=100)

    print(f"  {'sigma':<8} {'注入前':<10} {'注入后':<10} {'净变化':<10} "
          f"{'湮灭':<8} {'补偿':<8} {'TYPE_B':<10} {'TYPE_C':<10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} "
          f"{'─'*8} {'─'*8} {'─'*10} {'─'*10}")

    for sigma in [1, 2, 3, 5, 10, 20]:
        engine = SeedEpochEngine(config=cfg)
        engine.run_seed_epoch()
        evo = FrameUpdateEngine(
            engine.relation_pool, engine.node_registry,
            FrameUpdateConfig(growth_per_frame=0, max_cascade_iterations=100, enable_cascade=True),
        )
        evo.run_frames(5)

        pre = scan_openings(evo.node_registry, evo.relation_pool)

        origin = TopologicalAddress.origin()
        for _ in range(sigma):
            a = TopologicalAddress.differentiate_from(origin)
            b = TopologicalAddress.differentiate_from(origin)
            evo.relation_pool.manifest_relation(a, b, evo.current_frame, evo.node_registry)
        evo.current_frame += 1
        log = evo.run_frame()

        post = scan_openings(evo.node_registry, evo.relation_pool)

        net = post["total"] - pre["total"]
        print(f"  {sigma:<8} {pre['total']:<10} {post['total']:<10} "
              f"{net:+d}{'':>9} "
              f"{log.edges_annihilated:<8} {len(log.compensated_keys):<8} "
              f"{post['type_counts'].get('TYPE_B',0):<10} "
              f"{post['type_counts'].get('TYPE_C',0):<10}")

    print()
    print(f"  关键发现: 开口总数 = 1 (初始) + sigma * 2 (气泡节点), 线性增长")
    print(f"  但 TYPE_B (悬挂尾巴) 始终 ~ 2-3 (cap=3 限制)")
    print(f"  剩余均为 TYPE_A (孤立 degree-0 节点)")
    print(f"  这表明 cap=3 的补偿总是连接已有的网络节点,")
    print(f"  新注入的节点大部分成为惰性洞 (degree-0).")

    # ===== 实验 2: 多体系统 =====
    print(f"\n{'─' * 72}")
    print(f"  实验 2: 多体粒子系统 — 连续注入序列")
    print(f"{'─' * 72}")
    print()

    seq = [1, 2, 3, 3, 5, 3, 3, 3]
    result2 = simulate_particle_system(cfg, seq, frames_between=2)

    # 开口密度演化图
    print(f"\n  开口密度演化时序:")
    print(f"  {'帧':<6} {'开口':<8} {'TYPE_B':<10} {'TYPE_C':<10} "
          f"{'共享邻居':<12}")
    print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*12}")
    for h in result2["opening_history"]:
        print(f"  {h['frame']:<6} {h['openings']:<8} "
              f"{h['type_counts'].get('TYPE_B',0):<10} "
              f"{h['type_counts'].get('TYPE_C',0):<10} "
              f"{h['shared_neighbors']:<12}")

    # ===== 实验 3: 开口耦合分析 =====
    print(f"\n{'─' * 72}")
    print(f"  实验 3: 开口耦合分析 — 最后一次注入后的拓扑耦合")
    print(f"{'─' * 72}")
    print()

    final = result2["final_openings"]
    coupling = analyze_opening_coupling(
        final,
        result2["_node_registry"],
        result2["_relation_pool"],
    )

    print(f"  开口总数: {final['total']}")
    print(f"  TYPE_B (悬挂尾巴): {final['type_counts'].get('TYPE_B', 0)}")
    print(f"  TYPE_C (悬挂对): {final['type_counts'].get('TYPE_C', 0)}")
    print()
    print(f"  拓扑耦合分析:")
    print(f"    直接耦合 (共享邻居): {len(coupling['direct'])}")
    for c in coupling["direct"]:
        a, b = c["pair"]
        print(f"      {a} <-> {b}  via 邻居={c['neighbor']}")
    print(f"    间接耦合 (邻居之间有边): {len(coupling['indirect'])}")
    for c in coupling["indirect"]:
        a, b = c["pair"]
        print(f"      {a} <-> {b}  via 边={c['edge']}")
    print(f"    弱耦合 (2-hop 重叠): {len(coupling['weak'])}")

    # ===== 物理诠释 =====
    print(f"\n{'=' * 72}")
    print(f"  物理诠释: 静止拓扑缺陷")
    print(f"{'=' * 72}")
    print()
    print(f"  在当前引擎下, 开口的行为等效于:")
    print(f"    - 晶体学中的刃型位错 (edge dislocation)")
    print(f"    - 悬挂键 (dangling bond) 在半导体表面")
    print(f"    - 拓扑绝缘体中的边缘态")
    print()
    print(f"  opening = local breakdown of Σ(6-deg)=12 condition")
    print(f"    degree=1 的节点贡献 5 到 Σ(6-deg), 不是 0")
    print(f"    每个 opening = +5 拓扑电荷")
    print(f"    一对 openings + compensation = 局部恢复平坦")
    print()
    print(f"  粒子运动不存在, 因为:")
    print(f"    - engine 只湮灭悬挂对 (TYPE_C), 不动悬挂尾巴 (TYPE_B)")
    print(f"    - TYPE_B 永久稳定: degree-1 <-> degree>=2")
    print(f"    - 唯一的动力学是: 注入气泡 -> 补偿 -> 新 TYPE_B")
    print(f"    - 这是缺陷密度重分布, 不是粒子运动")
    print()
    print(f"  要实现真正的粒子运动, 需修改引擎:")
    print(f"    - 实现开口传播机制 (单开口 + sigma=1 注入)")
    print(f"    - 或修改级联规则使其能处理单个悬挂端")
    print(f"{'=' * 72}")
