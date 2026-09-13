"""
阿基米德真空称重 — 边界刚度调制净湮灭模拟
=====================================================

模拟对象:
    阿基米德实验 (Archimedes, SAR-GRAV, Sos Enattos):
    通过热调制高 Tc 铜氧化物超导体 (YBCO/GdBCO) 越过相变点,
    调制堆叠卡西米尔腔的边界反射率 -> 调制腔内真空能量 -> 检测样品重量变化。

SPUM 机制映射:
    - 样品 R = 网络中按度数选取的稠密闭合子图 (材料核)
    - 边界刚度 b ∈ [0,1] = 卡西米尔边界的反射率 (真空模式截断强度)
        b=0: 透明 (T > T_c, 正常态, 无模式截断 -> 内部涨落可逆自愈)
        b=1: 全反射 (T < T_c, 超导态, 全模式截断 -> 内部创生完全抑制)
    - 自发内部湮灭: 每帧 ann_rate 条 R 内部边断裂 (真空涨落, 与 b 无关)
    - 对偶转移: 湮灭以比例 (1-b) 在 R 内部重建 (自愈),
      比例 b 强制转移到背景 (闭合子图创生抑制)
    - delta(R) = 内部湮灭 - 内部补偿 = b * ann_rate (净湮灭率 = 引力源强度)
    - ΔF ∝ Δdelta(R): 边界调制 -> 净湮灭率变化 -> 重量变化
    - 内部边净损失 -> sigma_R 升高, sigma_out 降低 -> 密度梯度增强 (引力场)

运行: python openSPUM/tests/archimedes_vacuum_weight.py
"""

from __future__ import annotations
import sys
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

_root = str(Path(__file__).resolve().parent.parent)  # openSPUM/
if _root not in sys.path:
    sys.path.insert(0, _root)

from Phase_1.seed_epoch_engine import SeedEpochConfig, SeedEpochEngine
from Phase_1.node_registry import NodeRegistry
from Phase_1.relation_pool import RelationPool


# ============================================================
# 网络构建与区域选取
# ============================================================

def build_network(target_edges: int = 20000, seed_duration: int = 100):
    """确定性构建背景网络 (同一参数 -> 逐位相同网络)。"""
    config = SeedEpochConfig(target_edges=target_edges, seed_duration=seed_duration)
    engine = SeedEpochEngine(config=config)
    engine.run_seed_epoch()
    return engine.node_registry, engine.relation_pool


def select_region(registry: NodeRegistry, n_r: int = 40) -> Set[str]:
    """样品 R = 度数最高的 n_r 个节点 (材料核, 稠密闭合子图)。"""
    nodes = sorted(registry.nodes.values(), key=lambda n: -n.degree)
    return {n.address.uid for n in nodes[:n_r]}


def region_stats(pool: RelationPool, registry: NodeRegistry, R: Set[str]) -> Dict:
    """区域拓扑统计: 内部边数 / 边界边数 / sigma_R / sigma_out / dsigma。"""
    e_in = 0
    e_bound = 0
    for key in pool.manifest:
        a, b = key
        ia, ib = a in R, b in R
        if ia and ib:
            e_in += 1
        elif ia or ib:
            e_bound += 1
    p_in = len(R)
    p_out = max(1, registry.node_count() - len(R))
    e_out = max(1, pool.edge_count() - e_in)  # 至少一端在 R 外的边
    sigma_in = p_in / e_in if e_in else float("inf")
    sigma_out = p_out / e_out
    dsigma = sigma_in - sigma_out if math.isfinite(sigma_in) else float("nan")
    return {
        "P_in": p_in, "E_in": e_in, "E_bound": e_bound,
        "sigma_in": sigma_in, "sigma_out": sigma_out, "dsigma": dsigma,
    }


# ============================================================
# 边界刚度帧演化 (核心)
# ============================================================

def _internal_edge_list(pool: RelationPool, R: Set[str]) -> List[Tuple[str, str]]:
    """R 内部边 (确定性排序: 按 key 字典序)。"""
    return sorted(
        k for k in pool.manifest
        if not pool.is_cluster_edge(k) and k[0] in R and k[1] in R
    )


def run_frame_boundary(
    pool: RelationPool,
    registry: NodeRegistry,
    R: Set[str],
    b: float,
    frame: int,
    growth: int = 3,
    ann_rate: int = 5,
    max_iters: int = 100,
) -> Dict:
    """单帧演化 + 边界刚度调制。

    帧序列:
        1. 背景生长: growth 条全局新边 (候选优先低度数/悬挂 ->
           极少落入高度数样品区 R, 对应创生抑制)
        2. 自发内部湮灭: 确定性选取 ann_rate 条 R 内部边断裂
           (真空涨落, 速率与 b 无关)
        3. 补偿转移: n_inner = round((1-b)*ann_rate) 条在 R 内重建 (自愈);
           n_transfer = ann_rate - n_inner 条强制外部 (对偶转移)
        4. 级联消解: 悬挂对批量湮灭 + 补偿 (与引擎 cap=3 一致)
    """
    # ---- 1. 背景生长 ----
    if growth > 0:
        cands = pool.deterministic_candidates(registry, count=growth, layer="all")
        for la, lb in cands:
            pool.manifest_relation(la, lb, frame, registry)

    # ---- 2. 自发内部湮灭 ----
    internal = _internal_edge_list(pool, R)
    to_ann = internal[:ann_rate]
    ann_keys: Set[Tuple[str, str]] = set(to_ann)
    ann_pairs: List[Tuple[str, str, str]] = []  # (key, uid1, uid2)
    ann_inside = 0
    for key in to_ann:
        rel = pool.manifest.pop(key, None)
        if rel is None:
            continue
        registry.update_degree(rel.loc1.uid, -1)
        registry.update_degree(rel.loc2.uid, -1)
        ann_pairs.append((key, rel.loc1.uid, rel.loc2.uid))
        ann_inside += 1

    # ---- 3. 补偿转移 ----
    n_inner = int(round((1.0 - b) * ann_inside))
    n_transfer = ann_inside - n_inner
    comp_inside = 0
    comp_total = 0

    # 3a. 内部自愈: 重建前 n_inner 条被湮灭的内部边
    for key, u1, u2 in ann_pairs[:n_inner]:
        n1 = registry.nodes.get(u1)
        n2 = registry.nodes.get(u2)
        if n1 is None or n2 is None:
            continue
        if n1.is_saturated or n2.is_saturated:
            continue
        rel = pool.manifest_relation(n1.address, n2.address, frame, registry)
        if rel is not None:
            comp_inside += 1
            comp_total += 1

    # 3b. 外部转移: n_transfer 条边强制落到 R 外部
    if n_transfer > 0:
        cands = pool.deterministic_candidates(
            registry, count=n_transfer + 8, layer="all", exclude_keys=ann_keys,
        )
        placed_ext = 0
        used: Set[Tuple[str, str]] = set()
        for la, lb in cands:
            if placed_ext >= n_transfer:
                break
            if la.uid in R and lb.uid in R:
                continue  # 转移槽禁止内部
            key = tuple(sorted([la.uid, lb.uid]))
            if key in used or key in pool.manifest:
                continue
            rel = pool.manifest_relation(la, lb, frame, registry)
            if rel is None:
                continue
            used.add(key)
            placed_ext += 1
            comp_total += 1

    # ---- 4. 级联消解 ----
    ann_cas = comp_cas = 0
    for _it in range(max_iters):
        dang = registry.dangling_nodes()
        if not dang:
            break
        dang_uids = {n.address.uid for n in dang}
        batch: List[Tuple[str, str]] = []
        for key, rel in list(pool.manifest.items()):
            if pool.is_cluster_edge(key):
                continue
            if rel.loc1.uid in dang_uids and rel.loc2.uid in dang_uids:
                batch.append(key)
        if not batch:
            break
        batch_set = set(batch)
        for key in batch_set:
            rel = pool.manifest.pop(key, None)
            if rel is None:
                continue
            registry.update_degree(rel.loc1.uid, -1)
            registry.update_degree(rel.loc2.uid, -1)
            ann_cas += 1
            if key[0] in R and key[1] in R:
                ann_inside += 1
        comp_count = min(len(batch_set), 3)
        cands = pool.deterministic_candidates(
            registry, count=comp_count, layer="all", exclude_keys=batch_set,
        )
        for la, lb in cands:
            rel = pool.manifest_relation(la, lb, frame, registry)
            if rel is None:
                continue
            comp_cas += 1
            comp_total += 1
            if la.uid in R and lb.uid in R:
                comp_inside += 1

    ann_inside += ann_cas
    return {
        "ann_inside": ann_inside, "comp_inside": comp_inside,
        "ann_total": ann_inside, "comp_total": comp_total,
        "delta_in": ann_inside - comp_inside,
        "net": ann_inside - comp_total,
    }


# ============================================================
# 实验协议
# ============================================================

def sweep_boundary_stiffness(
    b_values: List[float],
    target_edges: int = 20000,
    seed_duration: int = 100,
    n_r: int = 40,
    frames: int = 40,
    warmup: int = 8,
    growth: int = 3,
) -> Dict:
    """对每个 b 重建确定性相同网络, 测量稳态 delta(R)、sigma_R、sigma_out。"""
    out = {}
    for b in b_values:
        registry, pool = build_network(target_edges, seed_duration)
        R = select_region(registry, n_r)
        ann_rate = max(1, n_r // 8)
        deltas: List[float] = []
        sig_in_list: List[float] = []
        sig_out_list: List[float] = []
        net_list: List[float] = []
        for f in range(frames):
            r = run_frame_boundary(pool, registry, R, b, f,
                                   growth=growth, ann_rate=ann_rate)
            net_list.append(r["net"])
            if f >= warmup:
                deltas.append(r["delta_in"])
                st = region_stats(pool, registry, R)
                sig_in_list.append(st["sigma_in"])
                sig_out_list.append(st["sigma_out"])
        mean_d = sum(deltas) / len(deltas)
        std_d = math.sqrt(sum((x - mean_d) ** 2 for x in deltas) / len(deltas))
        st_end = region_stats(pool, registry, R)
        out[b] = {
            "delta_in": mean_d, "delta_std": std_d,
            "sigma_in": sum(sig_in_list) / len(sig_in_list),
            "sigma_out": sum(sig_out_list) / len(sig_out_list),
            "net_mean": sum(net_list) / len(net_list),
            "E_in_end": st_end["E_in"],
            "E_final": pool.edge_count(),
        }
    return out


def step_response(
    b_lo: float = 0.0, b_hi: float = 0.6,
    target_edges: int = 20000, seed_duration: int = 100,
    n_r: int = 40, settle: int = 20, probe: int = 25,
    growth: int = 3,
) -> Dict:
    """阶跃响应: b_lo 平衡 -> 阶跃到 b_hi -> 追踪 delta(R) 与 sigma_R。

    预言: delta 在 1 帧内跳变 (同相, 无记忆);
          sigma_R 以恒定速率漂移 (净湮灭累积)。
    """
    registry, pool = build_network(target_edges, seed_duration)
    R = select_region(registry, n_r)
    ann_rate = max(1, n_r // 8)
    base = 0.0
    for f in range(settle):
        r = run_frame_boundary(pool, registry, R, b_lo, f,
                               growth=growth, ann_rate=ann_rate)
        base = r["delta_in"]  # 阶跃前基线的最后一帧
    series_d: List[float] = []
    series_sig: List[float] = []
    e_in_series: List[float] = []
    for f in range(probe):
        r = run_frame_boundary(pool, registry, R, b_hi, settle + f,
                               growth=growth, ann_rate=ann_rate)
        series_d.append(r["delta_in"])
        st = region_stats(pool, registry, R)
        series_sig.append(st["sigma_in"])
        e_in_series.append(st["E_in"])
    steady = sum(series_d[-5:]) / 5
    # delta 的响应时间: 达到稳态的首帧
    tau = None
    for i, v in enumerate(series_d):
        if abs(v - steady) <= 1e-9 and abs(steady - base) > 1e-9:
            tau = i + 1
            break
    return {
        "delta_series": series_d, "sigma_series": series_sig,
        "e_in_series": e_in_series,
        "steady": steady, "base": base, "tau_frames": tau,
        "sigma_drift": (series_sig[-1] - series_sig[0]) / max(1, len(series_sig)),
    }


def scaling_check(
    n_r_values: List[int] = (20, 30, 40, 60),
    b_lo: float = 0.0, b_hi: float = 0.6,
    target_edges: int = 20000, seed_duration: int = 100,
    frames: int = 40, warmup: int = 8, growth: int = 3,
) -> Dict:
    """标度律: 每节点净湮灭率 delta/|R| 是否与样品大小无关 (强度量)。"""
    out = {}
    for n_r in n_r_values:
        delta_lo = []
        delta_hi = []
        for _rep in range(2):  # 两次独立复现取平均
            registry, pool = build_network(target_edges, seed_duration)
            R = select_region(registry, n_r)
            ann_rate = max(1, n_r // 8)
            d_lo, d_hi = [], []
            for f in range(frames):
                r = run_frame_boundary(pool, registry, R, b_lo, f,
                                       growth=growth, ann_rate=ann_rate)
                if f >= warmup:
                    d_lo.append(r["delta_in"])
            for f in range(frames):
                r = run_frame_boundary(pool, registry, R, b_hi, f,
                                       growth=growth, ann_rate=ann_rate)
                if f >= warmup:
                    d_hi.append(r["delta_in"])
            delta_lo.append(sum(d_lo) / len(d_lo))
            delta_hi.append(sum(d_hi) / len(d_hi))
        m_lo = sum(delta_lo) / len(delta_lo)
        m_hi = sum(delta_hi) / len(delta_hi)
        out[n_r] = {
            "delta_lo": m_lo, "delta_hi": m_hi,
            "abs_delta": m_hi - m_lo,
            "per_node": (m_hi - m_lo) / n_r,
        }
    return out


# ============================================================
# 打印报告
# ============================================================

def print_report(
    net_summary: Dict,
    region0: Dict,
    sweep: Dict,
    step: Dict,
    scale: Dict,
) -> None:
    W = 72
    print("=" * W)
    print("  阿基米德真空称重 - 边界刚度调制净湮灭模拟报告")
    print("=" * W)
    print(f"\n  [网络] V={net_summary['V']}, E={net_summary['E']}, "
          f"晶子={net_summary['cryst']}, 悬挂={net_summary['dang']}")
    print(f"  [区域 R] P_R={region0['P_in']}, E_R(内)={region0['E_in']}, "
          f"E(边界)={region0['E_bound']}")
    print(f"  [初始] sigma_R={region0['sigma_in']:.4f}, "
          f"sigma_out={region0['sigma_out']:.4f}, "
          f"dsigma={region0['dsigma']:.4f}")

    # ---- b 扫描 ----
    print(f"\n{'─' * W}")
    print("  实验 1: delta(R) vs 边界刚度 b  (稳态净湮灭率, 每帧)")
    print(f"{'─' * W}")
    print(f"  {'b':<6}{'delta':<10}{'std':<8}{'sig_R':<10}"
          f"{'sig_out':<10}{'dsig':<10}{'E_R末':<10}{'守恒净':<8}")
    print(f"  {'─'*6}{'─'*10}{'─'*8}{'─'*10}{'─'*10}{'─'*10}{'─'*10}{'─'*8}")
    b_list = sorted(sweep.keys())
    xs, ys = [], []
    for b in b_list:
        d = sweep[b]
        dsigma = d["sigma_in"] - d["sigma_out"]
        print(f"  {b:<6.2f}{d['delta_in']:<10.3f}{d['delta_std']:<8.3f}"
              f"{d['sigma_in']:<10.4f}{d['sigma_out']:<10.4f}"
              f"{dsigma:<10.4f}{d['E_in_end']:<10d}{d['net_mean']:<8.3f}")
        xs.append(b)
        ys.append(d["delta_in"])

    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0
    intercept = my - slope * mx
    ann_rate = net_summary.get("ann_rate", 0)
    print(f"\n  线性拟合: delta(R;b) = {slope:.4f}*b + {intercept:.4f}")
    print(f"  理论预言: delta = b * ann_rate (ann_rate=|R|//8={ann_rate})")
    print(f"  判定: {'线性响应 (ΔF 正比调制深度)' if abs(slope - ann_rate) <= 1.5 * ann_rate else '偏离线性'}")

    # ---- 阶跃响应 ----
    print(f"\n{'─' * W}")
    print(f"  实验 2: 阶跃响应  b: {step.get('b_lo', 0.0)} -> {step.get('b_hi', 0.6)}")
    print(f"{'─' * W}")
    ds = step["delta_series"]
    print(f"  逐帧 delta(R): {[f'{v:.0f}' for v in ds]}")
    print(f"  响应时间: tau = {step['tau_frames']} 帧 "
          f"({'同相 (瞬时)' if step['tau_frames'] == 1 else '有限弛豫'})")
    print(f"  稳态 delta: {step['steady']:.2f} / 帧")
    ss = step["sigma_series"]
    print(f"  sigma_R 序列 (帧0->末): {ss[0]:.3f} -> {ss[-1]:.3f}, "
          f"漂移 {step['sigma_drift']:.4f}/帧 (净湮灭累积)")
    print(f"  E_R(内) 序列: {step['e_in_series'][0]:.0f} -> "
          f"{step['e_in_series'][-1]:.0f} (内部结构耗损)")

    # ---- 标度律 ----
    print(f"\n{'─' * W}")
    print("  实验 3: 标度律 (每节点净湮灭率是否强度量)")
    print(f"{'─' * W}")
    for n_r, d in sorted(scale.items()):
        print(f"  |R|={n_r:<4} delta(0)={d['delta_lo']:<8.3f} "
              f"delta(0.6)={d['delta_hi']:<8.3f} "
              f"Δdelta={d['abs_delta']:<8.3f} "
              f"Δdelta/|R|={d['per_node']:.4f}")

    print("=" * W)


def main() -> None:
    target_edges = 20000
    seed_duration = 100
    n_r = 40
    ann_rate = max(1, n_r // 8)

    # 网络 + 区域快照
    registry, pool = build_network(target_edges, seed_duration)
    R = select_region(registry, n_r)
    net_summary = {
        "V": registry.node_count(),
        "E": pool.edge_count(),
        "cryst": len(registry.crystallite_nodes()),
        "dang": len(registry.dangling_nodes()),
        "ann_rate": ann_rate,
    }
    region0 = region_stats(pool, registry, R)

    # 实验 1: b 扫描
    sweep = sweep_boundary_stiffness(
        b_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        target_edges=target_edges, seed_duration=seed_duration,
        n_r=n_r, frames=40, warmup=8, growth=3,
    )

    # 实验 2: 阶跃响应
    step = step_response(
        b_lo=0.0, b_hi=0.6,
        target_edges=target_edges, seed_duration=seed_duration,
        n_r=n_r, settle=20, probe=25, growth=3,
    )
    step["b_lo"] = 0.0
    step["b_hi"] = 0.6

    # 实验 3: 标度律
    scale = scaling_check(
        n_r_values=(20, 30, 40, 60), b_lo=0.0, b_hi=0.6,
        target_edges=target_edges, seed_duration=seed_duration,
        frames=40, warmup=8, growth=3,
    )

    print_report(net_summary, region0, sweep, step, scale)


if __name__ == "__main__":
    main()
