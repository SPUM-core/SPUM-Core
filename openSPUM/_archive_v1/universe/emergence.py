"""
SPUM 涌现观测层 — 从帧演化中读取宇宙级涌现结构。

本层只做"观测"，不改写引擎状态。全部指标从 ParticleArray 的
原始拓扑量（deg / connections / pos / radius / active）推导，
对应 SPUM 可证伪断言：

    E1. 边缘不完美   — 每帧必残留悬挂端 (deg < 3), 永不为 0
    E2. 拓扑守恒     — Σ(6−deg) = 6V − 2E 全局可追踪
    E3. 晶子涌现     — deg ≥ κ (42) 的饱和终极单元
    E4. 12 晶子闭环  — 12 个晶子互连成 30 边 / 每子图度 5
                       → 宇宙的第一个"原子" (正二十面体)
    E5. σ 空间密度   — σ = |P| / |ε|, 火形不均匀度 Φ = Var(deg)
    E6. 开口占比     — 悬挂端占比 ≤ 1/3 (带边界高斯-博内导出)

本体论约束:
    - 不存储时间戳、不引入概率、不假设坐标语义
    - position 仅用于几何检测 (正二十面体相切), 不参与拓扑判断
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from Phase_0.particle_array import ParticleArray
from Phase_0.constants import CRYSTALLITE_DEGREE_THRESHOLD, KAPPA


# ============================================================
# 基础观测
# ============================================================

def observe_topology(particles: ParticleArray) -> Dict:
    """基础拓扑观测 — 只读 deg/边/活性。"""
    active = particles.active
    degrees = particles.degree
    V = int(np.sum(active))
    E = int(np.sum(degrees[active])) // 2
    return {
        "V": V,
        "E": E,
        "sigma": (V / E) if E > 0 else float("inf"),
        "invariant": 6 * V - 2 * E,               # Σ(6−deg)
        "dangling": int(np.sum(active & (degrees < 3))),
        "crystallites": int(np.sum(degrees[active] >= CRYSTALLITE_DEGREE_THRESHOLD)),
        "max_degree": int(np.max(degrees[active])) if V else 0,
        "degree_mean": float(np.mean(degrees[active])) if V else 0.0,
        "degree_std": float(np.std(degrees[active])) if V else 0.0,
    }


def observe_fire(particles: ParticleArray) -> Dict:
    """火形观测 — σ 的空间不均匀性 (∇σ ≠ 0 是引力梯度场的拓扑定义)。

    Φ = Var(deg) / ⟨deg⟩² — 归一化不均匀指数。
    Φ → 0: 火衰 (均匀化, 水流停滞); Φ 大: 火亢 (梯度冲毁木形)。
    """
    active = particles.active
    degrees = particles.degree[active]
    if degrees.size == 0:
        return {"fire_index": 0.0, "sigma_gradient": 0.0}
    mean = float(np.mean(degrees))
    var = float(np.var(degrees))
    fire = var / (mean * mean) if mean > 0 else 0.0
    # σ 梯度: 高 σ (低 deg) 区域与低 σ (高 deg) 区域的空间分离度
    hi = degrees >= mean
    lo = degrees < mean
    grad = 0.0
    if np.sum(hi) and np.sum(lo):
        grad = float(np.mean(degrees[hi]) - np.mean(degrees[lo]))
    return {"fire_index": round(fire, 6), "sigma_gradient": round(grad, 6)}


def observe_opening_ratio(particles: ParticleArray) -> Dict:
    """开口占比 — 悬挂端/总节点, 由带边界离散高斯-博内导出 ≤ 1/3。"""
    active = particles.active
    V = int(np.sum(active))
    if V == 0:
        return {"opening_ratio": 0.0, "opening_bound_ok": True}
    n_dangling = int(np.sum(active & (particles.degree < 3)))
    ratio = n_dangling / V
    return {
        "opening_ratio": round(ratio, 6),
        "opening_bound_ok": ratio <= 1.0 / 3.0 + 1e-9,
    }


# ============================================================
# 12 晶子闭环 (正二十面体) 涌现检测
# ============================================================

def _crystallite_subgraph(particles: ParticleArray):
    """提取晶子子图: 节点 = deg ≥ κ 的活性粒子, 边 = 晶子间连接。

    Returns:
        (uids, positions, radii, sub_degrees, sub_edges)
        不足 12 晶子时返回 (None,)*5
    """
    active = particles.active
    deg = particles.degree
    is_crys = active & (deg >= CRYSTALLITE_DEGREE_THRESHOLD)
    crys_idx = np.where(is_crys)[0]
    if crys_idx.size < 12:
        return None, None, None, None, None

    idx_set = set(int(i) for i in crys_idx)
    uids = [str(particles.uid[i]) for i in crys_idx]
    positions = {
        uids[k]: (float(particles.pos[int(crys_idx[k])][0]),
                  float(particles.pos[int(crys_idx[k])][1]),
                  float(particles.pos[int(crys_idx[k])][2]))
        for k in range(len(crys_idx))
    }
    radii = {uids[k]: float(particles.radius[int(crys_idx[k])])
             for k in range(len(crys_idx))}

    # 晶子子图边
    sub_edges: List[Tuple[int, int]] = []
    for (i, j) in particles.connections:
        if i in idx_set and j in idx_set:
            sub_edges.append((i, j))

    sub_deg = {uids[k]: 0 for k in range(len(crys_idx))}
    for (i, j) in sub_edges:
        sub_deg[str(particles.uid[i])] += 1
        sub_deg[str(particles.uid[j])] += 1

    return uids, positions, radii, sub_deg, sub_edges


def _crystallite_components(particles) -> List[Dict]:
    """拆解晶子诱导子图为连通分量, 对每个分量做 12 闭环判定。

    多中心分布式初态下, 各簇晶子各自成环, 全局判定 (混合所有簇晶子)
    会破坏 30 边/度 5 判据而永不 formed。改为连通分量级:
    每个分量独立判定 12 节点 / 30 边 / 每节点度 5 / Σ(6−deg)=12。

    返回:
        分量列表, 每个含 {uids, positions, radii, sub_deg, n, n_edges,
        deg5, invariant, formed}
    """
    active = particles.active
    deg = particles.degree
    is_crys = active & (deg >= CRYSTALLITE_DEGREE_THRESHOLD)
    crys_idx = np.where(is_crys)[0]

    # 晶子诱导子图: 节点 = 晶子, 边 = 两晶子间连接
    idx_set = set(int(i) for i in crys_idx)
    adj: Dict[int, set] = {int(i): set() for i in crys_idx}
    for (i, j) in particles.connections:
        if i in idx_set and j in idx_set:
            adj[int(i)].add(int(j))
            adj[int(j)].add(int(i))

    # BFS 拆连通分量
    visited = set()
    components = []
    uid_map = {int(i): str(particles.uid[i]) for i in crys_idx}
    for start in adj:
        if start in visited:
            continue
        stack = [start]
        comp_nodes = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp_nodes.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    stack.append(nb)
        components.append(comp_nodes)

    out = []
    for comp in components:
        n = len(comp)
        comp_set = set(comp)
        uids = [uid_map[x] for x in comp]
        positions = {
            uid_map[x]: (float(particles.pos[x][0]),
                         float(particles.pos[x][1]),
                         float(particles.pos[x][2]))
            for x in comp
        }
        radii = {uid_map[x]: float(particles.radius[x]) for x in comp}
        # 分量内边 (仅晶子-晶子)
        comp_edges = [(i, j) for (i, j) in particles.connections
                      if i in comp_set and j in comp_set]
        sub_deg = {uid_map[x]: 0 for x in comp}
        for (i, j) in comp_edges:
            sub_deg[uid_map[i]] += 1
            sub_deg[uid_map[j]] += 1
        n_edges = len(comp_edges)
        deg5 = sum(1 for d in sub_deg.values() if d == 5)
        invariant = sum(6 - d for d in sub_deg.values())
        formed = (n == 12 and n_edges == 30 and deg5 == 12 and invariant == 12)
        out.append({
            "uids": uids,
            "positions": positions,
            "radii": radii,
            "sub_deg": sub_deg,
            "sub_edges": comp_edges,
            "n": n,
            "n_edges": n_edges,
            "deg5": deg5,
            "invariant": invariant,
            "formed": formed,
        })
    return out


def _find_cluster_id(particles, uid: str) -> Optional[str]:
    """从粒子 uid (surf_I_jj / cent_I) 推测簇号 I。"""
    prefix = str(uid)
    if prefix.startswith("cent_"):
        return prefix  # 返回 cent_ 完整 uid 作为簇标识
    if prefix.startswith("surf_"):
        parts = prefix.split("_")
        if len(parts) >= 2:
            try:
                return f"cent_{int(parts[1]):04d}"
            except ValueError:
                return None
    return None


def detect_12_crystallite_ring(particles: ParticleArray) -> Dict:
    """检测 12 晶子闭环 — 宇宙的第一个"原子" (连通分量级)。

    拓扑判据 (与几何无关, 纯组合):
        1. 通信分量内晶子数 = 12
        2. 分量内边 = 30 (握手引理: Σdeg/2)
        3. 每个晶子子图度 = 5
        4. Σ(6−deg) = 12 (欧拉恒等式强制解)

    多中心分布式初态下, 全局晶子混合会破坏判据; 改为逐个连通分量
    独立判定。任一分量满足即集群涌现闭环。

    几何确认 (可选): 12 晶子位置构成正二十面体 (相切约束)。
    """
    total_crys = int(np.sum(particles.active &
                            (particles.degree >= CRYSTALLITE_DEGREE_THRESHOLD)))
    result = {
        "ring_formed": False,
        "n_crystallites": total_crys,
        "n_ring_edges": 0,
        "deg5_count": 0,
        "invariant": 0,
        "icosahedron": None,
        "cluster_id": None,
        "component_uids": None,
    }
    if total_crys < 12:
        return result

    components = _crystallite_components(particles)
    # 全局口径: 汇总全部连通分量的边/度 (兼容单簇 U4 语义)
    n_edges_g = sum(c["n_edges"] for c in components)
    deg5_g = sum(c["deg5"] for c in components)
    inv_g = sum(c["invariant"] for c in components)
    result.update({
        "n_ring_edges": n_edges_g,
        "deg5_count": deg5_g,
        "invariant": inv_g,
    })

    # 连通分量级判定: 任一分量满足闭环
    for comp in components:
        if not comp["formed"]:
            continue
        uids = comp["uids"]
        positions = comp["positions"]
        radii = comp["radii"]
        sub_edges = comp["sub_edges"]
        cluster_id = None
        for u in uids:
            cid = _find_cluster_id(particles, u)
            if cid:
                cluster_id = cid
                break
        result.update({
            "ring_formed": True,
            "n_ring_edges": comp["n_edges"],
            "deg5_count": comp["deg5"],
            "invariant": comp["invariant"],
            "cluster_id": cluster_id,
            "component_uids": uids,
        })
        try:
            from Phase_3.icosahedron_assembly import IcosahedronDetector
            neighbors: Dict[str, List[str]] = {u: [] for u in uids}
            for (i, j) in sub_edges:
                ui, uj = str(particles.uid[i]), str(particles.uid[j])
                neighbors[ui].append(uj)
                neighbors[uj].append(ui)
            result["icosahedron"] = IcosahedronDetector().detect(
                positions, neighbors, radii)
        except Exception:
            result["icosahedron"] = None
        break
    return result


# ============================================================
# 统一观测入口
# ============================================================

def observe_universe(particles: ParticleArray) -> Dict:
    """完整涌现观测 — 一帧快照的全部可证伪量。"""
    topo = observe_topology(particles)
    fire = observe_fire(particles)
    opening = observe_opening_ratio(particles)
    ring = detect_12_crystallite_ring(particles)
    return {
        **topo,
        **fire,
        **opening,
        "ring": ring,
        # 局部闭合可达: 42 团簇无悬挂端 (不完美定理图论版前半)
        "local_closed": topo["dangling"] == 0,
        # 闭合子图不变量: 全局 Σ(6−deg) = 12 时构成闭合三角剖分
        "closed_subgraph": topo["invariant"] == 12,
    }
