"""
SPUM 局部演化规则集 — 元胞自动机的核心"物理"。

所有规则是纯函数：`next_deg = f(deg_self, {deg_neighbors})`
不访问任何全局状态、坐标、半径。

规则设计原则：
    - 悬挂边守恒：deg ≤ 1 的节点在邻居充足时被"救活"，否则回归潜态
    - 饱和限制：deg ≥ κ 的节点不再增加度数（晶子饱和）
    - 均值趋向：稳定节点向邻居均值靠拢（拓扑压力平滑）
    - 确定性：对同一输入永远给出同一输出
    - 空邻居列表 → 节点不在网络中，状态不变（避免 `all([])` 陷阱）
"""

from typing import List, Sequence

# ──────────────────────────────────────────────────────────
# 拓扑常数
# ──────────────────────────────────────────────────────────
KAPPA: int = 12          # 最小晶子阈值（接吻数）
CRYSTALLITE_DEGREE_THRESHOLD: int = 50  # 晶子饱和上限


# ──────────────────────────────────────────────────────────
# 规则族
# ──────────────────────────────────────────────────────────

def rule_mean_tracking(deg: int, neighbor_degs: Sequence[int]) -> int:
    """均值追踪规则 — 悬挂边守恒 + 均值趋向。

    空邻居列表 = 节点不在网络中，状态不变。
    """
    if not neighbor_degs:
        return deg

    if deg == 0:
        # 潜态：所有邻居稳定 → 激活为 deg=1
        if all(d >= 2 for d in neighbor_degs):
            return 1
        return 0

    if deg == 1:
        # 悬挂态：邻居稳定 → 提升到 deg=2；否则回归潜态
        if all(d >= 2 for d in neighbor_degs):
            return 2
        return 0

    if deg >= CRYSTALLITE_DEGREE_THRESHOLD:
        return deg

    # deg ≥ 2：向邻居均值靠拢
    avg = sum(neighbor_degs) / len(neighbor_degs)
    if avg > deg + 0.5:
        return min(deg + 1, CRYSTALLITE_DEGREE_THRESHOLD)
    elif avg < deg - 0.5:
        return max(deg - 1, 2)
    return deg


def rule_median_convergence(deg: int, neighbor_degs: Sequence[int]) -> int:
    """中位数收敛规则 — 比均值更快达到局部一致。

    悬挂边规则同均值规则，但稳定节点趋向邻居中位数。
    """
    if not neighbor_degs:
        return deg

    if deg == 0:
        if all(d >= 2 for d in neighbor_degs):
            return 1
        return 0

    if deg == 1:
        if all(d >= 2 for d in neighbor_degs):
            return 2
        return 0

    if deg >= CRYSTALLITE_DEGREE_THRESHOLD:
        return deg

    if len(neighbor_degs) <= 2:
        return deg

    sorted_degs = sorted(neighbor_degs)
    median = sorted_degs[len(sorted_degs) // 2]

    if median > deg:
        return min(deg + 1, CRYSTALLITE_DEGREE_THRESHOLD)
    elif median < deg:
        return max(deg - 1, 2)
    return deg


def rule_gradient_driven(deg: int, neighbor_degs: Sequence[int]) -> int:
    """梯度驱动规则 — 度数向局部密度梯度方向流动。

    类比 SPUM 的 ∇σ 驱动：度数从高密度区向低密度区扩散。
    """
    if not neighbor_degs:
        return deg

    if deg == 0:
        if all(d >= 2 for d in neighbor_degs):
            return 1
        return 0

    if deg == 1:
        if all(d >= 2 for d in neighbor_degs):
            return 2
        return 0

    if deg >= CRYSTALLITE_DEGREE_THRESHOLD:
        return deg

    # 计算"流向"：比自己高的邻居多 → 流入（度数增加）
    higher = sum(1 for d in neighbor_degs if d > deg)
    lower = sum(1 for d in neighbor_degs if d < deg)
    same = len(neighbor_degs) - higher - lower

    if higher > lower + same:
        return min(deg + 1, CRYSTALLITE_DEGREE_THRESHOLD)
    elif lower > higher + same:
        return max(deg - 1, 2)
    return deg


# ──────────────────────────────────────────────────────────
# 边演化
# ──────────────────────────────────────────────────────────

def evolve_edges_stable(neighbors: List[List[int]],
                        degrees: List[int],
                        degree_delta: List[int],
                        n: int) -> List[List[int]]:
    """稳定边演化：仅当节点度数变化时才增减边。

    比 edges_from_degrees 更保守——保留已有拓扑结构，
    只修复度数变化导致的边数不匹配。

    原则：
        - deg=0 的节点：删除所有边（回归潜态）
        - 度数增加的节点：从邻居池中按度数降序选取新邻居
        - 度数减少的节点：从邻居中度数最低的开始断开
        - 度数不变的节点：保持所有边
    """
    new_edges = [[] for _ in range(n)]
    edge_set = set()

    for i in range(n):
        for j in neighbors[i]:
            if i < j:
                edge_set.add((i, j))

    # 保留需要维持的边
    kept = set()
    deleted = set()

    for (i, j) in edge_set:
        if degrees[i] == 0 or degrees[j] == 0:
            # 任一节点归零 → 删除
            deleted.add((i, j))
        else:
            # 暂时保留（待会再检查数量）
            kept.add((i, j))

    # 度数变化的节点需要调整边数
    for i in range(n):
        if degrees[i] == 0:
            continue
        current_count = sum(1 for j in neighbors[i] if degrees[j] > 0
                            and (min(i, j), max(i, j)) in kept)
        target = degrees[i]

        if current_count < target:
            # 需要增加边：从未连接且 deg>0 的节点中选取
            candidates = [j for j in range(n)
                          if j != i and degrees[j] > 0
                          and (min(i, j), max(i, j)) not in kept
                          and (min(i, j), max(i, j)) not in deleted]
            candidates.sort(key=lambda j: degrees[j], reverse=True)
            needed = target - current_count
            for j in candidates[:needed]:
                edge = (min(i, j), max(i, j))
                kept.add(edge)

        elif current_count > target:
            # 需要减少边：断开度数最低的邻居
            connected = [j for j in neighbors[i]
                         if (min(i, j), max(i, j)) in kept]
            connected.sort(key=lambda j: degrees[j])
            excess = current_count - target
            for j in connected[:excess]:
                edge = (min(i, j), max(i, j))
                kept.discard(edge)

    # 写入
    for (i, j) in kept:
        new_edges[i].append(j)
        new_edges[j].append(i)

    return new_edges


def edges_from_degrees(current_edges: List[List[int]],
                       degrees: List[int],
                       n: int) -> List[List[int]]:
    """从度数推断应存在的边（全局协调版本）。

    保留已有拓扑中匹配度数要求的边，
    删除不匹配的边，按需创建新边。
    """
    new_edges = [[] for _ in range(n)]

    edge_set = set()
    for i in range(n):
        for j in current_edges[i]:
            if i < j:
                edge_set.add((i, j))

    kept_edges = set()
    for i in range(n):
        if degrees[i] == 0:
            continue
        neighbors = [j for j in current_edges[i] if degrees[j] > 0]
        neighbors.sort(key=lambda j: degrees[j], reverse=True)
        keep_count = min(degrees[i], len(neighbors))
        for j in neighbors[:keep_count]:
            edge = (min(i, j), max(i, j))
            kept_edges.add(edge)

    for (i, j) in kept_edges:
        new_edges[i].append(j)
        new_edges[j].append(i)

    return new_edges


# ──────────────────────────────────────────────────────────
# 规则查询
# ──────────────────────────────────────────────────────────

RULES = {
    "mean": rule_mean_tracking,
    "median": rule_median_convergence,
    "gradient": rule_gradient_driven,
}


def get_rule(name: str):
    """按名称获取规则函数。"""
    if name not in RULES:
        raise ValueError(f"未知规则: {name}，可选: {list(RULES.keys())}")
    return RULES[name]
