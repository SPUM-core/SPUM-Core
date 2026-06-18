"""
闭合子图分析 v2 — 聚焦晶子子图与局部闭合结构

SPUM 预测: 对于满足平面三角剖分的闭合子图，Σ(6−deg(v)) = 12。
现实约束: 当前代码产生一张大网，需要从晶子子图中找闭合结构。

分析层次:
  1. 晶子子图 (degree=50 节点) — 是否形成闭合簇？
  2. k-核分解 — 图中心最稠密的部分
  3. 最大团 — 完全连接的晶子集合
  4. 单节点局部 Σ(6−deg) — 每个节点自身的 "6 − deg" 分布

运行: python openSPUM/tests/verify_closed_subgraph.py
"""

import sys
from pathlib import Path
from collections import defaultdict, deque
import math

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from Phase_1.seed_epoch_engine import SeedEpochConfig, SeedEpochEngine
from Phase_1.constants import CRYSTALLITE_DEGREE_THRESHOLD


# ============================================================
# 图分析工具
# ============================================================

def build_adjacency(engine):
    adj = defaultdict(set)
    for (uid_a, uid_b) in engine.relation_pool.manifest:
        adj[uid_a].add(uid_b)
        adj[uid_b].add(uid_a)
    return dict(adj)


def subgraph_adj(adj, node_set):
    """Extract adjacency within a subset of nodes."""
    s = set(node_set)
    result = {}
    for u in node_set:
        result[u] = {v for v in adj.get(u, []) if v in s}
    return result


def deg_in_subgraph(adj_sub):
    """Degree within the subgraph."""
    return {u: len(nbs) for u, nbs in adj_sub.items()}


def find_connected_components(adj):
    visited = set()
    components = []
    for node in adj:
        if node in visited:
            continue
        queue = deque([node])
        comp = set()
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    queue.append(nb)
        if comp:
            components.append(comp)
    return components


def k_core_decomposition(adj):
    """k-core decomposition: returns {k: [nodes]}."""
    if not adj:
        return {}
    degrees = {u: len(nbs) for u, nbs in adj.items()}
    nodes = sorted(degrees.keys(), key=lambda u: degrees[u])
    bin_boundaries = {}
    max_deg = max(degrees.values()) if degrees else 0
    bins = [[] for _ in range(max_deg + 1)]
    pos = {u: i for i, u in enumerate(nodes)}
    for u in nodes:
        bins[degrees[u]].append(u)
    cores = {}
    for k in range(max_deg + 1):
        while bins[k]:
            u = bins[k].pop()
            cores[u] = k
            for v in adj.get(u, []):
                if degrees[v] > k:
                    old_deg = degrees[v]
                    degrees[v] -= 1
                    new_deg = degrees[v]
                    # move to correct bin
                    bins[old_deg].remove(v)
                    bins[new_deg].append(v)
    result = defaultdict(list)
    for node, k in cores.items():
        result[k].append(node)
    return dict(result)


def find_cliques_of_size(adj_sub, min_size=3, max_search=5000):
    """Simple brute-force for small subgraphs. Only works for V ≤ ~50."""
    nodes = list(adj_sub.keys())
    if len(nodes) > 30:
        return []  # too large for brute force

    cliques = []
    n = len(nodes)

    # Helper: check if a set is a clique
    def is_clique(node_set):
        s = list(node_set)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if s[j] not in adj_sub[s[i]]:
                    return False
        return True

    # Bron–Kerbosch (simplified: only find all maximal)
    def bron_kerbosch(r, p, x):
        if not p and not x:
            if len(r) >= min_size:
                cliques.append(set(r))
            return
        # use pivot
        union_px = p | x
        if union_px:
            pivot = next(iter(union_px))
            candidates = p - adj_sub[pivot] - {pivot}
        else:
            candidates = p
        for v in list(candidates):
            bron_kerbosch(
                r | {v},
                p & adj_sub[v],
                x & adj_sub[v],
            )
            p.remove(v)
            x.add(v)

    bron_kerbosch(set(), set(nodes), set())
    return sorted([sorted(c) for c in cliques], key=len, reverse=True)


# ============================================================
# 分析函数
# ============================================================

def analyze_full_graph(engine, label):
    """全图分析: k-core, 晶子子图, 局部不变量。"""
    adj = build_adjacency(engine)
    nodes = engine.node_registry.nodes
    total_v = len(nodes)
    total_e = engine.relation_pool.edge_count()

    print(f"\n{'=' * 72}")
    print(f"  闭合子图分析 v2: {label}")
    print(f"{'=' * 72}")
    print(f"  总节点数: {total_v}")
    print(f"  总边数:   {total_e}")
    print(f"  度数守恒: {sum(n.degree for n in nodes.values())} = 2×{total_e} "
          f"{'✓' if sum(n.degree for n in nodes.values()) == 2*total_e else '✗'}")

    # 度数分布
    degs = [n.degree for n in nodes.values()]
    print(f"\n  度数分布:")
    print(f"    最小: {min(degs)}")
    print(f"    中位: {sorted(degs)[len(degs)//2]}")
    print(f"    最大: {max(degs)}")
    print(f"    晶子 (≥50): {sum(1 for d in degs if d >= 50)}")

    # ----- 1. k-core 分解 -----
    cores = k_core_decomposition(adj)
    max_k = max(cores.keys())
    print(f"\n  [{1}] k-core 分解:")
    print(f"    最大 k-core: k_max = {max_k} ({len(cores[max_k])} 节点)")

    # 只看 k ≥ 3 的 core (有意义的结构)
    significant_cores = {k: v for k, v in cores.items() if k >= 3}
    if significant_cores:
        min_sig = min(significant_cores.keys())
        max_sig = max(significant_cores.keys())
        # 最高核心的节点集合
        top_core_nodes = cores[max_k]
        top_adj = subgraph_adj(adj, top_core_nodes)
        top_v = len(top_core_nodes)
        top_e = sum(len(nbs) for nbs in top_adj.values()) // 2
        top_deg = deg_in_subgraph(top_adj)
        top_invariant = sum(6 - d for d in top_deg.values())
        print(f"    k={max_k} core: V={top_v}, E={top_e}, "
              f"Σ(6−deg)={top_invariant}")
        print(f"    核心内部度数范围: [{min(top_deg.values())}, {max(top_deg.values())}]")

    # ----- 2. 晶子子图 (degree ≥ 50) — 仅使用聚簇边 ----
    crystallite_nodes = [uid for uid in nodes
                         if nodes[uid].degree >= CRYSTALLITE_DEGREE_THRESHOLD]
    crystal_nodes_all = [uid for uid in nodes
                         if nodes[uid].degree >= CRYSTALLITE_DEGREE_THRESHOLD]

    # 构建仅含聚簇边的晶子邻接表
    cry_cluster_adj = defaultdict(set)
    for (uid_a, uid_b) in engine.relation_pool.manifest:
        if not engine.relation_pool.is_cluster_edge((uid_a, uid_b)):
            continue
        na = nodes.get(uid_a)
        nb = nodes.get(uid_b)
        if na and nb and na.degree >= CRYSTALLITE_DEGREE_THRESHOLD and nb.degree >= CRYSTALLITE_DEGREE_THRESHOLD:
            cry_cluster_adj[uid_a].add(uid_b)
            cry_cluster_adj[uid_b].add(uid_a)

    print(f"\n  [{2}] 晶子子图 (degree ≥ 50):")
    print(f"    晶子总数: {len(crystal_nodes_all)}")

    if len(crystal_nodes_all) >= 2:
        cry_adj = subgraph_adj(adj, crystal_nodes_all)
        cry_components = find_connected_components(cry_adj)

        print(f"    晶子子图连通分量数: {len(cry_components)}")
        # 聚簇边信息
        cry_cluster_edges = sum(len(cry_cluster_adj[u]) for u in crystal_nodes_all) // 2
        print(f"    晶子间边数 (总): {sum(len(cry_adj[u]) for u in crystal_nodes_all) // 2}")
        print(f"    晶子间边数 (聚簇): {cry_cluster_edges}")

        if cry_components and cry_cluster_edges > 0:
            for idx, comp in enumerate(cry_components):
                if len(comp) < 2:
                    continue
                # 使用聚簇边分析
                comp_cluster = defaultdict(set)
                for u in comp:
                    for v in cry_cluster_adj.get(u, []):
                        if v in comp:
                            comp_cluster[u].add(v)
                            comp_cluster[v].add(u)
                if not comp_cluster:
                    continue
                deg_in = deg_in_subgraph(comp_cluster)
                v, e = len(comp_cluster), sum(deg_in.values()) // 2
                spum_val = sum(6 - d for d in deg_in.values())
                inside_degs = list(deg_in.values())
                closed = all(d >= 2 for d in inside_degs)
                print(f"    晶子簇 {idx} (聚簇): V聚簇={v}, E聚簇={e}, "
                      f"闭合={'✓' if closed else '✗'}, "
                      f"Σ(6−deg)={spum_val}, "
                      f"内部度数范围 [{min(inside_degs)}, {max(inside_degs)}]")

    # ----- 3. 全图 Σ(6−deg) 分析 -----
    print(f"\n  [{3}] 全图 Σ(6−deg) 分解:")

    # 全局
    global_spum = sum(6 - n.degree for n in nodes.values())
    total_deg = sum(n.degree for n in nodes.values())
    print(f"    全局: Σ(6−deg) = {global_spum}")
    print(f"           (公式: 6V−2E = {6*total_v} − {2*total_e} = {6*total_v - 2*total_e})")

    # 单节点 6−deg 分布
    inv_dist = defaultdict(int)
    for n in nodes.values():
        inv_dist[6 - n.degree] += 1
    top_invs = sorted(inv_dist.items(), key=lambda x: -x[0])[:8]
    bottom_invs = sorted(inv_dist.items(), key=lambda x: x[0])[:8]
    print(f"    单节点 6−deg 分布 (高值前 8):")
    for val, cnt in top_invs:
        print(f"      6−deg = {val:>3}: {cnt} 节点")
    print(f"    单节点 6−deg 分布 (低值前 8):")
    for val, cnt in bottom_invs:
        print(f"      6−deg = {val:>3}: {cnt} 节点")

    # ----- 4. 晶子间是否形成了正则图？-----
    print(f"\n  [{4}] 晶子子图结构分析:")

    if len(crystal_nodes_all) >= 3:
        cry_adj = subgraph_adj(adj, crystal_nodes_all)
        cry_deg = deg_in_subgraph(cry_adj)
        deg_vals = list(cry_deg.values())
        if deg_vals:
            print(f"    晶子子图度数: min={min(deg_vals)}, "
                  f"max={max(deg_vals)}, "
                  f"avg={sum(deg_vals)/len(deg_vals):.2f}")

            # 如果每个晶子之间都连接 → 完全图
            n_cry = len(crystal_nodes_all)
            complete_e = n_cry * (n_cry - 1) // 2
            actual_e = sum(deg_vals) // 2
            if actual_e == complete_e:
                print(f"    结构: 完全图 K_{{{n_cry}}} {'(每个晶子连接所有其他晶子)'}")
            else:
                density = actual_e / complete_e * 100 if complete_e > 0 else 0
                print(f"    密度: {actual_e}/{complete_e} = {density:.1f}%")

            # 尝试找小团
            if n_cry <= 15:
                cliques = find_cliques_of_size(cry_adj, min_size=3)
                if cliques:
                    print(f"    最大团大小: {len(cliques[0])}")
                    # 对每个团计算 Σ(6−deg)
                    for c in cliques[:5]:
                        c_adj = subgraph_adj(adj, c)
                        c_deg = deg_in_subgraph(c_adj)
                        v, e = len(c), sum(c_deg.values()) // 2
                        spum_val = sum(6 - d for d in c_deg.values())
                        print(f"      团 V={v}, E={e}, Σ(6−deg)={spum_val}")

    # ----- 5. 结论 -----
    print(f"\n  {'─' * 50}")
    print(f"  评估:")

    # 判断标准: 是否存在使 Σ(6−deg)=12 成立的闭合子图
    found_12 = False
    # 检查 k-core 最高层的核心
    if significant_cores:
        core_set = cores[max_k]
        core_adj_s = subgraph_adj(adj, core_set)
        core_deg_s = deg_in_subgraph(core_adj_s)
        core_spum = sum(6 - d for d in core_deg_s.values())
        core_v = len(core_set)
        core_e = sum(core_deg_s.values()) // 2
        expected_e = 3 * core_v - 6
        print(f"    最大 k-core: V={core_v}, E={core_e}, "
              f"预期三角剖分 E=3V−6={expected_e}, "
              f"偏离={core_e - expected_e}")
        print(f"    Σ(6−deg) = {core_spum}")
        if abs(core_spum) < 50:
            found_12 = True
            print(f"    *** Σ(6−deg) 接近理论值! ***")

    print(f"    {'12 已在图中涌现 ✓' if found_12 else '12 尚未在图结构中显现'}")

    return {
        "total_v": total_v,
        "total_e": total_e,
        "max_k": max_k,
        "n_crystallites": len(crystal_nodes_all),
        "found_12": found_12,
    }


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    config = SeedEpochConfig(target_edges=20000, seed_duration=300)
    engine = SeedEpochEngine(config=config)
    engine.run_seed_epoch()
    r1 = analyze_full_graph(engine, "10⁴ 量级 (20000 边, 630 节点)")

    config2 = SeedEpochConfig(target_edges=100000, seed_duration=500)
    engine2 = SeedEpochEngine(config=config2)
    engine2.run_seed_epoch()
    r2 = analyze_full_graph(engine2, "10⁵ 量级 (100000 边, 2226 节点)")

    print(f"\n{'=' * 72}")
    print(f"  最终判断")
    print(f"  10⁴: k_max={r1['max_k']}, 晶子={r1['n_crystallites']}")
    print(f"  10⁵: k_max={r2['max_k']}, 晶子={r2['n_crystallites']}")
    if r2.get("found_12"):
        print(f"  状态: 12 已在图中涌现 ✓")
        print(f"  k={r2['max_k']} 核心 Σ(6−deg) 接近 12，")
        print(f"  平面三角剖分正在形成中。")
    else:
        print(f"  状态: 12 尚未在图结构中显现（需更多迭代）")
    print(f"{'=' * 72}")
