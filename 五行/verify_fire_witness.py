'''
无火证人演化轨迹对比实验 (§9.3 / §5.1 数值化)
=============================================

目标: 数值复现无火不可证人 —— 两个图四形统计相同、仅火形(∇k)不同,
在完全相同的五步帧规则与随机设置下演化, 轨迹分岔, 证明方向信息 (S_火)
不可由其余四维 (水/木/土/金) 推出。

构造要点 (数学修正):
  图同构必然保持度序列 ⇒ "结构同构但 ∇σ 不同"不可能成立。
  本实验采用严格可行的构造: 随机搜索两个图, 满足
    - 四形统计量相同: (μ, 桥边数, 土节点数, 水节点数) 四元组相等
    - σ = N/M 相同 (同 N, M 自动满足)
    - 度梯度 ∇k (火) 显著不同
  即: 初始五维向量中前四维重合、仅 S_火 分离 —— 这正是"删去火则混淆"的
  数值版本。随后在相同帧规则 + 相同随机种子下分别演化 T 帧, 输出轨迹
  分岔数据。

用法:
    python 五行/verify_fire_witness.py
    python 五行/verify_fire_witness.py --N 30 --M 45 --samples 2500 --frames 150 --trials 8
'''

import sys
import os
import copy
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 复用 §9.1 脚本的图生成器与五步帧事件
from verify_wuxing_completeness import (
    erdos_renyi, connected_components, edge_key,
    degree_gradient, EVENT_FUNCS,
)


# ──────────────────────────────────────────────────────────────────────────────
# 一、快速帧内桥检测 (Tarjan, O(N+M)) —— 仅用于搜索, 正确性与 BFS 法交叉核对
# ──────────────────────────────────────────────────────────────────────────────


def tarjan_bridges(adj: dict) -> set:
    """Tarjan 桥算法 (帧内算法, O(N+M)): 返回桥边键集合。"""
    ids = {}
    low = {}
    parent = {}
    bridges = set()
    timer = 0
    visited = set()
    for root in adj:
        if root in visited:
            continue
        visited.add(root)
        parent[root] = None
        stack = [(root, iter(adj[root]))]
        while stack:
            v, it = stack[-1]
            if v not in ids:
                ids[v] = low[v] = timer
                timer += 1
            advanced = False
            for w in it:
                if w == parent[v]:
                    continue
                if w not in visited:
                    visited.add(w)
                    parent[w] = v
                    stack.append((w, iter(adj[w])))
                    advanced = True
                    break
                low[v] = min(low[v], ids[w])
            if not advanced:
                stack.pop()
                p = parent[v]
                if p is not None:
                    low[p] = min(low[p], low[v])
                    if low[v] > ids[p]:
                        bridges.add(edge_key(p, v))
    return bridges


def five_form_metrics_fast(adj: dict) -> dict:
    """五形指标 (Tarjan 版, 快速): 与 §9.1 的 five_phase_metrics 输出对齐。"""
    n = len(adj)
    m = sum(len(nbrs) for nbrs in adj.values()) // 2
    c = connected_components(adj)
    mu = m - n + c
    bridges = tarjan_bridges(adj)
    n_tu = n_shui = n_mu = 0
    for v in adj:
        d = len(adj[v])
        if d <= 1:
            n_tu += 1
        elif d == 2:
            n1, n2 = list(adj[v])
            k1, k2 = edge_key(v, n1), edge_key(v, n2)
            if k1 in bridges or k2 in bridges:
                n_shui += 1          # 至少一条边是桥 ⇒ 不在环上 ⇒ 水 (链段)
            else:
                n_mu += 1            # 两条边皆在环上 ⇒ 木 (环边界)
        else:
            n_mu += 1                # deg ≥ 3: 木 (骨架), 同时为火 (场源)
    return {
        'mu': mu, 'n_metal': len(bridges), 'n_tu': n_tu,
        'n_shui': n_shui, 'n_mu': n_mu,
        'grad': degree_gradient(adj), 'sigma': n / m if m else 0.0,
        'n': n, 'm': m,
    }


def cross_check_tarjan(samples: int, rng: random.Random) -> int:
    """将 Tarjan 桥判定与 §9.1 的 BFS 法在随机图上逐边交叉核对, 返回失配数。"""
    from verify_wuxing_completeness import classify_edges
    mismatch = 0
    for _ in range(samples):
        n = rng.randint(10, 40)
        p = rng.choice([0.03, 0.08, 0.15])
        g = erdos_renyi(n, p, rng)
        bfs = classify_edges(g)
        tar = tarjan_bridges(g)
        for k in bfs:
            if (bfs[k] == 'bridge') != (k in tar):
                mismatch += 1
    return mismatch


# ──────────────────────────────────────────────────────────────────────────────
# 二、证人对搜索: 四形统计相同、仅 ∇k 不同
# ──────────────────────────────────────────────────────────────────────────────


def search_witness_pair(N: int, M: int, samples: int, rng: random.Random):
    """随机搜索: 按四形四元组 (μ, 桥数, 土, 水) 分桶,
    在桶内找 ∇k 分离最大的图对 (四形统计相同、σ 相同、仅火不同)。
    返回 None 表示未找到。"""
    p = 2 * M / (N * (N - 1))
    buckets = defaultdict(list)
    for _ in range(samples):
        g = erdos_renyi(N, p, rng)
        met = five_form_metrics_fast(g)
        key = (met['mu'], met['n_metal'], met['n_tu'], met['n_shui'])
        buckets[key].append((g, met))

    candidates = []
    for key, items in buckets.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[1]['grad'])
        sep = items[-1][1]['grad'] - items[0][1]['grad']
        candidates.append((sep, items[0], items[-1], key))

    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        return None
    sep, lo, hi, key = candidates[0]
    g_lo, met_lo = lo
    g_hi, met_hi = hi
    return {
        'G_hom': g_lo, 'G_het': g_hi,
        'bucket': key,
        'met_hom': met_lo, 'met_het': met_hi,
        'grad_sep': sep,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 三、演化与轨迹分岔分析
# ──────────────────────────────────────────────────────────────────────────────


def evolve_trajectory(adj: dict, frames: int, seed: int,
                      event_types: list) -> dict:
    """在副本上按五步帧规则演化 frames 帧 (回退选择, 与 §9.1 一致),
    记录每帧后的五形指标轨迹。G₁/G₂ 使用相同种子 ⇒ 相同的随机设置。"""
    g = copy.deepcopy(adj)
    rng = random.Random(seed)
    traj = {k: [] for k in ('mu', 'n_metal', 'n_tu', 'n_shui', 'grad')}
    for frame_idx in range(frames):
        for offset in range(len(event_types)):
            ev = event_types[(frame_idx + offset) % len(event_types)]
            if EVENT_FUNCS[ev](g, rng) is not None:
                break
        met = five_form_metrics_fast(g)
        for k in traj:
            traj[k].append(met[k])
    return traj


def trajectory_distance(t1: dict, t2: dict) -> dict:
    """逐指标轨迹 L1 距离 (逐帧 |Δ| 求和) 与终态差。"""
    out = {}
    for k in t1:
        total = sum(abs(a - b) for a, b in zip(t1[k], t2[k]))
        final = t1[k][-1] - t2[k][-1]
        out[k] = {'l1': total, 'final': final}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 四、主入口
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='无火证人演化轨迹对比实验 (§9.3)')
    parser.add_argument('--seed', type=int, default=42, help='搜索随机种子')
    parser.add_argument('--N', type=int, default=30, help='图节点数')
    parser.add_argument('--M', type=int, default=45, help='图边数')
    parser.add_argument('--samples', type=int, default=2500, help='搜索样本数')
    parser.add_argument('--frames', type=int, default=150, help='演化帧数')
    parser.add_argument('--trials', type=int, default=8, help='演化重复种子数')
    parser.add_argument('--min_sep', type=float, default=0.3, help='梯度分离最小阈值')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    event_types = ['water_extend', 'wood_close', 'earth_attach', 'prune']

    print('=' * 72)
    print('无火证人演化轨迹对比实验 (N=%d, M=%d, samples=%d)' %
          (args.N, args.M, args.samples))
    print('=' * 72)

    # ── 0. Tarjan 与 BFS 桥判定交叉核对 ──
    print('\n[0] 桥检测算法交叉核对 (Tarjan vs BFS, 20 图)')
    mm = cross_check_tarjan(20, rng)
    print(f'    失配边数 = {mm}')
    assert mm == 0, 'Tarjan 与 BFS 判定不一致, 中止'

    # ── 1. 搜索证人对 ──
    print('\n[1] 搜索证人 G₁/G₂ (四形统计相同, 仅 ∇k 不同)')
    pair = search_witness_pair(args.N, args.M, args.samples, rng)
    assert pair is not None, '未找到四形相同的图对, 增大 --samples 或调整 N/M'
    assert pair['grad_sep'] >= args.min_sep, \
        f'梯度分离不足 ({pair["grad_sep"]:.3f} < {args.min_sep}), 增大 --samples'
    mh, mt = pair['met_hom'], pair['met_het']
    print(f'    桶 (μ, 桥, 土, 水) = {pair["bucket"]}')
    print(f'    G₁ (低火, 梯度小): σ={mh["sigma"]:.4f}  ∇k={mh["grad"]:.4f}  '
          f'μ={mh["mu"]} 桥={mh["n_metal"]} 土={mh["n_tu"]} 水={mh["n_shui"]} 木={mh["n_mu"]}')
    print(f'    G₂ (高火, 梯度大): σ={mt["sigma"]:.4f}  ∇k={mt["grad"]:.4f}  '
          f'μ={mt["mu"]} 桥={mt["n_metal"]} 土={mt["n_tu"]} 水={mt["n_shui"]} 木={mt["n_mu"]}')
    four_same = (pair['met_hom']['mu'] == pair['met_het']['mu'] and
                 pair['met_hom']['n_metal'] == pair['met_het']['n_metal'] and
                 pair['met_hom']['n_tu'] == pair['met_het']['n_tu'] and
                 pair['met_hom']['n_shui'] == pair['met_het']['n_shui'])
    print(f'    → 四形统计相同: {four_same}  '
          f'σ 相同: {mh["sigma"] == mt["sigma"]}  '
          f'火分离 Δ∇k = {pair["grad_sep"]:.4f}')

    # ── 2. 同规则 + 同种子演化, 轨迹分岔 ──
    print(f'\n[2] 相同帧规则 + 相同随机种子演化 {args.frames} 帧, 重复 {args.trials} 次')
    agg = {k: {'l1': 0.0, 'final': 0.0} for k in ('mu', 'n_metal', 'n_tu', 'n_shui', 'grad')}
    for trial in range(args.trials):
        seed = args.seed + trial
        t_hom = evolve_trajectory(pair['G_hom'], args.frames, seed, event_types)
        t_het = evolve_trajectory(pair['G_het'], args.frames, seed, event_types)
        dist = trajectory_distance(t_hom, t_het)
        for k in agg:
            agg[k]['l1'] += dist[k]['l1']
            agg[k]['final'] += abs(dist[k]['final'])

    print('    逐指标轨迹距离 (L1, 均值):')
    for k, v in agg.items():
        print(f'      S_{k:8s} L1(均值) = {v["l1"] / args.trials:8.2f}  '
              f'|终态差|(均值) = {v["final"] / args.trials:8.2f}')

    # 四形终态分岔检验: 初始四形相同, 终态四形必须出现分歧 (至少一形分歧)
    diverged_forms = [k for k in ('mu', 'n_metal', 'n_tu', 'n_shui') if agg[k]['final'] > 0]
    print(f'\n[3] 结论')
    print(f'    初始四形统计相同 (距离 = 0), 仅 S_火 分离 (Δ∇k = {pair["grad_sep"]:.4f})')
    print(f'    演化 {args.frames} 帧后四形终态分歧: {diverged_forms}')
    print(f'    → 轨迹分岔: {"PASS" if diverged_forms else "FAIL"}')
    assert diverged_forms, '演化后四形未出现分歧'

    print('\n' + '=' * 72)
    print('无火证人数值复现成功: 四形相同 + 火不同的两图, 同规则演化轨迹分岔。')
    print('方向信息 (S_火) 不可由其余四维推出 ⇒ 定理 I 无火不可的实证支撑。')
    print('=' * 72)


if __name__ == '__main__':
    main()
