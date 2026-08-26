'''
五形完备性定理 §9.1 数值验证脚本
================================

验证 `五行/spum-五形完备性.md` 的三条定理:

  测试 1 (命题 2) 边二分完备:   任意帧图上 环边 ⊎ 桥边 = 全部边 (互斥且穷尽),
                                并以"删边分量数 +1 ⇔ 桥"的经典引理复核分类正确性
  测试 2 (命题 1) 节点三分完备: 全部节点落入 {土(deg≤1), 水/木(deg=2), 木(deg≥3)},
                                无未分类节点
  测试 3 (定理 D) 动态落点覆盖: 五步帧事件的效应落在五形上, 且方向与事件类型一致;
                                五形全部产生效应 (无死相位), 每帧保持握手引理

全部谓词为帧内可判定 (GT-003/GT-005: 帧内信息封闭, 不跨帧累加, 不依赖全局坐标)。
实现复用 `src/spum_graph` 的 FrameGraph / HandshakingVerifier (L0.5 工具层)。

用法:
    python 五行/verify_wuxing_completeness.py            # 默认参数
    python 五行/verify_wuxing_completeness.py --seed 7 --frames 400
'''

import sys
import os
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from spum_graph import FrameGraph, HandshakingVerifier

# ──────────────────────────────────────────────────────────────────────────────
# 一、随机图生成 (三类: ER / 随机正则 / 幂律 BA) —— 覆盖多种 σ 区间
# ──────────────────────────────────────────────────────────────────────────────


def erdos_renyi(n: int, p: float, rng: random.Random) -> dict:
    """Erdős–Rényi 随机图: 每对节点以概率 p 连边。"""
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def random_regular(n: int, d: int, rng: random.Random, max_tries: int = 200) -> dict:
    """随机 d-正则图: 配置模型 (stub 配对), 拒绝自环/重边后重试。"""
    if n * d % 2 != 0:
        raise ValueError(f"n*d 必须为偶数: n={n}, d={d}")
    if d >= n:
        raise ValueError(f"d 必须 < n: n={n}, d={d}")
    for _ in range(max_tries):
        stubs = []
        for i in range(n):
            stubs.extend([i] * d)
        rng.shuffle(stubs)
        adj = {i: set() for i in range(n)}
        ok = True
        for k in range(0, len(stubs), 2):
            a, b = stubs[k], stubs[k + 1]
            if a == b or b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if ok:
            return adj
    raise RuntimeError(f"随机正则图构造失败 (n={n}, d={d})")


def barabasi_albert(n: int, m: int, rng: random.Random) -> dict:
    """Barabási–Albert 优先连接 (幂律度分布)。初始 m 个节点完全图。"""
    adj = {i: set() for i in range(m)}
    for i in range(m):
        for j in range(i + 1, m):
            adj[i].add(j)
            adj[j].add(i)
    for new in range(m, n):
        degs = [(len(adj[v]), v) for v in adj]
        total = sum(d for d, _ in degs)
        adj[new] = set()
        targets = set()
        while len(targets) < m:
            r = rng.random() * total
            acc = 0
            for d, v in degs:
                acc += d
                if r <= acc:
                    targets.add(v)
                    break
        for t in targets:
            if t != new and t not in adj[new]:
                adj[new].add(t)
                adj[t].add(new)
    return adj


# ──────────────────────────────────────────────────────────────────────────────
# 二、帧内局部谓词: 连通分量 / 桥判定 / 边二分 / 节点三分
# ──────────────────────────────────────────────────────────────────────────────


def connected_components(adj: dict) -> int:
    """帧内连通分量数 (单帧 BFS, 不跨帧)。"""
    seen = set()
    count = 0
    for v in adj:
        if v in seen:
            continue
        count += 1
        stack = [v]
        seen.add(v)
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    return count


def edge_key(u: int, v: int) -> tuple:
    """边规范键。"""
    return (u, v) if u < v else (v, u)


def is_cycle_edge(adj: dict, u: int, v: int) -> bool:
    """帧内桥判定引理 (命题 2 的局部化):
    不经边 (u,v), 从 u 出发 BFS 能否到达 v —— 可达则 e 在环上, 否则为桥。
    最坏情况访问整帧连通分量, 但单帧内完成, 不依赖帧外状态 (GT-005)。"""
    seen = {u}
    stack = [u]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if (x == u and y == v) or (x == v and y == u):
                continue  # 绕过待判定的边 e
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return v in seen


def classify_edges(adj: dict) -> dict:
    """全部边二分: {edge_key: 'cycle' | 'bridge'}。返回字典即保证每条边恰好一类。"""
    edge_cls = {}
    for u, nbrs in adj.items():
        for v in nbrs:
            k = edge_key(u, v)
            if k not in edge_cls:
                edge_cls[k] = 'cycle' if is_cycle_edge(adj, u, v) else 'bridge'
    return edge_cls


def node_class(adj: dict, edge_cls: dict, v: int) -> str:
    """节点角色三分 (命题 1):
        deg ≤ 1 → 土; deg = 2 → 两条边皆在环上则 木, 否则 水; deg ≥ 3 → 木。
    火形为密度源 (deg ≥ 3 核心) 与梯度场, 见 five_phase_metrics 的 gradient。"""
    d = len(adj[v])
    if d <= 1:
        return '土'
    if d == 2:
        n1, n2 = list(adj[v])
        c1 = edge_cls[edge_key(v, n1)]
        c2 = edge_cls[edge_key(v, n2)]
        return '木' if (c1 == 'cycle' and c2 == 'cycle') else '水'
    return '木'


# ──────────────────────────────────────────────────────────────────────────────
# 三、五形指标 (帧内可计算)
# ──────────────────────────────────────────────────────────────────────────────


def degree_gradient(adj: dict) -> float:
    """火形: 节点度梯度 ⟨∇k⟩ = (1/N) Σ_i (1/deg(i)) Σ_{j∈N(i)} |k_i − k_j|。"""
    total, cnt = 0.0, 0
    for v, nbrs in adj.items():
        d = len(nbrs)
        if d == 0:
            continue
        total += sum(abs(d - len(adj[x])) for x in nbrs) / d
        cnt += 1
    return total / cnt if cnt else 0.0


def five_phase_metrics(adj: dict) -> dict:
    """计算当前帧的五形指标:
        S_木 = 环基数 μ = M − N + C
        S_金 = 桥边数 (可修剪目标)
        S_土 = 度 ≤ 1 节点数 (悬挂储备)
        S_水 = 水形节点数 (deg=2 非环链段)
        S_火 = 度梯度 ⟨∇k⟩ 与空间密度 σ = N/M
    """
    n = len(adj)
    m = sum(len(nbrs) for nbrs in adj.values()) // 2
    c = connected_components(adj)
    mu = m - n + c
    edge_cls = classify_edges(adj)
    n_metal = sum(1 for cls in edge_cls.values() if cls == 'bridge')
    n_tu = n_shui = n_mu = 0
    for v in adj:
        cls = node_class(adj, edge_cls, v)
        if cls == '土':
            n_tu += 1
        elif cls == '水':
            n_shui += 1
        else:
            n_mu += 1
    return {
        'mu': mu,            # 木
        'n_metal': n_metal,  # 金
        'n_tu': n_tu,        # 土
        'n_shui': n_shui,    # 水
        'grad': degree_gradient(adj),   # 火 (∇k)
        'sigma': n / m if m else 0.0,   # 火 (σ)
        'n': n,
        'm': m,
        'n_mu': n_mu,
    }


def metrics_delta(after: dict, before: dict) -> dict:
    """帧前后五形指标差 (火形 grad 用符号变化判断)。"""
    return {
        'mu': after['mu'] - before['mu'],
        'n_metal': after['n_metal'] - before['n_metal'],
        'n_tu': after['n_tu'] - before['n_tu'],
        'n_shui': after['n_shui'] - before['n_shui'],
        'grad_sign': (1 if after['grad'] > before['grad'] else
                      -1 if after['grad'] < before['grad'] else 0),
        'sigma': after['sigma'] - before['sigma'],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 四、五步帧演化模拟器 (定理 D 的动态落点)
# ──────────────────────────────────────────────────────────────────────────────

# 事件 → 主落点相位 (文档 §4 表)
EVENT_PRIMARY = {
    'water_extend': '水',   # 创生 V⁺ 沿链延伸
    'wood_close':   '木',   # 创生 V⁺ 闭合新环
    'earth_attach': '土',   # 创生 V⁺ 附着新低度节点
    'prune':        '金',   # 删除 V⁻ 修剪悬挂边
}


def vp_water_extend(adj: dict, rng: random.Random):
    """V⁺ 水: 选悬挂节点 v (deg ≤ 1), 附着新节点 u。
    确定性效应: v 由 1 度变 2 度且两条边皆为桥 → v 归入 水 (Δn_shui = +1); u 为悬挂 (土)。"""
    hanging = [v for v in adj if len(adj[v]) <= 1]
    if not hanging:
        return None
    v = rng.choice(hanging)
    u = (max(adj) + 1) if adj else 0
    adj[u] = {v}
    adj[v].add(u)
    return 'water_extend'


def vp_wood_close(adj: dict, rng: random.Random):
    """V⁺ 木: 取同一连通分量内距离 2 的两节点 start, w, 加边闭合新环。
    确定性效应: Δμ = +1 (同分量加边, 路径 start→mid→w 与新增边构成环)。"""
    if len(adj) < 3:
        return None
    start = rng.choice(list(adj))
    # 帧内 BFS 两层: 求距离恰为 2 的节点集合
    seen = {start}
    frontier = {start}
    for _ in range(2):
        nxt = set()
        for x in frontier:
            nxt |= (adj[x] - seen)
        seen |= nxt
        frontier = nxt
    candidates = [w for w in frontier if w != start and w not in adj[start]]
    if not candidates:
        return None
    w = rng.choice(candidates)
    adj[start].add(w)
    adj[w].add(start)
    return 'wood_close'


def vp_earth_attach(adj: dict, rng: random.Random):
    """V⁺ 土: 选 deg ≥ 3 核心节点 v, 附着新节点 u。
    确定性效应: u 度 1 → Δn_tu = +1 (储备扩容)。"""
    cores = [v for v in adj if len(adj[v]) >= 3]
    if not cores:
        return None
    v = rng.choice(cores)
    u = (max(adj) + 1) if adj else 0
    adj[u] = {v}
    adj[v].add(u)
    return 'earth_attach'


def vp_prune(adj: dict, rng: random.Random):
    """V⁻ 金: 选悬挂边 (a,b) (deg(a) = 1, 优先 deg(b) = 2), 删除 a 及其边。
    确定性效应: 被删边必为桥 → Δn_metal = −1; 邻居 b 度降 1, 若 b 原为 2 度则释放为 土。"""
    hanging_edges = []
    for a, nbrs in adj.items():
        if len(nbrs) == 1:
            b = next(iter(nbrs))
            hanging_edges.append((a, b, len(adj[b])))
    if not hanging_edges:
        return None
    # 优先选邻居度为 2 的 (释放效果确定)
    preferred = [e for e in hanging_edges if e[2] == 2]
    pool = preferred if preferred else hanging_edges
    a, b, _ = rng.choice(pool)
    adj[b].discard(a)
    del adj[a]
    return 'prune'


EVENT_FUNCS = {
    'water_extend': vp_water_extend,
    'wood_close': vp_wood_close,
    'earth_attach': vp_earth_attach,
    'prune': vp_prune,
}


# ──────────────────────────────────────────────────────────────────────────────
# 五、测试 3: 动态落点覆盖
# ──────────────────────────────────────────────────────────────────────────────


def test3_dynamic_coverage(seed_adj: dict, rng: random.Random, frames: int) -> dict:
    """运行五步帧演化 (每帧一次 V⁺ 或 V⁻ 事件), 断言:
       (a) 每个事件的效应可归入五形, 且主落点方向与事件类型一致 (100% 确定性断言)
       (b) 火形 (σ) 每事件响应 (∇σ 由度数分布确定, 事件必改变边数)
       (c) 无死相位: 全部五形在演化中产生过非零效应
       (d) 每帧握手引理 Σdeg = 2M (复用 HandshakingVerifier)
       (e) GT-002 残余: 悬挂储备在各帧的残留率"""
    adj = {v: set(nbrs) for v, nbrs in seed_adj.items()}
    verifier = HandshakingVerifier()
    event_types = list(EVENT_FUNCS.keys())

    stats = {ev: {'n': 0, 'violations': []} for ev in event_types}
    phase_hits = {'水': 0, '木': 0, '土': 0, '金': 0, '火': 0}
    total_events = 0
    sigma_changed = 0
    frames_with_hanging = 0
    hs_failures = 0

    for frame_idx in range(frames):
        before = five_phase_metrics(adj)

        # 事件选择: 从当前序号轮转尝试全部事件类型 (预条件不满足则尝试下一种),
        # 保证每帧、每类事件都被公平尝试, 不因预条件漂移而饿死某类事件。
        applied = None
        for offset in range(len(event_types)):
            ev = event_types[(frame_idx + offset) % len(event_types)]
            applied = EVENT_FUNCS[ev](adj, rng)
            if applied is not None:
                break
        if applied is None:
            continue  # 全部预条件不满足 (图饱和), 跳过本帧
        total_events += 1
        ev = applied
        stats[ev]['n'] += 1

        after = five_phase_metrics(adj)
        d = metrics_delta(after, before)
        primary = EVENT_PRIMARY[ev]

        # (a) 主落点方向断言
        if ev == 'water_extend':
            ok = d['n_shui'] >= 1
        elif ev == 'wood_close':
            ok = d['mu'] >= 1
        elif ev == 'earth_attach':
            ok = d['n_tu'] >= 1
        else:  # prune
            ok = d['n_metal'] == -1
        if not ok:
            stats[ev]['violations'].append(d)

        # (b) 火形响应: σ 变化 (例外仅当 N == M 时 (N±1)/(M±1) 恰相等)
        if d['sigma'] != 0:
            sigma_changed += 1

        # (d) 握手引理 (帧内独立验证)
        fg = FrameGraph(frame_id=total_events)
        for u, nbrs in adj.items():
            for w in nbrs:
                if u < w:
                    fg.add_edge(u, w)
        if not verifier.verify_handshaking(fg)['passed']:
            hs_failures += 1

        # (c) 无死相位: 记录五形是否产生过效应
        if d['n_shui'] != 0:
            phase_hits['水'] += 1
        if d['mu'] != 0:
            phase_hits['木'] += 1
        if d['n_tu'] != 0:
            phase_hits['土'] += 1
        if d['n_metal'] != 0:
            phase_hits['金'] += 1
        if d['sigma'] != 0 or d['grad_sign'] != 0:
            phase_hits['火'] += 1

        # (e) GT-002 残余
        if after['n_tu'] >= 1:
            frames_with_hanging += 1

    return {
        'total_events': total_events,
        'stats': stats,
        'phase_hits': phase_hits,
        'sigma_change_rate': sigma_changed / total_events if total_events else 0.0,
        'hanging_rate': frames_with_hanging / frames if frames else 0.0,
        'hs_failures': hs_failures,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 六、测试 1 & 2: 边二分完备 / 节点三分完备
# ──────────────────────────────────────────────────────────────────────────────


def verify_bridge_by_removal(adj: dict, u: int, v: int) -> int:
    """经典引理复核: 删边 (u,v) 后连通分量增量。
    桥 ⇔ 增量 = 1; 环边 ⇔ 增量 = 0。"""
    import copy
    adj2 = copy.deepcopy(adj)
    adj2[u].discard(v)
    adj2[v].discard(u)
    return connected_components(adj2) - connected_components(adj)


def test1_edge_dichotomy(adj: dict) -> dict:
    """命题 2: 环边 ⊎ 桥边 = 全部边 (互斥且穷尽), 并用删边法复核。"""
    edge_cls = classify_edges(adj)
    m = sum(len(nbrs) for nbrs in adj.values()) // 2
    n_cycle = sum(1 for c in edge_cls.values() if c == 'cycle')
    n_bridge = sum(1 for c in edge_cls.values() if c == 'bridge')
    # 穷尽 + 互斥
    partition_ok = (n_cycle + n_bridge == m) and (len(edge_cls) == m)
    # 分类正确性: 桥 ⇔ 删边分量 +1
    mismatches = []
    for (u, v), cls in edge_cls.items():
        inc = verify_bridge_by_removal(adj, u, v)
        expect = 'bridge' if inc == 1 else 'cycle'
        if cls != expect:
            mismatches.append(((u, v), cls, expect, inc))
    return {'m': m, 'n_cycle': n_cycle, 'n_bridge': n_bridge,
            'partition_ok': partition_ok, 'mismatches': mismatches}


def test2_node_trichotomy(adj: dict) -> dict:
    """命题 1: 全部节点落入 {土, 水, 木}, 无未分类节点。"""
    edge_cls = classify_edges(adj)
    classes = defaultdict(int)
    unclassified = []
    for v in adj:
        cls = node_class(adj, edge_cls, v)
        if cls not in ('土', '水', '木'):
            unclassified.append(v)
        classes[cls] += 1
    return {'n': len(adj), 'classes': dict(classes), 'unclassified': unclassified}


# ──────────────────────────────────────────────────────────────────────────────
# 七、主入口
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='五形完备性定理 §9.1 数值验证')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--frames', type=int, default=400, help='测试 3 演化帧数')
    parser.add_argument('--n', type=int, default=120, help='测试 1/2 图规模')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print('=' * 72)
    print('五形完备性定理 §9.1 数值验证 (seed=%d, frames=%d)' % (args.seed, args.frames))
    print('=' * 72)

    # ── 生成图族: 覆盖多种 σ 区间 ──
    graphs = {}
    for p in (0.01, 0.03, 0.05, 0.1, 0.2):
        graphs[f'ER(n={args.n}, p={p})'] = erdos_renyi(args.n, p, rng)
    for d in (2, 3, 4):
        graphs[f'Reg(n=60, d={d})'] = random_regular(60, d, rng)
    for m in (1, 2, 3):
        graphs[f'BA(n={args.n}, m={m})'] = barabasi_albert(args.n, m, rng)

    # ── 测试 1: 边二分完备 ──
    print('\n[测试 1] 命题 2 — 边二分完备: 环边 ⊎ 桥边 = 全部边')
    t1_total_m, t1_cycle, t1_bridge = 0, 0, 0
    t1_ok = True
    for name, adj in graphs.items():
        res = test1_edge_dichotomy(adj)
        t1_total_m += res['m']
        t1_cycle += res['n_cycle']
        t1_bridge += res['n_bridge']
        status = 'PASS' if (res['partition_ok'] and not res['mismatches']) else 'FAIL'
        if status == 'FAIL':
            t1_ok = False
        print(f"  [{status}] {name:22s} M={res['m']:5d} "
              f"环边={res['n_cycle']:5d} 桥边={res['n_bridge']:5d} "
              f"核对偏差={len(res['mismatches'])}")
        if res['mismatches'][:3]:
            for mm in res['mismatches'][:3]:
                print(f"      MISMATCH: {mm}")
    print(f"  → 汇总: 检验边数 {t1_total_m} = 环边 {t1_cycle} + 桥边 {t1_bridge} "
          f"({'一致' if t1_cycle + t1_bridge == t1_total_m else '不一致'})")
    assert t1_ok, '测试 1 失败: 边二分不完备或分类错误'

    # ── 测试 2: 节点三分完备 ──
    print('\n[测试 2] 命题 1 — 节点三分完备: 全部节点 ∈ {土, 水, 木}')
    t2_ok = True
    t2_total, t2_unclassified = 0, 0
    for name, adj in graphs.items():
        res = test2_node_trichotomy(adj)
        t2_total += res['n']
        t2_unclassified += len(res['unclassified'])
        status = 'PASS' if not res['unclassified'] else 'FAIL'
        if status == 'FAIL':
            t2_ok = False
        c = res['classes']
        print(f"  [{status}] {name:22s} N={res['n']:4d} "
              f"土={c.get('土', 0):4d} 水={c.get('水', 0):4d} 木={c.get('木', 0):4d} "
              f"未分类={len(res['unclassified'])}")
    print(f"  → 汇总: 检验节点 {t2_total}, 未分类 {t2_unclassified}")
    assert t2_ok, '测试 2 失败: 存在未分类节点'

    # ── 测试 3: 动态落点覆盖 ──
    print(f'\n[测试 3] 定理 D — 动态落点覆盖 (演化 {args.frames} 帧)')
    # 稀疏种子图 BA(60,1): 树状结构, 天然含悬挂端与链段,
    # 使水(链延伸)/木(环闭合)/土(储备)/金(修剪) 四类事件预条件均可满足
    seed_adj = barabasi_albert(60, 1, rng)
    res3 = test3_dynamic_coverage(seed_adj, rng, args.frames)

    print(f'  → 有效事件: {res3["total_events"]}')
    for ev, s in res3['stats'].items():
        status = 'PASS' if not s['violations'] else f"FAIL ({len(s['violations'])})"
        print(f"  [{status}] {ev:14s} 主落点={EVENT_PRIMARY[ev]}  "
              f"次数={s['n']:4d} 方向违规={len(s['violations'])}")
    hits = res3['phase_hits']
    no_dead = all(hits[p] > 0 for p in ('水', '木', '土', '金', '火'))
    print(f'  → 五形效应命中次数: 水={hits["水"]} 木={hits["木"]} 土={hits["土"]} '
          f'金={hits["金"]} 火={hits["火"]}  (无死相位: {"PASS" if no_dead else "FAIL"})')
    print(f'  → 火形(σ)事件响应率: {res3["sigma_change_rate"]:.1%}')
    print(f'  → GT-002 悬挂储备残留率: {res3["hanging_rate"]:.1%}')
    print(f'  → 握手引理违规: {res3["hs_failures"]}')

    t3_ok = True
    for ev, s in res3['stats'].items():
        if s['violations']:
            t3_ok = False
    assert t3_ok, '测试 3 失败: 事件主落点方向违规'
    assert no_dead, '测试 3 失败: 存在死相位 (某形从未产生效应)'
    assert res3['sigma_change_rate'] >= 0.95, '测试 3 失败: 火形响应率过低'
    assert res3['hs_failures'] == 0, '测试 3 失败: 握手引理被违反'

    # ── 总结 ──
    print('\n' + '=' * 72)
    print('总结: 三条定理全部通过数值验证')
    print('  测试 1 (命题 2) 边二分完备       : 环边 ⊎ 桥边 = 全部边, 删边法复核一致')
    print('  测试 2 (命题 1) 节点三分完备     : 全部节点 ∈ {土, 水, 木}, 无未分类')
    print('  测试 3 (定理 D) 动态落点覆盖     : 事件效应落点与方向一致, 五形无死相位')
    print('=' * 72)


if __name__ == '__main__':
    main()
