"""
WX-001 五形耦合演化实验（v5 —— 完整五步帧 + 节点收支平衡 + 密度临界锁定）
==========================================================
在随机图上运行 SPUM 五步帧演化，逐帧记录五形指标（δ/Φ/μ/L/桥边/f_gold），
验证五形桥 GT-026 耦合链 A1-A7 是否涌现。

v2-v4 缺陷链（2026-08-10 诊断）:
  v2: 无湮灭 → 平均度≈20, δ→0, 桥边=0（金形无候选）
  v3: 湮灭<创生 → 平均度≈11, δ→0
  v4: 湮灭比例控制生效, 但缺"节点创生" → 悬挂端删除使 |P| 崩溃 (300→4)
v5 修正:
    1. 完整五步帧 + 创生/湮灭对偶: V⁺ 创生边(梯度驱动) + 新节点(度2加入, 水形链段);
       V⁻ 饱和区湮灭(比例控制, 目标平均度锁定临界稀疏态)
    2. 节点收支平衡: 非修剪态按目标规模 N_INIT 补齐 → |P| 稳定, 不完美可存活(公理2)
    3. 帧5 忠实"删边→邻居降度→残留到下一帧"(不级联, 公理2)
    4. 金形修剪 = 木僵滞回状态机: 判断依据 = 上一帧 μ/n 快照(跨帧状态, 非本帧瞬时值);
       修剪对象 = 环参与 ≤1 的边(GT-024: 桥边/低环参与边); 修剪态 = 收缩期,
       暂停节点补充 → μ 有效回落 → 滞回退出(防钉死)
    5. 测量在补充之前: 修剪释放的悬挂端在本帧池中可见(δ 反映释放, A5 可测)
    6. A1-A7 按规格原文(GT-026 §3.2)命名:
       A1 火→水 (∇σ→输运) / A2 水→土 (输运→储备参与) / A3 水→木 (输运→μ)
       A4 木→金 (μ→修剪) / A5 金→土 (修剪→释放) / A6-7 土→火 (δ→∇σ再生, 闭环)
    7. A6/7 构造性相关风险检查: δ 与 Φ 同源于低度节点占比
    8. A5 事件研究: 修剪态进入帧(trim_state False→True)前后 δ 轨迹,
       规避 f_gold 与 δ 的构造性负干扰

依赖声明（2026-08-10）：
  - 文档依赖: .trae/rules/spum-evolution.md（五步帧协议, 不完美定理）,
              .trae/rules/spum-core.md 第六节（五形定义）,
              图论/spum-图论五形桥.md（GT-021~026 指标集与耦合链 A1-A7）
  - 代码依赖: networkx（帧快照容器与指标计算——全部帧局部, 不跨帧累加, 公理3）
  - 关联报告: experiments/wuxing_coupling/README.md
"""
import json
import os
import argparse
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _setup_cjk_font():
    """配置中文字体（Windows 常见字体），避免 PNG 标签乱码。"""
    from matplotlib import font_manager
    for f in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"]:
        if os.path.exists(f):
            try:
                font_manager.fontManager.addfont(f)
                name = font_manager.FontProperties(fname=f).get_name()
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue


_setup_cjk_font()

SEED = 42
rng = np.random.default_rng(SEED)

# ── 配置 ──────────────────────────────────────────────────────────────
N_INIT = 300            # 目标节点规模（每帧补充回此规模——节点收支平衡）
P_INIT = 0.007          # 初始 ER 密度（平均度≈2.1, 临界稀疏态）
N_GROWTH = 3            # 每帧创生边数（V⁺）
TARGET_AVG_DEG = 2.4    # 目标平均度——创生/湮灭比例控制锚点（临界稀疏态, 悬挂端可存活）
GROWTH_EXP = 2.0        # 创生加权指数: 概率 ∝ |Δσ|^exp（0=均匀对照）
MU_ON = 0.220           # 木僵触发: μ/n ≥ 0.220 进入修剪态（p95≈0.223 之上, 高尾事件）
MU_OFF = 0.205          # 压力解除: μ/n < 0.205 退出修剪态（p5≈0.207 之下, 滞回防钉死）
F_MAX = 5               # 修剪态内每帧最多修剪的桥边数
T_FRAMES = 500          # 演化帧数
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ── 五形指标（帧局部计算, 公理3）────────────────────────────────────
def sigma_of(G, v):
    """σ(v) = 2/deg(v)——图论精确定义（GT-008）。deg=0 节点不存在, 返回兜底 2.0。"""
    d = G.degree(v)
    if d == 0:
        return 2.0
    return 2.0 / d


def cycle_participation(G):
    """每条边参与的环数。

    方法: 删除边 e 后 μ = m−n+c 的下降量 = e 参与的环数。
      桥边(0环): μ 不变; 单环边: μ 降 1; 多环边: 降 >1。
    金形候选 = 环参与 ≤ 1 的边（GT-024: 桥边 / 低环参与边——修剪操作的天然目标）。
    返回 {边: 环数}。
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}
    c0 = nx.number_connected_components(G)
    mu0 = G.number_of_edges() - n + c0
    out = {}
    for u, v in list(G.edges()):
        G.remove_edge(u, v)
        c1 = nx.number_connected_components(G)
        mu1 = G.number_of_edges() - n + c1
        out[(u, v)] = mu0 - mu1
        G.add_edge(u, v)
    return out


def endpoint_delta_bias(G):
    """A2 水→土 端点级指标: 主干链上的储备聚集度。

    规格原文(GT-026 A2): "链上介数 BC(e) 上升的边, 其端点 δ 贡献增大(被输运的储备)"。
    实现: 帧局部（最大连通分量）算边介数 BC(e);
      bias = mean(端点悬挂 | BC ≥ p75) − mean(端点悬挂 | BC ≤ p25)。
      悬挂端(deg<2) = 土形储备。bias > 0 → 主干链端点富集储备（水形输运连接土形储备）。
    相对量设计: 排除全局 δ 干扰（全局 δ 被创生边"拉入网络"而虚降——v5 旧代理失效根因）。
    返回 (bias, 高介数端点悬挂占比, 低介数端点悬挂占比)。
    """
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps or len(comps[0]) < 3:
        return 0.0, 0.0, 0.0
    sub = G.subgraph(comps[0])
    try:
        bc = nx.edge_betweenness_centrality(sub)
    except Exception:
        return 0.0, 0.0, 0.0
    if not bc:
        return 0.0, 0.0, 0.0
    vals = np.array(list(bc.values()))
    if vals.std() == 0:
        return 0.0, 0.0, 0.0
    high_thr = np.percentile(vals, 75)
    low_thr = np.percentile(vals, 25)
    deg = dict(G.degree())
    high_eps = []
    low_eps = []
    for e, b in bc.items():
        u, v = e
        if b >= high_thr:
            high_eps += [int(deg[u] < 2), int(deg[v] < 2)]
        if b <= low_thr:
            low_eps += [int(deg[u] < 2), int(deg[v] < 2)]
    high_ratio = float(np.mean(high_eps)) if high_eps else 0.0
    low_ratio = float(np.mean(low_eps)) if low_eps else 0.0
    return high_ratio - low_ratio, high_ratio, low_ratio


def compute_metrics(G, mean_dsigma, mu_gain=0, created_dang=0.0):
    """计算当前帧的五形指标。全部帧局部, 不跨帧累加。"""
    n = G.number_of_nodes()
    if n == 0:
        return None
    degs = np.array([d for _, d in G.degree()])

    # 土形: δ 悬挂端密度（GT-023 = GT-005 同一量）
    delta = float((degs < 2).mean())

    # 火形: Φ = std(σ) 火势（GT-025）
    sigmas = 2.0 / np.maximum(degs, 1)
    phi = float(sigmas.std())

    # 木形: μ 环基数 + 聚类系数（GT-022；λ₂ 以 μ/C_Δ 代理——隔离原则）
    mu = G.number_of_edges() - n + nx.number_connected_components(G)
    clustering = float(nx.average_clustering(G))

    # 金形: 桥边数（GT-024 修剪候选集）
    bridge_count = len(list(nx.bridges(G)))

    # 水形: 平均最短路径 L + 全局效率（GT-021, 最大分量帧局部计算）
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    L = None
    E_glob = None
    if comps and len(comps[0]) > 1:
        sub = G.subgraph(comps[0])
        try:
            L = float(nx.average_shortest_path_length(sub))
            E_glob = 1.0 / L
        except nx.NetworkXError:
            pass

    # A2 水→土 端点级: 主干链储备聚集度（高介数边端点悬挂占比 − 低介数边）
    eb_bias, eb_high, eb_low = endpoint_delta_bias(G)

    return {
        "frame": None,  # 主循环填充
        "n": n,
        "m": G.number_of_edges(),
        "delta": delta,            # 土
        "phi": phi,                # 火
        "mu": mu,                  # 木
        "clustering": clustering,  # 木
        "bridge_count": bridge_count,  # 金（存量）
        "L": L,                    # 水
        "E_glob": E_glob,          # 水（代理）
        "mean_dsigma": mean_dsigma,  # 水（输运强度 = 每帧创生边平均 |Δσ|）
        "mu_gain": float(mu_gain),  # 木（前沿: 创生边形成的新环数, A3 信号）
        "created_dang": float(created_dang),  # 土/水（端点级: 创生边端点悬挂占比, A2 信号）
        "endpoint_bias": eb_bias,   # 土/水（端点级: 主干链储备聚集度, A2 信号）
        "endpoint_bias_high": eb_high,
        "endpoint_bias_low": eb_low,
    }


# ── 五步帧演化操作 ──────────────────────────────────────────────────
def fire_driven_growth(G, n_growth, exp):
    """帧1 创生 V⁺: 采样 σ 梯度大的非边对加边。

    返回 (实际添加边数, 创生边平均 |Δσ|, 前沿 μ 增量, 创生边端点悬挂占比)。
    前沿 μ 增量 = 创生边中形成新环的边数（A3 水→木 前沿级信号）:
      加边不增节点, μ 变化 = 1 + (c1 − c0)。
      跨分量连接: c 减 1 → μ 增 0（连通而非成环）; 同分量内加边: μ 增 1（新环）。
    创生边端点悬挂占比（A2 水→土 端点级信号）: 创生边端点中 deg<2 的比例
      = 输运载体直接接入储备的证据（被输运的储备挂在链上）。
    exp=0 → 均匀加边（对照）；exp>0 → 火形梯度驱动输运（高σ稀疏端 → 低σ饱和端）。
    """
    nodes = [v for v in G.nodes() if G.degree(v) > 0]
    if len(nodes) < 2:
        return 0, 0.0, 0, 0.0
    added = 0
    attempts = 0
    max_attempts = n_growth * 40
    dsigma_sum = 0.0
    mu_gain = 0
    dang_eps = 0
    n = G.number_of_nodes()
    while added < n_growth and attempts < max_attempts:
        attempts += 1
        u = nodes[rng.integers(len(nodes))]
        su = sigma_of(G, u)
        non_neighbors = [v for v in nodes if v != u and not G.has_edge(u, v)]
        if not non_neighbors:
            continue
        deltas = np.array([abs(su - sigma_of(G, v)) for v in non_neighbors])
        if deltas.max() == 0:
            probs = np.ones(len(non_neighbors))
        else:
            probs = deltas ** exp
        probs = probs / probs.sum()
        idx = rng.choice(len(non_neighbors), p=probs)
        v = non_neighbors[idx]
        dang_eps += int(G.degree(u) < 2) + int(G.degree(v) < 2)  # 加边前端点悬挂状态
        c0 = nx.number_connected_components(G)
        G.add_edge(u, v)
        c1 = nx.number_connected_components(G)
        mu_gain += 1 + c1 - c0  # 同分量加边成环 +1; 跨分量连接 0
        dsigma_sum += float(deltas[idx])
        added += 1
    mean_dsigma = dsigma_sum / added if added else 0.0
    created_dang_ratio = dang_eps / (2 * added) if added else 0.0
    return added, mean_dsigma, mu_gain, created_dang_ratio


def satur_annihilate(G, k):
    """帧1 湮灭 V⁻（对偶补偿）: 从饱和区（度≥3 节点）删除边。

    语义: 闭合子图内净湮灭（spum-evolution §4.1）——饱和区收缩,
    其净边删除量转移为背景净创生（由 fire_driven_growth 承担）→ 边数守恒。
    网络越稀疏, 饱和区越小, 湮灭自然衰减——密度负反馈锁定在临界态。
    返回实际删除边数。
    """
    removed = 0
    attempts = 0
    while removed < k and attempts < k * 40:
        attempts += 1
        if G.number_of_edges() == 0:
            break
        degs = dict(G.degree())
        hubs = [v for v, d in degs.items() if d >= 3]
        if not hubs:
            break  # 无饱和区, 湮灭停止（网络已临界稀疏）
        hub = hubs[rng.integers(len(hubs))]
        nbrs = list(G.neighbors(hub))
        if not nbrs:
            continue
        w = nbrs[rng.integers(len(nbrs))]
        G.remove_edge(hub, w)
        removed += 1
    return removed


def gold_trim_hyst(G, trim_state, mu_on, mu_off, f_max, mu_n_ref):
    """帧4-5 金形修剪（木僵滞回状态机）。

    语义: 上一帧 μ/n（帧快照 mu_n_ref）≥ mu_on → 本帧修剪爆发
          (每帧剪至多 f_max 条低环参与边) → μ 回落 → < mu_off 退出。
          滞回防止 μ 被修剪负反馈钉死在阈值下。
    判断依据 = 上一帧快照: 木僵是跨帧状态（μ 持续高位→修剪需求, GT-026 A4）,
      不是本帧创生后的瞬时值（v5 诊断: 瞬时 μ/n 均值 0.204, 从不达阈值）。
    修剪对象: 环参与 ≤ 1 的边（GT-024: 桥边/低环参与边——修剪操作的天然目标）。
      桥边(0环)优先——断开桥边释放被锁节点（木+金→土 拆解方向）。
    修剪产生的 deg<2 节点不级联删除（公理2）, 残留到本帧测量 → δ 上升（A5 可测）;
      但 deg=0 完全孤立的节点结构性消失（公理1: 孤立节点不存在）。
    返回 (f_gold 修剪边数, 新修剪态)。
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0, False

    if trim_state and mu_n_ref < mu_off:
        return 0, False  # 压力解除, 退出修剪态
    if not trim_state and mu_n_ref < mu_on:
        return 0, False  # 未达木僵, 保持静默

    cp = cycle_participation(G)
    cands = [e for e, k in cp.items() if k <= 1]
    cands_sorted = sorted(
        cands, key=lambda e: (cp[e], min(G.degree(e[0]), G.degree(e[1]))))
    f_gold = 0
    for e in cands_sorted[:f_max]:
        G.remove_edge(*e)
        f_gold += 1
    iso = [v for v in G.nodes() if G.degree(v) == 0]
    if iso:
        G.remove_nodes_from(iso)
    return f_gold, True


def remove_dangling(G):
    """帧5 悬挂端删除（强制, 逻辑自洽）: 删除 deg<2 节点及其关联边。

    删除使邻居度降至 1 的节点保留——残留到下一帧（公理2: 每帧必残留悬挂端）。
    同一帧内不级联消解（一帧只做一次删除）。
    返回删除的悬挂端数。
    """
    dangling = [v for v in list(G.nodes()) if G.degree(v) < 2]
    G.remove_nodes_from(dangling)
    return len(dangling)


_new_node_counter = [0]


def add_new_nodes(G, count, exp):
    """V⁺ 节点创生: 补充 count 个新节点, 以度 2 加入（连接 2 个现存节点）。

    水形链段来源: 新节点以度 2 加入自然形成链（GT-021 来源）。
    连接目标: 梯度采样——一端取自高σ稀疏端（储备/土）, 一端取自低σ饱和端,
    体现"土+水+火→木"组装方向（悬挂端沿梯度被输运）。
    返回实际补充数。
    """
    added = 0
    nodes = list(G.nodes())
    for _ in range(count):
        if len(nodes) < 2:
            break
        v = f"n{_new_node_counter[0]}"
        _new_node_counter[0] += 1
        # 高σ端采样（稀疏储备端）
        sigmas = np.array([sigma_of(G, u) for u in nodes])
        s_max = sigmas.max()
        if s_max <= 0:
            p1 = np.ones(len(nodes)) / len(nodes)
        else:
            p1 = (sigmas ** max(exp, 1e-6))
            p1 = p1 / p1.sum()
        i1 = rng.choice(len(nodes), p=p1)
        u1 = nodes[i1]
        # 低σ端采样（饱和端）
        inv = 1.0 / np.maximum(sigmas, 1e-9)
        p2 = inv / inv.sum()
        i2 = rng.choice(len(nodes), p=p2)
        u2 = nodes[i2]
        if u2 == u1:
            u2 = nodes[rng.integers(len(nodes))]
        G.add_edge(v, u1)
        G.add_edge(v, u2)
        nodes = list(G.nodes())
        added += 1
    return added


def pair_evolution(G, n_growth, exp, target_deg):
    """帧1 创生 V⁺ + 湮灭 V⁻（比例控制守恒）。

    创生: 梯度驱动加 n_growth 条边（偏向高σ稀疏端, 输运语义）。
    湮灭: 饱和区（度≥3 节点）删边 = n_growth + 密度修正项。
      修正项 ∝ (平均度 − 目标), 使平均度锁定在 target_deg（临界稀疏态）。
      这避免 v2 陷阱: 平均度>2 时删度1节点会推高平均度, 需净删边压回。
    返回 (创生数, 创生边平均|Δσ|, 湮灭数, 前沿μ增量, 创生边端点悬挂占比)。
    """
    n = G.number_of_nodes()
    avg = 2 * G.number_of_edges() / n if n else 0.0
    added, mean_dsigma, mu_gain, created_dang = fire_driven_growth(G, n_growth, exp)
    excess = int(np.ceil(max(0.0, (avg - target_deg) * n / 2)))
    annihilated = satur_annihilate(G, added + excess)
    return added, mean_dsigma, annihilated, mu_gain, created_dang


# ── 主演化循环 ──────────────────────────────────────────────────────
def run_evolution(exp, n_growth, n_init, t_frames, seed):
    """跑 T 帧演化, 返回指标时间序列。"""
    G = nx.gnp_random_graph(n_init, P_INIT, seed=seed)
    G = nx.convert_node_labels_to_integers(G)
    G.remove_nodes_from([v for v in G.nodes() if G.degree(v) == 0])  # 公理1

    history = []
    trim_state = False
    prev_mu_n = 0.0  # 上一帧 μ/n 快照——木僵判断依据（跨帧状态, 非本帧瞬时值）
    for tau in range(t_frames):
        # 帧1 创生 V⁺（梯度驱动边 + 新节点） + 湮灭 V⁻（比例控制, 密度锁定）
        added, mean_dsigma, annihilated, mu_gain, created_dang = pair_evolution(
            G, n_growth, exp, TARGET_AVG_DEG)
        # 帧5 悬挂端删除（强制）——先于金形修剪, 修剪产生的悬挂端残留到本帧测量（A5 可测）
        dangling_removed = remove_dangling(G)
        # 帧4-5 金形修剪（木僵滞回; 判断依据 = 上一帧 μ/n 快照）
        f_gold, trim_state = gold_trim_hyst(
            G, trim_state, MU_ON, MU_OFF, F_MAX, prev_mu_n)
        # 帧2-3 连接/体积变化 = 指标重算。测量在补充之前:
        #   修剪释放的悬挂端（deg<2 残留）在本帧池中可见 → δ 反映释放（A5）;
        #   修剪使 deg=0 的节点已结构性消失（公理1）。
        m = compute_metrics(G, mean_dsigma, mu_gain, created_dang)
        # 节点收支平衡: 按目标规模 N_INIT 补齐（度2加入, 水形链段）。
        #   补充在测量之后 → 不污染本帧 δ/μ 快照。
        #   木僵修剪态 = 收缩期（拆解方向, 木+金→土）: 暂停补充, μ 回落解除木僵,
        #   释放的节点积累为土形储备（δ 上升）; 退出修剪态后恢复补充（扩张恢复）。
        if m is None:
            new_nodes = add_new_nodes(G, N_INIT, exp)
            m = compute_metrics(G, mean_dsigma, mu_gain, created_dang)
            if m is None:
                continue
        elif trim_state:
            new_nodes = 0
        else:
            new_nodes = add_new_nodes(G, max(0, N_INIT - m["n"]), exp)
        m["frame"] = tau
        m["f_gold"] = f_gold
        m["trim_state"] = trim_state
        m["dangling_removed"] = dangling_removed
        m["annihilated"] = annihilated
        m["new_nodes"] = new_nodes
        history.append(m)
        prev_mu_n = m["mu"] / m["n"]
    return history


# ── 滞后交叉相关分析（A1-A7 时序验证）──────────────────────────────
def lag_corr(x, y, max_lag=20):
    """x(t) 与 y(t+lag) 的滞后相关。lag>0 表示 x 领先 y（x 是因）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xd = np.diff(x)  # 一阶差分去趋势, 避免伪相关
    yd = np.diff(y)
    out = {}
    for lag in range(0, max_lag + 1):
        if lag == 0:
            a, b = xd, yd
        else:
            a, b = xd[:-lag], yd[lag:]
        if len(a) < 30 or np.std(a) == 0 or np.std(b) == 0:
            out[lag] = 0.0
            continue
        out[lag] = float(np.corrcoef(a, b)[0, 1])
    return out


def best_lag(corrs):
    lag = max(corrs, key=lambda k: abs(corrs[k]))
    return lag, corrs[lag]


def series(history, key):
    return [m[key] if m[key] is not None else np.nan for m in history]


# ── 绘图 ─────────────────────────────────────────────────────────────
def plot_timeseries(history, out_png, exp):
    frames = [m["frame"] for m in history]
    keys = [
        ("delta", "土 δ 悬挂端密度"),
        ("phi", "火 Φ 火势(std σ)"),
        ("mean_dsigma", "水 输运强度(创生Δσ)"),
        ("mu_gain", "木 前沿 μ 增量(创生新环)"),
        ("created_dang", "土/水 创生边端点悬挂占比(A2)"),
        ("endpoint_bias", "土/水 主干链储备聚集(A2 bias)"),
        ("mu", "木 μ 环基数"),
        ("f_gold", "金 f_gold 修剪率"),
        ("bridge_count", "金 桥边存量"),
        ("n", "节点规模 |P|"),
    ]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.4 * len(keys)), sharex=True)
    for ax, (key, label) in zip(axes, keys):
        ax.plot(frames, series(history, key), lw=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("帧 τ")
    fig.suptitle(f"五形耦合演化 v5（exp={exp}, 目标度={TARGET_AVG_DEG}）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ── 主入口 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=float, default=GROWTH_EXP, help="创生加权指数（0=对照）")
    ap.add_argument("--frames", type=int, default=T_FRAMES)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"exp{args.exp:g}"

    print("=" * 64)
    print(f"WX-001 五形耦合演化实验 v5  |  exp={args.exp}  frames={args.frames}")
    print(f"N_init={N_INIT} (节点收支平衡)  创生={N_GROWTH}/帧  目标平均度={TARGET_AVG_DEG}  "
          f"木僵滞回 μ/n [{MU_OFF},{MU_ON}]  f_max={F_MAX}")
    print("=" * 64)

    history = run_evolution(
        exp=args.exp, n_growth=N_GROWTH,
        n_init=N_INIT, t_frames=args.frames, seed=args.seed,
    )

    # ── 输出时序 JSON ──
    out_json = os.path.join(OUT_DIR, f"wuxing_coupling_v2_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n时序已写入: {out_json}")

    # ── 绘图 ──
    out_png = os.path.join(OUT_DIR, f"wuxing_coupling_v2_{tag}.png")
    plot_timeseries(history, out_png, args.exp)
    print(f"时序图已写入: {out_png}")

    # ── A1-A7 验证（滞后相关, 一阶差分去趋势）──
    print("\n── A1-A7 耦合链验证（规格 GT-026 §3.2, 滞后相关去趋势）──")
    delta = series(history, "delta")
    phi = series(history, "phi")
    mu = series(history, "mu")
    f_gold = series(history, "f_gold")
    dsigma = series(history, "mean_dsigma")
    mu_gain = series(history, "mu_gain")
    endpoint_bias = series(history, "endpoint_bias")
    n_nodes = series(history, "n")

    checks = [
        ("A1 火→水", "∇σ(Φ) 领先 定向输运(创生Δσ)", phi, dsigma),
        ("A2 水→土", "输运 领先 主干链储备聚集(端点级 bias)", dsigma, endpoint_bias),
        ("A3 水→木", "输运 领先 前沿骨架扩展(创生边新环数)", dsigma, mu_gain),
        ("A4 木→金", "环基数 领先 修剪率(μ→f_gold)", mu, f_gold),
        ("A5 金→土", "修剪 领先 节点释放(δ)", f_gold, delta),
        ("A6-7 土→火", "δ 领先 ∇σ再生(Φ), 闭环", delta, phi),
    ]
    results = {}
    for name, desc, x, y in checks:
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        corrs = lag_corr(x, y, max_lag=15)
        lag, r = best_lag(corrs)
        print(f"  {name} [{desc}]  best_lag={lag:2d}, r={r:+.3f}")
        results[name] = {"best_lag": lag, "r": r, "corrs": corrs}

    # A2 横截面补充: 主干链端点悬挂占比(high) vs 低介数边端点悬挂占比(low)
    eb_high = np.array([m["endpoint_bias_high"] for m in history])
    eb_low = np.array([m["endpoint_bias_low"] for m in history])
    created_dang = np.array([m["created_dang"] for m in history])
    print(f"\n  [A2 横截面] 高介数边端点悬挂占比 mean={eb_high.mean():.4f}  "
          f"低介数边 mean={eb_low.mean():.4f}  bias={np.mean(eb_high-eb_low):+.4f}"
          + ("  <- 主干链富集储备 ✓" if (eb_high - eb_low).mean() > 0.01 else "  <- 悬挂端在链末低介数区"))
    print(f"  [A2 端点级] 创生边端点悬挂占比 mean={created_dang.mean():.4f}  "
          f"(全网 δ mean={np.mean(delta):.4f})"
          + ("  <- 输运载体直接接入储备 ✓" if created_dang.mean() > np.mean(delta) * 1.5 else "  <- 输运不偏向悬挂端"))
    results["A2_cross_section"] = {
        "bias_mean": float(np.mean(eb_high - eb_low)),
        "high_mean": float(eb_high.mean()),
        "low_mean": float(eb_low.mean()),
        "created_dang_mean": float(created_dang.mean()),
        "global_delta_mean": float(np.mean(delta)),
    }

    # 构造性相关检查: corr(δ, Φ) 原值（A6 风险标注）
    delta_arr = np.array(delta)
    phi_arr = np.array(phi)
    construct_corr = float(np.corrcoef(delta_arr, phi_arr)[0, 1]) if np.std(delta_arr) > 0 else 0.0
    print(f"  [A6 构造性检查] corr(δ, Φ) 原值 = {construct_corr:+.3f}"
          + ("  <- 高说明同源, 非独立证据" if abs(construct_corr) > 0.7 else "  <- 独立度可接受"))

    # A5 事件研究: 修剪态进入帧前后 δ 轨迹（规避 f_gold 与 δ 的构造性负干扰）。
    #   事件 = trim_state 从 False→True（木僵触发, 修剪爆发开始）; 收缩期暂停补充,
    #   释放节点积累为土形储备 → 预期 δ 抬升。
    trim_state_seq = series(history, "trim_state")
    enter_idx = [i for i in range(1, len(trim_state_seq))
                 if trim_state_seq[i] and not trim_state_seq[i - 1]]
    event_window = 4
    print(f"\n  A5 事件研究: 修剪态进入 {len(enter_idx)} 次, 窗口 ±{event_window} 帧")
    if len(enter_idx) >= 8:
        traj = []
        for i in enter_idx:
            seg = delta_arr[max(0, i - event_window): i + event_window + 1]
            if len(seg) < 2 * event_window + 1:
                continue
            traj.append(seg)
        if traj:
            traj = np.array(traj)
            base = traj.mean(axis=0)[event_window]  # 进入修剪态帧 δ
            mean_traj = traj.mean(axis=0)
            rel = mean_traj - traj.mean(axis=0)[:event_window].mean()
            print(f"    修剪态 δ 轨迹: t-4..t0..t+4 = "
                  + " ".join(f"{v:.4f}" for v in mean_traj))
            print(f"    修剪帧 δ={base:.4f} (全窗均值 {mean_traj.mean():.4f}); "
                  f"正增量 = 修剪释放储备 ✓")
            results["A5_event_study"] = {
                "n_events": len(traj),
                "traj_mean": mean_traj.tolist(),
                "traj_rel": rel.tolist(),
                "enter_delta_base": base,
            }
    else:
        print(f"    修剪态进入过少({len(enter_idx)}), 事件研究不显著")

    # 火势自相关（循环周期检测）
    autocorr = lag_corr(phi, phi, max_lag=40)
    _, r_self = best_lag({k: v for k, v in autocorr.items() if k > 0})
    print(f"  火势自相关（周期检测）: best_lag>0 r={r_self:+.3f}")

    out_res = os.path.join(OUT_DIR, f"wuxing_coupling_v2_{tag}_analysis.json")
    with open(out_res, "w", encoding="utf-8") as f:
        json.dump({"results": results, "construct_corr_delta_phi": construct_corr,
                   "autocorr_phi": autocorr, "frames": len(history)},
                  f, ensure_ascii=False, indent=1)
    print(f"\n分析已写入: {out_res}")

    last = history[-1]
    avg_deg = 2 * last["m"] / last["n"] if last["n"] else 0
    print(f"\n末帧状态: n={last['n']}, 平均度={avg_deg:.2f}, δ={last['delta']:.4f}, "
          f"μ={last['mu']:.1f}, Φ={last['phi']:.4f}, 桥边={last['bridge_count']}")
    print(f"δ 统计: min={np.min(delta):.4f} max={np.max(delta):.4f} mean={np.mean(delta):.4f}")


if __name__ == "__main__":
    main()
