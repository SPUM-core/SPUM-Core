"""
SPUM 跨域桥接 — 7 领域概念 → ⟨P, ε⟩ 映射
=============================================

同一套公理系统在 7 个领域的 ⟨P, ε⟩ 归约。
所有映射从公理派生，不是类比或隐喻。

领域:
    物理学 — 净湮灭效应 + σ 梯度
    社会学 — 社会网络子图
    经济学 — 交换边集子图
    语言学 — 认知信号协议
    数学 — V⁺/V⁻ 本原操作
    五形 — 五种拓扑相位
    儒释道哲学 — 经典概念拓扑归约

用法:
    from spum.domain import DomainBridge
    bridge = DomainBridge()
    result = bridge.map("physics", "gravity")
    # → {"domain": "physics", "concept": "gravity",
    #    "spum_reduction": "∇σ from net annihilation",
    #    "axioms": ["axiom5", "derived.sigma"]}
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════
# 概念映射记录
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ConceptMapping:
    """单个概念的 SPUM 归约记录。

    Attributes:
        domain: 领域名
        concept: 概念名
        spum_reduction: SPUM 归约描述
        axioms: 涉及的公理列表
        formula: 若有形式化公式，在此
        path: 文件系统中的文档路径
        derived_from: 从哪些 SPUM 核心节点推导
    """
    domain: str
    concept: str
    spum_reduction: str
    axioms: List[str] = field(default_factory=list)
    formula: str = ""
    path: str = ""
    derived_from: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# 领域定义
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Domain:
    """领域定义。"""
    name: str
    description: str
    primitives: List[str]
    concepts: Dict[str, ConceptMapping] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# 跨域桥接器
# ══════════════════════════════════════════════════════════════════════

class DomainBridge:
    """跨域桥接器 — 将 ⟨P, ε⟩ 原语映射到 7 个领域的概念。

    设计原则:
        - 所有映射从公理派生，不是类比
        - 每条映射标注推导路径（涉及的节点、公理、文档）
        - 不预设跨域类比的存在——只在推导路径明确时建立映射

    用法:
        bridge = DomainBridge()

        # 查询映射
        result = bridge.map("physics", "gravity")
        if result:
            print(result.spum_reduction)

        # 列出领域
        print(bridge.list_domains())

        # 列出领域内概念
        print(bridge.list_concepts("wuxing"))

        # SPUM 原语 → 领域概念（反向查询）
        results = bridge.reverse_map("σ")
    """

    def __init__(self):
        self._domains: Dict[str, Domain] = {}
        self._init_domains()

    def _init_domains(self) -> None:
        """初始化所有领域的概念映射。"""
        self._init_physics()
        self._init_sociology()
        self._init_economics()
        self._init_linguistics()
        self._init_math()
        self._init_wuxing()
        self._init_rushidao()
        self._init_graph()

    def _add_mapping(self, domain: str, concept: str, reduction: str,
                     axioms: List[str], formula: str = "",
                     path: str = "",
                     derived_from: Optional[List[str]] = None) -> None:
        """添加一条概念映射。"""
        if domain not in self._domains:
            raise KeyError(f"未知领域: {domain}")
        mapping = ConceptMapping(
            domain=domain,
            concept=concept,
            spum_reduction=reduction,
            axioms=axioms,
            formula=formula,
            path=path,
            derived_from=derived_from or [],
        )
        self._domains[domain].concepts[concept] = mapping

    # ── 物理学 ─────────────────────────────────────────────────

    def _init_physics(self) -> None:
        dom = Domain(
            name="physics",
            description="物理现象: 净湮灭效应 + σ 梯度 + 拓扑响应",
            primitives=["σ", "V⁺/V⁻", "∇σ", "闭合子图", "永恒粒子"],
        )
        self._domains["physics"] = dom

        self._add_mapping("physics", "gravity",
            "引力不是力，是净湮灭效应的宏观表现。物质粒子内部持续的净湮灭事件导致周围σ升高，"
            "形成从内向外递减的梯度场。a = −2kc²m/r³，与牛顿万有引力形式一致(G=2kc²)。",
            axioms=["axiom5", "derived.sigma"],
            formula="σ(r)=σ₀+km/r² → a=−2kc²m/r³",
            path="物理学/基本相互作用/spum-引力.md",
            derived_from=["N020", "N013"],
        )

        self._add_mapping("physics", "light",
            "光=永恒粒子。晶子组合超过12个→完美闭壳被打破→必然产生无法愈合的缺口。"
            "缺口产生内禀方向→关系流引擎驱动永不停歇地改变坐标。光速c=4κ/τ。",
            axioms=["axiom3", "axiom4"],
            formula="c=4κ/τ",
            path="物理学/光学/spum-光.md",
            derived_from=["N018"],
        )

        self._add_mapping("physics", "dark_matter",
            "暗物质=子图协动(SPUM替代方案)。星系外围平坦旋转曲线由空间粒子子图的"
            "整体拓扑运动主导，不依赖不可见物质。SPARC 175星系94.3%有效。",
            axioms=["axiom4", "axiom5"],
            formula="v_co(r)=Ω·r·f(r) (f=边缘泄露衰减)",
            path="docs/Subgraph_CoMotion/SPUM_Subgraph_CoMotion.md",
            derived_from=["N006", "N020"],
        )

        self._add_mapping("physics", "time",
            "没有'时间本身'，只有'网络演化到第几步'。时间不是背景、不是河流、"
            "不是均匀流逝的实体。dt=τ(离散帧)，不是无穷小。",
            axioms=["axiom3"],
            formula="dt=τ, dv/dt≤const",
            derived_from=["N012"],
        )

        self._add_mapping("physics", "space",
            "没有空容器。泡泡(空间粒子)本身就是空间。接触关系定义'这里'与'那里'。"
            "空间是关系凝固的痕迹，而非关系的容器。",
            axioms=["axiom1"],
            derived_from=["N004"],
        )

        self._add_mapping("physics", "matter",
            "物质=缺口互锁的静止态。多个光子式缺口结构碰撞嵌套→缺口矢量彼此对称锁"
            "定→净缺口矢量归零→可以静止。静质量即被锁闭的动能。",
            axioms=["axiom4", "axiom5"],
            path="物理学/基本相互作用/spum-强力.md",
            derived_from=["N018", "N019"],
        )

        self._add_mapping("physics", "alpha_1_137",
            "精细结构常数α≈1/137是拓扑约束+认知投影的涌现结果，不是自由参数。"
            "由κ网络的自相似嵌套层级+π的认知投影因子共同决定。",
            axioms=["axiom4", "derived.sigma"],
            formula="α≈(4π²·σ₀)/(2^N)",
            path="物理学/spum-精细结构常数.md" if False else "SPUM2610.md",
            derived_from=["N017"],
        )

        self._add_mapping("physics", "rotation_curve",
            "子图协动: 闭合子图的全局拓扑运动。仅3参数(Υ,V_co,r_co)，中位数χ²_red=0.68，"
            "94.3%的SPARC星系有效拟合。与NFW比AIC/BIC在~2/3星系更优。",
            axioms=["axiom4"],
            formula="v(r)=√(Υv_bar²+v_co²·r²/(r²+r_co²))",
            path="docs/Subgraph_CoMotion/SPUM_Subgraph_CoMotion.md",
            derived_from=["N006", "N020"],
        )

    # ── 社会学 ─────────────────────────────────────────────────

    def _init_sociology(self) -> None:
        dom = Domain(
            name="sociology",
            description="社会关系网络: 权力、群体、制度、不平等的图论表达",
            primitives=["deg", "介数", "V⁺(社会)", "V⁻(社会)", "σ_social"],
        )
        self._domains["sociology"] = dom

        self._add_mapping("sociology", "power",
            "权力 = α·deg + β·介数 + γ·V⁻能力。度数是直接控制力，介数是信息控制力，"
            "V⁻能力是切断关系的能力。三者独立但相关。",
            axioms=["axiom1", "axiom2"],
            formula="power = α·deg(v) + β·betweenness(v) + γ·V⁻(v)",
            path="社会学/spum-权力与层级.md",
            derived_from=["N004", "N005"],
        )

        self._add_mapping("sociology", "group_formation",
            "群体 = σ 耦合高聚类子图。内群体偏好 = σ 同构认知经济。"
            "邓巴数 = d_max (认知维护的上限连接数)。",
            axioms=["axiom1", "derived.sigma"],
            path="社会学/spum-群体形成.md",
        )

        self._add_mapping("sociology", "inequality",
            "不平等 = σ 分布不均匀。马太效应 = V⁺ 正反馈：已有高度数的节点更容易获得新连接。"
            "社会流动 = 沿 ∇σ 迁移。",
            axioms=["axiom5", "derived.sigma"],
            path="社会学/spum-不平等.md",
        )

        self._add_mapping("sociology", "institution",
            "制度 = 跨帧稳定子图 + 自指边确认。通过三重锁定(认知、规范、规制)维持跨帧稳定性。",
            axioms=["axiom3", "axiom5"],
            path="社会学/spum-制度.md",
        )

    # ── 经济学 ─────────────────────────────────────────────────

    def _init_economics(self) -> None:
        dom = Domain(
            name="economics",
            description="经济交换子图: 市场、货币、价格的 ⟨P, ε⟩ 表达",
            primitives=["V⁺(交易)", "V⁻(交易)", "σ_econ", "∇σ_resource"],
        )
        self._domains["economics"] = dom

        self._add_mapping("economics", "money",
            "货币 = 标准化 V⁺/V⁻ 对偶令牌。价值 = ∇σ_resource 拓扑信号。"
            "价格 = σ 交换比（两种资源的 σ 比值）。",
            axioms=["axiom1", "derived.sigma"],
            path="经济学/spum-货币与价值.md",
        )

        self._add_mapping("economics", "market",
            "市场 = 高连通替代路径子图。竞争 = 替代路径 ≥ 2。"
            "均衡 = ∇σ 趋零（资源分布均匀化）。",
            axioms=["axiom4", "derived.sigma"],
            path="经济学/spum-市场与价格.md",
        )

        self._add_mapping("economics", "crisis",
            "经济危机 = V⁻ 级联。系统性风险 = 全连通无防火墙子图。"
            "σ 重构: 危机后 σ 分布重排。",
            axioms=["axiom5", "axiom2"],
            path="经济学/spum-经济危机.md",
        )

    # ── 语言学 ─────────────────────────────────────────────────

    def _init_linguistics(self) -> None:
        dom = Domain(
            name="linguistics",
            description="认知信号协议: 语言作为 G_ling 子图间的 V⁺/V⁻ 信号",
            primitives=["σ_ling", "G_ling", "V⁺(语言)", "V⁻(语言)"],
        )
        self._domains["linguistics"] = dom

        self._add_mapping("linguistics", "syntax",
            "句法 = 图→序列线性化规则。递归 = 自指 V⁺（深度受 deg 约束）。"
            "普遍语法(UG) = 共享拓扑约束。",
            axioms=["axiom3", "axiom1"],
            path="语言学/spum-句法与结构.md",
        )

        self._add_mapping("linguistics", "meaning",
            "词 = (信号, 概念子图) 映射边。语义 = 概念子图拓扑位置。"
            "语义场 = 节点社区。",
            axioms=["axiom1"],
            path="语言学/spum-词汇与语义.md",
        )

    # ── 数学 ───────────────────────────────────────────────────

    def _init_math(self) -> None:
        dom = Domain(
            name="math",
            description="离散关系本体数学: 全部运算还原为 V⁺/V⁻",
            primitives=["V⁺", "V⁻", "N(自然数)"],
        )
        self._domains["math"] = dom

        self._add_mapping("math", "addition",
            "加法 = V⁺（加边）。a+b = 在分离子图间建立 b 条边。",
            axioms=["axiom1"],
            formula="a+b = a 上的 V⁺ 操作 × b",
            path="数学/spum-数学公理.md",
        )

        self._add_mapping("math", "subtraction",
            "减法 = V⁻（删边）。a-b = 删除 a 中的 b 条边。",
            axioms=["axiom1"],
            formula="a-b = a 上的 V⁻ 操作 × b",
            path="数学/spum-数学公理.md",
        )

        self._add_mapping("math", "pi",
            "π 不是宇宙固有常数。它是人类以六方向直角坐标认知各向同性球体时的"
            "认知压缩因子。宇宙本体只记录整数12和拓扑约束。",
            axioms=["axiom4"],
            derived_from=["N017"],
        )

        self._add_mapping("math", "continuum",
            "连续是离散关系网络在认知投影中的有效近似。离散系统(计算机、AI)真实存在"
            "于宇宙中，且能完美模拟一切连续现象。模拟不是模仿，是还原。",
            axioms=["axiom3"],
            path="数学/spum-数学分界.md",
        )

    # ── 五行/五形 ──────────────────────────────────────────────

    def _init_wuxing(self) -> None:
        dom = Domain(
            name="wuxing",
            description="五形: ⟨P, ε⟩ 演化中必然涌现的五种拓扑相位",
            primitives=["水形", "木形", "土形", "金形", "火形", "S向量"],
        )
        self._domains["wuxing"] = dom

        self._add_mapping("wuxing", "water",
            "水形 = 链式传输结构(低度节点路径)。网络中度≤2的极大无分支路径。"
            "核心指标: 平均最短路径L、全局效率E_glob、边介数中心性。",
            axioms=["axiom1", "axiom3"],
            formula="S_水 = f(L, E_glob, betweenness)",
            path="五行/README.md",
            derived_from=["N004", "N005"],
        )

        self._add_mapping("wuxing", "wood",
            "木形 = 闭合环路骨架。环基数 μ = M - N + C。"
            "拓扑守恒公理强制闭合子图满足Σ(6-deg)=12。木形骨架是物质粒子的刚性支撑。",
            axioms=["axiom4"],
            formula="S_木 = f(μ, C_Δ, λ₂)",
            path="五行/README.md",
            derived_from=["N006", "N014"],
        )

        self._add_mapping("wuxing", "earth",
            "土形 = 分散储备池(低度节点+小规模分量)。悬挂边删除的必然副产品。"
            "不完美定理保证每帧必残留悬挂端。",
            axioms=["axiom2", "axiom5"],
            formula="S_土 = f(deg≤2占比, 碎片化指数F)",
            path="五行/README.md",
            derived_from=["N013"],
        )

        self._add_mapping("wuxing", "metal",
            "金形 = 修剪与更新(桥边/割边删除)。永恒粒子充当'拓扑剪刀'，"
            "以局部火形梯度驱动的概率修剪冗余边。",
            axioms=["axiom3", "axiom5"],
            formula="S_金 = f(桥边数, 节点割集, 修剪潜力)",
            path="五行/README.md",
            derived_from=["N011", "N018"],
        )

        self._add_mapping("wuxing", "fire",
            "火形 = 密度梯度与定向驱动(σ不均匀分布)。"
            "∇σ是体积流定向运动的唯一驱动因——不需要'力'。"
            "高σ(热) → 低σ(冷)的σ差值自动规定流向。",
            axioms=["derived.sigma"],
            formula="S_火 = f(⟨∇k⟩, 拉普拉斯谱, 火势Φ=Var(ΔV_i))",
            path="五行/README.md",
            derived_from=["N004", "N010"],
        )

        self._add_mapping("wuxing", "five_phase_coupling",
            "五形耦合动力学(从公理自发生成的离散反应-扩散系统): "
            "火形∇σ→驱动水形链定向输送土形节点→水形将土形储备输送到木形生长前沿→"
            "木形环路吸收新节点→μ超临界→金形修剪激活→释放节点回归土形→"
            "产生新的σ不均匀→火形梯度再生。",
            axioms=["axiom1", "axiom2", "axiom3", "axiom4", "axiom5",
                    "derived.sigma"],
            formula="S = (S_水, S_木, S_土, S_金, S_火)",
            path="五行/README.md",
        )

    # ── 儒释道哲学 ─────────────────────────────────────────────

    def _init_rushidao(self) -> None:
        dom = Domain(
            name="rushidao",
            description="儒释道经典概念的 SPUM 拓扑归约",
            primitives=["道=ε", "仁", "空", "无"],
        )
        self._domains["rushidao"] = dom

        self._add_mapping("rushidao", "dao",
            "道 = ε（边集）——字面即定义。道生一 = 边→节点。"
            "一生二 = 节点度≥2→产生差异。二生三 = 三种操作(V⁺/V⁻/帧)。"
            "三生万物 = 帧演化→全部现象。",
            axioms=["axiom1"],
            formula="道 = ε, 道生一 = ⟨P, {ε}⟩",
            path="儒释道哲学/spum-道德经.md",
            derived_from=["N001", "N005"],
        )

        self._add_mapping("rushidao", "ren",
            "仁 = 边连通度最优区间。既不是孤立(d=0)，也不是过度连接(d=d_max)。"
            "礼 = S∈H 维护协议(向量保持在健康凸多面体内)。",
            axioms=["axiom1", "axiom2"],
            path="儒释道哲学/spum-论语.md",
        )

        self._add_mapping("rushidao", "emptiness",
            "空 = L1 认知投影 ≠ L0 本体。五蕴(色受想行识)=S向量五分量的认知投影皆空。"
            "色不异空=投影不异于本体认知。",
            axioms=["axiom1", "axiom3"],
            formula="空 = [L1] ≠ [L0]",
            path="儒释道哲学/spum-心经.md",
            derived_from=["N021"],
        )

        self._add_mapping("rushidao", "wuweifa",
            "无为法 = ⟨P, ε⟩ 演化不依赖任何外部推动力。"
            "演化驱动力是内生的几何矛盾: 填补空隙必生更多空隙。",
            axioms=["axiom3", "axiom5"],
            path="儒释道哲学/spum-道德经.md",
        )

    # ── SPUM-图论 ──────────────────────────────────────────────

    def _init_graph(self) -> None:
        dom = Domain(
            name="graph_theory",
            description="SPUM-图论 v2.0: 关系张力场的几何投影理论",
            primitives=["deg", "σ", "δ", "V⁺", "V⁻", "d_topo"],
        )
        self._domains["graph_theory"] = dom

        self._add_mapping("graph_theory", "edge",
            "边 = 排他性确认事件。不是'节点间的连接'——是先有确认事件，"
            "节点由此被定义。连接优先于属性。",
            axioms=["axiom1"],
            path="图论/spum-图论公理.md",
        )

        self._add_mapping("graph_theory", "space",
            "空间 = 张力不可容纳时的拉开现象。球体 = 各向同性张力平衡的等势面。"
            "不是容器，是关系张力场的几何投影。",
            axioms=["axiom1", "axiom4"],
            path="图论/spum-图论公理.md",
        )

        self._add_mapping("graph_theory", "d_topo",
            "d_topo = 帧间距离。跨帧状态的图编辑距离，度量网络在两帧之间的变化量。"
            "d_topo = |V⁺| + |V⁻|。",
            axioms=["axiom3"],
            formula="d_topo = |V⁺| + |V⁻|",
            path="图论/spum-图论指标.md",
        )

    # ═══════════════════════════════════════════════════════════
    # 公有接口
    # ═══════════════════════════════════════════════════════════

    def list_domains(self) -> List[str]:
        """列出所有已注册领域。"""
        return sorted(self._domains.keys())

    def list_concepts(self, domain: str) -> List[str]:
        """列出某领域的所有已映射概念。

        Args:
            domain: 领域名

        Returns:
            概念名列表
        """
        if domain not in self._domains:
            return []
        return sorted(self._domains[domain].concepts.keys())

    def get_domain(self, domain: str) -> Optional[Domain]:
        """获取领域对象。"""
        return self._domains.get(domain)

    def map(self, domain: str, concept: str) -> Optional[ConceptMapping]:
        """正向映射: 领域概念 → SPUM 归约。

        Args:
            domain: 领域名 (physics/sociology/economics/linguistics/math/wuxing/rushidao)
            concept: 概念名

        Returns:
            ConceptMapping 或 None
        """
        if domain not in self._domains:
            return None
        return self._domains[domain].concepts.get(concept)

    def reverse_map(self, spum_primitive: str) -> List[ConceptMapping]:
        """反向映射: SPUM 原语 → 所有使用了它的领域概念。

        Args:
            spum_primitive: SPUM 原语名 (如 "σ", "V⁺", "axiom1", "deg")

        Returns:
            包含该原语的所有概念映射列表
        """
        results = []
        for domain in self._domains.values():
            for mapping in domain.concepts.values():
                if (spum_primitive in mapping.axioms or
                    spum_primitive in mapping.formula or
                    spum_primitive in mapping.spum_reduction or
                    spum_primitive in domain.primitives):
                    results.append(mapping)
        return results

    def search(self, keyword: str) -> List[ConceptMapping]:
        """全文搜索所有领域的概念映射。

        Args:
            keyword: 关键词

        Returns:
            匹配的概念映射列表
        """
        results = []
        kw = keyword.lower()
        for domain in self._domains.values():
            for mapping in domain.concepts.values():
                if (kw in mapping.concept.lower() or
                    kw in mapping.spum_reduction.lower() or
                    kw in mapping.domain.lower()):
                    results.append(mapping)
        return results

    def cross_domain_chain(self, concept1: str, concept2: str) -> Optional[dict]:
        """寻找两个概念间通过 ⟨P, ε⟩ 的联系路径。

        Args:
            concept1: 第一个概念名
            concept2: 第二个概念名

        Returns:
            {"path": [...], "shared_axioms": [...], "bridge": ...} 或 None
        """
        c1_results = self.search(concept1)
        c2_results = self.search(concept2)

        if not c1_results or not c2_results:
            return None

        # 找共同公理
        axioms1 = set()
        for m in c1_results:
            axioms1.update(m.axioms)
        axioms2 = set()
        for m in c2_results:
            axioms2.update(m.axioms)
        shared = axioms1 & axioms2

        return {
            "concept1_search": concept1,
            "concept2_search": concept2,
            "matches_1": [m.concept for m in c1_results],
            "matches_2": [m.concept for m in c2_results],
            "shared_axioms": sorted(shared),
            "bridge_via_spum": f"两者都通过{'、'.join(sorted(shared))}从⟨P,ε⟩派生" if shared
                              else "未找到共享公理桥接路径",
        }

    def summary(self) -> str:
        """所有领域的汇总统计。"""
        total_concepts = sum(len(d.concepts) for d in self._domains.values())
        lines = [
            f"SPUM 跨域桥接 | {len(self._domains)} 领域 | {total_concepts} 概念映射",
        ]
        for name in sorted(self._domains):
            d = self._domains[name]
            lines.append(f"  {name}: {len(d.concepts)} 概念 — {d.description}")
        return "\n".join(lines)
