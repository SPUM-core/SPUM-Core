#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPUM 架构评分系统 v5.0
=====================
对齐 SPUM 版本号生成与审核规范 v5.0

改进 (v4.0 → v5.0):
  - 词汇库对齐 N001-N022 (替换过时术语: 关节→节点, 连杆→边, VSPT→永恒粒子)
  - 新增 L0.5 SPUM-图论层检测 (FrameGraph/DanglingDetector/HandshakingVerifier 提及)
  - B 维度: 双重闭合判据 (δ + Δμ)
  - C 维度: 溯源至 N001-N022 (原 N001-N020)
  - LE 维度: L0.5 经典图论依赖检测
  - 内容类型扩展: +graph +core
  - 可选集成: src/core/ FrameProtocol / ReviewBridge

用法:
  python score_calculator.py <文件路径> <内容类型>
  python score_calculator.py knowledge.md theory
  python score_calculator.py 08_SPUM_Graph_Theory.md graph
  python score_calculator.py experiments/n016/run.py exp --delta 0.04 --drift 0.008
"""

import json, time, hashlib, os, sys, re
from typing import Dict, List, Set, Optional, Tuple

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# ══════════════════════════════════════════════════════════════════════
# 术语体系 (对齐 spum-vocabulary.md + nodes.txt, 2026-06)
# ══════════════════════════════════════════════════════════════════════

# L0 本体论术语 (来自 nodes.txt N001-N021)
L0_ENTITIES: Set[str] = {
    # 基本构成
    "节点", "空间粒子", "κ-particle", "边", "连接", "度数", "deg",
    "晶子", "永恒粒子", "最小度数",
    # 演化机制
    "创生事件", "湮灭事件", "创生-湮灭对偶", "悬挂边删除",
    "离散帧", "净湮灭效应", "空间密度梯度",
    "dv/dt", "拓扑容量上限",
    # 结构属性
    "拓扑常数12", "分形约束生长", "几何发生学", "认知投影",
    "闭合网络", "闭合子图", "总边数守恒",
    # 认知方法
    "π的降级", "离散系统本体论证明", "认知压缩因子",
    # 定理/原理
    "欧拉恒等式", "高斯-博内", "角度亏损总和恒为4π",
    "不完美", "悬挂端", "双层不完美", "锚点漂移率",
}

# L0.5 SPUM-图论术语 (来自 N022 + src/spum_graph/)
L05_ENTITIES: Set[str] = {
    "FrameGraph", "DanglingDetector", "HandshakingVerifier",
    "SPUM-图论", "帧快照", "公理1-连接优先", "公理2-悬挂端不可消除",
    "公理3-帧快照", "公理4-完美极限", "公理5-局部闭合",
    "悬挂端密度", "δ", "锚点漂移", "Δμ", "双层闭合判据",
    "dangling_density", "compute_dangling", "euler_characteristic",
    "angle_deficit_sum", "握手引理",
}

# L1 认知投影术语
L1_ENTITIES: Set[str] = {
    "球体", "表面积", "体积", "π", "三维坐标", "欧氏几何",
    "正二十面体", "接吻数", "笛卡尔坐标", "微分", "积分",
    "连续", "极限",
}

# L2 传统物理借词 (必须重新定义为 L0 后才能作为推理前提)
L2_BORROWED: Set[str] = {
    "力", "场", "能量", "质量", "电荷", "引力", "光速",
    "光子", "电子", "质子", "中子", "暗物质", "暗能量",
    "大爆炸奇点", "引力子", "绝对时空", "熵", "波函数",
    "自旋", "角动量",
}

# 经典图论借词 (L2，必须与 SPUM-图论隔离)
L2_GRAPH: Set[str] = {
    "连通分量", "最短路径", "聚类系数", "介数中心性",
    "度分布", "随机图", "小世界网络", "无标度网络",
    "谱聚类", "拉普拉斯矩阵",
}

# 连续统/模糊词 (出现即违规)
BANNED_WORDS: Set[str] = {
    "容器", "背景", "舞台", "弯曲", "时空", "均匀流逝",
    "过去", "未来", "奇点", "大爆炸", "动能", "势能",
    "波粒二象性", "概率云", "叠加态", "观测坍缩",
    "独立实体", "内禀属性", "点粒子", "弦", "膜", "额外维度",
}

# 模糊推理词
VAGUE_WORDS: Set[str] = {
    "显然", "自然如此", "自然而然", "偶然", "随机", "碰巧",
    "神奇", "不可思议", "理应如此",
}

# 基底指纹
BASE_FINGERPRINT = "a7f3e2d8"

# 内容类型枚举
CONTENT_TYPES = {"theory", "graph", "exp", "data", "dial", "code", "core"}

# ══════════════════════════════════════════════════════════════════════
# 逻辑图构建器 (v5.0)
# ══════════════════════════════════════════════════════════════════════

class LogicGraph:
    """文本 → 逻辑图解析器。

    解析公理/推导/现象层级关系。
    """
    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        self.G = nx.DiGraph() if HAS_NX else None
        self.axioms: List[str] = []
        self.derivations: List[str] = []
        self.phenomena: List[str] = []
        self.l05_nodes: List[str] = []  # SPUM-图论相关节点
        self.entity: Dict[str, str] = {}  # node_id → "L0"/"L05"/"L1"/"L2"/"L2_graph"
        self.vague_count = 0
        self.banned_count = 0
        self.l2_graph_count = 0
        self._tokens: List[str] = []

    def tokenize(self) -> List[str]:
        """简单分词：提取中文词、英文标识符、数字。"""
        if self._tokens:
            return self._tokens
        tokens = []
        for line in self.lines:
            # 中文字符序列
            for m in re.finditer(r'[\u4e00-\u9fff]+', line):
                tokens.append(m.group())
            # 英文标识符
            for m in re.finditer(r'[A-Za-z_][A-Za-z0-9_]*', line):
                tokens.append(m.group())
            # δ, Δμ 等希腊字母
            for m in re.finditer(r'[α-ωΑ-Ω]', line):
                tokens.append(m.group())
        self._tokens = tokens
        return tokens

    def parse(self):
        """解析文本为逻辑节点。"""
        tokens = self.tokenize()

        for i, line in enumerate(self.lines):
            clean = line.strip()
            if not clean:
                continue

            # 逐行分类
            is_header = clean.startswith("#")
            is_axiom = any(k in clean for k in [
                "公设", "公理", "定义", "基底", "第一性", "根本原因",
                "是……的必然", "强制解",
            ])
            is_deriv = any(k in clean for k in [
                "因此", "所以", "推导", "可得", "由……", "→",
                "符合", "一致", "验证",
            ])
            is_pheno = any(k in clean for k in [
                "现象", "实验", "预言", "观测", "效应", "结果",
                "实验验证", "实测",
            ])

            etype = self._judge_entity(clean, is_header)

            if is_axiom or (is_header and any(e in clean for e in ["公理", "定义"])):
                nid = f"axiom_{i}"
                if HAS_NX:
                    self.G.add_node(nid, type="axiom", text=line)
                self.axioms.append(nid)
                self.entity[nid] = etype
            elif is_pheno:
                nid = f"pheno_{i}"
                if HAS_NX:
                    self.G.add_node(nid, type="phenomenon", text=line)
                self.phenomena.append(nid)
                self.entity[nid] = etype
            elif is_deriv:
                nid = f"deriv_{i}"
                if HAS_NX:
                    self.G.add_node(nid, type="derivation", text=line)
                    if self.axioms:
                        self.G.add_edge(self.axioms[-1], nid)
                self.derivations.append(nid)
                self.entity[nid] = etype

            # 检测 L0.5 图论提及
            if any(t in line for t in L05_ENTITIES):
                if is_deriv or is_axiom:
                    self.l05_nodes.append(f"deriv_{i}" if is_deriv else f"axiom_{i}")

        # 统计违规词 (全局: 仅计算非否定语境的出现)
        negate_kw = ["不依赖", "抛弃", "无需", "批判", "不是", "否定",
                     "禁止", "降级", "不存在", "无背景", "无舞台",
                     "黑名单", "违规", "禁用词", "没有", "不能",
                     "不是……而是", "拒绝", "反对"]
        # 词汇列表行自动豁免: 双引号引用的术语列表行
        def _is_vocabulary_line(l: str) -> bool:
            return bool(re.search(r'\*\*[^*]+类\*\*', l))
        text_flat = "".join(self.lines)
        self.vague_count = sum(1 for w in VAGUE_WORDS if w in text_flat)
        for w in BANNED_WORDS:
            for i, line in enumerate(self.lines):
                if w in line:
                    if _is_vocabulary_line(line):
                        continue  # 术语列表行自动豁免
                    ctx = line
                    if i > 0:
                        ctx = self.lines[i-1] + " " + ctx
                    if i < len(self.lines) - 1:
                        ctx = ctx + " " + self.lines[i+1]
                    if not any(k in ctx for k in negate_kw):
                        self.banned_count += 1
                        break

        # L2 经典图论检测: 检查是否引用了 L2 图论概念但未声明隔离
        for t in L2_GRAPH:
            if t in text_flat:
                # 检查附近是否有 "经典图论" 或 "L2" 隔离声明
                self.l2_graph_count += 1

        return self.G

    def _judge_entity(self, line: str, is_header: bool = False) -> str:
        """判定行内术语的认知层级。"""
        # 批判/否定语境 → 可豁免 L2 污染
        negate = any(w in line for w in [
            "不依赖", "抛弃", "无需", "批判", "不是", "否定",
            "禁止", "降级", "不存在", "无背景", "无舞台",
            "黑名单", "违规", "禁用词", "没有", "不能",
            "拒绝", "反对", "绝无",
        ])
        # 词汇列表行自动豁免
        if re.search(r'\*\*[^*]+类\*\*', line):
            negate = True

        l05 = sum(1 for e in L05_ENTITIES if e in line)
        l0 = sum(1 for e in L0_ENTITIES if e in line)
        l1 = sum(1 for e in L1_ENTITIES if e in line)
        l2 = sum(1 for e in L2_BORROWED if e in line)
        l2g = sum(1 for e in L2_GRAPH if e in line)
        banned = sum(1 for w in BANNED_WORDS if w in line)

        # 致命: 连续统词在非否定语境 → L2 污染
        if banned > 0 and not negate:
            return "L2_banned"

        # L2 先判定 (可被否定豁免)
        if l2 > 0:
            return "L1" if negate else "L2"
        if l2g > 0:
            return "L05" if negate else "L2_graph"

        # 正向层级
        if l05 > 0:
            return "L05"
        if l0 > 0:
            return "L0"
        if l1 > 0:
            return "L1"

        # 标题默认 L0 (公理/定义级)
        if is_header:
            return "L0"

        return "L0"  # 默认本体


# ══════════════════════════════════════════════════════════════════════
# 四维评分器 v5.0
# ══════════════════════════════════════════════════════════════════════

class SPUMScorerV5:
    """v5.0 评分器: L0.5 + 双层闭合判据 + N001-N022。"""

    def __init__(
        self,
        lg: LogicGraph,
        content_type: str = "theory",
        delta: Optional[float] = None,    # 悬挂端密度 (用于双层判据)
        drift: Optional[float] = None,    # 锚点漂移率
        delta_th: float = 0.05,
        drift_th: float = 0.01,
    ):
        self.lg = lg
        self.ctype = content_type
        self.G = lg.G
        self.scores: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.LE = 0
        self.delta = delta
        self.drift = drift
        self.delta_th = delta_th
        self.drift_th = drift_th
        self.warnings: List[str] = []

    # ── A: 架构纯净度 ──────────────────────────────────────────────

    def calc_A(self) -> int:
        """检查公理节点是否仅含 L0/L0.5 原生概念。"""
        ax = [n for n in self.lg.axioms]
        if not ax:
            return 70  # 无明确公理声明，默认中等

        score = 100
        l2_banned = 0
        l2_undeclared = 0
        l05_l2_mix = self.lg.l2_graph_count

        for nid in ax:
            e = self.lg.entity.get(nid, "L0")
            if e == "L2_banned":
                l2_banned += 1
                score = min(score, 30)
            elif e == "L2":
                l2_undeclared += 1
                score -= 15
            elif e == "L2_graph":
                score -= 20  # L0.5 与 L2 图论混淆
            elif e == "L05":
                pass  # L0.5 正常

        # L0.5 层特殊检测
        if self.ctype == "graph" and l05_l2_mix > 3:
            score -= 15
            self.warnings.append(f"L0.5-L2图论混淆: {l05_l2_mix}处")

        # 致命词直接降至 30 以下
        if l2_banned > 0:
            score = min(score, 28)
            self.warnings.append(f"致命违规: {l2_banned} 处连续统词")

        # 模糊词扣分
        score -= self.lg.vague_count * 5

        return max(0, min(100, score))

    # ── B: 推导严密性 ──────────────────────────────────────────────

    def calc_B(self) -> int:
        """检查推导链 + 双层闭合判据。"""
        score = 100

        # 图分析 (需要 networkx)
        if HAS_NX and self.G is not None and len(self.G) > 0:
            # 环检测
            if not nx.is_directed_acyclic_graph(self.G):
                cycles_found = True
                score -= 40
                self.warnings.append("推导图存在循环")
            else:
                cycles_found = False

            # 悬挂节点 (非公理的零入度节点)
            dangling = [
                n for n in self.G.nodes
                if self.G.in_degree(n) == 0
                and self.G.nodes[n].get("type") != "axiom"
            ]
            score -= min(40, len(dangling) * 10)
            if dangling:
                self.warnings.append(f"悬空节点: {len(dangling)}个")
        else:
            # 无 networkx: 文本级检测
            if self.lg.banned_count >= 3:
                score -= 20
                self.warnings.append(f"违规词: {self.lg.banned_count}")

        # 模糊词
        score -= self.lg.vague_count * 5

        # ── 双层闭合判据 (v5.0 新增) ──
        if self.delta is not None and self.drift is not None:
            delta_ok = self.delta < self.delta_th
            drift_ok = self.drift < self.drift_th

            if delta_ok and drift_ok:
                # 完整闭合: B 可达 90+
                pass
            elif delta_ok and not drift_ok:
                # 仅边缘闭合: B ≤ 85
                score = min(score, 85)
                self.warnings.append(
                    f"仅边缘闭合: δ={self.delta:.4f}<{self.delta_th} "
                    f"但 Δμ={self.drift:.4f}≥{self.drift_th}"
                )
            else:
                # 无闭合: B ≤ 70
                score = min(score, 70)
                self.warnings.append(
                    f"未闭合: δ={self.delta:.4f} Δμ={self.drift:.4f}"
                )

        return max(0, min(100, score))

    # ── C: 归约完备度 ──────────────────────────────────────────────

    def calc_C(self) -> int:
        """检查现象→公理的归约路径 + N001-N022 覆盖。"""
        score = 100

        # 现象归约
        phenos = self.lg.phenomena
        axioms = self.lg.axioms

        if phenos and axioms and HAS_NX and self.G is not None:
            connected = 0
            for p in phenos:
                for a in axioms:
                    try:
                        if nx.has_path(self.G, a, p):
                            connected += 1
                            break
                    except (nx.NodeNotFound, nx.NetworkXError):
                        pass
            if phenos:
                coverage = connected / len(phenos)
                if coverage < 0.5:
                    score -= int((1 - coverage) * 60)

        # N001-N022 覆盖检测 (文本级)
        if self.ctype in ("theory", "graph"):
            text = self.lg.text
            n021_found = "N021" in text or "拓扑认知" in text
            n022_found = "N022" in text or "SPUM-图论" in text
            if not n021_found:
                score -= 5
                self.warnings.append("未覆盖 N021 (拓扑认知)")
            if not n022_found:
                score -= 5
                self.warnings.append("未覆盖 N022 (SPUM-图论)")

        # L2 概念未归约
        l2_nodes = [n for n, e in self.lg.entity.items() if e == "L2"]
        score -= min(30, len(l2_nodes) * 5)

        return max(0, min(100, score))

    # ── D: 方法论透明度 ──────────────────────────────────────────────

    def calc_D(self) -> int:
        """检查数学工具声明 + 帧协议透明度。"""
        score = 85

        # 数学工具声明检测
        text = self.lg.text
        has_tool_decl = any(k in text for k in [
            "数学工具", "描述工具", "认知投影", "近似", "模型",
            "不是物理实体", "L1",
        ])
        if not has_tool_decl:
            score -= 10
            self.warnings.append("未声明数学工具属性")

        # 离散-连续过渡声明
        has_transition = any(k in text for k in [
            "离散-连续", "近似", "统计投影", "认知投影",
        ])
        if not has_transition and self.ctype == "theory":
            score -= 10
            self.warnings.append("未声明离散-连续过渡")

        # 帧协议透明度 (v5.0: core 类型特殊加分)
        if self.ctype == "core":
            has_frame = any(k in text for k in [
                "FrameProtocol", "FrameState", "FrameLogger",
                "begin_frame", "end_frame",
            ])
            has_review = any(k in text for k in [
                "ReviewBridge", "ReviewResult", "rollback",
            ])
            if has_frame:
                score += 10
            if has_review:
                score += 5

        # 图论代码特殊检查
        if self.ctype == "graph":
            if "L05" in text or "L0.5" in text:
                score += 5
            if "L2" in text and "隔离" in text:
                score += 5

        return max(0, min(100, score))

    # ── LE: 梯子依赖指数 ───────────────────────────────────────────

    def calc_LE(self) -> int:
        """移除 L2 概念后逻辑图连通性 + L0.5 独立检查。"""
        if not HAS_NX or self.G is None:
            # 无图分析: 文本级检测
            l2_count = sum(1 for e in self.lg.entity.values()
                          if e in ("L2", "L2_banned"))
            l2g_count = sum(1 for e in self.lg.entity.values()
                           if e == "L2_graph")
            if l2_count >= 3 or l2g_count >= 2:
                return 3
            elif l2_count >= 1:
                return 2
            else:
                return 0 if self.lg.vague_count == 0 else 1

        # 移除 L2 概念节点
        l2_nodes = [
            n for n, e in self.lg.entity.items()
            if e in ("L2", "L2_banned", "L2_graph")
        ]
        gc = self.G.copy()
        gc.remove_nodes_from([n for n in l2_nodes if n in gc])

        if len(gc) == 0:
            return 3

        # 检查推导链完整性
        ax = [n for n, d in gc.nodes(data=True)
              if d.get("type") == "axiom"]
        deriv = [n for n, d in gc.nodes(data=True)
                 if d.get("type") == "derivation"]

        if not deriv:
            return 0

        reachable: Set[str] = set()
        for a in ax:
            try:
                reachable.update(nx.dfs_preorder_nodes(gc, a))
            except (nx.NetworkXError, KeyError):
                pass

        if not reachable:
            return 3

        rate = len([d for d in deriv if d in reachable]) / len(deriv)

        # L0.5 独立检查
        l05_ok = True
        if self.ctype == "graph" and self.lg.l2_graph_count > 0:
            l05_ok = False

        if rate >= 0.9 and l05_ok:
            return 0
        elif rate >= 0.7:
            return 1
        elif rate >= 0.4:
            return 2
        else:
            return 3

    # ── 运行 ───────────────────────────────────────────────────────

    def run(self) -> Dict[str, int]:
        self.scores["A"] = self.calc_A()
        self.scores["B"] = self.calc_B()
        self.scores["C"] = self.calc_C()
        self.scores["D"] = self.calc_D()
        self.LE = self.calc_LE()
        return {
            "A": self.scores["A"],
            "B": self.scores["B"],
            "C": self.scores["C"],
            "D": self.scores["D"],
            "LE": self.LE,
        }

    def verdict(self) -> str:
        """v5.0 阈值判决。"""
        s = self.scores
        le = self.LE
        if any(s[d] < 50 for d in "ABCD") or le >= 2:
            return "ROLLBACK"
        if any(s[d] < 70 for d in "ABCD"):
            return "WARN"
        return "PASS"

    def format_report(self) -> str:
        """评审报告 (内部格式)。"""
        s = self.scores
        lines = [
            "=" * 56,
            f"REVIEW REPORT — SPUM v5.0 ({self.ctype})",
            "=" * 56,
            f"A 架构纯净度: {s['A']:3d}/100",
            f"B 推导严密性: {s['B']:3d}/100",
            f"C 归约完备度: {s['C']:3d}/100",
            f"D 方法论透明度: {s['D']:3d}/100",
            f"LE 梯子依赖:  {self.LE}",
        ]
        if self.delta is not None:
            lines.append(f"  δ = {self.delta:.4f} (th={self.delta_th})")
        if self.drift is not None:
            lines.append(f"  Δμ = {self.drift:.4f} (th={self.drift_th})")
        if self.warnings:
            lines.append(f"警告: {len(self.warnings)} 项")
            for w in self.warnings[:5]:
                lines.append(f"  - {w}")
        lines.append(f"判定: {self.verdict()}")
        lines.append("=" * 56)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 版本号生成 v5.0
# ══════════════════════════════════════════════════════════════════════

def make_version(
    scores: Dict[str, int],
    ctype: str,
    content_path: str,
    merkle_prefix: Optional[str] = None,
) -> str:
    """生成 v5.0 格式版本号。

    Args:
        scores: {"A": int, "B": int, "C": int, "D": int, "LE": int}
        ctype: 内容类型 (theory/graph/exp/data/dial/code/core)
        content_path: 内容文件路径
        merkle_prefix: 可选 Merkle 前缀，不提供则从 content_path 生成
    """
    if ctype not in CONTENT_TYPES:
        raise ValueError(f"无效内容类型 '{ctype}'。合法: {CONTENT_TYPES}")

    # 验证指纹
    if merkle_prefix:
        vfp = merkle_prefix[:12].ljust(12, "0")
    else:
        try:
            with open(content_path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()[:12]
        except (FileNotFoundError, OSError):
            h = hashlib.sha256(content_path.encode()).hexdigest()[:12]
        vfp = h

    ts = int(time.time())

    return (
        f"{BASE_FINGERPRINT}."
        f"{ctype}."
        f"{scores['A']:02d}."
        f"{scores['B']:02d}."
        f"{scores['C']:02d}."
        f"{scores['D']:02d}."
        f"{scores['LE']}."
        f"{vfp}."
        f"{ts}"
    )


def make_draft_version(version: str) -> str:
    """在版本号末尾添加 -draft 标记。"""
    return f"{version}-draft"


# ══════════════════════════════════════════════════════════════════════
# CLI 主入口
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> dict:
    """解析命令行参数。"""
    args = {
        "path": None,
        "ctype": "theory",
        "delta": None,
        "drift": None,
        "delta_th": 0.05,
        "drift_th": 0.01,
        "output": None,
        "draft": False,
        "merkle": None,
    }

    positional = []
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a.startswith("--"):
            key = a[2:]
            if key == "delta":
                i += 1
                args["delta"] = float(sys.argv[i])
            elif key == "drift":
                i += 1
                args["drift"] = float(sys.argv[i])
            elif key == "delta-th":
                i += 1
                args["delta_th"] = float(sys.argv[i])
            elif key == "drift-th":
                i += 1
                args["drift_th"] = float(sys.argv[i])
            elif key == "output":
                i += 1
                args["output"] = sys.argv[i]
            elif key == "draft":
                args["draft"] = True
            elif key == "merkle":
                i += 1
                args["merkle"] = sys.argv[i]
            elif key == "help":
                print_help()
                sys.exit(0)
            i += 1
        else:
            positional.append(a)
            i += 1

    if len(positional) >= 1:
        args["path"] = positional[0]
    if len(positional) >= 2:
        args["ctype"] = positional[1]

    return args


def print_help():
    print("""
SPUM 架构评分系统 v5.0
======================

用法:
  python score_calculator.py <文件路径> [内容类型] [选项]

内容类型: theory | graph | exp | data | dial | code | core

选项:
  --delta FLOAT      悬挂端密度 δ (用于双层闭合判据)
  --drift FLOAT      锚点漂移率 Δμ (用于双层闭合判据)
  --delta-th FLOAT   δ 阈值 (默认: 0.05)
  --drift-th FLOAT   Δμ 阈值 (默认: 0.01)
  --merkle STRING    Merkle 根前缀 (代替文件哈希)
  --draft            生成草稿版本号 (-draft 后缀)
  --output PATH      输出 JSON 结果到文件
  --help             显示此帮助

示例:
  python score_calculator.py knowledge.md theory
  python score_calculator.py 08_SPUM_Graph_Theory.md graph
  python score_calculator.py experiments/n016/run.py exp --delta 0.04 --drift 0.008
  python score_calculator.py src/core/frame.py core
""")


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    if args["path"] is None:
        print("错误: 请指定文件路径")
        print("用法: python score_calculator.py <文件路径> [内容类型]")
        sys.exit(1)

    path = args["path"]
    ctype = args["ctype"]

    if not os.path.exists(path):
        print(f"错误: 文件不存在: {path}")
        sys.exit(1)

    if ctype not in CONTENT_TYPES:
        print(f"警告: 未知内容类型 '{ctype}'，使用默认 'theory'")
        ctype = "theory"

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 解析
    lg = LogicGraph(text)
    lg.parse()

    # 评分
    scorer = SPUMScorerV5(
        lg,
        content_type=ctype,
        delta=args["delta"],
        drift=args["drift"],
        delta_th=args["delta_th"],
        drift_th=args["drift_th"],
    )
    scores = scorer.run()

    # 输出报告
    print(scorer.format_report())
    print()

    # 版本号
    version = make_version(
        scores, ctype, path,
        merkle_prefix=args["merkle"],
    )
    if args["draft"]:
        version = make_draft_version(version)
    print(f"版本号: {version}")

    # 输出 JSON
    result = {
        "version": version,
        "content_type": ctype,
        "scores": scores,
        "verdict": scorer.verdict(),
        "warnings": scorer.warnings,
        "timestamp": int(time.time()),
        "v5_features": {
            "L05_detection": bool(lg.l05_nodes),
            "L2_graph_count": lg.l2_graph_count,
            "double_layer": (
                args["delta"] is not None
                and args["drift"] is not None
            ),
        },
    }

    output_path = args["output"] or "score_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")
