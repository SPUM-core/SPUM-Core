"""
SPUM API — AI 之家
====================

将全部公理和推导规则编码为 AI 可直接调用的 API。
只需 `from spum import SPUM` 即可调用全部能力。

快速开始:
    from spum import SPUM
    spum = SPUM()

    # 图操作
    spum.add_edge("A", "B")
    print(f"delta={spum.delta:.3f}, sigma={spum.sigma:.3f}")

    # 帧演化
    spum.new_frame()
    spum.v_plus("X", "Y")
    spum.commit_frame()

    # 跨域映射
    result = spum.map_concept("physics", "gravity")
    print(result["spum_reduction"])

    # 推理链
    chain = spum.derive("N005", "N013")
    print(chain["path"])

    # 评审
    review = spum.review("...")
    print(review["report"])

    # 整体状态
    print(spum.summary())

公理系统:
    Axiom 1: 关系第一性 — 连接定义存在
    Axiom 2: 悬挂端不可消除（有限帧内）
    Axiom 3: 离散帧快照 — 图状态是帧序列的快照
    Axiom 4: 拓扑守恒 — Sigma(6-deg)=12, chi=V-E+F
    Axiom 5: 不完美 — 双层判据 (delta<theta) and (Dmu<theta)

跨域桥接 (7 领域):
    physics     — 物理现象: 净湮灭效应 + sigma 梯度
    sociology   — 社会关系: 权力、群体、制度
    economics   — 经济交换: 市场、货币、危机
    linguistics — 认知信号: 句法、语义
    math        — 离散数学: V+/V- 本原操作
    wuxing      — 五形拓扑: 五种拓扑相位
    rushidao    — 儒释道哲学: 归约

知识图谱:
    22 核心节点 (N001-N022), 33+ 推导边
"""

from .api import SPUM
from .axioms import Axioms, Constants
from .frame import FrameEngine, FrameState, FrameType
from .domain import DomainBridge, ConceptMapping
from .reason import (
    ReasoningEngine, DerivationChain, DerivationStep,
    MultiPathResult, ReviewResult, ReviewVerdict, ReviewViolation,
    CORE_NODES, CORE_EDGES,
)

__all__ = [
    # 主入口
    "SPUM",
    # 公理
    "Axioms", "Constants",
    # 帧引擎
    "FrameEngine", "FrameState", "FrameType",
    # 跨域桥接
    "DomainBridge", "ConceptMapping",
    # 推理
    "ReasoningEngine", "DerivationChain", "DerivationStep",
    "MultiPathResult", "ReviewResult", "ReviewVerdict", "ReviewViolation",
    # 知识图谱
    "CORE_NODES", "CORE_EDGES",
]

__version__ = "4.0"
__description__ = "SPUM API — 将全部公理和推导规则编码为 AI 可直接调用的 API"
