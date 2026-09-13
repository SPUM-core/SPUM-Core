"""
OpenSPUM Phase 1 — Cosmic Kernel 确定性核心运行时

包含:
    - constants:             基础常量（κ, τ, 晶子阈值等）
    - topological_address:   拓扑地址系统（差异化驱动）
    - node_registry:         节点注册表（被动响应关系显化）
    - relation_pool:         确定性关系池（无随机候选生成）
    - seed_epoch_engine:     种子创生期引擎（等比序列确定增长）
"""

from .constants import (
    KAPPA,
    TAU,
    CRYSTALLITE_DEGREE_THRESHOLD,
    MAX_CONTACTS,
    CRYSTALLITE_DIAMETER_MAX,
    CRYSTALLITE_CAPACITY,
    GEOMETRIC_TOLERANCE,
    MAX_CRYSTALLITE_CLUSTER_SIZE,
    MAX_CRYSTALLITE_PLANAR_DEGREE,
    MAX_CLUSTER_ITERATIONS,
    DANGLING_QUOTA,
    MIDRANGE_QUOTA,
    NEWNODE_QUOTA,
)
from .topological_address import TopologicalAddress, reset_differentiation_counter
from .node_registry import NodeRegistry, NodeState
from .relation_pool import RelationPool, Relation, RelationState, ContactType
from .seed_epoch_engine import SeedEpochConfig, SeedEpochEngine

__all__ = [
    "KAPPA",
    "TAU",
    "CRYSTALLITE_DEGREE_THRESHOLD",
    "MAX_CONTACTS",
    "CRYSTALLITE_DIAMETER_MAX",
    "CRYSTALLITE_CAPACITY",
    "GEOMETRIC_TOLERANCE",
    "MAX_CRYSTALLITE_CLUSTER_SIZE",
    "MAX_CRYSTALLITE_PLANAR_DEGREE",
    "MAX_CLUSTER_ITERATIONS",
    "DANGLING_QUOTA",
    "MIDRANGE_QUOTA",
    "NEWNODE_QUOTA",
    "TopologicalAddress",
    "reset_differentiation_counter",
    "NodeRegistry",
    "NodeState",
    "RelationPool",
    "Relation",
    "RelationState",
    "ContactType",
    "SeedEpochConfig",
    "SeedEpochEngine",
]
