"""青檬引擎 · 领域适配层 —— 理论可变。

将同一拓扑结构映射到不同语义空间；适配器可插件化注册。
模块:
    registry  DomainAdapter 基类 + DomainRegistry 注册表 + 内置示例适配器
    boundary  边界效应 —— 领域/节点集间耦合强度与帧间衰减
"""

from .registry import DomainAdapter, DomainRegistry, PhysicsAdapter, SociologyAdapter
from .boundary import InteractionBoundary, BoundaryConfig

__all__ = [
    "DomainAdapter",
    "DomainRegistry",
    "PhysicsAdapter",
    "SociologyAdapter",
    "InteractionBoundary",
    "BoundaryConfig",
]
