"""
青檬引擎 · 领域适配层 — 理论可变的语义映射
============================================

同一张 ⟨P, ε⟩ 拓扑结构可映射到不同语义空间（物理/社会/经济/语言…）。
本层以"适配器注册表"模式实现：领域适配器 = 把拓扑指标翻译为领域术语。

分层保障：
  - core 层不可变（公理底线）
  - domain 层可插件化——开发者挂载自己的解释器，不污染内核

用法:
    from qingmeng.domain import DomainRegistry
    reg = DomainRegistry()
    reg.register("physics", PhysicsAdapter())
    adapter = reg.get("physics")
    adapter.interpret(structural_metrics)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class DomainAdapter(ABC):
    """领域适配器基类——把拓扑信号翻译为领域语义。"""

    name: str = "base"

    @abstractmethod
    def interpret(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> Dict:
        """把结构指标翻译为领域解释。

        Args:
            metrics: 拓扑结构指标（来自 graph/topology 或 emergence）
            context: 领域上下文（如目标概念、历史状态）

        Returns:
            领域语义解释（dict，含可审计的映射依据）
        """
        ...


class DomainRegistry:
    """适配器注册表——领域名称 → 适配器实例。"""

    def __init__(self) -> None:
        self._adapters: Dict[str, DomainAdapter] = {}

    def register(self, adapter: DomainAdapter) -> None:
        """注册一个领域适配器（同名覆盖）。"""
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> DomainAdapter:
        """获取适配器；未注册时抛 KeyError。"""
        if name not in self._adapters:
            raise KeyError(f"未注册的领域适配器: {name}")
        return self._adapters[name]

    def names(self) -> list:
        return sorted(self._adapters)

    def interpret(self, name: str, metrics: Dict[str, Any],
                  context: Dict[str, Any] | None = None) -> Dict:
        """便捷入口：取适配器并解释。"""
        return self.get(name).interpret(metrics, context or {})


# ── 内置示例适配器 ────────────────────────────────────────────────

class PhysicsAdapter(DomainAdapter):
    """物理领域：把 σ 梯度映射为温度/密度语义。"""

    name = "physics"

    def interpret(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> Dict:
        sigma = metrics.get("sigma", 0.0)
        # 高 σ（稀疏）→ 高温；低 σ（饱和）→ 低温（spum-core.md §4）
        zone = "高温(稀疏)" if sigma > 1.0 else ("中温" if sigma > 0.5 else "低温(饱和)")
        return {
            "domain": "physics",
            "reduction": f"σ={sigma:.3f} → {zone}（拓扑密度 → 温度语义）",
            "mapping": {"sigma": sigma, "temperature_zone": zone},
        }


class SociologyAdapter(DomainAdapter):
    """社会领域：把聚类系数/桥边映射为群体/权力语义。"""

    name = "sociology"

    def interpret(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> Dict:
        clustering = metrics.get("clustering", 0.0)
        bridges = metrics.get("bridges", 0)
        # 高聚类 → 内群体偏好；桥边 → 群体间连接
        cohesion = "强凝聚(高聚类)" if clustering > 0.5 else "松散(低聚类)"
        return {
            "domain": "sociology",
            "reduction": f"聚类={clustering:.3f} → {cohesion}，桥边={bridges}（结构连接语义）",
            "mapping": {"clustering": clustering, "bridges": bridges, "cohesion": cohesion},
        }
