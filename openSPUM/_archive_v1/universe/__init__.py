"""openSPUM.universe — 统一宇宙模拟层。"""

from .simulator import UniverseSimulator, UniverseConfig, UniverseReport
from .emergence import (
    observe_universe, observe_topology, observe_fire,
    observe_opening_ratio, detect_12_crystallite_ring,
)

__all__ = [
    "UniverseSimulator", "UniverseConfig", "UniverseReport",
    "observe_universe", "observe_topology", "observe_fire",
    "observe_opening_ratio", "detect_12_crystallite_ring",
]
