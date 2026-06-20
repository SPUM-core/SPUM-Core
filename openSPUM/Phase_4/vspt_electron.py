"""
VSPT 电子耦合引擎 — 从纯几何分支结构计算电子密度（v2 — 无拟合参数）

物理模型（v2 重写 — 诚实性修正）：
    v1 使用自抑制公式 children(l) = 3/(1+γ·N(l)·l/N_ref) 试图拟合 1s 轨道形状，
    其中 γ=0.08, N_ref=600 是手动调参，无 ⟨P, ε⟩ 依据。
    
    v2 改为纯几何途径：
    - VSPT 节点以 3 分支/节点纯几何生长（正二十面体三角网格约束）
    - 不施加任何电子密度反馈调制
    - 电子径向概率分布 P(l) 作为后验量计算：P(l) = N(l) / total_nodes
    - 结合能用拓扑量度表达（非 eV），不做量子力学数据拟合

    当前局限（诚实声明）：
    - 纯几何生长下 N(l) ∝ 3^l 单调增长，不出现 1s 轨道的径向峰
    - 量子力学径向概率 |R(r)|² ∝ r²e^{-2r/a₀} 需要附加拓扑约束才能涌现
    - 这些约束（电子-核自旋耦合、占据数排斥、Pauli 拓扑》）是 Phase 5+ 的工作
    - 当前版本仅计算纯几何 VSPT + 后验拓扑量，不做 eV 能量换算

    对比 v1：
        γ=0.08    → 已移除（手动调参）
        N_ref=600 → 已移除（手动选择）
        13.6 eV   → 已移除（来自玻尔模型，非 SPUM）
        avg_deg/4 → 已移除（经验修正因子）
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Phase_4.eternal_particle import EternalParticle, Handedness
from Phase_4.vspt_growth import VSPTNode, VSPTTree, VSPTConfig, VSPTEngine
from Phase_4.vspt_growth import _normalize, _triangular_mesh_directions

from Phase_1.constants import KAPPA, TAU


# ============================================================
# 电子态配置 (v2 — 精简)
# ============================================================

@dataclass
class ElectronConfig:
    """电子-VSPT 配置（v2 — 无拟合参数）。

    Attributes:
        electron_count: 电子数 (= 原子序数 Z), 当前仅用于结构计数
    """
    electron_count: int = 1


# ============================================================
# 电子-VSPT 引擎 (v2 — 纯几何 + 后验拓扑量)
# ============================================================

class ElectronVSPTEngine:
    """VSPT 电子引擎 — 纯几何生长 + 后验拓扑量计算。

    与 v1 的关键区别：
        - 生长阶段无电子密度反馈（无 γ, 无 N_ref）
        - children_per_node = 3 始终（三角网格约束）
        - 电子密度 = 归一化层节点数 P(l) = N(l) / total_nodes
        - 结合能 = 拓扑量纲（非 eV），报告原始拓扑量
    """

    def __init__(self, config: Optional[VSPTConfig] = None,
                 electron_cfg: Optional[ElectronConfig] = None):
        self.config = config or VSPTConfig()
        self.electron_cfg = electron_cfg or ElectronConfig()
        self._cpu_engine = VSPTEngine(config=config)
        self._node_counter = 0
        self._rng = random.Random(self.config.seed)
        # 后验拓扑量
        self.electron_density: Dict[int, float] = {}
        self.electron_peak_layer: int = 0
        self.topology_binding_index: float = 0.0

    def _next_uid(self, prefix: str = "v") -> str:
        self._node_counter += 1
        return f"{prefix}{self._node_counter:06d}"

    def seed_tree(
        self,
        nucleus_uid: str,
        face_index: int,
        handedness: Handedness,
        surface_pos: Tuple[float, float, float],
    ) -> VSPTTree:
        """同 VSPTEngine.seed_tree。"""
        return self._cpu_engine.seed_tree(
            nucleus_uid, face_index, handedness, surface_pos
        )

    def run_growth(
        self,
        outward_solid_faces: int,
        nucleus_uid: str,
        nucleus_particles: Dict[str, EternalParticle],
        nucleus_position_map: Dict[str, Tuple[float, float, float]],
    ) -> Dict:
        """运行纯几何 VSPT 生长 + 后验拓扑量计算。

        生长规则（全部从 ⟨P, ε⟩ 几何约束推导）：
            - 每个实面种 1 棵 VSPT 树
            - 每节点产生 3 个子节点（正二十面体三角网格约束）
            - 子节点方向 = 120° 分支角（三角网格顶点法向）
            - 无电子密度反馈，无自抑制

        后验拓扑量：
            - 层分布 layer_distribution: {l: N(l)}
            - 电子密度: P(l) = N(l) / total_nodes
            - 拓扑结合指数: Σ(N(l) × (avg_deg(l) - 3)) / total_nodes
              反映 VSPT 网络的整体连接紧密程度

        Returns:
            {n_trees, total_nodes, layer_distribution, max_layer,
             electron_density, electron_peak_layer,
             topology_binding_index, avg_degree, trees}
        """
        # 核半径
        radii = [math.sqrt(sum(p[i] ** 2 for i in range(3)))
                 for p in nucleus_position_map.values()]
        nucleus_radius = sum(radii) / len(radii) if radii else 1.0

        # === 播种 ===
        self._cpu_engine.trees = {}
        self._cpu_engine._node_counter = 0

        for ep_uid, ep in nucleus_particles.items():
            if ep.orientation.name == "OUTWARD":
                continue
            for fi in range(4):
                pos = nucleus_position_map.get(ep_uid, (0.0, 0.0, 0.0))
                r = math.sqrt(sum(p * p for p in pos))
                radial = tuple(p / r for p in pos) if r > 1e-12 else (1.0, 0.0, 0.0)
                surface_pos = tuple(radial[i] * (r + ep.radius) for i in range(3))
                self._cpu_engine.seed_tree(
                    nucleus_uid=f"{nucleus_uid}_{ep_uid}",
                    face_index=fi,
                    handedness=ep.handedness,
                    surface_pos=surface_pos,
                )

        trees = self._cpu_engine.trees
        tree_count = len(trees)

        # === 纯几何逐层生长 ===
        layer_dist = {0: tree_count}
        children_per = float(self.config.children_per_node)  # = 3.0

        for layer in range(1, self.config.max_layers + 1):
            layer_new = 0
            layer_radius = nucleus_radius + layer * self.config.layer_spacing

            for tree in trees.values():
                if tree.node_count >= self.config.max_nodes_per_tree:
                    continue
                current_layer = tree.max_layer
                next_layer = current_layer + 1
                if next_layer > self.config.max_layers:
                    continue

                current_uids = tree.layer_nodes.get(current_layer, [])
                new_nodes = []
                for parent_uid in current_uids:
                    parent = tree.nodes.get(parent_uid)
                    if parent is None or not parent.is_solid:
                        continue

                    parent_dir = _normalize(parent.position)
                    child_dirs = _triangular_mesh_directions(
                        parent_dir, tree.handedness
                    )

                    for di in range(self.config.children_per_node):
                        if di >= len(child_dirs):
                            break

                        d = child_dirs[di]
                        child_uid = self._next_uid()
                        child_pos = (
                            d[0] * layer_radius,
                            d[1] * layer_radius,
                            d[2] * layer_radius,
                        )

                        child = VSPTNode(
                            uid=child_uid,
                            layer=next_layer,
                            parent_uid=parent.uid,
                            position=child_pos,
                            branch_angle=120.0,
                            degree=0,
                            is_solid=True,
                        )
                        tree.nodes[child_uid] = child
                        parent.children.append(child_uid)
                        new_nodes.append(child)

                # 更新度数
                for node in tree.nodes.values():
                    node.degree = len(node.children)
                    if node.parent_uid and node.parent_uid != "nucleus":
                        node.degree += 1

                if new_nodes:
                    tree.layer_nodes[next_layer] = [n.uid for n in new_nodes]
                    layer_new += len(new_nodes)

            if layer_new > 0:
                layer_dist[layer] = layer_dist.get(layer, 0) + layer_new
            else:
                break

        # === 横向连接 ===
        for tree in trees.values():
            for layer, uids in tree.layer_nodes.items():
                if layer == 0:
                    continue
                thresh = self.config.lateral_distance_threshold * self.config.layer_spacing
                nodes = [tree.nodes[uid] for uid in uids if uid in tree.nodes]
                for i, ni in enumerate(nodes):
                    for j in range(i + 1, len(nodes)):
                        nj = nodes[j]
                        dx = ni.position[0] - nj.position[0]
                        dy = ni.position[1] - nj.position[1]
                        dz = ni.position[2] - nj.position[2]
                        dist = math.sqrt(dx**2 + dy**2 + dz**2)
                        if dist < thresh:
                            ni.degree += 1
                            nj.degree += 1

        # === 后验拓扑量计算 ===
        total_nodes = sum(layer_dist.values())

        # 电子密度 = 归一化层节点数
        electron_prob = {}
        max_p = 0.0
        peak_l = 0
        for layer in sorted(layer_dist.keys()):
            p = layer_dist[layer] / max(total_nodes, 1)
            electron_prob[layer] = p
            if p > max_p:
                max_p = p
                peak_l = layer

        self.electron_density = electron_prob
        self.electron_peak_layer = peak_l

        # 拓扑结合指数 (无量纲)
        # = Σ(N(l) × (avg_deg(l) - 3)) / total_nodes
        # avg_deg(l) - 3 = 横向连接带来的额外束缚
        # 纯树状结构 avg_deg ≈ 3, 结合指数 = 0
        # 密堆结构 avg_deg → 6, 结合指数 → 3
        all_avg_deg = 0.0
        total_connection = 0.0
        for layer in sorted(layer_dist.keys()):
            layer_nodes_list = []
            for tree in trees.values():
                uids = tree.layer_nodes.get(layer, [])
                layer_nodes_list.extend(
                    tree.nodes[uid] for uid in uids if uid in tree.nodes
                )
            if layer_nodes_list:
                l_avg = sum(n.degree for n in layer_nodes_list) / len(layer_nodes_list)
                total_connection += layer_dist[layer] * (l_avg - 3.0)
            all_avg_deg += layer_dist[layer]

        if total_nodes > 0:
            self.topology_binding_index = total_connection / total_nodes

        # 全局平均度数
        all_nodes = []
        for tree in trees.values():
            all_nodes.extend(tree.nodes.values())
        avg_degree = sum(n.degree for n in all_nodes) / max(len(all_nodes), 1) if all_nodes else 0

        return {
            "n_trees": len(trees),
            "total_nodes": total_nodes,
            "layer_distribution": dict(sorted(layer_dist.items())),
            "max_layer": max(layer_dist.keys()) if layer_dist else 0,
            "trees": trees,
            "electron_density": electron_prob,
            "electron_peak_layer": peak_l,
            "topology_binding_index": self.topology_binding_index,
            "avg_degree": avg_degree,
        }
