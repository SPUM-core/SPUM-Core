"""
VSPT 分支生长引擎 — 从原子核实面生长的树状分支结构

SPUM 定义（元素化学/SPUM-VSPT.md §3）：
    VSPT（Vacant-Solid Proliferation Topology）是从原子核表面实面
    沿三角形网格模板向外生长的树状分支结构。

VSPT 几何三律（SPUM-VSPT.md §3.3）：
    1. k≥3（最小连通度）：每个分支节点至少被 3 个邻接关系锚定
    2. 分支角 120°：分支角锁定为正二十面体二面角 ≈ 120°
    3. ρ∝r⁻³（密度幂律）：节点数密度沿径向 r⁻³ 衰减

生长算法：
    - 核表面每块实面播种一棵 VSPT 树
    - 每棵树的节点沿三角形网格模板向外生长
    - 每层生成 spherical shell 上的分支点
    - 分支方向锁定在切平面 120° 方位
    - 壳层节点数受球面面积约束，密度自然衰减
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Phase_4.eternal_particle import (
    EternalParticle,
    Handedness,
)


# ============================================================
# VSPT 节点
# ============================================================

@dataclass
class VSPTNode:
    """VSPT 分支树中的一个节点。

    Attributes:
        uid:         节点全局 ID
        layer:       所在壳层（0=核表面, 1=第一壳层, ...）
        parent_uid:  父节点 UID（核表面节点父为 "nucleus"）
        children:    子节点 UID 列表
        position:    相对核心的球坐标 (x, y, z)
        branch_angle:与父节点的分支角（度）
        degree:      当前度数（= 邻接连接数）
        is_solid:    是否为实面（可继续生长）
    """
    uid: str
    layer: int = 0
    parent_uid: Optional[str] = None
    children: List[str] = field(default_factory=list)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    branch_angle: float = 120.0
    degree: int = 0
    is_solid: bool = True

    @property
    def radius(self) -> float:
        return math.sqrt(sum(p * p for p in self.position))


# ============================================================
# VSPT 树
# ============================================================

@dataclass
class VSPTTree:
    """从单个实面生长出的 VSPT 分支树。"""
    tree_id: str
    nucleus_uid: str
    face_index: int
    handedness: Handedness
    root_uid: Optional[str] = None
    nodes: Dict[str, VSPTNode] = field(default_factory=dict)
    # 壳层节点索引 {layer: [uid, ...]}
    layer_nodes: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def max_layer(self) -> int:
        return max((n.layer for n in self.nodes.values()), default=0)

    @property
    def density_profile(self) -> Dict[int, int]:
        profile = {}
        for n in self.nodes.values():
            profile[n.layer] = profile.get(n.layer, 0) + 1
        return dict(sorted(profile.items()))


# ============================================================
# VSPT 引擎配置
# ============================================================

@dataclass
class VSPTConfig:
    """VSPT 生长引擎配置。"""
    max_layers: int = 8
    max_nodes_per_tree: int = 500
    branch_angle: float = 120.0
    min_degree: int = 3
    # 每节点的子节点数 — 使用三角网格分支
    children_per_node: int = 3
    # 横向连接距离阈值（× Δr），基于三角网格最小间距
    # 同一球壳上相邻节点间的弦距离 ≈ 2·r·sin(Δr/(2r)) ≈ Δr
    lateral_distance_threshold: float = 1.2
    # 壳层间距（相对于核半径）
    layer_spacing: float = 0.8
    # 随机种子
    seed: Optional[int] = None


# ============================================================
# 三角网格几何工具
# ============================================================

def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """单位化向量"""
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n < 1e-15:
        return (1.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a: Tuple[float, float, float],
           b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _tangent_basis(normal: Tuple[float, float, float]) -> Tuple:
    """构建切平面正交基 (t1, t2)。"""
    ref = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0)
    t1 = _cross(normal, ref)
    t1 = _normalize(t1)
    t2 = _cross(normal, t1)
    t2 = _normalize(t2)
    return t1, t2


def _triangular_mesh_directions(
    base_direction: Tuple[float, float, float],
    handedness: Handedness,
) -> List[Tuple[float, float, float]]:
    """在切平面生成 3 个 120° 间隔的分支方向。

    三角网格约束：
        从父节点出发的 3 个子节点位于切平面，间隔 120°。
        手性 L/D 决定旋转方向（逆时针/顺时针）。

    Returns:
        3 个单位方向向量
    """
    normal = _normalize(base_direction)
    t1, t2 = _tangent_basis(normal)

    h = handedness.value
    dirs = []
    for k in range(3):
        angle = k * 2.0 * math.pi / 3.0  # 0°, 120°, 240°
        # 切平面方向
        tangent = (
            math.cos(angle) * t1[0] + h * math.sin(angle) * t2[0],
            math.cos(angle) * t1[1] + h * math.sin(angle) * t2[1],
            math.cos(angle) * t1[2] + h * math.sin(angle) * t2[2],
        )
        # 合成方向 = 法线 + spread × 切向，然后归一化
        # 120° 分支角在切平面投影下精确保持
        spread = 1.0
        d = (
            normal[0] + spread * tangent[0],
            normal[1] + spread * tangent[1],
            normal[2] + spread * tangent[2],
        )
        dirs.append(_normalize(d))

    return dirs


def _calculate_branch_angle(
    parent_pos: Tuple[float, float, float],
    child1_pos: Tuple[float, float, float],
    child2_pos: Tuple[float, float, float],
) -> float:
    """计算两个子节点从父节点视角的夹角（度）。"""
    v1 = tuple(child1_pos[i] - parent_pos[i] for i in range(3))
    v2 = tuple(child2_pos[i] - parent_pos[i] for i in range(3))
    n1 = math.sqrt(sum(v * v for v in v1))
    n2 = math.sqrt(sum(v * v for v in v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


# ============================================================
# VSPT 引擎
# ============================================================

class VSPTEngine:
    """VSPT 生长引擎 — 三角网格约束的球面分支生长。

    生长规则：
        1. 每块实面播种一棵树（根节点在核表面）
        2. 每层生长时，每个节点沿三角网格方向产生 3 个子节点
        3. 子节点位于半径为 r₀ + layer·Δr 的球壳上
        4. 三个子节点在切平面内间隔 120°
        5. 壳层节点数受球面面积约束，超出面积上限时停止分支

    最终输出的密度分布自然遵循 ρ ∝ N(l) / (4πr²·Δr)，
    其中 N(l) ∝ r²，因此 ρ ∝ 常量不是 r⁻³。
    但对于有限总节点数 N_total 分布在半径 R 内，
    整体平均密度 ∝ N_total / R³ ∝ R⁻³。
    """

    def __init__(self, config: Optional[VSPTConfig] = None):
        self.config = config or VSPTConfig()
        self.trees: Dict[str, VSPTTree] = {}
        self._node_counter = 0
        self._rng = random.Random(self.config.seed)

    def _next_uid(self, prefix: str = "v") -> str:
        self._node_counter += 1
        return f"{prefix}{self._node_counter:06d}"

    def _spherical_position(
        self,
        center_dir: Tuple[float, float, float],
        tangent_dir: Tuple[float, float, float],
        radial_dist: float,
        tangent_scale: float,
    ) -> Tuple[float, float, float]:
        """计算球面上一点的位置。

        center_dir:  径向单位向量（从球心指向父节点）
        tangent_dir: 切平面方向
        radial_dist: 球半径（从球心到该层）
        tangent_scale: 切向偏移量
        """
        # 位置 = 径向 + 切向偏移
        pos = (
            center_dir[0] * radial_dist + tangent_dir[0] * tangent_scale,
            center_dir[1] * radial_dist + tangent_dir[1] * tangent_scale,
            center_dir[2] * radial_dist + tangent_dir[2] * tangent_scale,
        )
        return _normalize(pos)

    def seed_tree(
        self,
        nucleus_uid: str,
        face_index: int,
        handedness: Handedness,
        surface_pos: Tuple[float, float, float],
    ) -> VSPTTree:
        """在核表面播下一棵 VSPT 树。

        根节点位于核表面，layer=0。
        """
        tree_id = f"{nucleus_uid}_f{face_index}"
        tree = VSPTTree(
            tree_id=tree_id,
            nucleus_uid=nucleus_uid,
            face_index=face_index,
            handedness=handedness,
        )

        root_uid = self._next_uid("r")
        root = VSPTNode(
            uid=root_uid,
            layer=0,
            parent_uid="nucleus",
            position=surface_pos,
            degree=0,
            is_solid=True,
        )
        tree.nodes[root_uid] = root
        tree.root_uid = root_uid
        tree.layer_nodes[0] = [root_uid]

        self.trees[tree_id] = tree
        return tree

    def grow_tree(
        self,
        tree: VSPTTree,
        nucleus_radius: float,
    ) -> int:
        """生长一棵 VSPT 树一个壳层。

        每层从当前叶子节点向三角网格方向生成子节点。

        Returns:
            新增节点数
        """
        if tree.node_count >= self.config.max_nodes_per_tree:
            return 0

        current_layer = tree.max_layer
        next_layer = current_layer + 1

        if next_layer > self.config.max_layers:
            return 0

        # 当前层所有节点
        current_uids = tree.layer_nodes.get(current_layer, [])

        # 该层的理论球面半径
        layer_radius = nucleus_radius + next_layer * self.config.layer_spacing

        # 该层球面面积
        shell_area = 4.0 * math.pi * (layer_radius ** 2)

        # 球面面积允许的最大节点数（以三角形网格间距为 Δr 估计）
        triangle_area = (math.sqrt(3) / 4.0) * (self.config.layer_spacing ** 2)
        max_nodes_this_layer = max(3, int(shell_area / triangle_area))

        new_nodes = []
        total_children = 0

        for parent_uid in current_uids:
            parent = tree.nodes.get(parent_uid)
            if parent is None or not parent.is_solid:
                continue
            if parent.degree >= self.config.min_degree:
                continue

            # 三角网格：每个节点产生 children_per_node 个子节点
            # 方向在切平面间隔 120°，由三角网格模板约束
            parent_dir = _normalize(parent.position)
            child_dirs = _triangular_mesh_directions(
                parent_dir, tree.handedness
            )

            for di in range(self.config.children_per_node):
                if di >= len(child_dirs) or total_children >= max_nodes_this_layer:
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
                total_children += 1

        # 更新度数（包含：父子连接 + 同层横向连接）
        # 横向连接通过同层节点间的空间邻近关系确定
        for node in tree.nodes.values():
            node.degree = len(node.children)
            if node.parent_uid and node.parent_uid != "nucleus":
                node.degree += 1  # +1 for parent connection

        # 注册新节点到壳层索引
        if new_nodes:
            tree.layer_nodes[next_layer] = [n.uid for n in new_nodes]

        return len(new_nodes)

    def _add_lateral_connections(self, tree: VSPTTree,
                                  nucleus_radius: float) -> int:
        """为 VSPT 树的各层添加横向连接（三角网格邻居）。

        在三角网格球面上，每个节点最多与 6 个同层节点相邻。
        横向连接使高层节点（仅有 1 个子节点）也能满足 k≥3。

        Returns:
            添加的横向连接数
        """
        lateral_count = 0
        for layer, uids in tree.layer_nodes.items():
            if layer == 0:
                continue  # 核表面层不计
            layer_radius = nucleus_radius + layer * self.config.layer_spacing
            # 横向连接距离阈值：基于三角网格最小间距
            # 同一球壳上相邻节点弦距离 ≈ 2·r·sin(Δr/(2r)) ≈ Δr
            distance_threshold = self.config.lateral_distance_threshold * self.config.layer_spacing

            nodes = [tree.nodes[uid] for uid in uids if uid in tree.nodes]
            for i, ni in enumerate(nodes):
                for j in range(i + 1, len(nodes)):
                    nj = nodes[j]
                    dx = ni.position[0] - nj.position[0]
                    dy = ni.position[1] - nj.position[1]
                    dz = ni.position[2] - nj.position[2]
                    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if dist < distance_threshold:
                        # 横向连接：双方度数 +1
                        ni.degree += 1
                        nj.degree += 1
                        lateral_count += 1
        return lateral_count

    def run_growth(
        self,
        outward_solid_faces: int,
        nucleus_uid: str,
        nucleus_particles: Dict[str, EternalParticle],
        nucleus_position_map: Dict[str, Tuple[float, float, float]],
    ) -> Dict:
        """运行完整 VSPT 生长。

        流程：
            1. 遍历所有朝内永恒粒子，每块实面播种一棵树
            2. 逐层生长（0 → max_layers），每层各树同步生长
            3. 壳层节点数受球面面积约束

        Returns:
            {n_trees, total_nodes, layer_distribution, ...}
        """
        self.trees = {}
        self._node_counter = 0

        # 计算核半径
        radii = [math.sqrt(sum(p[i] ** 2 for i in range(3)))
                 for p in nucleus_position_map.values()]
        nucleus_radius = sum(radii) / len(radii) if radii else 1.0

        # === 播种：每块朝内粒子的每块实面 ===
        tree_count = 0
        for ep_uid, ep in nucleus_particles.items():
            if ep.orientation.name == "OUTWARD":
                continue  # 朝外永恒粒子无实面对外暴露
            for fi in range(4):  # 每粒子 4 块实面
                pos = nucleus_position_map.get(ep_uid, (0.0, 0.0, 0.0))
                r = math.sqrt(sum(p * p for p in pos))

                if r > 1e-12:
                    radial = tuple(p / r for p in pos)
                else:
                    radial = (1.0, 0.0, 0.0)

                surface_pos = tuple(
                    radial[i] * (r + ep.radius) for i in range(3)
                )
                self.seed_tree(
                    nucleus_uid=f"{nucleus_uid}_{ep_uid}",
                    face_index=fi,
                    handedness=ep.handedness,
                    surface_pos=surface_pos,
                )
                tree_count += 1

        # === 逐层生长 ===
        total_nodes = tree_count
        layer_dist = {0: tree_count}
        n_trees = len(self.trees)

        for layer in range(1, self.config.max_layers + 1):
            layer_new = 0
            for tree in self.trees.values():
                added = self.grow_tree(tree, nucleus_radius)
                layer_new += added

            if layer_new > 0:
                layer_dist[layer] = layer_new
                total_nodes += layer_new
            else:
                break

        # === 添加横向连接（同层三角网格邻居）===
        total_lateral = 0
        for tree in self.trees.values():
            lat = self._add_lateral_connections(tree, nucleus_radius)
            total_lateral += lat

        return {
            "n_trees": n_trees,
            "total_nodes": total_nodes,
            "layer_distribution": dict(sorted(layer_dist.items())),
            "max_layer": max(layer_dist.keys()) if layer_dist else 0,
            "trees": self.trees,
        }
