"""
VSPT GPU 并行生长引擎 — PyTorch CUDA 加速版

架构：
    使用 PyTorch 张量操作在 GPU 上批量计算 VSPT 分支生长。
    所有子节点方向计算通过批量张量运算完成，无需手动 CUDA kernel。

数据流（每层）：
    CPU → GPU: 父节点位置 (N, 3)  →  批量计算切平面基 + 3×120° 方向
    GPU:       child_pos = normalize(normal + spread × tangent) × layer_radius
    GPU → CPU: 子节点位置 → 转为 VSPTNode 对象

性能：
    40棵树 × 6层 × 每层~100节点 × 3子节点 = ~72,000 子节点/帧
    GPU 批量操作：几毫秒即可完成
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from Phase_4.eternal_particle import EternalParticle, Handedness
from Phase_4.vspt_growth import VSPTNode, VSPTTree, VSPTConfig


# ============================================================
# GPU 设备检测
# ============================================================

_HAVE_CUDA = torch.cuda.is_available()
_DEVICE = torch.device("cuda:0") if _HAVE_CUDA else torch.device("cpu")


def _batch_tangent_basis(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """批量计算切平面正交基 (t1, t2) 每行。

    Args:
        normals: (N, 3) 归一化法线向量

    Returns:
        t1, t2: (N, 3) 切平面正交基
    """
    ref = torch.tensor([1.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype)
    # 对法线 x 分量接近 1 的，换参考向量
    mask = normals[:, 0].abs() > 0.9
    ref_expanded = ref.expand_as(normals).clone()
    if mask.any():
        alt_ref = torch.tensor([0.0, 1.0, 0.0], device=normals.device, dtype=normals.dtype)
        ref_expanded[mask] = alt_ref

    # t1 = cross(normal, ref)
    t1 = torch.cross(normals, ref_expanded, dim=1)
    t1_norm = torch.norm(t1, dim=1, keepdim=True).clamp(min=1e-15)
    t1 = t1 / t1_norm

    # t2 = cross(normal, t1)
    t2 = torch.cross(normals, t1, dim=1)
    t2_norm = torch.norm(t2, dim=1, keepdim=True).clamp(min=1e-15)
    t2 = t2 / t2_norm

    return t1, t2


def _batch_grow_children(
    parent_positions: torch.Tensor,
    handedness: torch.Tensor,
    layer_radius: float,
    spread: float = 1.0,
) -> torch.Tensor:
    """GPU 批量生成子节点位置。

    Args:
        parent_positions: (N, 3) 父节点位置（未归一化）
        handedness: (N,) 手性 (+1 或 -1)
        layer_radius: 子节点壳层半径
        spread: 切向散布系数

    Returns:
        children: (N, 3, 3) 子节点位置 [n_parent, k_child, xyz]
    """
    normals = torch.nn.functional.normalize(parent_positions, dim=1)
    t1, t2 = _batch_tangent_basis(normals)

    # 3 个 120° 角度
    angles = torch.tensor(
        [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0],
        device=parent_positions.device, dtype=parent_positions.dtype
    )  # (3,)

    # cosθ, sinθ → (1, 3), (1, 3)
    cos_a = torch.cos(angles).unsqueeze(0)  # (1, 3)
    sin_a = torch.sin(angles).unsqueeze(0)  # (1, 3)

    # h × sinθ → (N, 3)
    h_sin = handedness.unsqueeze(1) * sin_a  # (N, 3)

    # 切平面方向: cosθ·t1 + h·sinθ·t2  →  (N, 3, 3)
    t1_exp = t1.unsqueeze(1)  # (N, 1, 3)
    t2_exp = t2.unsqueeze(1)  # (N, 1, 3)
    cos_exp = cos_a.unsqueeze(2)  # (1, 3, 1)
    h_sin_exp = h_sin.unsqueeze(2)  # (N, 3, 1)

    tangent_dir = cos_exp * t1_exp + h_sin_exp * t2_exp  # (N, 3, 3)

    # 合成方向 = normalize(normal + spread × tangent)
    normal_exp = normals.unsqueeze(1).expand(-1, 3, -1)  # (N, 3, 3)
    raw_dir = normal_exp + spread * tangent_dir  # (N, 3, 3)
    child_dir = torch.nn.functional.normalize(raw_dir, dim=2)  # (N, 3, 3)

    # 子节点位置 = child_dir × layer_radius
    children = child_dir * layer_radius  # (N, 3, 3)

    return children


def _batch_lateral_connections(
    positions: torch.Tensor,
    layers: torch.Tensor,
    distance_threshold: float,
    degree_inc: torch.Tensor,
) -> torch.Tensor:
    """GPU 批量计算横向连接。

    对同层节点计算 pairwise 距离，距离 < 阈值则 degree += 1。

    Args:
        positions: (N, 3) 节点位置
        layers: (N,) 层号
        distance_threshold: 距离阈值
        degree_inc: (N,) 度数增量（初始化为 0，原地更新）

    Returns:
        更新后的 degree_inc
    """
    N = positions.shape[0]
    if N < 2:
        return degree_inc

    # 层掩码矩阵: same_layer[i,j] = (layers[i] == layers[j] 且 layers[i] != 0)
    layer_eq = layers.unsqueeze(1) == layers.unsqueeze(0)  # (N, N)
    not_root = (layers != 0).unsqueeze(1)  # (N, 1)
    mask = layer_eq & not_root  # (N, N)

    # pairwise 距离矩阵
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
    dist = torch.norm(diff, dim=2)  # (N, N)

    # 距离 < 阈值 且 同层非根
    close = (dist < distance_threshold) & mask  # (N, N)

    # 每个节点的邻居计数（排除自身）
    close = close.float()
    close.fill_diagonal_(0.0)
    counts = close.sum(dim=1).long()  # (N,)

    degree_inc += counts
    return degree_inc


# ============================================================
# GPU VSPT 引擎
# ============================================================

class GPUEngine:
    """GPU 加速的 VSPT 生长引擎。

    与 CPU 版 VSPTEngine 接口兼容，
    run_growth() 返回相同数据格式。

    内部工作流：
        1. CPU 播种（VSPTTree 种子节点）
        2. 每层生长：
           a. 收集当前层所有树的父节点
           b. GPU 批量生成子节点位置
           c. 子节点写回对应树的 nodes/layer_nodes
        3. 全部层完成后：GPU 批量横向连接 → 更新度数
    """

    def __init__(self, config: Optional[VSPTConfig] = None):
        self.config = config or VSPTConfig()
        self.trees: Dict[str, VSPTTree] = {}
        self._node_counter = 0
        self._rng = random.Random(self.config.seed)

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

    def _grow_layer_gpu(
        self,
        tree_entries: List[Tuple[str, int, VSPTNode]],
        layer_radius: float,
    ) -> List[Tuple[str, str, Tuple[float, float, float], str]]:
        """GPU 批量生成子节点，跨树并行。

        Args:
            tree_entries: [(tree_id, parent_index_in_batch, parent_node), ...]
            layer_radius: 子节点壳层半径

        Returns:
            [(tree_id, parent_uid, (child_x, child_y, child_z), child_uid), ...]
        """
        n_parents = len(tree_entries)
        if n_parents == 0:
            return []

        # 准备 GPU 输入：父节点位置 + 手性
        pos_list = []
        hand_list = []
        tree_ids = []
        parent_uids = []

        for tid, _, pnode in tree_entries:
            pos_list.append(pnode.position)
            hand_list.append(self.trees[tid].handedness.value)
            tree_ids.append(tid)
            parent_uids.append(pnode.uid)

        pos_tensor = torch.tensor(pos_list, dtype=torch.float32, device=_DEVICE)
        hand_tensor = torch.tensor(hand_list, dtype=torch.float32, device=_DEVICE)

        # GPU 批量计算子节点位置
        child_tensor = _batch_grow_children(pos_tensor, hand_tensor, layer_radius, spread=1.0)
        # child_tensor: (N, 3, 3) → 展平为 (N*3, 3)

        child_tensor_flat = child_tensor.reshape(-1, 3)  # (N*3, 3)

        # 读回 CPU
        child_np = child_tensor_flat.cpu().numpy()  # (N*3, 3)

        # 构建结果
        results = []
        for pi in range(n_parents):
            for k in range(3):
                ci = pi * 3 + k
                cx, cy, cz = child_np[ci]
                child_uid = self._next_uid()
                results.append((
                    tree_ids[pi],
                    parent_uids[pi],
                    (float(cx), float(cy), float(cz)),
                    child_uid,
                ))

        return results

    def _add_lateral_gpu(self) -> None:
        """GPU 批量处理所有树的横向连接。"""
        for tree in self.trees.values():
            all_nodes = list(tree.nodes.values())
            if len(all_nodes) < 2:
                continue

            pos_list = [nd.position for nd in all_nodes]
            layer_list = [nd.layer for nd in all_nodes]

            pos_tensor = torch.tensor(pos_list, dtype=torch.float32, device=_DEVICE)
            layer_tensor = torch.tensor(layer_list, dtype=torch.int32, device=_DEVICE)
            deg_tensor = torch.zeros(len(all_nodes), dtype=torch.int32, device=_DEVICE)

            threshold = self.config.lateral_distance_threshold * self.config.layer_spacing
            _batch_lateral_connections(pos_tensor, layer_tensor, threshold, deg_tensor)

            deg_np = deg_tensor.cpu().numpy()
            for i, nd in enumerate(all_nodes):
                nd.degree += int(deg_np[i])

    def _grow_layer_cpu(self, tree, nucleus_radius, next_layer, layer_radius):
        """CPU 回退：单棵树单层生长。"""
        from Phase_4.vspt_growth import _normalize, _triangular_mesh_directions
        current_uids = tree.layer_nodes.get(next_layer - 1, [])
        shell_area = 4.0 * math.pi * (layer_radius ** 2)
        triangle_area = (math.sqrt(3) / 4.0) * (self.config.layer_spacing ** 2)
        max_nodes = max(3, int(shell_area / triangle_area))

        new_nodes = []
        total_children = 0

        for parent_uid in current_uids:
            parent = tree.nodes.get(parent_uid)
            if parent is None or not parent.is_solid:
                continue

            parent_dir = _normalize(parent.position)
            child_dirs = _triangular_mesh_directions(parent_dir, tree.handedness)

            for di in range(self.config.children_per_node):
                if di >= len(child_dirs) or total_children >= max_nodes:
                    break
                d = child_dirs[di]
                child_uid = self._next_uid()
                child_pos = (d[0] * layer_radius, d[1] * layer_radius, d[2] * layer_radius)
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

        for node in tree.nodes.values():
            node.degree = len(node.children)
            if node.parent_uid and node.parent_uid != "nucleus":
                node.degree += 1

        if new_nodes:
            tree.layer_nodes[next_layer] = [n.uid for n in new_nodes]
        return len(new_nodes)

    def run_growth(
        self,
        outward_solid_faces: int,
        nucleus_uid: str,
        nucleus_particles: Dict[str, EternalParticle],
        nucleus_position_map: Dict[str, Tuple[float, float, float]],
    ) -> Dict:
        """运行 GPU 加速的 VSPT 生长。

        同 VSPTEngine.run_growth 接口，返回相同格式。

        Returns:
            {n_trees, total_nodes, layer_distribution, max_layer, trees}
        """
        self.trees = {}
        self._node_counter = 0

        # 核半径
        radii = [math.sqrt(sum(p[i] ** 2 for i in range(3)))
                 for p in nucleus_position_map.values()]
        nucleus_radius = sum(radii) / len(radii) if radii else 1.0

        # === 播种 ===
        tree_count = 0
        for ep_uid, ep in nucleus_particles.items():
            if ep.orientation.name == "OUTWARD":
                continue
            for fi in range(4):
                pos = nucleus_position_map.get(ep_uid, (0.0, 0.0, 0.0))
                r = math.sqrt(sum(p * p for p in pos))
                radial = tuple(p / r for p in pos) if r > 1e-12 else (1.0, 0.0, 0.0)
                surface_pos = tuple(radial[i] * (r + ep.radius) for i in range(3))
                self.seed_tree(
                    nucleus_uid=f"{nucleus_uid}_{ep_uid}",
                    face_index=fi,
                    handedness=ep.handedness,
                    surface_pos=surface_pos,
                )
                tree_count += 1

        # === 逐层生长（GPU 批量跨树并行）===
        total_nodes = tree_count
        layer_dist = {0: tree_count}
        n_trees = len(self.trees)

        for layer in range(1, self.config.max_layers + 1):
            layer_new = 0
            layer_radius = nucleus_radius + layer * self.config.layer_spacing

            # 球面面积上限：该层允许的最大子节点数
            shell_area = 4.0 * math.pi * (layer_radius ** 2)
            triangle_area = (math.sqrt(3) / 4.0) * (self.config.layer_spacing ** 2)
            max_nodes_this_layer = max(3, int(shell_area / triangle_area))

            # 收集所有树的当前层父节点，跨树 GPU 批量计算
            batch_entries = []  # [(tree_id, parent_idx_in_batch, parent_node)]
            total_possible_children = 0

            for tree_id, tree in self.trees.items():
                if tree.node_count >= self.config.max_nodes_per_tree:
                    continue
                current_layer = tree.max_layer
                next_layer = current_layer + 1
                if next_layer > self.config.max_layers:
                    continue

                parent_uids = tree.layer_nodes.get(current_layer, [])
                parent_nodes = [tree.nodes[uid] for uid in parent_uids
                              if uid in tree.nodes]
                if not parent_nodes:
                    continue

                for pn in parent_nodes:
                    batch_entries.append((tree_id, len(batch_entries), pn))
                total_possible_children += len(parent_nodes) * 3

            if not batch_entries:
                break

            # 如果可能子节点数超过面积上限，随机采样
            if total_possible_children > max_nodes_this_layer:
                # 比例采样父节点
                n_parents = len(batch_entries)
                total_full = n_parents * 3
                keep_ratio = max_nodes_this_layer / total_full
                if keep_ratio < 1.0:
                    import random as _rnd
                    _rnd.Random(self.config.seed if self.config.seed else 42)
                    # 按比例保留父节点
                    n_keep = max(1, int(n_parents * keep_ratio))
                    keep_idx = set(_rnd.sample(range(n_parents), n_keep))
                    batch_entries = [be for i, be in enumerate(batch_entries) if i in keep_idx]

            # GPU 批量生成子节点
            if _HAVE_CUDA and len(batch_entries) >= 4:
                children_data = self._grow_layer_gpu(batch_entries, layer_radius)
            else:
                # CPU 回退
                for tree_id, tree in self.trees.items():
                    if tree_id in tree_offsets:
                        added = self._grow_layer_cpu(tree, nucleus_radius,
                                                      tree.max_layer + 1,
                                                      layer_radius)
                        layer_new += added
                if layer_new > 0:
                    layer_dist[layer] = layer_dist.get(layer, 0) + layer_new
                    total_nodes += layer_new
                continue

            # 分发子节点到各树
            tree_new_nodes = {tid: [] for tid in self.trees}
            for tree_id, parent_uid, child_pos, child_uid in children_data:
                tree = self.trees[tree_id]
                parent = tree.nodes[parent_uid]
                next_layer = parent.layer + 1

                child = VSPTNode(
                    uid=child_uid,
                    layer=next_layer,
                    parent_uid=parent_uid,
                    position=child_pos,
                    branch_angle=120.0,
                    degree=0,
                    is_solid=True,
                )
                tree.nodes[child_uid] = child
                parent.children.append(child_uid)
                tree_new_nodes[tree_id].append(child)

            # 更新度数和壳层索引
            for tree_id, tree in self.trees.items():
                if tree_id in tree_new_nodes and tree_new_nodes[tree_id]:
                    # 度数更新
                    for node in tree.nodes.values():
                        node.degree = len(node.children)
                        if node.parent_uid and node.parent_uid != "nucleus":
                            node.degree += 1

                    # 壳层索引
                    new = tree_new_nodes[tree_id]
                    next_layer = new[0].layer
                    tree.layer_nodes[next_layer] = [n.uid for n in new]
                    layer_new += len(new)

            if layer_new > 0:
                layer_dist[layer] = layer_dist.get(layer, 0) + layer_new
                total_nodes += layer_new
            else:
                break

        # === 横向连接（GPU 加速）===
        if _HAVE_CUDA:
            self._add_lateral_gpu()
        else:
            # CPU 回退
            from Phase_4.vspt_growth import _add_lateral_connections_impl
            for tree in self.trees.values():
                for layer, uids in tree.layer_nodes.items():
                    if layer == 0:
                        continue
                    thresh = self.config.lateral_distance_threshold * self.config.layer_spacing
                    nodes_at_layer = [tree.nodes[uid] for uid in uids if uid in tree.nodes]
                    for i, ni in enumerate(nodes_at_layer):
                        for j in range(i + 1, len(nodes_at_layer)):
                            nj = nodes_at_layer[j]
                            dx = ni.position[0] - nj.position[0]
                            dy = ni.position[1] - nj.position[1]
                            dz = ni.position[2] - nj.position[2]
                            dist = math.sqrt(dx**2 + dy**2 + dz**2)
                            if dist < thresh:
                                ni.degree += 1
                                nj.degree += 1

        return {
            "n_trees": n_trees,
            "total_nodes": total_nodes,
            "layer_distribution": dict(sorted(layer_dist.items())),
            "max_layer": max(layer_dist.keys()) if layer_dist else 0,
            "trees": self.trees,
        }
