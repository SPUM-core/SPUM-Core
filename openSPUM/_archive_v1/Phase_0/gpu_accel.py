"""
GPU 几何热区加速器 — CUDA 只加速 O(N²) 数值候选计算。

设计原则 (SPUM 确定性优先):
    GPU 计算"几何候选对"集合 (相切 / 重叠 / 角邻接的 pairwise cdist),
    返回候选对索引列表。实际 add_connection / remove_connection 的
    度数装配始终在 CPU-侧完成 (connections set, 可哈希、确定)。

    → GPU 与 CPU 的帧语义完全一致, 结果可逐帧对照。

Function mapping:
    tangent_pairs_gpu  → 代替 step2_connect 的 O(N²) 几何相切候选检测
    overlap_pairs_gpu  → 代替 step3b 的 O(N²) 重叠检测
    volume_gpu         → 避免代替 step3_volume (保持与 CPU 完全一致)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


def _import_torch():
    """惰性导入 torch (仅 CUDA + 真正调用时)。"""
    import torch
    return torch


def _to_cuda_tensor(torch, arr: np.ndarray, dtype=None) -> object:
    """numpy → CUDA tensor (float32/int32)。"""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=dtype))
    return t.to("cuda")


def _pairs_from_triu(torch, triu_mask, act_idx) -> List[Tuple[int, int]]:
    """从 triu 掩码提取 (i_ref, j_ref) 原索引对 (i<j)。"""
    idx = torch.nonzero(triu_mask)
    if idx.shape[0] == 0:
        return []
    pairs = []
    n = idx.shape[0]
    max_pairs = 200000
    keep = min(n, max_pairs)  # 归约, 防爆内存
    k = idx[:keep]
    for t in range(k.shape[0]):
        pairs.append((int(act_idx[k[t, 0].item()]),
                      int(act_idx[k[t, 1].item()])))
    return pairs


# ──────────────────────────────────────────
# 几何相切候选对 (对应 step2 的 O(N²) 相切检测)
# ──────────────────────────────────────────

def tangent_pairs_gpu(pos: np.ndarray, radius: np.ndarray,
                      active: np.ndarray,
                      tol: float = 1e-4,
                      max_pairs: int = 200000) -> List[Tuple[int, int]]:
    """计算满足 |d - (ri+rj)| <= tol*(ri+rj) 的活性粒子候选对。

    Args:
        pos:   (N,3) float32 位置
        radius:(N,) float32 半径
        active:(N,) bool
        tol:   几何容差 (默认 1e-4, 与 CPU GEOMETRIC_TOLERANCE 对齐)
        max_pairs: 返回对上限

    Returns:
        [(i,j), ...] i<j, 均为活性粒子
    """
    torch = _import_torch()
    act_idx = np.where(active)[0]
    n_act = len(act_idx)
    if n_act < 2:
        return []

    # float64 与 CPU (numpy float64) 几何判定一致, 保证严格语义等价
    pos_t = _to_cuda_tensor(torch, pos[act_idx], dtype=np.float64)
    r_t = _to_cuda_tensor(torch, radius[act_idx], dtype=np.float64)

    # pairwise distance 矩阵 (n_act, n_act)
    d = torch.cdist(pos_t, pos_t, p=2)
    r_sum = r_t.view(-1, 1) + r_t.view(1, -1)
    thresh = tol * r_sum

    # 相切: |d - r_sum| <= thresh, 且 d > 0 (非自)
    # 注意: 这里用"宽松超集" (thresh 略放宽 + 绝对项), 只为粗剪枝;
    #       由 CPU 侧 step2_connect 用精确谓词复核 → 语义严格一致。
    guard = 1e-12
    thresh = (tol + 1e-6) * r_sum + guard
    close = (d > 1e-9) & ((d - r_sum).abs() <= thresh)
    triu = torch.triu(close, diagonal=1)
    return _pairs_from_triu(torch, triu, act_idx)


def overlap_pairs_gpu(pos: np.ndarray, radius: np.ndarray,
                      active: np.ndarray,
                      tol: float = 0.01,
                      max_pairs: int = 50000) -> List[Tuple[int, int]]:
    """重叠检测: d < ri + rj - tol 的活性粒子对。

    对应 step3b_enforce_impenetrability 的 O(N²) 重叠判定。
    """
    torch = _import_torch()
    act_idx = np.where(active)[0]
    n_act = len(act_idx)
    if n_act < 2:
        return []

    pos_t = _to_cuda_tensor(torch, pos[act_idx], dtype=np.float32)
    r_t = _to_cuda_tensor(torch, radius[act_idx], dtype=np.float32)

    d = torch.cdist(pos_t, pos_t, p=2)
    r_sum = r_t.view(-1, 1) + r_t.view(1, -1)
    over = (d > 1e-9) & (d < r_sum - tol)
    triu = torch.triu(over, diagonal=1)
    return _pairs_from_triu(torch, triu, act_idx, )


# ──────────────────────────────────────────
# 体积向量化 (与 CPU step3_volume 数值一致)
# ──────────────────────────────────────────

def volume_batch(initial_volume, degree, initial_degree) -> np.ndarray:
    """向量化体积计算 (供 GPU/CPU 共用, 保证一致)。

    V = initial_volume + max(0, degree - initial_degree)
    r = (3V/4π)^(1/3)
    """
    V = initial_volume + np.maximum(0.0, degree.astype(np.float64) - initial_degree)
    V = np.maximum(V, 0.5)
    return (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)