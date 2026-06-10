"""
悬挂端检测器 — 公理 2 和公理 4 的实操实现

实现基于锚点距离的 δ 密度计算（L1 认知投影中的悬挂端检测），
与经典 P0/P1/P2 实验代码对齐。

悬挂端定义:
    到最近两个锚点的距离差 < ε 的样本。
    即：样本处于多锚点的模糊地带，语义位置不确定。

双层判据（N015/N016 修正）:
    完整闭合 = (δ < θ_δ) 且 (Δμ < θ_μ) 持续 N 帧
    单靠 δ 下降不能判定闭合 —— N013 被证伪。
"""

import torch
from typing import Tuple, Optional


def compute_dangling(
    hidden: torch.Tensor,
    anchors: torch.Tensor,
    eps: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算悬挂端标志和距离差。

    Args:
        hidden: 形状 (N, D)，样本的隐藏层表示
        anchors: 形状 (C, D)，各类别的锚点（类均值）
        eps: 悬挂端距离差阈值

    Returns:
        is_dangling: 形状 (N,) 布尔张量，True=悬挂端
        dist_diff: 形状 (N,) 浮点张量，最近两个锚点的距离差

    算法:
        1. 计算每个样本到每个锚点的欧氏距离 → (N, C)
        2. 对距离排序，取最近两个 → d1, d2
        3. dist_diff = d2 - d1
        4. is_dangling = (dist_diff < eps)

    含义:
        dist_diff 很小 → 样本到最近两个锚点的距离差不多
                    → 样本处于两个类别的语义边界
                    → 悬挂端（认知歧义节点）
    """
    # (N, D) × (C, D) → (N, C)
    dists = torch.cdist(hidden.unsqueeze(1), anchors.unsqueeze(0)).squeeze(1)

    # 最近两个锚点的距离
    sorted_dists, _ = dists.sort(dim=1)
    dist_diff = sorted_dists[:, 1] - sorted_dists[:, 0]

    is_dangling = dist_diff < eps
    return is_dangling, dist_diff


class DanglingDetector:
    """悬挂端检测器 — 封装 δ 密度 + 锚点漂移双层判据。

    用法:
        detector = DanglingDetector(eps=0.3, drift_threshold=0.01)
        detector.update(hidden, labels, epoch)

        if detector.is_fully_closed():
            print("完整闭合近似达到")
    """

    def __init__(
        self,
        eps: float = 0.3,
        drift_threshold: float = 0.01,
        closure_patience: int = 3
    ):
        self.eps = eps
        self.drift_threshold = drift_threshold
        self.closure_patience = closure_patience

        self._history = []          # 每帧的 (delta, drift, epoch)
        self._closure_counter = 0   # 连续闭合帧计数
        self._last_anchors: Optional[torch.Tensor] = None

    def compute_anchors(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """计算各类别的锚点（类均值隐藏表示）。

        Args:
            hidden: (N, D) 隐藏层输出
            labels: (N,) 类别标签
            num_classes: 类别数

        Returns:
            anchors: (C, D) 各类锚点
        """
        anchors = torch.zeros(num_classes, hidden.shape[1], device=hidden.device)
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                anchors[c] = hidden[mask].mean(dim=0)
        return anchors

    def update(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        epoch: int
    ) -> dict:
        """执行一帧的悬挂端/锚点更新。

        Returns:
            dict with keys: delta, drift, is_delta_stable, is_drift_stable, is_closed
        """
        # 1. 计算锚点
        anchors = self.compute_anchors(hidden, labels, num_classes)

        # 2. 计算 δ 密度
        is_dangling, _ = compute_dangling(hidden, anchors, self.eps)
        delta = is_dangling.float().mean().item()

        # 3. 计算锚点漂移 Δμ
        drift = None
        if self._last_anchors is not None:
            drift = torch.norm(anchors - self._last_anchors).item()

        self._last_anchors = anchors.clone()

        # 4. 双层判据
        is_delta_stable = delta < self.eps
        is_drift_stable = drift is not None and drift < self.drift_threshold
        is_closed = is_delta_stable and is_drift_stable

        # 5. 连续闭合计数
        if is_closed:
            self._closure_counter += 1
        else:
            self._closure_counter = 0

        result = {
            "epoch": epoch,
            "delta": delta,
            "drift": drift,
            "dangling_count": is_dangling.sum().item(),
            "total_samples": hidden.shape[0],
            "is_delta_stable": is_delta_stable,
            "is_drift_stable": is_drift_stable,
            "is_closed": is_closed,
            "closure_frames": self._closure_counter,
        }
        self._history.append(result)
        return result

    def is_fully_closed(self) -> bool:
        """检查是否达到完整闭合（双层判据 + 持续 patience 帧）。"""
        return self._closure_counter >= self.closure_patience

    def history(self) -> list:
        """返回所有帧状态记录。"""
        return self._history

    def summary(self) -> dict:
        """返回汇总统计。"""
        if not self._history:
            return {}
        deltas = [h["delta"] for h in self._history]
        drifts = [h["drift"] for h in self._history if h["drift"] is not None]
        closed_frames = sum(1 for h in self._history if h["is_closed"])
        return {
            "total_frames": len(self._history),
            "min_delta": min(deltas),
            "final_delta": deltas[-1],
            "min_drift": min(drifts) if drifts else None,
            "final_drift": drifts[-1] if drifts else None,
            "closed_frames": closed_frames,
            "fully_closed": self.is_fully_closed(),
        }
