"""
SPUM 训练钩子 — 在训练循环中注入 δ 密度、Δμ 锚点漂移、完整闭合检测

用法:
    hook = SPUMHook(eps=0.3, drift_threshold=0.01, closure_patience=3)

    for epoch in range(N):
        train_one_epoch(model, loader)
        result = hook.on_epoch_end(model, loader, epoch, num_classes=10)
        print(f"Epoch {epoch}: δ={result['delta']:.4f}, drift={result['drift']:.2f}")
"""

import torch
import torch.nn as nn
from typing import Optional

# 复用 SPUM-图论模块的检测器
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from spum_graph.dangling import DanglingDetector


class SPUMHook:
    """训练循环中的 SPUM 监控钩子。

    每个 epoch 结束时调用 on_epoch_end，自动:
    1. 计算隐藏层锚点
    2. 计算 δ 密度（悬挂端比例）
    3. 计算 Δμ 锚点漂移率
    4. 输出双层闭合判据结果
    """

    def __init__(
        self,
        eps: float = 0.3,
        drift_threshold: float = 0.01,
        closure_patience: int = 3,
    ):
        self.eps = eps
        self.drift_threshold = drift_threshold
        self.closure_patience = closure_patience
        self._detector = DanglingDetector(
            eps=eps,
            drift_threshold=drift_threshold,
            closure_patience=closure_patience,
        )

    @torch.no_grad()
    def on_epoch_end(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        num_classes: int,
    ) -> dict:
        """在一个 epoch 结束时调用。

        收集所有训练样本的隐藏层表示，计算锚点和悬挂端状态。

        Args:
            model: 训练中的模型（自动调用 eval 模式）
            loader: 训练数据加载器
            epoch: 当前 epoch 编号
            num_classes: 分类类别数

        Returns:
            dict with keys: epoch, delta, drift, dangling_count, total_samples,
                           is_delta_stable, is_drift_stable, is_closed, closure_frames
        """
        model.eval()
        device = next(model.parameters()).device

        all_hidden = []
        all_labels = []

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # 要求模型支持 return_hidden=True
            if hasattr(model, "forward") and "return_hidden" in model.forward.__code__.co_varnames:
                _, h = model(x, return_hidden=True)
            else:
                # DeepCNN: 通过 self.h 暴露隐藏层
                _ = model(x)
                h = model.h
            all_hidden.append(h)
            all_labels.append(y)

        hidden = torch.cat(all_hidden, dim=0)
        labels = torch.cat(all_labels, dim=0)
        model.train()

        return self._detector.update(hidden, labels, num_classes, epoch)

    def is_fully_closed(self) -> bool:
        """检查是否达到完整闭合（双层判据 + 持续 patience 帧）。"""
        return self._detector.is_fully_closed()

    def summary(self) -> dict:
        """返回所有帧的汇总统计。"""
        return self._detector.summary()

    def history(self) -> list:
        """返回所有帧状态记录。"""
        return self._detector.history()
