"""
共享实验组件 — 跨 P0/P1/P2/N015/N016 复用的模型、数据加载、SPUM 钩子。

组件:
    models.py        — MLP, DeepCNN 模型定义
    data.py          — 合成数据生成, MNIST 加载
    spum_integration.py — SPUM 训练钩子 (δ密度, Δμ锚点漂移, 完整闭合检测)
"""
from .models import MLP, DeepCNN
from .data import load_mnist, make_synthetic, get_data_loaders
from .spum_integration import SPUMHook

__all__ = ["MLP", "DeepCNN", "load_mnist", "make_synthetic", "get_data_loaders", "SPUMHook"]
