"""
共享数据加载 — 合成数据 + MNIST

统一接口，使各实验脚本可从 shared 导入而非各自实现。
"""

import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def make_synthetic(
    n_samples: int = 20000,
    n_features: int = 20,
    n_classes: int = 4,
    class_sep: float = 0.8,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成合成分类数据（P0/P2 实验标准配置）。

    Returns:
        X_train, X_test, y_train, y_test
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=4,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def load_mnist(
    data_dir: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """加载 MNIST 数据集（torchvision 方式）。

    Args:
        data_dir: 数据缓存目录，默认 experiments/mnist_raw 的父目录
        batch_size: 批大小
        num_workers: 并行加载线程数

    Returns:
        train_loader, test_loader
    """
    import os
    from torchvision import datasets, transforms

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
    )
    return train_loader, test_loader


def load_mnist_npy(
    npy_path: str,
    train_size: int = 5000,
    val_size: int = 10000,
    seed: int = 42,
    batch_size: int = 128,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """从 .npz 文件加载 MNIST（N016 实验方式，无 torchvision 依赖）。

    Args:
        npy_path: mnist_data.npz 路径
        train_size: 训练样本数
        val_size: 验证样本数

    Returns:
        train_loader, val_loader
    """
    data = np.load(npy_path)
    X, y = data["X"].astype(np.float32), data["y"]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=val_size, random_state=seed
    )
    X_tr, _, y_tr, _ = train_test_split(
        X_tr, y_tr, train_size=train_size, random_state=seed
    )
    X_tr = X_tr.reshape(-1, 1, 28, 28)
    X_va = X_va.reshape(-1, 1, 28, 28)

    def _loader(X_arr, y_arr, shuffle=False):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_arr, dtype=torch.float32).contiguous(),
                torch.tensor(y_arr, dtype=torch.long),
            ),
            batch_size, shuffle=shuffle, num_workers=0,
        )
    return _loader(X_tr, y_tr, True), _loader(X_va, y_va, False)


def get_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """将 numpy 数组包装为 DataLoader（合成数据常用）。"""
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test).long()
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader
