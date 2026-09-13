"""
统一后端抽象 — 显卡部署的探测 / 选择 / 降级。

原则 (SPUM 确定性优先):
    GPU 只加速 O(N²) 几何热区 (相切 / 重叠 / 体积向量化)。
    边集 (connections) 始终驻留 CPU set — 保证确定性、可哈希、
    与帧语义严格一致。GPU 结果必须与 CPU 数值一致 (可对照)。

    → 显卡部署 = 不动帧拓扑逻辑, 只替换数值热区为 CUDA 内核。
      无 GPU / 无 torch 时自动回退 numpy (现有 CPU 管线原样运行)。

后端状态:
    BACKEND ∈ {'cuda', 'cpu'}
    torch   : 惰性导入 (仅 CUDA 可用才真正 import, 避免无 GPU 机器报错)
"""

from __future__ import annotations

from typing import Optional

# 后端能力探测结果 (模块级缓存)
_DEVICE_NAME: Optional[str] = None
_HAS_CUDA: Optional[bool] = None
_BACKEND: Optional[str] = None


def _probe() -> None:
    """探测 CUDA 可用性 (惰性, 仅一次)。"""
    global _HAS_CUDA, _DEVICE_NAME, _BACKEND
    if _BACKEND is not None:
        return
    try:
        import torch  # 惰性 — 无 torch 时不报错
        _HAS_CUDA = bool(torch.cuda.is_available())
        if _HAS_CUDA:
            _DEVICE_NAME = torch.cuda.get_device_name(0)
            _BACKEND = "cuda"
        else:
            _DEVICE_NAME = None
            _BACKEND = "cpu"
    except Exception:
        _HAS_CUDA = False
        _DEVICE_NAME = None
        _BACKEND = "cpu"


def cuda_available() -> bool:
    """CUDA 是否可用。"""
    _probe()
    return _HAS_CUDA is True


def device_name() -> Optional[str]:
    """当前 GPU 设备名 (无 GPU 返回 None)。"""
    _probe()
    return _DEVICE_NAME


def active_backend() -> str:
    """返回当前生效后端: 'cuda' | 'cpu'。"""
    _probe()
    return _BACKEND or "cpu"


def resolve(backend: Optional[str]) -> str:
    """把用户请求的后端解析为实际后端。

    'cuda' / 'gpu' → 强制 GPU (不可用则抛错)
    'cpu'         → 强制 CPU
    'auto' / None → 探测: 有 CUDA 用 GPU, 否则 CPU (优雅降级)
    """
    want = (backend or "auto").lower()
    _probe()
    if want in ("cuda", "gpu"):
        if not cuda_available():
            raise RuntimeError(
                "请求 GPU 后端但 CUDA 不可用。"
                f"当前: backend={active_backend()}"
            )
        return "cuda"
    if want == "cpu":
        return "cpu"
    return "cuda" if cuda_available() else "cpu"