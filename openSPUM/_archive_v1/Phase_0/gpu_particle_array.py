"""PyTorch GPU 粒子数组 — GPU SoA 布局。"""

import math
import numpy as np
import torch

from .constants import KAPPA, CRYSTALLITE_DEGREE_THRESHOLD


class TorchParticleArray:
    """GPU-resident SoA (Structure of Arrays) 粒子数组。

    使用 PyTorch CUDA tensor 作为后端。
    与 ParticleArray 接口兼容，所有数据在 GPU 上。

    仅在 GPU 管理的数据: pos, radius, degree, active, initial_volume, initial_degree
    仍在 CPU 管理: uid (字符串), connections (集合)
    """

    def __init__(self, n: int = 0, max_n: int = 65536, device: str = 'cuda'):
        self.N = n
        self.max_n = max_n
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.connections = set()
        self._alloc(n)

    def _alloc(self, n: int):
        n = max(n, 1)
        self.N = n
        self.pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.radius = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.degree = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.uid = np.empty(n, dtype=object)
        self.initial_volume = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.initial_degree = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.connections.clear()

    def resize(self, new_n: int):
        if new_n <= self.N:
            return
        old = {
            'pos': self.pos.clone(),
            'radius': self.radius.clone(),
            'degree': self.degree.clone(),
            'active': self.active.clone(),
            'uid': self.uid.copy(),
            'initial_volume': self.initial_volume.clone(),
            'initial_degree': self.initial_degree.clone(),
        }
        old_connections = self.connections.copy()
        self._alloc(new_n)
        for k in old:
            v = old[k]
            if isinstance(v, torch.Tensor):
                getattr(self, k)[:len(v)] = v
            else:
                getattr(self, k)[:len(v)] = v
        self.connections = old_connections

    def add_particle(self, pos, uid, degree=0,
                     initial_volume=1.0, initial_degree=0) -> int:
        active_cpu = (~self.active).cpu().numpy() if self.N > 0 else np.array([], dtype=bool)
        latent = np.where(active_cpu)[0]
        if len(latent) > 0:
            idx = int(latent[0])
        else:
            idx = int(self.active.sum().item()) if self.N > 0 else 0
            if idx >= self.max_n:
                raise RuntimeError(f"Particle count exceeds {self.max_n}")
            if idx >= self.N:
                new_n = min(self.N * 2 + 1, self.max_n)
                self.resize(new_n)

        V = float(initial_volume) + max(0, float(degree) - float(initial_degree))
        r = (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.pos[idx] = torch.tensor(pos, dtype=torch.float32, device=self.device)
        self.radius[idx] = r
        self.degree[idx] = degree
        self.active[idx] = True
        self.uid[idx] = uid
        self.initial_volume[idx] = initial_volume
        self.initial_degree[idx] = initial_degree
        return idx

    def remove_particle(self, idx: int):
        self.active[idx] = False
        self.degree[idx] = 0

    def active_count(self) -> int:
        return int(self.active.sum().item())

    def latent_count(self) -> int:
        return int(((~self.active) & (torch.tensor([u is not None for u in self.uid],
                                                   device=self.device))).sum().item())

    def degree_histogram(self) -> np.ndarray:
        d = self.degree[self.active].cpu().numpy()
        d_clipped = np.clip(d, 0, CRYSTALLITE_DEGREE_THRESHOLD)
        hist = np.zeros(CRYSTALLITE_DEGREE_THRESHOLD + 1, dtype=np.int32)
        for i in range(CRYSTALLITE_DEGREE_THRESHOLD + 1):
            hist[i] = int(np.sum(d_clipped == i))
        return hist

    def dangling_mask(self) -> torch.Tensor:
        return self.active & (self.degree < 3)

    def spum_invariant(self) -> int:
        V = self.active_count()
        E = int(self.degree[self.active].sum().item()) // 2
        return 6 * V - 2 * E

    # ---- Edge management (CPU) ----

    def add_connection(self, i: int, j: int) -> bool:
        if i < 0 or j < 0:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in self.connections:
            return False
        self.connections.add((a, b))
        self.degree[a] += 1
        self.degree[b] += 1
        return True

    def remove_connection(self, i: int, j: int) -> bool:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in self.connections:
            return False
        self.connections.discard((a, b))
        self.degree[a] = max(0, int(self.degree[a].item()) - 1)
        self.degree[b] = max(0, int(self.degree[b].item()) - 1)
        return True

    def connected(self, i: int, j: int) -> bool:
        a, b = (i, j) if i < j else (j, i)
        return (a, b) in self.connections

    def remove_all_connections(self, idx: int):
        to_remove = [(a, b) for (a, b) in self.connections if a == idx or b == idx]
        for a, b in to_remove:
            self.connections.discard((a, b))
            self.degree[a] = max(0, int(self.degree[a].item()) - 1)
            self.degree[b] = max(0, int(self.degree[b].item()) - 1)

    def clear_connections(self):
        self.connections.clear()
        self.degree[:] = 0

    def copy_to_host(self) -> dict:
        return {
            'pos': self.pos.cpu().numpy(),
            'radius': self.radius.cpu().numpy(),
            'degree': self.degree.cpu().numpy(),
            'active': self.active.cpu().numpy(),
            'uid': self.uid.copy(),
            'initial_volume': self.initial_volume.cpu().numpy(),
            'initial_degree': self.initial_degree.cpu().numpy(),
        }

    def snapshot(self, frame_number: int) -> 'FrameSnapshot':
        from .metadata_decoder import FrameSnapshot
        dang = self.dangling_mask().cpu().numpy()
        return FrameSnapshot(
            frame_number=frame_number,
            degree_histogram=self.degree_histogram(),
            active_count=self.active_count(),
            latent_count=self.latent_count(),
            crystallite_count=int((self.degree[self.active] >= CRYSTALLITE_DEGREE_THRESHOLD).sum().item()),
            spum_invariant=self.spum_invariant(),
            dangling_count=int(np.sum(dang)),
        )
