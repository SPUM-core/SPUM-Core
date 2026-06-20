"""GPU 帧内核 — PyTorch 加速版（实验性：未集成到主线）。

注意：当前主线使用 CPU (numpy) 版 frame_kernels.py，
此文件仅保留作为 PyTorch CUDA 加速的参考实现。
尚未经过充分的正确性验证，不推荐用于生产。
"""

import math
import numpy as np
import torch

from .constants import GEOMETRIC_TOLERANCE


# ──────────────────────────────────────────
# Step 3: 变化体积 (vectorized, O(N))
# ──────────────────────────────────────────

def torch_step3_volume(particles):
    """GPU 加速体积更新。

    V = initial_volume + (degree - initial_degree)
    r = (3V/4π)^(1/3)
    """
    act = particles.active
    iv = particles.initial_volume[act]
    id_ = particles.initial_degree[act].float()
    deg = particles.degree[act].float()

    V = iv + (deg - id_)
    V = torch.clamp(V, min=0.5)

    particles.radius[act] = (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)


# ──────────────────────────────────────────
# Step 2: 表面角邻接 — pairwise dot product (O(N²))
# ──────────────────────────────────────────

def torch_surface_adjacency(pos, radii, surf_indices, center_idx):
    """GPU 加速表面角邻接检测。

    计算所有表面粒子对之间的角距，检测角半径重叠。

    Returns:
        List of (i, j) 相邻对
    """
    n_surf = len(surf_indices)
    if n_surf < 2 or center_idx < 0:
        return []

    # Surface positions and radii (already on GPU)
    surf_pos = pos[surf_indices]  # (n_surf, 3)
    surf_r = radii[surf_indices]  # (n_surf,)

    # Distance from center
    R = torch.norm(surf_pos[0]).item()

    # Angular radius = arcsin(r / R)
    theta = torch.asin(torch.clamp(surf_r / R, -1.0, 1.0))  # (n_surf,)

    # Pairwise dot products → angular distances
    norms = torch.norm(surf_pos, dim=1)  # (n_surf,)
    normed = surf_pos / norms.view(-1, 1)  # (n_surf, 3)
    dots = torch.clamp(normed @ normed.T, -1.0, 1.0)  # (n_surf, n_surf)

    # Angular distance between each pair
    ang_dist = torch.acos(dots)  # (n_surf, n_surf)

    # Sum of angular radii
    ang_sum = theta.view(-1, 1) + theta.view(1, -1)  # (n_surf, n_surf)

    # Adjacent if |ang_dist - ang_sum| < 5% tolerance
    tol = 0.05 * torch.clamp(ang_sum, min=0.01)
    adjacent = (ang_dist > 0) & (torch.abs(ang_dist - ang_sum) < tol)

    # Extract pairs (i < j)
    triu = torch.triu(adjacent, diagonal=1)
    pairs = torch.nonzero(triu)

    return [(int(surf_indices[pairs[k, 0].item()]),
             int(surf_indices[pairs[k, 1].item()])) for k in range(pairs.shape[0])]


# ──────────────────────────────────────────
# Step 3b: 不可入性 — pairwise distance (O(N²))
# ──────────────────────────────────────────

def torch_detect_overlaps(particles, max_pairs=50000):
    """GPU 加速重叠检测。

    使用所有活性粒子对的 pairwise distance，
    检测 distance < r_i + r_j - tol 的重叠对。

    Returns:
        List of (i, j) 重叠对
    """
    act_idx = torch.where(particles.active)[0]
    n_act = len(act_idx)
    if n_act < 2:
        return []

    pos = particles.pos[act_idx]  # (n_act, 3)
    r = particles.radius[act_idx]  # (n_act,)

    # Pairwise distance matrix (n_act, n_act)
    dist = torch.cdist(pos, pos, p=2)

    # Pairwise radius sum matrix
    r_sum = r.view(-1, 1) + r.view(1, -1)

    # Overlap: dist < r_sum - tol
    overlap = (dist > 0) & (dist < r_sum - 0.01)

    triu = torch.triu(overlap, diagonal=1)
    pairs = torch.nonzero(triu)

    n_found = pairs.shape[0]
    if n_found == 0:
        return []

    if n_found > max_pairs:
        # Subsample: keep the most overlapping pairs
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        overlap_amount = (r_sum[idx_i, idx_j] - dist[idx_i, idx_j])
        keep = torch.argsort(overlap_amount, descending=True)[:max_pairs]
        pairs = pairs[keep]

    return [(int(act_idx[pairs[k, 0].item()].item()),
             int(act_idx[pairs[k, 1].item()].item())) for k in range(pairs.shape[0])]


# ──────────────────────────────────────────
# Enforce tangency
# ──────────────────────────────────────────

def torch_enforce_tangency(particles):
    """GPU 辅助相切强制。

    BFS 序由 CPU 管理，每个粒子的坐标用 PyTorch 更新。
    """
    center_idx = _find_center_torch(particles)
    if center_idx < 0:
        return

    N = particles.N
    active = particles.active
    pos = particles.pos
    radius = particles.radius

    # Fix center at origin
    pos[center_idx] = 0.0

    # Build adjacency on CPU
    adj = {i: set() for i in range(N) if active[i].item()}
    for a, b in particles.connections:
        if active[a].item() and active[b].item():
            adj[a].add(b)
            adj[b].add(a)

    # BFS from center
    placed = {center_idx}
    queue = [center_idx]
    order = []

    while queue:
        current = queue.pop(0)
        for nbr in adj[current]:
            if nbr not in placed:
                placed_nbrs = [n for n in adj[nbr] if n in placed]
                if placed_nbrs:
                    placed.add(nbr)
                    queue.append(nbr)
                    order.append((nbr, placed_nbrs))

    # Position updates
    for idx, placed_nbrs in order:
        r_idx = radius[idx].item()
        pn_indices = torch.tensor(placed_nbrs, device=pos.device)
        pn_pos = pos[pn_indices]  # (k, 3)
        pn_r = radius[pn_indices]  # (k,)

        # Direction from each placed neighbor to current position
        d_vec = pos[idx].unsqueeze(0) - pn_pos  # (k, 3)
        d_norm = torch.norm(d_vec, dim=1)  # (k,)

        # Target position: pn_pos + (d_vec / d_norm) * (pn_r + r_idx)
        target_dist = pn_r + r_idx
        mask = d_norm > 1e-12
        target = pn_pos.clone()
        target[mask] = pn_pos[mask] + (d_vec[mask] / d_norm[mask].view(-1, 1)) * target_dist[mask].view(-1, 1)
        if not mask.all():
            # Fallback for collapsed particles
            fallback = (~mask).nonzero().flatten()
            for fi in fallback:
                target[fi] = pn_pos[fi] + torch.tensor([1.0, 0.0, 0.0], device=pos.device) * target_dist[fi]

        # Average position
        pos[idx] = target.mean(dim=0)

    # Iterative refinement (3 passes)
    all_indices = [i for i in range(N) if active[i].item() and i != center_idx]
    for _ in range(3):
        for idx in all_indices:
            nbrs = [n for n in adj[idx] if n in placed and n != idx]
            if len(nbrs) < 2:
                continue

            pn_indices = torch.tensor(nbrs, device=pos.device)
            pn_pos = pos[pn_indices]
            pn_r = radius[pn_indices]
            r_idx = radius[idx].item()

            d_vec = pos[idx].unsqueeze(0) - pn_pos
            d_norm = torch.norm(d_vec, dim=1)
            target_dist = pn_r + r_idx
            mask = d_norm > 1e-12
            target = pn_pos.clone()
            target[mask] = pn_pos[mask] + (d_vec[mask] / d_norm[mask].view(-1, 1)) * target_dist[mask].view(-1, 1)
            if not mask.all():
                fb = (~mask).nonzero().flatten()
                for fi in fb:
                    target[fi] = pn_pos[fi] + torch.tensor([1.0, 0.0, 0.0], device=pos.device) * target_dist[fi]

            pos[idx] = target.mean(dim=0)


def _find_center_torch(particles) -> int:
    """Find center particle (uid starts with 'cent_')."""
    for i in range(particles.N):
        if particles.active[i].item() and str(particles.uid[i]).startswith('cent_'):
            return i
    return -1


# ──────────────────────────────────────────
# Full GPU-accelerated frame
# ──────────────────────────────────────────

def torch_run_full_frame(particles, star_mode=False, no_purge=False,
                         skip_center=False, gap_fill_per_frame=20,
                         seed_purge_only=False, min_active=0):
    """PyTorch 加速的全帧运算。

    GPU 执行:
        - step3_volume (torch vectorized)
        - step2 surface adj (torch pairwise dot)
        - step3b overlap detection (torch cdist)

    CPU 执行:
        - step1_create / step1b_gap_fill (逻辑密集)
        - step4/5 dangling/purge (简单)
        - reincarnate (逻辑密集)
        - enforce_tangency (BFS 序需 CPU)
    """
    from .frame_kernels import (
        step1_create, step1b_gap_fill, step2_connect,
        step4_dangling, step5_purge, reincarnate_to_minimum,
        enforce_tangency, _find_center
    )

    active_before = particles.active_count()

    # Step 3: Volume (GPU)
    torch_step3_volume(particles)

    # Step 3c: Tangency (GPU-assisted)
    torch_enforce_tangency(particles)

    # Step 1: Gap fill (CPU)
    created = step1_create(particles)
    if created == 0:
        created = step1b_gap_fill(particles, max_per_frame=gap_fill_per_frame)

    # Step 2: Connect (GPU surface adj)
    connected = 0
    if star_mode:
        center_idx = _find_center_torch(particles)
        if center_idx >= 0:
            surf_indices = [i for i in range(particles.N)
                           if particles.active[i].item() and i != center_idx]
            if surf_indices:
                pairs = torch_surface_adjacency(particles.pos, particles.radius,
                                                surf_indices, center_idx)
                for i, j in pairs:
                    if particles.add_connection(i, j):
                        connected += 1

    # Step 2b: pre-growth knn (CPU fallback)
    if not star_mode or connected == 0:
        connected += step2_connect(particles, star_mode=False,
                                   skip_center=skip_center)

    # Step 3b: Impenetrability (GPU)
    overlaps_resolved = 0
    overlaps = torch_detect_overlaps(particles)
    for i, j in overlaps:
        if not particles.active[i].item() or not particles.active[j].item():
            continue
        d = float(torch.norm(particles.pos[i] - particles.pos[j]).item())
        min_d = particles.radius[i].item() + particles.radius[j].item()
        if d < min_d - 0.01 and particles.connected(i, j):
            particles.remove_connection(i, j)
            overlaps_resolved += 1

    # Step 4/5: Dangling
    if no_purge:
        purged = 0
        dangling_count = 0
    else:
        dangling = step4_dangling(particles, seed_only=seed_purge_only)
        dangling_count = int(np.sum(dangling))
        purged = step5_purge(particles, dangling)

    # Step 6: Reincarnate
    if min_active > 0 and particles.active_count() < min_active:
        reincarnated = reincarnate_to_minimum(particles, min_active=min_active)
    else:
        reincarnated = 0

    active_after = particles.active_count()

    return {
        "created": created,
        "connected": connected,
        "overlaps_resolved": overlaps_resolved,
        "dangling_marked": dangling_count,
        "purged": purged,
        "reincarnated": reincarnated,
        "active_before": active_before,
        "active_after": active_after,
        "degree_histogram": particles.degree_histogram(),
    }
