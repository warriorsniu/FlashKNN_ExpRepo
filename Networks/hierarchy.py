"""Build the four-level DeLA neighbour hierarchy with FlashKNN."""

from __future__ import annotations

import math

import torch


def _voxel_first_cuda(xyz: torch.Tensor, size: float) -> torch.Tensor:
    # Sorting integer voxel coordinates gives a deterministic representative.
    try:
        from FlashKNN import xyz2key
    except ImportError:  # source-tree execution before pip installation
        from functions import xyz2key

    grid = torch.floor(xyz / size).long()
    key = xyz2key(grid[:, 0], grid[:, 1], grid[:, 2], None)
    order = key.argsort()
    sorted_key = key[order]
    first = torch.ones_like(sorted_key, dtype=torch.bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    return order[first].sort().values


def build_flash_hierarchy(
    metric_xyz: torch.Tensor,
    voxel_sizes: tuple[float, ...] = (0.06, 0.12, 0.24, 0.48),
    ks: tuple[int, ...] = (24, 24, 24, 24),
    alpha: int = 16,
) -> list[torch.Tensor]:
    """Return indices in the order expected by DeLA/DeepLA.

    The timing boundary of callers should include this entire function.  It
    includes hierarchy downsampling, all KNN queries and nearest interpolation
    maps, while assuming the input scan is already on the GPU.
    """
    try:
        from FlashKNN import FlashKNN
    except ImportError:  # source-tree execution before pip installation
        from functions import FlashKNN

    exponent = round(math.log2(alpha))
    if alpha < 2 or 2**exponent != alpha:
        raise ValueError("alpha must be a power of two")
    if len(voxel_sizes) != len(ks):
        raise ValueError("voxel_sizes and ks must have equal length")

    full_xyz = metric_xyz
    current_xyz = metric_xyz
    cumulative = torch.arange(len(metric_xyz), device=metric_xyz.device)
    batch = torch.zeros(len(metric_xyz), device=metric_xyz.device, dtype=torch.long)
    forward: list[torch.Tensor] = []
    cumulative_levels: list[torch.Tensor] = []

    for level, (size, k) in enumerate(zip(voxel_sizes, ks)):
        if level:
            local = _voxel_first_cuda(current_xyz, size)
            current_xyz = current_xyz[local].contiguous()
            cumulative = cumulative[local]
            batch = batch[local]
            forward.append(local)
            cumulative_levels.append(cumulative)
        grid = torch.floor(current_xyz / size).long().contiguous()
        neighbours = FlashKNN(num_nbr=k, num_down=exponent).query(
            grid, batch, current_xyz, memory_mode="SM", sorting_mode="PS"
        )
        forward.append(neighbours)

    # The recursive reference implementation appends these while unwinding:
    # deepest-to-shallowest, each mapping all level-0 points to that level.
    backward: list[torch.Tensor] = []
    full_batch = torch.zeros(len(full_xyz), device=full_xyz.device, dtype=torch.long)
    for level in range(len(cumulative_levels) - 1, -1, -1):
        size = voxel_sizes[level + 1]
        back = FlashKNN(num_nbr=1).back_query(
            full_xyz,
            cumulative_levels[level],
            query_grid_size=size * 2,
            down_grid_size=size,
            batch_idx=full_batch,
        )
        backward.append(back)
    return forward + backward
