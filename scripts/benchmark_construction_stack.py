#!/usr/bin/env python3
"""Compare FlashKNN's pure-PyTorch construction path across Torch stacks."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


ENCODE_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def xyz2key(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            batch: torch.Tensor | None = None) -> torch.Tensor:
    def expand(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        key = torch.zeros_like(a)
        for bit in range(8):
            mask = 1 << bit
            key = key | ((a & mask) << (2 * bit + 2)) \
                | ((b & mask) << (2 * bit + 1)) \
                | ((c & mask) << (2 * bit))
        return key

    if x.device not in ENCODE_CACHE:
        values = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        ENCODE_CACHE[x.device] = (
            expand(values, zero, zero).to(x.device),
            expand(zero, values, zero).to(x.device),
            expand(zero, zero, values).to(x.device),
        )
    encode_x, encode_y, encode_z = ENCODE_CACHE[x.device]
    x, y, z = x.long(), y.long(), z.long()
    key = encode_x[x & 255] | encode_y[y & 255] | encode_z[z & 255]
    mask = 255
    key16 = encode_x[(x >> 8) & mask] | encode_y[(y >> 8) & mask] \
        | encode_z[(z >> 8) & mask]
    key = key16 << 24 | key
    if batch is not None:
        key = batch.long() << 48 | key
    return key


def prepare_grid(room: Path, crop_points: int) -> torch.Tensor:
    coord = torch.as_tensor(np.load(room / "coord.npy"), dtype=torch.float32)
    coord -= coord.amin(dim=0, keepdim=True)
    coord = (torch.round(coord * 1000.0) / 1000.0).cuda().contiguous()
    grid = torch.floor(coord / 0.02).long()
    key = xyz2key(grid[:, 0], grid[:, 1], grid[:, 2])
    order = key.argsort()
    sorted_key = key[order]
    first = torch.ones_like(sorted_key, dtype=torch.bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    support_grid = grid[order[first]].contiguous()
    if len(support_grid) < crop_points:
        raise RuntimeError(f"Room has only {len(support_grid)} voxelized points")
    generator = torch.Generator(device="cpu").manual_seed(47)
    center = int(torch.randint(len(support_grid), (1,), generator=generator))
    distance = (support_grid - support_grid[center]).square().sum(1)
    return support_grid[distance.argsort()[:crop_points]].contiguous()


def construct(grid_coord: torch.Tensor) -> None:
    device = grid_coord.device
    grid_range = torch.tensor([0, -1, 1]).long().to(device)
    grid_offset = torch.meshgrid(
        grid_range, grid_range, grid_range, indexing="ij"
    )
    grid_offset = torch.stack(grid_offset, dim=-1).view(-1, 3)
    batch = torch.zeros(len(grid_coord), device=device, dtype=torch.long)
    key = xyz2key(
        grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2], batch
    )
    order = torch.argsort(key)
    key = key[order]
    ordered_grid = grid_coord[order]
    inverse_order = torch.zeros(len(ordered_grid), device=device, dtype=torch.long)
    inverse_order[order] = torch.arange(
        len(ordered_grid), device=device, dtype=torch.long
    )
    key_down = key >> 6
    grid_down = ordered_grid >> 2
    unique_key, child_to_parent, count = torch.unique_consecutive(
        key_down, return_inverse=True, return_counts=True
    )
    steps = functional.pad(torch.cumsum(count, dim=0), (1, 0))
    grid_down = grid_down[steps[:len(unique_key)]]
    neighbor_grid = (
        grid_down.unsqueeze(1) + grid_offset.unsqueeze(0)
    ).view(-1, 3)
    neighbor_key = xyz2key(
        neighbor_grid[:, 0], neighbor_grid[:, 1], neighbor_grid[:, 2],
        batch[steps[:len(unique_key)]].unsqueeze(1).repeat((1, 27)).view(-1),
    )
    full_neighbor, inverse = torch.unique(
        torch.cat([neighbor_key[::27], neighbor_key]), return_inverse=True
    )
    valid = inverse[:len(unique_key)]
    lookup = -torch.ones(len(full_neighbor), dtype=torch.long, device=device)
    lookup[valid] = torch.arange(len(unique_key), dtype=torch.long, device=device)
    neighbors = lookup[inverse[len(unique_key):]].reshape(-1, 27)
    functional.pad(
        torch.cumsum(functional.pad(count, (0, 1))[neighbors], dim=1),
        (1, 0),
    )
    torch.zeros((len(ordered_grid), 32), device=device, dtype=torch.int32)
    # Keep references live through the synchronization, matching the wrapper.
    _ = inverse_order, child_to_parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    grid = prepare_grid(args.room, 250_000)
    timings = []
    for iteration in range(args.warmups + args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        construct(grid)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        if iteration >= args.warmups:
            timings.append(elapsed)
    print({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "points": len(grid),
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "min_ms": min(timings), "max_ms": max(timings),
    })


if __name__ == "__main__":
    main()
