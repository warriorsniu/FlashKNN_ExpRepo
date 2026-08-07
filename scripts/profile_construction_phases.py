#!/usr/bin/env python3
"""Profile FlashKNN construction phases and neighbor-lookup alternatives."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


def main() -> None:
    """Run phase-level construction benchmarks on one deterministic S3DIS crop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--warmups", default=5, type=int)
    parser.add_argument("--repeats", default=30, type=int)
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "Query"))
    sys.path.insert(0, str(repo / "FlashKNN"))

    import torch
    import torch.nn.functional as functional
    from benchmark_s3dis import load_xyz, prepare, room_paths
    from FlashKNN import xyz2key
    from FlashKNN.CuFun import (
        FlashKNN_Morton_Encode,
        FlashKNN_Morton_Neighbor_Keys,
    )

    generator = torch.Generator(device="cpu").manual_seed(47)
    selected = None
    selected_path = None
    for path in room_paths(args.data_root.resolve()):
        selected = prepare(
            torch, xyz2key, load_xyz(torch, path), path, "pre", "sample_part",
            0.02, 250_000, None, generator,
        )
        if selected is not None:
            selected_path = path
            break
    if selected is None or selected_path is None:
        raise RuntimeError("No room contains 250,000 voxelized points")
    _, grid_coord, _, _ = selected
    batch_idx = torch.zeros(len(grid_coord), device="cuda", dtype=torch.long)
    offsets = torch.tensor([0, -1, 1], device="cuda", dtype=torch.long)
    offsets = torch.stack(
        torch.meshgrid(offsets, offsets, offsets, indexing="ij"), dim=-1
    ).view(-1, 3)

    def run(lookup_mode: str) -> tuple[dict[str, float], tuple[torch.Tensor, ...]]:
        """Execute construction once and return CUDA-event phase times."""
        names = (
            "encode", "sort", "reorder_inverse", "parent_unique", "neighbor_keys",
            "neighbor_lookup", "neighbor_counts", "output_alloc",
        )
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]
        events[0].record()
        if lookup_mode == "fused":
            key = FlashKNN_Morton_Encode(grid_coord, batch_idx)
        else:
            key = xyz2key(
                grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2], batch_idx
            )
        events[1].record()
        order = torch.argsort(key)
        events[2].record()
        key = key[order]
        ordered_grid = grid_coord[order]
        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(len(order), device=order.device)
        events[3].record()
        key_down = key >> 6
        unique_key, child_to_parent, count = torch.unique_consecutive(
            key_down, return_inverse=True, return_counts=True
        )
        steps = functional.pad(torch.cumsum(count, dim=0), (1, 0))
        grid_down = (ordered_grid >> 2)[steps[:-1]]
        events[4].record()
        parent_batch = batch_idx[steps[:-1]].contiguous()
        if lookup_mode == "fused":
            neighbor_key = FlashKNN_Morton_Neighbor_Keys(
                grid_down.contiguous(), parent_batch
            )
        else:
            neighbor_grid = (
                grid_down[:, None, :] + offsets[None, :, :]
            ).reshape(-1, 3)
            neighbor_key = xyz2key(
                neighbor_grid[:, 0], neighbor_grid[:, 1], neighbor_grid[:, 2],
                parent_batch.unsqueeze(1).expand(-1, 27).reshape(-1),
            )
        events[5].record()
        if lookup_mode == "unique":
            all_keys, inverse = torch.unique(
                torch.cat((neighbor_key[::27], neighbor_key)), return_inverse=True
            )
            valid = inverse[:len(unique_key)]
            lookup = torch.full(
                (len(all_keys),), -1, dtype=torch.long, device=grid_coord.device
            )
            lookup[valid] = torch.arange(len(unique_key), device=grid_coord.device)
            neighbors = lookup[inverse[len(unique_key):]].reshape(-1, 27)
        elif lookup_mode in ("searchsorted", "fused"):
            parent_keys = neighbor_key[::27].contiguous()
            positions = torch.searchsorted(parent_keys, neighbor_key)
            safe_positions = positions.clamp_max(len(parent_keys) - 1)
            present = (positions < len(parent_keys)) & (
                parent_keys[safe_positions] == neighbor_key
            )
            neighbors = torch.where(present, safe_positions, -1).reshape(-1, 27)
        else:
            raise ValueError(f"Unknown lookup mode: {lookup_mode}")
        events[6].record()
        padded_count = functional.pad(count, (0, 1))
        neighbor_counts = functional.pad(
            torch.cumsum(padded_count[neighbors], dim=1), (1, 0)
        )
        events[7].record()
        output = torch.zeros(
            (len(ordered_grid), 32), device=grid_coord.device, dtype=torch.int32
        )
        events[8].record()
        torch.cuda.synchronize()
        timings = {
            name: events[index].elapsed_time(events[index + 1])
            for index, name in enumerate(names)
        }
        return timings, (
            order, inverse_order, child_to_parent, count, steps, neighbors,
            neighbor_counts, output,
        )

    results: dict[str, dict[str, list[float]]] = {
        mode: defaultdict(list) for mode in ("unique", "searchsorted", "fused")
    }
    references = None
    for iteration in range(args.warmups + args.repeats):
        for mode in ("unique", "searchsorted", "fused"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            timings, tensors = run(mode)
            wall_ms = (time.perf_counter() - start) * 1000.0
            if iteration == args.warmups and mode == "unique":
                references = tuple(t.detach().clone() for t in tensors[:-1])
            if iteration == args.warmups and mode in ("searchsorted", "fused"):
                if references is None:
                    raise RuntimeError("Missing unique-mode reference")
                for index, (reference, actual) in enumerate(zip(references, tensors[:-1])):
                    if not torch.equal(reference, actual):
                        raise RuntimeError(f"Lookup alternatives differ at tensor {index}")
            if iteration >= args.warmups:
                for name, value in timings.items():
                    results[mode][name].append(value)
                results[mode]["cuda_total"].append(sum(timings.values()))
                results[mode]["wall_total"].append(wall_ms)

    print({
        "room": selected_path.relative_to(args.data_root.resolve()).as_posix(),
        "points": len(grid_coord),
        "parents": len(references[3]) if references is not None else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    })
    for mode, metrics in results.items():
        print(mode)
        for name, values in metrics.items():
            print(f"  {name}: median={statistics.median(values):.4f} ms "
                  f"mean={statistics.mean(values):.4f} ms")


if __name__ == "__main__":
    main()
