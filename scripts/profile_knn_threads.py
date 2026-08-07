#!/usr/bin/env python3
"""Launch one paper-scale kNN query for Nsight Compute microarchitecture metrics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    """Profile one backend on a deterministic 250k-point S3DIS crop."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("cukd", "flash-smps", "flash-gmss"), required=True
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--room-index", type=int, default=0)
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "Query"))
    sys.path.insert(0, str(repo / "FlashKNN"))

    import torch
    import Cukd.CuFun as cukd
    from benchmark_s3dis import load_xyz, prepare, room_paths
    from FlashKNN import FlashKNN, xyz2key

    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    generator = torch.Generator(device="cpu").manual_seed(47)
    selected = None
    selected_path = None
    for path in room_paths(args.data_root.resolve())[args.room_index:]:
        coord = load_xyz(torch, path)
        sample = prepare(
            torch, xyz2key, coord, path, "pre", "sample_part",
            0.02, 250_000, None, generator,
        )
        if sample is not None:
            selected = sample
            selected_path = path
            break
    if selected is None or selected_path is None:
        raise RuntimeError("No S3DIS room contains 250,000 voxelized points")

    support, grid, _, _ = selected
    torch.cuda.synchronize()
    if args.backend == "cukd":
        indices = torch.empty(
            (len(support), args.k), device="cuda", dtype=torch.int32
        )
        distances = torch.empty_like(indices, dtype=torch.float32)
        cukd.CukdKnnQueryTorch(
            support, support, args.k, indices, distances, torch.zeros(2), True
        )
    else:
        batch = torch.zeros(len(support), device="cuda", dtype=torch.long)
        memory_mode, sorting_mode = {
            "flash-smps": ("SM", "PS"),
            "flash-gmss": ("GM", "SS"),
        }[args.backend]
        knn = FlashKNN(num_nbr=args.k, num_down=2, debug=False)
        knn.query(
            grid, batch, support,
            memory_mode=memory_mode, sorting_mode=sorting_mode,
        )
    torch.cuda.synchronize()
    print({
        "backend": args.backend,
        "room": selected_path.relative_to(args.data_root.resolve()).as_posix(),
        "support": len(support),
        "query": len(support),
        "k": args.k,
        "gpu": torch.cuda.get_device_name(0),
    })


if __name__ == "__main__":
    main()
