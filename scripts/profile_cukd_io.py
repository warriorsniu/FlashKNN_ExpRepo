#!/usr/bin/env python3
"""Launch one paper-scale cudaKDTree k=32 query for Nsight Compute."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--room-index", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "Query"))
    sys.path.insert(0, str(repo / "FlashKNN"))

    import torch
    import Cukd.CuFun as cukd
    from benchmark_s3dis import load_xyz, prepare, room_paths
    from FlashKNN import xyz2key

    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    generator = torch.Generator(device="cpu").manual_seed(47)
    paths = room_paths(args.data_root.resolve())
    selected = None
    selected_path = None
    for path in paths[args.room_index:]:
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
    support, _, _, _ = selected
    query = support
    k = 32
    indices = torch.empty((len(query), k), device="cuda", dtype=torch.int32)
    distances = torch.empty((len(query), k), device="cuda", dtype=torch.float32)
    timing = torch.zeros(2)
    torch.cuda.synchronize()
    cukd.CukdKnnQueryTorch(
        support, query, k, indices, distances, timing, True
    )
    torch.cuda.synchronize()
    print({
        "room": selected_path.relative_to(args.data_root.resolve()).as_posix(),
        "support": len(support), "query": len(query), "k": k,
        "construction_s": float(timing[0]), "query_s": float(timing[1]),
        "gpu": torch.cuda.get_device_name(0),
    })


if __name__ == "__main__":
    main()
