#!/usr/bin/env python3
"""Compare fixed/adaptive recall by fixed-alpha candidate-count region."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path


EXPECTED_K = (8, 16, 24, 32, 48, 64)


def per_query_recall(torch, exact, predicted):
    predicted = predicted.long().sort(1).values
    positions = torch.searchsorted(predicted, exact.long())
    safe = positions.clamp_max(predicted.shape[1] - 1)
    return (
        (positions < predicted.shape[1])
        & (predicted.gather(1, safe) == exact)
    ).sum(1).float() / exact.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k", nargs="+", type=int, default=EXPECTED_K)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    shift = args.alpha.bit_length() - 1
    if 1 << shift != args.alpha:
        raise SystemExit("alpha must be a positive power of two")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import Cukd.CuFun as cukd
    from benchmark_ablation import co_tenant_snapshot, git_identity, sha256
    from benchmark_s3dis import gpu_info, load_xyz, prepare, room_paths
    try:
        from FlashKNN import (
            AdaptiveNeighborhoodFlashKNN, FlashKNN,
            build_adaptive_octree, xyz2key,
        )
    except ImportError:
        from functions import (
            AdaptiveNeighborhoodFlashKNN, FlashKNN,
            build_adaptive_octree, xyz2key,
        )

    repo = Path(__file__).resolve().parents[1]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    aggregate = {
        str(k): {
            name: {"queries": 0, "fixed_recall_sum": 0.0,
                   "adaptive_recall_sum": 0.0}
            for name in ("below_2k", "within_2k_8k", "above_8k")
        }
        for k in args.k
    }
    rooms = []
    for path in room_paths(args.data_root.resolve()):
        coord = load_xyz(torch, path)
        sample = prepare(
            torch, xyz2key, coord, path, "pre", "sample_part",
            args.voxel_size, args.crop_points, None, generator,
        )
        if sample is None:
            continue
        support, grid, _, crop_center = sample
        batch = torch.zeros(len(support), device="cuda", dtype=torch.long)
        hierarchy = build_adaptive_octree(grid, batch)
        fixed_level = hierarchy.levels[shift]
        sorted_candidates = fixed_level.candidate_counts[
            fixed_level.point_to_node.long()
        ]
        fixed_candidates = sorted_candidates[hierarchy.inverse_order]
        rooms.append({
            "room": path.relative_to(args.data_root.resolve()).as_posix(),
            "crop_center": crop_center,
        })
        for k in args.k:
            exact = torch.empty((len(support), k), device="cuda", dtype=torch.int32)
            exact_distance = torch.empty_like(exact, dtype=torch.float32)
            cukd.CukdKnnQueryTorch(
                support, support, k, exact, exact_distance,
                torch.zeros(2), True,
            )
            fixed = FlashKNN(num_nbr=k, num_down=shift, debug=False).query(
                grid, batch, support
            )
            adaptive = AdaptiveNeighborhoodFlashKNN(
                num_nbr=k, min_candidates_factor=2,
                max_candidates_factor=8,
            ).query(grid, batch, support)
            fixed_recall = per_query_recall(torch, exact, fixed)
            adaptive_recall = per_query_recall(torch, exact, adaptive)
            masks = {
                "below_2k": fixed_candidates < 2 * k,
                "within_2k_8k": (
                    (fixed_candidates >= 2 * k)
                    & (fixed_candidates <= 8 * k)
                ),
                "above_8k": fixed_candidates > 8 * k,
            }
            for name, mask in masks.items():
                count = int(mask.sum())
                bucket = aggregate[str(k)][name]
                bucket["queries"] += count
                bucket["fixed_recall_sum"] += float(fixed_recall[mask].sum())
                bucket["adaptive_recall_sum"] += float(
                    adaptive_recall[mask].sum()
                )
            del exact, exact_distance, fixed, adaptive
        del coord, support, grid, batch, hierarchy, fixed_candidates
        torch.cuda.empty_cache()

    summary = {}
    for k, buckets in aggregate.items():
        summary[k] = {}
        for name, bucket in buckets.items():
            count = bucket["queries"]
            fixed_mean = bucket["fixed_recall_sum"] / count if count else None
            adaptive_mean = (
                bucket["adaptive_recall_sum"] / count if count else None
            )
            summary[k][name] = {
                "queries": count,
                "query_fraction": count / (len(rooms) * args.crop_points),
                "fixed_recall": fixed_mean,
                "adaptive_recall": adaptive_mean,
                "adaptive_minus_fixed": (
                    adaptive_mean - fixed_mean if count else None
                ),
            }

    payload = {
        "metadata": {
            "dataset": "S3DIS", "scope": "sample_part", "mode": "pre",
            "gpu": gpu_info(torch, args.gpu),
            "physical_gpu_index": args.gpu,
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "git": git_identity(repo),
            "source_sha256": {
                name: sha256(repo / name) for name in (
                    "FlashKNN/functions/adaptive_octree.py",
                    "FlashKNN/functions/FlashKnnWrapper.py",
                    "analysis/analyze_adaptive_sparse_bins.py",
                )
            },
            "alpha": args.alpha, "octree_shift": shift,
            "adaptive_min_factor": 2, "adaptive_max_factor": 8,
            "voxel_size_m": args.voxel_size,
            "crop_points": args.crop_points, "seed": args.seed,
            "k": args.k, "co_tenant_end": co_tenant_snapshot(),
        },
        "summary": summary,
        "rooms": rooms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved {len(rooms)} rooms to {args.output}")


if __name__ == "__main__":
    main()
