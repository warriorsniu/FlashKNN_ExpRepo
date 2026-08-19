#!/usr/bin/env python3
"""Measure the per-query 3x3x3 candidate count for fixed alpha."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    if args.alpha <= 0 or args.alpha & (args.alpha - 1):
        raise SystemExit("alpha must be a positive power of two")
    shift = args.alpha.bit_length() - 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from benchmark_ablation import co_tenant_snapshot, git_identity, sha256
    from benchmark_s3dis import gpu_info, load_xyz, prepare, room_paths
    try:
        from FlashKNN import build_adaptive_octree, xyz2key
    except ImportError:
        from functions import build_adaptive_octree, xyz2key

    repo = Path(__file__).resolve().parents[1]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    records = []
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
        level = hierarchy.levels[shift]
        per_query = level.candidate_counts[
            level.point_to_node.long()
        ].float()
        records.append({
            "room": path.relative_to(args.data_root.resolve()).as_posix(),
            "crop_center": crop_center,
            "num_support": len(support),
            "candidate_mean": float(per_query.mean()),
            "candidate_median": float(per_query.median()),
            "candidate_p05": float(torch.quantile(per_query, 0.05)),
            "candidate_p25": float(torch.quantile(per_query, 0.25)),
            "candidate_p75": float(torch.quantile(per_query, 0.75)),
            "candidate_p95": float(torch.quantile(per_query, 0.95)),
            "candidate_min": int(per_query.min()),
            "candidate_max": int(per_query.max()),
        })
        del coord, support, grid, batch, hierarchy, per_query
        torch.cuda.empty_cache()

    payload = {
        "metadata": {
            "dataset": "S3DIS",
            "scope": "sample_part",
            "mode": "pre",
            "gpu": gpu_info(torch, args.gpu),
            "physical_gpu_index": args.gpu,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "git": git_identity(repo),
            "source_sha256": {
                "FlashKNN/functions/adaptive_octree.py": sha256(
                    repo / "FlashKNN/functions/adaptive_octree.py"
                ),
                "analysis/analyze_fixed_candidate_counts.py": sha256(
                    Path(__file__).resolve()
                ),
            },
            "alpha": args.alpha,
            "octree_shift": shift,
            "voxel_size_m": args.voxel_size,
            "crop_points": args.crop_points,
            "seed": args.seed,
            "manifold_prediction": (3 * args.alpha) ** 2,
            "co_tenant_end": co_tenant_snapshot(),
        },
        "summary": {
            field: statistics.fmean(record[field] for record in records)
            for field in (
                "candidate_mean", "candidate_median", "candidate_p05",
                "candidate_p25", "candidate_p75", "candidate_p95",
            )
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved {len(records)} rooms to {args.output}")


if __name__ == "__main__":
    main()
