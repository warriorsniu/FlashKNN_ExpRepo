#!/usr/bin/env python3
"""Benchmark fixed and octree-adaptive 3x3x3 FlashKNN neighborhoods."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from pathlib import Path
from typing import Any

from benchmark_ablation import co_tenant_snapshot, git_identity, sha256
from benchmark_s3dis import atomic_json, gpu_info, load_xyz, prepare, room_paths


EXPECTED_K = (8, 16, 24, 32, 48, 64)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k", nargs="+", type=int, default=EXPECTED_K)
    parser.add_argument("--adaptive-min-factor", type=int, default=2)
    parser.add_argument("--adaptive-max-factor", type=int, default=8)
    parser.add_argument("--num-down", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250_000)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def index_set_recall(torch: Any, exact: Any, predicted: Any) -> dict[str, float]:
    predicted = predicted.long().sort(1).values
    positions = torch.searchsorted(predicted, exact.long())
    safe = positions.clamp_max(predicted.shape[1] - 1)
    per_query = (
        (positions < predicted.shape[1])
        & (predicted.gather(1, safe) == exact)
    ).sum(1).float() / exact.shape[1]
    return {
        "mean": float(per_query.mean()),
        "minimum": float(per_query.min()),
        "p01": float(torch.quantile(per_query, 0.01)),
        "p05": float(torch.quantile(per_query, 0.05)),
    }


def distance_check(torch: Any, xyz: Any, indices: Any, distances: Any) -> dict[str, Any]:
    actual = ((xyz[indices.long()] - xyz[:, None, :]) ** 2).sum(2)
    delta = (actual - distances).abs()
    return {
        "allclose": bool(torch.allclose(actual, distances, rtol=1e-5, atol=1e-7)),
        "max_abs_diff": float(delta.max()),
    }


def main() -> None:
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import Cukd.CuFun as cukd
    try:
        from FlashKNN import AdaptiveNeighborhoodFlashKNN, FlashKNN, xyz2key
        import FlashKNN.CuFun as flash_cuda
    except ImportError:
        from functions import AdaptiveNeighborhoodFlashKNN, FlashKNN, xyz2key
        from functions import CuFun as flash_cuda

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if any(k not in EXPECTED_K for k in args.k):
        raise SystemExit(f"benchmark supports k in {EXPECTED_K}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    crop_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    paths = room_paths(args.data_root.resolve())
    if args.max_samples is not None:
        paths = paths[:args.max_samples]
    repo = Path(__file__).resolve().parents[1]
    variants = (
        "fixed_3x3x3",
        f"adaptive_{args.adaptive_min_factor}k_{args.adaptive_max_factor}k",
        "cuda_kdtree_exact",
    )
    payload: dict[str, Any] = {
        "metadata": {
            "dataset": "S3DIS",
            "scope": "sample_part",
            "mode": "pre",
            "gpu": gpu_info(torch, args.gpu),
            "physical_gpu_index": args.gpu,
            "pid": os.getpid(),
            "benchmark_pids": [os.getpid()],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "git": git_identity(repo),
            "source_sha256": {
                relative: sha256(repo / relative)
                for relative in (
                    "FlashKNN/functions/adaptive_octree.py",
                    "FlashKNN/functions/FlashKnnWrapper.py",
                    "Query/benchmark_adaptive_neighborhood.py",
                )
            },
            "extension": {
                "path": str(Path(flash_cuda.__file__).resolve()),
                "sha256": sha256(Path(flash_cuda.__file__).resolve()),
            },
            "voxel_size_m": args.voxel_size,
            "crop_points": args.crop_points,
            "num_down_fixed": args.num_down,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seed": args.seed,
            "k": list(args.k),
            "variants": list(variants),
            "adaptive_rule": (
                "start at the coarsest occupied Morton level; descend while "
                "the current 3x3x3 candidate count exceeds "
                f"{args.adaptive_max_factor}k; use the child only when it "
                f"contains at least {args.adaptive_min_factor}k candidates; "
                "no geometric guard and no post-query retry"
            ),
            "adaptive_min_candidates_factor": args.adaptive_min_factor,
            "adaptive_max_candidates_factor": args.adaptive_max_factor,
            "kernel_abi": (
                "unchanged production FlashKNN_Query_Dynamic_Load inputs; "
                "selected levels flattened into compatible query/support "
                "descriptors before one query-kernel call"
            ),
            "timing_boundary": (
                "CUDA-ready 250k crop; construction, adaptive level selection, "
                "compatible-input construction, and query recorded separately; "
                "total includes all four"
            ),
            "variant_ordering": "balanced cyclic rotation by room and k",
            "co_tenant_start": co_tenant_snapshot(),
        },
        "records": [],
    }

    eligible_room_index = 0
    for path in paths:
        coord = load_xyz(torch, path)
        sample = prepare(
            torch, xyz2key, coord, path, "pre", "sample_part",
            args.voxel_size, args.crop_points, None, crop_generator,
        )
        if sample is None:
            del coord
            continue
        support, grid, _, crop_center = sample
        batch = torch.zeros(len(support), device="cuda", dtype=torch.long)
        room = path.relative_to(args.data_root.resolve()).as_posix()
        for k_index, k in enumerate(args.k):
            exact_indices = torch.empty(
                (len(support), k), device="cuda", dtype=torch.int32
            )
            exact_distances = torch.empty_like(exact_indices, dtype=torch.float32)
            cukd.CukdKnnQueryTorch(
                support, support, k, exact_indices, exact_distances,
                torch.zeros(2), True,
            )
            torch.cuda.synchronize()

            rotation = (eligible_room_index + k_index) % len(variants)
            order = variants[rotation:] + variants[:rotation]
            record: dict[str, Any] = {
                "room": room,
                "k": k,
                "num_support": len(support),
                "crop_center": crop_center,
                "measurement_order": list(order),
                "variants": {},
            }
            for name in order:
                samples: list[dict[str, float]] = []
                predicted = distances = None
                model: Any
                if name == "fixed_3x3x3":
                    model = FlashKNN(num_nbr=k, num_down=args.num_down, debug=True)
                elif name.startswith("adaptive_"):
                    model = AdaptiveNeighborhoodFlashKNN(
                        num_nbr=k,
                        min_candidates_factor=args.adaptive_min_factor,
                        max_candidates_factor=args.adaptive_max_factor,
                    )
                else:
                    model = None
                baseline_allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                for repeat in range(args.warmups + args.repeats):
                    if name == "cuda_kdtree_exact":
                        exact_timing = torch.zeros(2)
                        cukd.CukdKnnQueryTorch(
                            support, support, k,
                            exact_indices, exact_distances,
                            exact_timing, True,
                        )
                        torch.cuda.synchronize()
                        predicted = exact_indices
                        sample_timing = {
                            "construction_ms": float(exact_timing[0]) * 1000.0,
                            "selection_ms": 0.0,
                            "compatibility_ms": 0.0,
                            "query_ms": float(exact_timing[1]) * 1000.0,
                            "total_ms": float(exact_timing.sum()) * 1000.0,
                        }
                    elif name == "fixed_3x3x3":
                        predicted = model.query(grid, batch, support)
                        timing = model.time_list[-1]
                        sample_timing = {
                            "construction_ms": timing["预处理耗时"] * 1000.0,
                            "selection_ms": 0.0,
                            "compatibility_ms": 0.0,
                            "query_ms": timing["查询耗时"] * 1000.0,
                        }
                        sample_timing["total_ms"] = (
                            sample_timing["construction_ms"]
                            + sample_timing["compatibility_ms"]
                            + sample_timing["query_ms"]
                        )
                    else:
                        predicted, distances = model.query(
                            grid, batch, support, return_distances=True
                        )
                        sample_timing = {
                            field: float(model.last_stats[field])
                            for field in (
                                "construction_ms", "selection_ms",
                                "compatibility_ms", "query_ms", "total_ms",
                            )
                        }
                    if repeat >= args.warmups:
                        samples.append(sample_timing)
                assert predicted is not None
                result: dict[str, Any] = {
                    "timings": samples,
                    "mean_ms": {
                        field: statistics.fmean(item[field] for item in samples)
                        for field in (
                            "construction_ms", "selection_ms",
                            "compatibility_ms", "query_ms", "total_ms",
                        )
                    },
                    "peak_incremental_allocated_mib": max(
                        0.0,
                        (
                            torch.cuda.max_memory_allocated()
                            - baseline_allocated
                        ) / (1024.0 * 1024.0),
                    ),
                    "recall_vs_cukd": index_set_recall(
                        torch, exact_indices, predicted
                    ),
                }
                if name == "cuda_kdtree_exact":
                    result["exact"] = True
                elif name.startswith("adaptive_"):
                    assert distances is not None
                    result["distance_check"] = distance_check(
                        torch, support, predicted, distances
                    )
                    result["octree"] = {
                        key: model.last_stats[key]
                        for key in (
                            "octree_levels", "level_node_counts",
                            "selected_levels", "min_candidates_factor",
                            "max_candidates_factor", "selection_band_points",
                            "query_kernel_launches",
                            "compatible_group_count",
                            "compatible_support_descriptor_count",
                            "compatible_support_copy_count",
                            "compatible_point_count", "compatible_point_ratio",
                        )
                    }
                record["variants"][name] = result
                print(
                    f"{room} k={k} {name} "
                    f"total={result['mean_ms']['total_ms']:.4f} ms "
                    f"recall={result['recall_vs_cukd']['mean']:.6f}",
                    flush=True,
                )
            payload["records"].append(record)
            atomic_json(args.output, payload)
            del exact_indices, exact_distances
        del support, grid, batch, coord
        torch.cuda.empty_cache()
        eligible_room_index += 1

    payload["metadata"]["co_tenant_end"] = co_tenant_snapshot()
    atomic_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output}")


if __name__ == "__main__":
    main()
