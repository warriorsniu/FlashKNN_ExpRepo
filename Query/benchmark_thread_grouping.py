#!/usr/bin/env python3
"""Benchmark fixed and adaptive FlashKNN thread-group strategies."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from pathlib import Path
from typing import Any

from benchmark_ablation import co_tenant_snapshot, git_identity, sha256
from benchmark_s3dis import (
    atomic_json,
    gpu_info,
    load_xyz,
    normalized_timings,
    prepare,
    room_paths,
)


VARIANTS: dict[str, int | None] = {
    "adaptive": None,
    "fixed_8": 8,
    "fixed_16": 16,
    "fixed_32": 32,
}
EXPECTED_K = (8, 16, 24, 32, 48, 64)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k", nargs="+", type=int, default=EXPECTED_K)
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS),
        default=tuple(VARIANTS),
    )
    parser.add_argument("--num-down", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250_000)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def mean_query_ms(timings: list[dict[str, float]]) -> float:
    return statistics.fmean(item["query_s"] for item in timings) * 1000.0


def sorted_squared_distances(torch: Any, support: Any, indices: Any) -> Any:
    neighbors = support[indices.long()]
    return ((neighbors - support[:, None, :]) ** 2).sum(2).sort(1).values


def index_set_recall(torch: Any, exact: Any, predicted: Any) -> dict[str, float]:
    """Count each exact neighbor at most once, even if prediction repeats."""
    exact = exact.long()
    predicted = predicted.long().sort(1).values
    positions = torch.searchsorted(predicted, exact)
    safe_positions = positions.clamp_max(predicted.shape[1] - 1)
    matched = (positions < predicted.shape[1]) & (
        predicted.gather(1, safe_positions) == exact
    )
    per_query = matched.sum(1).float() / exact.shape[1]
    return {"mean": float(per_query.mean()), "minimum": float(per_query.min())}


def main() -> None:
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import Cukd.CuFun as cukd
    try:
        from FlashKNN import FlashKNN, xyz2key
        import FlashKNN.CuFun as flash_cuda
    except ImportError:
        from functions import FlashKNN, xyz2key
        from functions import CuFun as flash_cuda

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if any(k not in EXPECTED_K for k in args.k):
        raise SystemExit(f"Thread grouping benchmark supports k in {EXPECTED_K}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    crop_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    paths = room_paths(args.data_root.resolve())
    if args.max_samples is not None:
        paths = paths[:args.max_samples]

    repo = Path(__file__).resolve().parents[1]
    selected_variants = list(dict.fromkeys(args.variants))
    if "adaptive" not in selected_variants:
        raise SystemExit(
            "adaptive must be selected so fixed strategies can be checked "
            "against it"
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
                    "FlashKNN/csrc/api.cpp",
                    "FlashKNN/csrc/flash_knn_query.h",
                    "FlashKNN/csrc/flash_knn_bitonic_top_p.cuh",
                    "FlashKNN/csrc/flash_knn_query_dynamic_load.cu",
                    "FlashKNN/functions/FlashKnnWrapper.py",
                    "Query/benchmark_ablation.py",
                    "Query/benchmark_thread_grouping.py",
                )
            },
            "extension": {
                "path": str(Path(flash_cuda.__file__).resolve()),
                "sha256": sha256(Path(flash_cuda.__file__).resolve()),
            },
            "voxel_size_m": args.voxel_size,
            "crop_points": args.crop_points,
            "num_down": args.num_down,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seed": args.seed,
            "k": list(args.k),
            "variants": {
                name: {"thread_group_size": VARIANTS[name]}
                for name in selected_variants
            },
            "adaptive_rule": "8 threads for k<=8; 16 for 8<k<=16; 32 for k>16",
            "variant_ordering": (
                "balanced cyclic rotation by room and k; adaptive reference "
                "query is outside timed samples"
            ),
            "sorting_revision": "generated_bitonic_top_p",
            "recall_definition": (
                "row-wise exact-index set recall versus CUKD; each exact "
                "neighbor is counted at most once"
            ),
            "strategy_equivalence": {
                "reference": "adaptive",
                "criterion": "sorted squared neighbor distances per query",
                "rtol": 1e-6,
                "atol": 1e-8,
                "note": (
                    "Index-set recall is also reported, but equal-distance "
                    "neighbors may have different valid indices."
                ),
            },
            "timing_boundary": (
                "CUDA-ready 250k crop; construction and query recorded "
                "separately; excludes file I/O, voxelization, crop and H2D"
            ),
            "co_tenant_start": co_tenant_snapshot(),
        },
        "records": [],
    }
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        identity_fields = (
            "torch", "torch_cuda", "voxel_size_m", "crop_points",
            "num_down", "warmups", "repeats", "seed", "adaptive_rule",
            "variant_ordering", "sorting_revision", "strategy_equivalence",
            "source_sha256", "recall_definition", "extension",
        )
        changed = {
            field: (
                previous.get("metadata", {}).get(field),
                payload["metadata"].get(field),
            )
            for field in identity_fields
            if previous.get("metadata", {}).get(field)
            != payload["metadata"].get(field)
        }
        old_uuid = previous.get("metadata", {}).get("gpu", {}).get("uuid")
        new_uuid = payload["metadata"].get("gpu", {}).get("uuid")
        if old_uuid != new_uuid:
            changed["gpu.uuid"] = (old_uuid, new_uuid)
        if changed:
            raise SystemExit(
                f"Refusing to resume incompatible output {args.output}: {changed}"
            )
        payload = previous
        benchmark_pids = payload["metadata"].setdefault(
            "benchmark_pids", [payload["metadata"].get("pid")]
        )
        if os.getpid() not in benchmark_pids:
            benchmark_pids.append(os.getpid())
        payload["metadata"]["pid"] = os.getpid()
        payload["metadata"]["co_tenant_resume"] = co_tenant_snapshot()

    record_positions = {
        (record["room"], int(record["k"])): position
        for position, record in enumerate(payload["records"])
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
            key = (room, k)
            if key in record_positions:
                record = payload["records"][record_positions[key]]
            else:
                record = {
                    "room": room,
                    "k": k,
                    "num_support": len(support),
                    "num_query": len(support),
                    "crop_center": crop_center,
                    "variants": {},
                }
                record_positions[key] = len(payload["records"])
                payload["records"].append(record)

            missing = [
                name for name in selected_variants
                if name not in record.get("variants", {})
            ]
            if not missing:
                continue

            rotation = (eligible_room_index + k_index) % len(selected_variants)
            measurement_order = (
                selected_variants[rotation:] + selected_variants[:rotation]
            )
            previous_order = record.get("measurement_order")
            if previous_order is not None and previous_order != measurement_order:
                raise RuntimeError(
                    f"{key} measurement order changed: "
                    f"{previous_order} != {measurement_order}"
                )
            record["measurement_order"] = measurement_order

            exact_indices = torch.empty(
                (len(support), k), device="cuda", dtype=torch.int32,
            )
            exact_distances = torch.empty_like(
                exact_indices, dtype=torch.float32,
            )
            cukd.CukdKnnQueryTorch(
                support, support, k, exact_indices, exact_distances,
                torch.zeros(2), True,
            )
            torch.cuda.synchronize()

            reference_knn = FlashKNN(
                num_nbr=k, num_down=args.num_down, debug=False,
            )
            reference_indices = reference_knn.query(
                grid, batch, support,
                memory_mode="SM",
                sorting_mode="PS",
                candidate_mode="register",
                enable_skip=True,
                thread_group_size=None,
            )
            adaptive_distances = sorted_squared_distances(
                torch, support, reference_indices,
            )
            for name in measurement_order:
                if name not in missing:
                    continue
                group_size = VARIANTS[name]
                knn = FlashKNN(num_nbr=k, num_down=args.num_down, debug=True)
                predicted = None
                for _ in range(args.warmups + args.repeats):
                    predicted = knn.query(
                        grid, batch, support,
                        memory_mode="SM",
                        sorting_mode="PS",
                        candidate_mode="register",
                        enable_skip=True,
                        thread_group_size=group_size,
                    )
                timings = normalized_timings(knn.time_list[args.warmups:])
                if predicted is None:
                    raise RuntimeError(f"{name} produced no indices")
                predicted_distances = sorted_squared_distances(
                    torch, support, predicted,
                )
                distance_close = torch.isclose(
                    predicted_distances, adaptive_distances,
                    rtol=1e-6, atol=1e-8,
                )
                differing_distance_rows = (~distance_close.all(1)).sum()
                distance_delta = (
                    predicted_distances - adaptive_distances
                ).abs()
                record.setdefault("variants", {})[name] = {
                    "configuration": {"thread_group_size": group_size},
                    "timings": timings,
                    "recall_vs_cukd": index_set_recall(
                        torch, exact_indices, predicted,
                    ),
                    "equivalence_vs_adaptive": {
                        "squared_distance_allclose": bool(
                            distance_close.all().item()
                        ),
                        "squared_distance_differing_queries": int(
                            differing_distance_rows.item()
                        ),
                        "squared_distance_max_abs_diff": float(
                            distance_delta.max().item()
                        ),
                    },
                }
                atomic_json(args.output, payload)
                print(
                    f"{room} k={k} {name} "
                    f"query={mean_query_ms(timings):.4f} ms "
                    f"recall={record['variants'][name]['recall_vs_cukd']['mean']:.6f}",
                    flush=True,
                )
            del exact_indices, exact_distances
        del support, grid, batch, coord
        torch.cuda.empty_cache()
        eligible_room_index += 1

    payload["metadata"]["co_tenant_end"] = co_tenant_snapshot()
    atomic_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output}")


if __name__ == "__main__":
    main()
