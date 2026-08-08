#!/usr/bin/env python3
"""Benchmark Pointcept's CUDA ball query against exact kNN on S3DIS crops."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def arguments() -> argparse.Namespace:
    """Parse the matched-radius ball-query benchmark configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", nargs="+", choices=("pre", "post"), default=("pre", "post"))
    parser.add_argument("--k", nargs="+", type=int, default=(24, 32, 48))
    parser.add_argument("--percentile", nargs="+", type=float, default=(0.9,))
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250_000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--calibration-points", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist resumable benchmark output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_info(torch: Any, physical_gpu: int) -> dict[str, Any]:
    """Read stable GPU identity fields for result provenance."""
    fields = "name,uuid,driver_version,memory.total"
    line = subprocess.check_output(
        ["nvidia-smi", "-i", str(physical_gpu), f"--query-gpu={fields}",
         "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()[0]
    name, uuid, driver, memory = [part.strip() for part in line.split(",")]
    return {"name": name, "uuid": uuid, "driver": driver, "memory_mib": int(memory)}


def exact_knn(torch: Any, cukd: Any, support: Any, query: Any, k: int) -> tuple[Any, Any]:
    """Return exact neighbor indices and squared distances without mutating support."""
    indices = torch.empty((len(query), k), device="cuda", dtype=torch.int32)
    distances = torch.empty((len(query), k), device="cuda", dtype=torch.float32)
    cukd.CukdKnnQueryTorch(
        support.clone(), query, k, indices, distances, torch.zeros(2), True
    )
    order = distances.argsort(dim=1)
    return indices.gather(1, order), distances.gather(1, order)


def set_recall(torch: Any, exact: Any, predicted: Any, chunk_size: int = 16_384) -> dict[str, float]:
    """Compute set recall while ignoring negative ball-query padding indices."""
    values = []
    for start in range(0, len(exact), chunk_size):
        reference = exact[start:start + chunk_size].long()
        candidate = predicted[start:start + chunk_size].long()
        matches = (reference[:, :, None] == candidate[:, None, :]) & (candidate[:, None, :] >= 0)
        values.append(matches.any(dim=2).float().mean(dim=1).cpu())
    per_query = torch.cat(values)
    return {"mean": float(per_query.mean()), "minimum": float(per_query.min())}


def timed_ball_query(
    torch: Any, ball_query: Any, support: Any, query: Any, k: int, radius: float,
    warmups: int, repeats: int,
) -> tuple[Any, list[float]]:
    """Run a synchronous CUDA-event timing loop and return the final indices."""
    support_offset = torch.tensor([len(support)], device="cuda", dtype=torch.int32)
    query_offset = torch.tensor([len(query)], device="cuda", dtype=torch.int32)
    predicted = None
    timings = []
    for iteration in range(warmups + repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        predicted, _ = ball_query(
            k, radius, 0.0, support, support_offset, query, query_offset
        )
        end.record()
        end.synchronize()
        if iteration >= warmups:
            timings.append(begin.elapsed_time(end) / 1000.0)
    assert predicted is not None
    return predicted, timings


def main() -> None:
    """Calibrate global radii, then benchmark every eligible S3DIS room."""
    args = arguments()
    if any(k < 1 or k >= 64 for k in args.k):
        raise SystemExit("k must be in [1, 63] so k+1 can detect truncation")
    if any(percentile <= 0 or percentile >= 1 for percentile in args.percentile):
        raise SystemExit("percentiles must be in (0, 1)")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import Cukd.CuFun as cukd
    from FlashKNN import xyz2key
    from benchmark_s3dis import load_xyz, prepare, room_paths
    from pointops import ball_query

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    paths = room_paths(args.data_root.resolve())
    crop_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    crop_centers: dict[str, int] = {}
    calibration: dict[tuple[str, int], list[Any]] = {
        (mode, k): [] for mode in args.mode for k in args.k
    }
    eligible_paths = []
    max_exact_k = max(args.k) + 1

    for path in paths:
        coord = load_xyz(torch, path)
        relative = path.relative_to(args.data_root.resolve()).as_posix()
        crop_center = None
        eligible = False
        for mode in args.mode:
            sample = prepare(
                torch, xyz2key, coord, path, mode, "sample_part", args.voxel_size,
                args.crop_points, crop_center, crop_generator,
            )
            if sample is None:
                continue
            support, _, query_indices, crop_center = sample
            query = support if mode == "pre" else support[query_indices].contiguous()
            _, squared_distances = exact_knn(torch, cukd, support, query, max_exact_k)
            stride = max(1, len(query) // args.calibration_points)
            for k in args.k:
                calibration[(mode, k)].append(
                    torch.sqrt(squared_distances[::stride, k - 1]).cpu()
                )
            eligible = True
            del support, query_indices, query, squared_distances
        if eligible and crop_center is not None:
            crop_centers[relative] = int(crop_center)
            eligible_paths.append(path)
            print(f"calibrated {relative}", flush=True)
            if args.max_samples is not None and len(eligible_paths) >= args.max_samples:
                break
        del coord

    if not eligible_paths:
        raise RuntimeError("No room contains enough voxelized points for the requested crop")

    radii: dict[tuple[str, int, float], float] = {}
    for mode in args.mode:
        for k in args.k:
            samples = torch.cat(calibration[(mode, k)])
            for percentile in args.percentile:
                radii[(mode, k, percentile)] = float(torch.quantile(samples, percentile))

    new_payload: dict[str, Any] = {
        "metadata": {
            "dataset": "S3DIS",
            "operator": "Pointcept pointops.ball_query",
            "operator_semantics": "fixed radius with nsample truncation/padding",
            "gpu": gpu_info(torch, args.gpu),
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "voxel_size_m": args.voxel_size,
            "crop_points": args.crop_points, "warmups": args.warmups,
            "repeats": args.repeats, "seed": args.seed,
            "timing_boundary": "CUDA events around allocation/zero, ball-query kernel and sqrt; excludes data preparation and exact-reference construction",
            "radius_calibration": "global quantile of sampled exact kth-neighbor distances over all evaluated rooms",
            "eligible_rooms": len(eligible_paths),
        },
        "radii": [
            {"mode": mode, "k": k, "percentile": percentile,
             "radius_m": radii[(mode, k, percentile)]}
            for mode in args.mode for k in args.k for percentile in args.percentile
        ],
        "records": [],
    }
    if args.resume and args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        old = payload.get("metadata", {})
        current = new_payload["metadata"]
        fields = ("dataset", "operator", "torch", "torch_cuda", "voxel_size_m",
                  "crop_points", "warmups", "repeats", "seed", "eligible_rooms")
        changed = {field: (old.get(field), current.get(field))
                   for field in fields if old.get(field) != current.get(field)}
        if old.get("gpu", {}).get("uuid") != current["gpu"].get("uuid"):
            changed["gpu.uuid"] = (old.get("gpu", {}).get("uuid"), current["gpu"].get("uuid"))
        if payload.get("radii") != new_payload["radii"]:
            changed["radii"] = (payload.get("radii"), new_payload["radii"])
        if changed:
            raise SystemExit(f"Refusing to resume incompatible output {args.output}: {changed}")
    else:
        payload = new_payload
        atomic_json(args.output, payload)
    completed = {
        (record["room"], record["mode"], int(record["k"]), float(record["percentile"]))
        for record in payload["records"]
    }

    for path in eligible_paths:
        coord = load_xyz(torch, path)
        relative = path.relative_to(args.data_root.resolve()).as_posix()
        crop_center = crop_centers[relative]
        for mode in args.mode:
            sample = prepare(
                torch, xyz2key, coord, path, mode, "sample_part", args.voxel_size,
                args.crop_points, crop_center, None,
            )
            assert sample is not None
            support, _, query_indices, _ = sample
            query = support if mode == "pre" else support[query_indices].contiguous()
            exact_indices, squared_distances = exact_knn(
                torch, cukd, support, query, max_exact_k
            )
            for k in args.k:
                exact = exact_indices[:, :k]
                kth_distance = torch.sqrt(squared_distances[:, k - 1])
                next_distance = torch.sqrt(squared_distances[:, k])
                for percentile in args.percentile:
                    record_key = (relative, mode, k, percentile)
                    if record_key in completed:
                        continue
                    radius = radii[(mode, k, percentile)]
                    predicted, timings = timed_ball_query(
                        torch, ball_query, support, query, k, radius,
                        args.warmups, args.repeats,
                    )
                    record = {
                        "room": relative, "mode": mode, "scope": "sample_part",
                        "k": k, "percentile": percentile, "radius_m": radius,
                        "num_support": len(support), "num_query": len(query),
                        "query_timings_s": timings,
                        "valid_neighbor_ratio": float((predicted >= 0).float().mean()),
                        "insufficient_query_ratio": float((kth_distance >= radius).float().mean()),
                        "truncated_query_ratio": float((next_distance < radius).float().mean()),
                        "recall_vs_cukd": set_recall(torch, exact, predicted),
                    }
                    payload["records"].append(record)
                    completed.add(record_key)
                    atomic_json(args.output, payload)
                    print(
                        f"{relative} {mode} k={k} p={percentile:g} "
                        f"latency={np.median(timings) * 1000:.3f}ms "
                        f"recall={record['recall_vs_cukd']['mean']:.6f}",
                        flush=True,
                    )
            del support, query_indices, query, exact_indices, squared_distances
        del coord


if __name__ == "__main__":
    main()
