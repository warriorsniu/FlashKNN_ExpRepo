#!/usr/bin/env python3
"""Benchmark PyTorch3D ball_query on the matched S3DIS crop/radius protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_ball_query import atomic_json, exact_knn, gpu_info, set_recall


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radii-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytorch3d-commit", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", nargs="+", choices=("pre", "post"), default=("pre", "post"))
    parser.add_argument("--k", nargs="+", type=int, default=(24, 32, 48))
    parser.add_argument("--percentile", type=float, default=0.9)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250_000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_radii(args: argparse.Namespace) -> tuple[dict[tuple[str, int], float], dict[str, Any]]:
    payload = json.loads(args.radii_source.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    required = {
        "dataset": "S3DIS",
        "crop_points": args.crop_points,
        "seed": args.seed,
        "voxel_size_m": args.voxel_size,
    }
    changed = {
        key: (metadata.get(key), value)
        for key, value in required.items()
        if metadata.get(key) != value
    }
    if changed:
        raise SystemExit(f"Radii source does not match requested protocol: {changed}")
    radii = {
        (entry["mode"], int(entry["k"])): float(entry["radius_m"])
        for entry in payload["radii"]
        if float(entry["percentile"]) == args.percentile
    }
    missing = [(mode, k) for mode in args.mode for k in args.k if (mode, k) not in radii]
    if missing:
        raise SystemExit(f"Radii source is missing requested settings: {missing}")
    return radii, metadata


def timed_ball_query(
    torch: Any,
    ball_query: Any,
    support: Any,
    query: Any,
    k: int,
    radius: float,
    warmups: int,
    repeats: int,
) -> tuple[Any, list[float]]:
    predicted = None
    timings = []
    query_batch = query.unsqueeze(0)
    support_batch = support.unsqueeze(0)
    query_lengths = torch.tensor([len(query)], device="cuda", dtype=torch.int64)
    support_lengths = torch.tensor([len(support)], device="cuda", dtype=torch.int64)
    with torch.inference_mode():
        for iteration in range(warmups + repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = ball_query(
                query_batch,
                support_batch,
                lengths1=query_lengths,
                lengths2=support_lengths,
                K=k,
                radius=radius,
                return_nn=False,
                skip_points_outside_cube=True,
            )
            _ = torch.sqrt(result.dists)
            end.record()
            end.synchronize()
            predicted = result.idx[0]
            if iteration >= warmups:
                timings.append(begin.elapsed_time(end) / 1000.0)
    assert predicted is not None
    return predicted, timings


def main() -> None:
    args = arguments()
    if any(k < 1 for k in args.k):
        raise SystemExit("k must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import pytorch3d
    import Cukd.CuFun as cukd
    from FlashKNN import xyz2key
    from benchmark_s3dis import load_xyz, prepare, room_paths
    from pytorch3d.ops import ball_query

    radii, source_metadata = load_radii(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    paths = room_paths(args.data_root.resolve())
    crop_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    crop_centers: dict[str, int] = {}
    eligible_paths = []

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
            if sample is not None:
                _, _, _, crop_center = sample
                eligible = True
        if eligible and crop_center is not None:
            crop_centers[relative] = int(crop_center)
            eligible_paths.append(path)
            if args.max_samples is not None and len(eligible_paths) >= args.max_samples:
                break
        del coord
    if not eligible_paths:
        raise RuntimeError("No room contains enough voxelized points for the requested crop")

    metadata = {
        "dataset": "S3DIS",
        "operator": "PyTorch3D pytorch3d.ops.ball_query",
        "operator_semantics": "fixed radius; first K support-order matches; -1 padding",
        "api": {"return_nn": False, "skip_points_outside_cube": True},
        "pytorch3d_version": getattr(pytorch3d, "__version__", "unknown"),
        "pytorch3d_commit": args.pytorch3d_commit,
        "gpu": gpu_info(torch, args.gpu),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "voxel_size_m": args.voxel_size,
        "crop_points": args.crop_points,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "timing_boundary": "CUDA events around ball_query and output-distance sqrt; excludes data preparation, exact reference, I/O and H2D",
        "radius_calibration": source_metadata["radius_calibration"],
        "radii_source": str(args.radii_source.resolve()),
        "radii_source_sha256": sha256(args.radii_source),
        "eligible_rooms": len(eligible_paths),
    }
    new_payload: dict[str, Any] = {
        "metadata": metadata,
        "radii": [
            {"mode": mode, "k": k, "percentile": args.percentile, "radius_m": radii[(mode, k)]}
            for mode in args.mode for k in args.k
        ],
        "records": [],
    }
    if args.resume and args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        fields = (
            "dataset", "operator", "api", "pytorch3d_commit", "torch", "torch_cuda",
            "voxel_size_m", "crop_points", "warmups", "repeats", "seed", "eligible_rooms",
            "radii_source_sha256",
        )
        changed = {
            field: (payload.get("metadata", {}).get(field), metadata.get(field))
            for field in fields
            if payload.get("metadata", {}).get(field) != metadata.get(field)
        }
        if payload.get("radii") != new_payload["radii"]:
            changed["radii"] = (payload.get("radii"), new_payload["radii"])
        if changed:
            raise SystemExit(f"Refusing to resume incompatible output: {changed}")
    else:
        payload = new_payload
        atomic_json(args.output, payload)
    completed = {
        (record["room"], record["mode"], int(record["k"]))
        for record in payload["records"]
    }

    max_exact_k = max(args.k) + 1
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
                record_key = (relative, mode, k)
                if record_key in completed:
                    continue
                radius = radii[(mode, k)]
                predicted, timings = timed_ball_query(
                    torch, ball_query, support, query, k, radius,
                    args.warmups, args.repeats,
                )
                kth_distance = torch.sqrt(squared_distances[:, k - 1])
                next_distance = torch.sqrt(squared_distances[:, k])
                record = {
                    "room": relative,
                    "mode": mode,
                    "scope": "sample_part",
                    "k": k,
                    "percentile": args.percentile,
                    "radius_m": radius,
                    "num_support": len(support),
                    "num_query": len(query),
                    "query_timings_s": timings,
                    "valid_neighbor_ratio": float((predicted >= 0).float().mean()),
                    "insufficient_query_ratio": float((kth_distance >= radius).float().mean()),
                    "truncated_query_ratio": float((next_distance < radius).float().mean()),
                    "recall_vs_cukd": set_recall(torch, exact_indices[:, :k], predicted),
                }
                payload["records"].append(record)
                completed.add(record_key)
                atomic_json(args.output, payload)
                print(
                    f"{relative} {mode} k={k} latency={np.median(timings) * 1000:.3f}ms "
                    f"recall={record['recall_vs_cukd']['mean']:.6f}",
                    flush=True,
                )
            del support, query_indices, query, exact_indices, squared_distances
        del coord


if __name__ == "__main__":
    main()
