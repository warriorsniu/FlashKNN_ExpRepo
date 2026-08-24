#!/usr/bin/env python3
"""Cross-GPU S3DIS kNN benchmark using one shared Pointcept data download."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", nargs="+", choices=("pre", "post"), default=("pre", "post"))
    parser.add_argument("--scope", nargs="+", choices=("sample_part", "full"), default=("sample_part", "full"))
    parser.add_argument("--k", nargs="+", type=int, default=(8, 16, 24, 32, 48, 64))
    parser.add_argument("--num-down", type=int, default=2,
                        help="Historical FlashKNN search expansion exponent")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true",
                        help="Skip paper baselines FLANN-CUDA and nanoflann")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_info(torch: Any, physical_gpu: int) -> dict[str, Any]:
    try:
        fields = "name,uuid,driver_version,memory.total"
        line = subprocess.check_output(
            ["nvidia-smi", "-i", str(physical_gpu), f"--query-gpu={fields}",
             "--format=csv,noheader,nounits"], text=True,
        ).strip().splitlines()[0]
        name, uuid, driver, memory = [part.strip() for part in line.split(",")]
        return {"name": name, "uuid": uuid, "driver": driver, "memory_mib": int(memory)}
    except Exception:
        return {"name": torch.cuda.get_device_name(0)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(repo: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
    ).strip())
    return {"commit": commit, "dirty": dirty}


def co_tenant_snapshot() -> dict[str, Any]:
    commands = {
        "gpus": [
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.used,"
            "utilization.gpu,pstate", "--format=csv,noheader,nounits",
        ],
        "compute_processes": [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,"
            "used_memory", "--format=csv,noheader,nounits",
        ],
    }
    snapshot: dict[str, Any] = {}
    for name, command in commands.items():
        try:
            snapshot[name] = subprocess.check_output(
                command, text=True,
            ).splitlines()
        except (OSError, subprocess.CalledProcessError) as error:
            snapshot[name] = [f"unavailable: {error}"]
    try:
        process_lines = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,args="], text=True,
        ).splitlines()
        snapshot["training_processes"] = [
            line.strip() for line in process_lines
            if "tools/train.py" in line
        ]
    except (OSError, subprocess.CalledProcessError) as error:
        snapshot["training_processes"] = [f"unavailable: {error}"]
    return snapshot


def room_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("Area_*/*.pth"))
    if not paths:
        paths = sorted(root.rglob("*.pth"))
    if not paths:
        area_dirs = sorted([*root.glob("Area_*"), *root.glob("area_*")])
        paths = sorted(
            room for area in area_dirs for room in area.iterdir()
            if room.is_dir() and (room / "coord.npy").is_file()
        )
    if not paths:
        paths = sorted(root.glob("Area_*.npy"))
    if not paths:
        raise SystemExit(
            f"No Pointcept PTH/per-field-NPY or historical s3disfull NPY rooms below {root}"
        )
    return paths


def load_xyz(torch: Any, path: Path) -> Any:
    if path.is_dir():
        coord = torch.as_tensor(np.load(path / "coord.npy"), dtype=torch.float32)
        coord -= coord.amin(dim=0, keepdim=True)
        coord = torch.round(coord * 1000.0) / 1000.0
        return coord.cuda(non_blocking=True).contiguous()
    if path.suffix == ".npy":
        coord = torch.as_tensor(np.load(path, mmap_mode="r")[:, :3].copy(), dtype=torch.float32)
        return coord.cuda(non_blocking=True).contiguous()
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "coord" not in data:
        raise ValueError(f"Expected Pointcept room dict with coord: {path}")
    coord = torch.as_tensor(np.asarray(data["coord"]), dtype=torch.float32)
    # The historical s3disfull/raw arrays are room-local. Pointcept stores
    # globally aligned coordinates, so undo only the translation. The official
    # annotation coordinates have millimetre precision; re-rounding after the
    # large global translation avoids float32 cancellation changing points that
    # lie exactly on a 2 cm voxel boundary.
    coord = coord - coord.amin(dim=0, keepdim=True)
    coord = torch.round(coord * 1000.0) / 1000.0
    return coord.cuda(non_blocking=True).contiguous()


def prepare(torch: Any, xyz2key: Any, coord: Any, path: Path, mode: str,
            scope: str, voxel_size: float, crop_points: int,
            crop_center: int | None = None, crop_generator: Any | None = None):
    grid = torch.floor(coord / voxel_size).long()
    key = xyz2key(grid[:, 0], grid[:, 1], grid[:, 2])
    order = key.argsort()
    sorted_key = key[order]
    first = torch.ones_like(sorted_key, dtype=torch.bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    support = coord[order[first]].contiguous()
    support_grid = grid[order[first]].contiguous()
    support_key = sorted_key[first]

    if scope == "sample_part":
        if len(support) < crop_points:
            return None
        if crop_center is None:
            crop_center = int(torch.randint(
                high=len(support), size=(1,), generator=crop_generator,
            ))
        distance = (support_grid - support_grid[crop_center]).square().sum(1)
        # argsort, rather than topk, preserves the historical EdgeAggr crop
        # including its deterministic ordering for equal-distance points.
        selected = distance.argsort()[:crop_points]
    else:
        selected = torch.arange(len(support), device="cuda")

    if mode == "post":
        selected = selected[support_key[selected].argsort()]
    support = support[selected].contiguous()
    support_grid = support_grid[selected].contiguous()
    selected_key = support_key[selected]
    if mode == "pre":
        query_indices = torch.arange(len(support), device="cuda")
    else:
        _, counts = torch.unique_consecutive(selected_key >> 3, return_counts=True)
        steps = torch.nn.functional.pad(torch.cumsum(counts, 0), (1, 0))
        query_indices = ((steps[:-1] + steps[1:]) // 2).long()
    return support, support_grid, query_indices, crop_center


def normalized_timings(items: list[dict]) -> list[dict[str, float]]:
    return [{
        "construction_s": float(item["预处理耗时"]),
        "query_s": float(item["查询耗时"]),
        "total_s": float(item["预处理耗时"] + item["查询耗时"]),
    } for item in items]


def recall(torch: Any, exact: Any, predicted: Any) -> dict[str, float]:
    """Compute set recall without counting repeated predictions twice."""
    exact = exact.long()
    predicted = predicted.long().sort(1).values
    positions = torch.searchsorted(predicted, exact)
    safe = positions.clamp_max(predicted.shape[1] - 1)
    values = (
        (positions < predicted.shape[1])
        & (predicted.gather(1, safe) == exact)
    ).sum(1).float() / exact.shape[1]
    duplicate_slots = (predicted[:, 1:] == predicted[:, :-1]).sum(1)
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "duplicate_queries": int((duplicate_slots > 0).sum()),
        "duplicate_slots": int(duplicate_slots.sum()),
    }


def main() -> None:
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    try:
        from FlashKNN import FlashKNN, xyz2key
        import FlashKNN.CuFun as flash_cuda
    except ImportError:
        from functions import FlashKNN, xyz2key
        from functions import CuFun as flash_cuda
    import Cukd.CuFun as cukd

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    crop_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    paths = room_paths(args.data_root)
    if args.max_samples is not None:
        paths = paths[:args.max_samples]
    repo = Path(__file__).resolve().parents[1]
    source_files = (
        "FlashKNN/csrc/api.cpp",
        "FlashKNN/csrc/flash_knn_query.h",
        "FlashKNN/csrc/flash_knn_query_dynamic_load.cu",
        "FlashKNN/functions/FlashKnnWrapper.py",
        "Query/benchmark_s3dis.py",
    )
    payload = {
        "metadata": {
            "dataset": "S3DIS",
            "source_format": (
                "historical s3disfull/raw .npy" if paths[0].suffix == ".npy" else
                "Pointcept per-field NPY; translated and rounded to room-local millimetres"
                if paths[0].is_dir() else
                "Pointcept legacy PTH; translated and rounded to room-local millimetres"
            ),
            "gpu": gpu_info(torch, args.gpu),
            "physical_gpu_index": args.gpu,
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "voxel_size_m": args.voxel_size,
            "crop_points": args.crop_points, "num_down": args.num_down,
            "warmups": args.warmups, "repeats": args.repeats,
            "git": git_identity(repo),
            "source_sha256": {
                relative: sha256(repo / relative) for relative in source_files
            },
            "extension": {
                "path": str(Path(flash_cuda.__file__).resolve()),
                "sha256": sha256(Path(flash_cuda.__file__).resolve()),
            },
            "flashknn_configuration": {
                "memory_mode": "SM", "sorting_mode": "PS",
                "candidate_mode": "register", "enable_skip": True,
                "thread_group_size": "adaptive", "alpha": 4,
                "sorting_revision": "generated_bitonic_top_p",
            },
            "co_tenant_start": co_tenant_snapshot(),
            "timing_boundary": "CUDA-ready inputs; excludes file I/O, voxelization, cropping and H2D",
        },
        "records": [],
    }
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        current = payload["metadata"]
        old = previous.get("metadata", {})
        identity_fields = (
            "source_format", "torch", "torch_cuda", "voxel_size_m",
            "crop_points", "num_down", "warmups", "repeats",
            "source_sha256", "extension", "flashknn_configuration",
        )
        changed = {
            field: (old.get(field), current.get(field))
            for field in identity_fields if old.get(field) != current.get(field)
        }
        old_uuid = old.get("gpu", {}).get("uuid")
        new_uuid = current.get("gpu", {}).get("uuid")
        if old_uuid != new_uuid:
            changed["gpu.uuid"] = (old_uuid, new_uuid)
        if changed:
            raise SystemExit(
                f"Refusing to resume incompatible output {args.output}: {changed}"
            )
        payload = previous
        payload["metadata"]["co_tenant_resume"] = co_tenant_snapshot()
    required_methods = {"flashknn", "cuda_kdtree"}
    if not args.skip_legacy:
        required_methods.update(("flann_cuda", "nanoflann"))
    if not args.skip_faiss:
        required_methods.update(("faiss_flat", "faiss_ivf"))
    record_positions = {
        (r["room"], r["mode"], r["scope"], int(r["k"])): position
        for position, r in enumerate(payload["records"])
    }
    completed = {
        key for key, position in record_positions.items()
        if required_methods.issubset(payload["records"][position].get("methods", {}))
    }
    for path in paths:
        coord = load_xyz(torch, path)
        for scope in args.scope:
            crop_center = None
            for mode in args.mode:
                sample = prepare(
                    torch, xyz2key, coord, path, mode, scope,
                    args.voxel_size, args.crop_points, crop_center, crop_generator,
                )
                if sample is None:
                    continue
                support, grid, query_indices, crop_center = sample
                query = support if mode == "pre" else support[query_indices].contiguous()
                batch = torch.zeros(len(support), device="cuda", dtype=torch.long)
                for k in args.k:
                    key = (path.relative_to(args.data_root).as_posix(), mode, scope, k)
                    if key in completed:
                        continue
                    flash = FlashKNN(num_nbr=k, num_down=args.num_down, debug=True)
                    predicted = None
                    for _ in range(args.warmups + args.repeats):
                        if mode == "pre":
                            predicted = flash.query(
                                grid, batch, support, memory_mode="SM", sorting_mode="PS"
                            )
                        else:
                            predicted = flash.selected_query(
                                support, grid, query_indices, batch,
                                dynamic_load=True, memory_mode="SM",
                            )
                    flash_times = normalized_timings(flash.time_list[args.warmups:])

                    exact_indices = torch.empty((len(query), k), device="cuda", dtype=torch.int32)
                    exact_distances = torch.empty((len(query), k), device="cuda", dtype=torch.float32)
                    exact_times = []
                    for iteration in range(args.warmups + args.repeats):
                        timing = torch.zeros(2)
                        cukd.CukdKnnQueryTorch(
                            support, query, k, exact_indices, exact_distances, timing, True
                        )
                        torch.cuda.synchronize()
                        if iteration >= args.warmups:
                            exact_times.append({
                                "construction_s": float(timing[0]),
                                "query_s": float(timing[1]),
                                "total_s": float(timing.sum()),
                            })
                    flash_recall = recall(torch, exact_indices, predicted)
                    methods: dict[str, Any] = {
                        "flashknn": {
                            "timings": flash_times,
                            "recall_vs_cukd": flash_recall,
                        },
                        "cuda_kdtree": {"timings": exact_times, "exact": True},
                    }
                    if not args.skip_legacy:
                        from legacy_backends import benchmark_legacy_methods
                        legacy = benchmark_legacy_methods(
                            torch, support, query, k, args.warmups, args.repeats
                        )
                        for method in ("flann_cuda", "nanoflann"):
                            legacy_indices = legacy[method].pop("indices")
                            methods[method] = {
                                **legacy[method],
                                "recall_vs_cukd": recall(
                                    torch, exact_indices, legacy_indices.to(exact_indices.device)
                                ),
                            }
                    if not args.skip_faiss:
                        from faiss_backends import benchmark_faiss_methods
                        faiss_result = benchmark_faiss_methods(
                            support, query, k, args.warmups, args.repeats,
                            target_recall=flash_recall["mean"], seed=args.seed,
                        )
                        methods["faiss_flat"] = {
                            "timings": normalized_timings(faiss_result["faiss_flat_time_info"]),
                            "exact": True,
                        }
                        methods["faiss_ivf"] = {
                            "timings": normalized_timings(faiss_result["faiss_ivf_time_info"]),
                            "training_s": faiss_result["faiss_ivf_training_time"],
                            "nlist": faiss_result["faiss_ivf_nlist"],
                            "nprobe": faiss_result["faiss_ivf_nprobe"],
                            "target_recall": faiss_result["faiss_ivf_target_recall"],
                            "calibration": faiss_result["faiss_ivf_calibration"],
                            "calibration_queries": faiss_result[
                                "faiss_ivf_calibration_queries"
                            ],
                            "recall_vs_faiss_flat": faiss_result["faiss_ivf_mean_recall"],
                        }
                    record = {
                        "room": key[0], "mode": mode, "scope": scope, "k": k,
                        "num_support": len(support), "num_query": len(query), "methods": methods,
                    }
                    if key in record_positions:
                        payload["records"][record_positions[key]] = record
                    else:
                        record_positions[key] = len(payload["records"])
                        payload["records"].append(record)
                    completed.add(key)
                    atomic_json(args.output, payload)
                    print(f"{key[0]} {scope} {mode} k={k} "
                          f"recall={methods['flashknn']['recall_vs_cukd']['mean']:.6f}", flush=True)
                del support, grid, query_indices, query, batch
        del coord
        torch.cuda.empty_cache()
    payload["metadata"]["co_tenant_end"] = co_tenant_snapshot()
    atomic_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output}")


if __name__ == "__main__":
    main()
