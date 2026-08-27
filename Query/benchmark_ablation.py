#!/usr/bin/env python3
"""Run final-kernel FlashKNN design ablations on deterministic S3DIS crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_s3dis import (
    atomic_json,
    gpu_info,
    load_xyz,
    normalized_timings,
    prepare,
    recall,
    room_paths,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "smps": {
        "memory_mode": "SM", "sorting_mode": "PS",
        "candidate_mode": "register", "enable_skip": True,
    },
    "smss": {
        "memory_mode": "SM", "sorting_mode": "SS",
        "candidate_mode": "register", "enable_skip": True,
    },
    "gmps": {
        "memory_mode": "GM", "sorting_mode": "PS",
        "candidate_mode": "register", "enable_skip": True,
    },
    "gmss": {
        "memory_mode": "GM", "sorting_mode": "SS",
        "candidate_mode": "register", "enable_skip": True,
    },
    "candidate_shared": {
        "memory_mode": "SM", "sorting_mode": "PS",
        "candidate_mode": "shared", "enable_skip": True,
    },
    "no_skip": {
        "memory_mode": "SM", "sorting_mode": "PS",
        "candidate_mode": "register", "enable_skip": False,
    },
    "candidate_shared_no_skip": {
        "memory_mode": "SM", "sorting_mode": "PS",
        "candidate_mode": "shared", "enable_skip": False,
    },
}

DEFAULT_VARIANTS = (
    "smps", "smss", "gmps", "candidate_shared", "no_skip",
    "candidate_shared_no_skip",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--k", nargs="+", type=int,
        default=(8, 16, 24, 32, 40, 48, 56, 64),
    )
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS),
        default=DEFAULT_VARIANTS,
    )
    parser.add_argument("--num-down", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250_000)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def git_identity(repo: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
    ).strip())
    return {"commit": commit, "dirty": dirty}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            snapshot[name] = subprocess.check_output(command, text=True).splitlines()
        except (OSError, subprocess.CalledProcessError) as error:
            snapshot[name] = [f"unavailable: {error}"]
    try:
        snapshot["load_average"] = list(os.getloadavg())
    except OSError:
        pass
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


def mean_query_ms(timings: list[dict[str, float]]) -> float:
    return statistics.fmean(item["query_s"] for item in timings) * 1000.0


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
    if any(k < 2 or k > 64 for k in args.k):
        raise SystemExit("The current ablation kernels support 2 <= k <= 64")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    crop_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    paths = room_paths(args.data_root.resolve())
    if args.max_samples is not None:
        paths = paths[:args.max_samples]

    repo = Path(__file__).resolve().parents[1]
    selected_variants = list(dict.fromkeys(args.variants))
    payload: dict[str, Any] = {
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
                relative: sha256(repo / relative)
                for relative in (
                    "FlashKNN/csrc/flash_knn_bitonic_top_p.cuh",
                    "FlashKNN/csrc/flash_knn_query_dynamic_load.cu",
                    "FlashKNN/csrc/flash_knn_query_GMPS.cu",
                    "FlashKNN/csrc/flash_knn_query_global_memory.cu",
                    "FlashKNN/functions/FlashKnnWrapper.py",
                    "Query/benchmark_ablation.py",
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
            "variants": {name: VARIANTS[name] for name in selected_variants},
            "sorting_revision": "generated_bitonic_top_p",
            "selection_revision_by_variant": {
                name: (
                    "serial_max_heap" if name == "gmss"
                    else "generated_bitonic_top_p"
                )
                for name in selected_variants
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
            "num_down", "warmups", "repeats", "seed",
            "sorting_revision", "source_sha256", "extension",
        )
        changed = {
            field: (previous.get("metadata", {}).get(field),
                    payload["metadata"].get(field))
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
        payload["metadata"]["co_tenant_resume"] = co_tenant_snapshot()

    record_positions = {
        (record["room"], int(record["k"])): position
        for position, record in enumerate(payload["records"])
    }
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
        for k in args.k:
            key = (room, k)
            if key in record_positions:
                record = payload["records"][record_positions[key]]
            else:
                record = {
                    "room": room, "k": k,
                    "num_support": len(support), "num_query": len(support),
                    "crop_center": crop_center, "variants": {},
                }
                record_positions[key] = len(payload["records"])
                payload["records"].append(record)

            missing = [
                name for name in selected_variants
                if name not in record.get("variants", {})
            ]
            if not missing:
                continue

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

            for name in missing:
                configuration = VARIANTS[name]
                knn = FlashKNN(
                    num_nbr=k, num_down=args.num_down, debug=True,
                )
                predicted = None
                for _ in range(args.warmups + args.repeats):
                    predicted = knn.query(
                        grid, batch, support,
                        memory_mode=configuration["memory_mode"],
                        sorting_mode=configuration["sorting_mode"],
                        candidate_mode=configuration["candidate_mode"],
                        enable_skip=configuration["enable_skip"],
                    )
                timings = normalized_timings(knn.time_list[args.warmups:])
                if predicted is None:
                    raise RuntimeError(f"{name} produced no indices")
                record.setdefault("variants", {})[name] = {
                    "configuration": configuration,
                    "timings": timings,
                    "recall_vs_cukd": recall(
                        torch, exact_indices, predicted,
                    ),
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

    payload["metadata"]["co_tenant_end"] = co_tenant_snapshot()
    atomic_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output}")


if __name__ == "__main__":
    main()
