#!/usr/bin/env python3
"""Direct fixed-grid kNN diagnostic against the upstream torch_knnquery code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--torch-knnquery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--crop-points", type=int, default=250000)
    parser.add_argument("--max-points-per-voxel", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def co_tenant_snapshot() -> dict[str, list[str]]:
    commands = {
        "gpus": [
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.used,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        "compute_processes": [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    }
    result: dict[str, list[str]] = {}
    for name, command in commands.items():
        try:
            result[name] = subprocess.check_output(command, text=True).splitlines()
        except (OSError, subprocess.CalledProcessError) as error:
            result[name] = [f"unavailable: {error}"]
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def recall_and_valid_fraction(torch: Any, exact: Any, predicted: Any) -> dict[str, float]:
    predicted = predicted.reshape(len(exact), exact.shape[1]).long()
    exact = exact.long()
    matched = 0
    valid = 0
    for begin in range(0, len(exact), 25000):
        pred_chunk = predicted[begin:begin + 25000]
        exact_chunk = exact[begin:begin + 25000]
        matched += int(
            (pred_chunk[:, :, None] == exact_chunk[:, None, :])
            .any(dim=2).sum().item()
        )
        valid += int((pred_chunk >= 0).sum().item())
    denominator = len(exact) * exact.shape[1]
    return {
        "recall_vs_cukd": matched / denominator,
        "valid_neighbor_fraction": valid / denominator,
    }


def series_stats(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values) if count > 1 else 0.0
    if count > 1:
        try:
            from scipy.stats import t as student_t
            critical = float(student_t.ppf(0.975, count - 1))
        except ImportError:
            critical = 1.96
        ci95 = critical * sample_sd / count ** 0.5
    else:
        ci95 = 0.0
    return {"count": count, "mean": mean, "sample_sd": sample_sd, "ci95": ci95}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    room_means: dict[str, list[float]] = {
        "torch_knnquery_public_query_ms": [],
        "torch_knnquery_core_query_ms": [],
        "flashknn_query_ms": [],
        "flashknn_gmss_query_ms": [],
        "cuda_kdtree_query_ms": [],
    }
    recall: dict[str, list[float]] = {
        "torch_knnquery": [], "flashknn": [],
    }
    for record in records:
        room_means["torch_knnquery_public_query_ms"].append(
            statistics.mean(record["methods"]["torch_knnquery"]["public_query_ms"])
        )
        room_means["torch_knnquery_core_query_ms"].append(
            statistics.mean(record["methods"]["torch_knnquery"]["core_query_ms"])
        )
        room_means["flashknn_query_ms"].append(
            statistics.mean(record["methods"]["flashknn"]["query_ms"])
        )
        room_means["flashknn_gmss_query_ms"].append(
            statistics.mean(record["methods"]["flashknn_gmss"]["query_ms"])
        )
        room_means["cuda_kdtree_query_ms"].append(
            statistics.mean(record["methods"]["cuda_kdtree"]["query_ms"])
        )
        recall["torch_knnquery"].append(
            record["methods"]["torch_knnquery"]["recall"]["recall_vs_cukd"]
        )
        recall["flashknn"].append(
            record["methods"]["flashknn"]["recall"]["recall_vs_cukd"]
        )
    summary = {name: series_stats(values) for name, values in room_means.items()}
    summary["recall_vs_cukd"] = {
        name: series_stats(values) for name, values in recall.items()
    }
    if records:
        flash = summary["flashknn_query_ms"]["mean"]
        summary["flashknn_speedup"] = {
            "vs_torch_knnquery_public": (
                summary["torch_knnquery_public_query_ms"]["mean"] / flash
            ),
            "vs_torch_knnquery_core": (
                summary["torch_knnquery_core_query_ms"]["mean"] / flash
            ),
            "vs_flashknn_gmss": (
                summary["flashknn_gmss_query_ms"]["mean"] / flash
            ),
        }
    return summary


def main() -> None:
    args = arguments()
    if args.k > 20:
        raise SystemExit("Upstream torch_knnquery requires k <= 20; use k=16 for this diagnostic")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch_root = args.torch_knnquery_root.resolve()
    sys.path.insert(0, str(torch_root))

    import torch
    from benchmark_s3dis import load_xyz, prepare, room_paths
    from functions import FlashKNN, xyz2key
    import Cukd.CuFun as cukd
    from torch_knnquery import VoxelGrid

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    crop_generator = torch.Generator(device="cpu").manual_seed(args.seed)

    upstream_commit = git_output(torch_root, "rev-parse", "HEAD")
    upstream_diff = git_output(torch_root, "diff", "--", "src/knnquery.cu")
    expected_patch_tokens = ("points.scalar_type()", "raypos_tensor.scalar_type()")
    if upstream_diff and not all(token in upstream_diff for token in expected_patch_tokens):
        raise SystemExit("torch_knnquery has changes beyond the expected PyTorch API patch")

    repo = Path(__file__).resolve().parents[1]
    metadata = {
        "dataset": "S3DIS",
        "scope": "sample_part",
        "mode": "pre",
        "crop_points": args.crop_points,
        "k": args.k,
        "voxel_size_m": args.voxel_size,
        "alpha": args.alpha,
        "kernel_size": [3, 3, 3],
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "upstream": {
            "repository": "https://github.com/janericlenssen/torch_knnquery",
            "commit": upstream_commit,
            "source_sha256_after_compatibility_patch": sha256(torch_root / "src/knnquery.cu"),
            "compatibility_patch_only": bool(upstream_diff),
            "native_k_limit": 20,
        },
        "flashknn_commit": git_output(repo, "rev-parse", "HEAD"),
        "benchmark_sha256": sha256(Path(__file__).resolve()),
        "timing_boundaries": {
            "common": "CUDA-ready inputs; excludes file I/O, 0.02 m voxelization, crop, and H2D",
            "torch_knnquery_public": "wall-clock synchronized VoxelGrid.query, including ray mask/compaction and output allocation",
            "torch_knnquery_core": "CUDA event around upstream query_along_ray 3x3x3 scan and sequential top-k kernel only",
            "flashknn": "existing synchronized FlashKNN query timer, including output allocation/remapping",
            "flashknn_gmss": "same FlashKNN construction/layout, global-memory support traversal, one thread per query, serial heap top-k",
            "cuda_kdtree": "existing native construction/query timers",
        },
        "co_tenant_start": co_tenant_snapshot(),
    }
    payload: dict[str, Any] = {"metadata": metadata, "records": [], "summary": {}}
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        old = payload.get("metadata", {})
        keys = ("crop_points", "k", "voxel_size_m", "alpha", "warmups", "repeats", "seed")
        changed = {key: (old.get(key), metadata.get(key)) for key in keys if old.get(key) != metadata.get(key)}
        if old.get("upstream", {}).get("commit") != upstream_commit:
            changed["upstream.commit"] = (old.get("upstream", {}).get("commit"), upstream_commit)
        if changed:
            raise SystemExit(f"Refusing to resume incompatible result: {changed}")

    completed = {record["room"] for record in payload["records"]}
    eligible_seen = 0
    paths = room_paths(args.data_root.resolve())
    for path in paths:
        coord = load_xyz(torch, path)
        sample = prepare(
            torch, xyz2key, coord, path, "pre", "sample_part",
            args.voxel_size, args.crop_points, None, crop_generator,
        )
        if sample is None:
            del coord
            continue
        eligible_seen += 1
        if args.max_samples is not None and eligible_seen > args.max_samples:
            del coord
            break
        support, support_grid, _, crop_center = sample
        room = path.relative_to(args.data_root.resolve()).as_posix()
        if room in completed:
            del coord, support, support_grid
            continue

        n = len(support)
        points = support.view(1, n, 3)
        actual = torch.tensor([n], dtype=torch.int32, device="cuda")
        voxel_grid = VoxelGrid(
            voxel_size=(args.voxel_size,) * 3,
            voxel_scale=(float(args.alpha),) * 3,
            kernel_size=(3, 3, 3),
            max_points_per_voxel=args.max_points_per_voxel,
            max_occ_voxels_per_example=n,
            ranges=(-1e6, -1e6, -1e6, 1e6, 1e6, 1e6),
        ).cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        voxel_grid.set_pointset(points, actual)
        torch.cuda.synchronize()
        construction_ms = (time.perf_counter() - started) * 1000
        max_occupancy = int(voxel_grid.occ_numpnts_tensor.max().item())
        occupied_voxels = int((voxel_grid.occ_numpnts_tensor > 0).sum().item())
        if max_occupancy > args.max_points_per_voxel:
            raise RuntimeError(
                f"{room}: voxel occupancy {max_occupancy} exceeds P={args.max_points_per_voxel}"
            )

        raypos = points.view(1, n, 1, 3)
        public_times: list[float] = []
        public_indices = None
        public_mask = None
        for iteration in range(args.warmups + args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            public_indices, _, public_mask = voxel_grid.query(
                raypos, k=args.k, radius_limit_scale=0,
                max_shading_points_per_ray=1,
            )
            torch.cuda.synchronize()
            if iteration >= args.warmups:
                public_times.append((time.perf_counter() - started) * 1000)
        assert public_indices is not None and public_mask is not None
        if int(public_mask.sum().item()) != n or len(public_indices) != n:
            raise RuntimeError(f"{room}: not all support points survived the ray wrapper")

        ray_to_batch = torch.zeros(n, dtype=torch.int32, device="cuda")
        sample_locations = support.view(n, 1, 3)
        sample_mask = torch.ones((n, 1), dtype=torch.int32, device="cuda")
        core_indices = torch.full((n, 1, args.k), -1, dtype=torch.int32, device="cuda")
        core_times: list[float] = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for iteration in range(args.warmups + args.repeats):
            core_indices.fill_(-1)
            start_event.record()
            voxel_grid.query_along_ray(
                voxel_grid.points, ray_to_batch, n, 1, voxel_grid.max_o,
                voxel_grid.P, args.k, voxel_grid.grid_size_vol, 0.0,
                voxel_grid.d_coord_shift, voxel_grid.scaled_vdim,
                voxel_grid.scaled_vsize, voxel_grid.kernel_size,
                voxel_grid.occ_numpnts_tensor, voxel_grid.occ_2_pnts_tensor,
                voxel_grid.coor_2_occ_tensor, sample_locations, sample_mask,
                core_indices,
            )
            end_event.record()
            end_event.synchronize()
            if iteration >= args.warmups:
                core_times.append(start_event.elapsed_time(end_event))

        batch = torch.zeros(n, device="cuda", dtype=torch.long)
        flash = FlashKNN(num_nbr=args.k, num_down=2, debug=True)
        flash_indices = None
        for _ in range(args.warmups + args.repeats):
            flash_indices = flash.query(
                support_grid, batch, support, memory_mode="SM", sorting_mode="PS"
            )
        assert flash_indices is not None
        flash_times = flash.time_list[args.warmups:]

        gmss = FlashKNN(num_nbr=args.k, num_down=2, debug=True)
        gmss_indices = None
        for _ in range(args.warmups + args.repeats):
            gmss_indices = gmss.query(
                support_grid, batch, support, memory_mode="GM", sorting_mode="SS"
            )
        assert gmss_indices is not None
        gmss_times = gmss.time_list[args.warmups:]

        exact_indices = torch.empty((n, args.k), device="cuda", dtype=torch.int32)
        exact_distances = torch.empty((n, args.k), device="cuda", dtype=torch.float32)
        cukd_construction: list[float] = []
        cukd_query: list[float] = []
        for iteration in range(args.warmups + args.repeats):
            timing = torch.zeros(2)
            cukd.CukdKnnQueryTorch(
                support, support, args.k, exact_indices, exact_distances, timing, True
            )
            torch.cuda.synchronize()
            if iteration >= args.warmups:
                cukd_construction.append(float(timing[0]) * 1000)
                cukd_query.append(float(timing[1]) * 1000)

        record = {
            "room": room,
            "crop_center": crop_center,
            "points": n,
            "methods": {
                "torch_knnquery": {
                    "construction_ms": construction_ms,
                    "public_query_ms": public_times,
                    "core_query_ms": core_times,
                    "occupied_voxels": occupied_voxels,
                    "max_points_in_voxel": max_occupancy,
                    "max_points_per_voxel": args.max_points_per_voxel,
                    "recall": recall_and_valid_fraction(
                        torch, exact_indices, public_indices
                    ),
                    "core_matches_public": bool(torch.equal(core_indices, public_indices)),
                },
                "flashknn": {
                    "construction_ms": [
                        float(item["预处理耗时"]) * 1000 for item in flash_times
                    ],
                    "query_ms": [
                        float(item["查询耗时"]) * 1000 for item in flash_times
                    ],
                    "recall": recall_and_valid_fraction(
                        torch, exact_indices, flash_indices
                    ),
                },
                "flashknn_gmss": {
                    "configuration": {
                        "support_access": "global_memory",
                        "query_mapping": "one_thread_per_query",
                        "selection": "serial_max_heap",
                    },
                    "construction_ms": [
                        float(item["预处理耗时"]) * 1000 for item in gmss_times
                    ],
                    "query_ms": [
                        float(item["查询耗时"]) * 1000 for item in gmss_times
                    ],
                    "recall": recall_and_valid_fraction(
                        torch, exact_indices, gmss_indices
                    ),
                },
                "cuda_kdtree": {
                    "construction_ms": cukd_construction,
                    "query_ms": cukd_query,
                    "exact": True,
                },
            },
        }
        payload["records"].append(record)
        payload["summary"] = summarize(payload["records"])
        atomic_json(args.output, payload)
        print(
            f"[{len(payload['records'])}] {room}: "
            f"torch core={statistics.mean(core_times):.3f} ms, "
            f"public={statistics.mean(public_times):.3f} ms, "
            f"GMSS={statistics.mean(record['methods']['flashknn_gmss']['query_ms']):.3f} ms, "
            f"FlashKNN={statistics.mean(record['methods']['flashknn']['query_ms']):.3f} ms"
        )
        del coord, support, support_grid, points, raypos, voxel_grid
        del public_indices, public_mask, core_indices, flash_indices, gmss_indices
        del exact_indices, exact_distances
        torch.cuda.empty_cache()

    payload["metadata"]["co_tenant_end"] = co_tenant_snapshot()
    payload["summary"] = summarize(payload["records"])
    atomic_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
