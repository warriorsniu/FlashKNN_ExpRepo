#!/usr/bin/env python3
"""Benchmark FlashKNN on a deterministic SemanticKITTI pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU id")
    parser.add_argument("--mode", choices=("pre", "post"), nargs="+", default=["pre", "post"])
    parser.add_argument("--k", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
    parser.add_argument("--alpha", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument(
        "--ivf-match-alpha",
        type=int,
        default=4,
        help="FlashKNN alpha whose recall is the FAISS IVF calibration target",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--exact-backend", choices=("auto", "cukd", "faiss"), default="auto")
    parser.add_argument("--include-faiss-timing", action="store_true",
                        help="Deprecated compatibility flag; FAISS runs by default")
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true",
                        help="Skip paper baselines FLANN-CUDA and nanoflann")
    parser.add_argument("--output", type=Path, default=Path("experiments/lidar/results/query.json"))
    return parser.parse_args()


def import_faiss(torch: Any) -> Any:
    if not hasattr(torch.Tensor, "untyped_storage"):
        torch.Tensor.untyped_storage = torch.Tensor.storage
    import faiss
    import faiss.contrib.torch_utils  # noqa: F401
    if not hasattr(faiss, "StandardGpuResources"):
        raise ImportError("FAISS was installed without GPU support")
    return faiss


def recall_intersection(torch: Any, exact: Any, predicted: Any) -> dict[str, float]:
    """Compute set recall without counting repeated predictions twice."""
    exact = exact.long()
    predicted = predicted.long().sort(dim=1).values
    positions = torch.searchsorted(predicted, exact)
    safe = positions.clamp_max(predicted.shape[1] - 1)
    recall = (
        (positions < predicted.shape[1])
        & (predicted.gather(1, safe) == exact)
    ).sum(dim=1).float() / exact.shape[1]
    duplicate_slots = (predicted[:, 1:] == predicted[:, :-1]).sum(dim=1)
    return {
        "mean": float(recall.mean()), "minimum": float(recall.min()),
        "p01": float(torch.quantile(recall, 0.01)),
        "p05": float(torch.quantile(recall, 0.05)),
        "duplicate_queries": int((duplicate_slots > 0).sum()),
        "duplicate_slots": int(duplicate_slots.sum()),
    }


def gpu_info(physical_gpu: int, torch: Any) -> dict[str, Any]:
    try:
        query = "name,uuid,driver_version,memory.total"
        line = subprocess.check_output(
            ["nvidia-smi", "-i", str(physical_gpu), f"--query-gpu={query}",
             "--format=csv,noheader,nounits"], text=True,
        ).strip().splitlines()[0]
        name, uuid, driver, memory = [item.strip() for item in line.split(",")]
        return {"name": name, "uuid": uuid, "driver": driver, "memory_mib": int(memory)}
    except Exception:
        return {"name": torch.cuda.get_device_name(0)}


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    for alpha in args.alpha:
        exponent = round(math.log2(alpha))
        if alpha < 2 or 2 ** exponent != alpha:
            raise SystemExit(f"alpha must be a power of two, got {alpha}")
    if not args.skip_faiss and args.ivf_match_alpha not in args.alpha:
        raise SystemExit(
            f"--ivf-match-alpha={args.ivf_match_alpha} is absent from --alpha={args.alpha}"
        )
    # Select the physical card before importing torch or CUDA extensions.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    try:
        from FlashKNN import FlashKNN
    except ImportError:  # source-tree fallback used by the historical local environment
        from functions.FlashKnnWrapper import FlashKNN

    try:
        import Cukd.CuFun as cukd_extension
    except ImportError:
        cukd_extension = None
    try:
        faiss = import_faiss(torch)
    except ImportError:
        faiss = None

    if args.exact_backend == "cukd" or (args.exact_backend == "auto" and cukd_extension is not None):
        if cukd_extension is None:
            raise SystemExit("Cukd is not installed")
        exact_backend = "cukd"
    else:
        if faiss is None:
            raise SystemExit("Neither Cukd nor GPU FAISS is available for exact ground truth")
        exact_backend = "faiss"

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][:args.max_samples]
    for entry in samples:
        sample_path = args.data_dir / entry["file"]
        actual = sha256(sample_path)
        if actual != entry["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {sample_path}: {actual}")
    payload: dict[str, Any] = {
        "metadata": {
            "dataset": "SemanticKITTI", "manifest": manifest,
            "gpu": gpu_info(args.gpu, torch), "python": platform.python_version(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "exact_backend": exact_backend, "warmups": args.warmups,
            "repeats": args.repeats,
            "faiss_ivf_match_alpha": args.ivf_match_alpha,
            "timing_boundary": "CUDA inputs ready; excludes file I/O, voxelization and H2D",
        },
        "samples": [],
    }
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("metadata", {}).get("manifest") != manifest:
            raise SystemExit(f"Existing output uses a different manifest: {args.output}")
        old = previous.get("metadata", {})
        current = payload["metadata"]
        identity_fields = (
            "torch", "torch_cuda", "exact_backend", "warmups", "repeats",
            "faiss_ivf_match_alpha",
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
    required_fields = {"exact", "flashknn"}
    if not args.skip_legacy:
        required_fields.update(("flann_cuda", "nanoflann"))
    if not args.skip_faiss:
        required_fields.update(("faiss_flat", "faiss_ivf"))
    record_positions = {
        (item["sample"], item["mode"], int(item["k"])): position
        for position, item in enumerate(payload["samples"])
    }
    completed = {
        key for key, position in record_positions.items()
        if required_fields.issubset(payload["samples"][position])
    }

    faiss_resources = faiss.StandardGpuResources() if faiss is not None else None
    for sample_number, entry in enumerate(samples, 1):
        archive = np.load(args.data_dir / entry["file"])
        support = torch.from_numpy(archive["support_xyz"]).cuda().contiguous()
        grid = torch.from_numpy(archive["grid_coord"]).cuda().long().contiguous()
        post_indices = torch.from_numpy(archive["post_query_indices"]).cuda().long().contiguous()
        batch = torch.zeros(len(support), device="cuda", dtype=torch.long)

        for mode in args.mode:
            query_indices = torch.arange(len(support), device="cuda") if mode == "pre" else post_indices
            query = support if mode == "pre" else support[query_indices].contiguous()
            for k in args.k:
                record_key = (entry["file"], mode, k)
                if record_key in completed:
                    print(f"[{sample_number}/{len(samples)}] skip completed {entry['file']} {mode} k={k}")
                    continue
                exact_indices = torch.empty((len(query), k), device="cuda", dtype=torch.int32)
                exact_distances = torch.empty((len(query), k), device="cuda", dtype=torch.float32)
                exact_times = []
                if exact_backend == "cukd":
                    for iteration in range(args.warmups + args.repeats):
                        timing = torch.zeros(2)
                        cukd_extension.CukdKnnQueryTorch(
                            support, query, k, exact_indices, exact_distances, timing, True
                        )
                        torch.cuda.synchronize()
                        if iteration >= args.warmups:
                            exact_times.append({"construction_seconds": float(timing[0]),
                                                "query_seconds": float(timing[1])})
                else:
                    config = faiss.GpuIndexFlatConfig()
                    config.device = 0
                    index = faiss.GpuIndexFlatL2(faiss_resources, 3, config)
                    exact_indices = exact_indices.long()
                    for iteration in range(args.warmups + args.repeats):
                        index.reset(); torch.cuda.synchronize(); start = time.perf_counter()
                        index.add(support); torch.cuda.synchronize(); build = time.perf_counter() - start
                        start = time.perf_counter()
                        index.search(query, k, exact_distances, exact_indices)
                        torch.cuda.synchronize(); search = time.perf_counter() - start
                        if iteration >= args.warmups:
                            exact_times.append({"construction_seconds": build, "query_seconds": search})

                record: dict[str, Any] = {
                    "sample": entry["file"], "mode": mode, "k": k,
                    "num_support": len(support), "num_query": len(query),
                    "exact": {"method": exact_backend, "timings": exact_times},
                    "flashknn": [],
                }
                for alpha in args.alpha:
                    exponent = round(math.log2(alpha))
                    knn = FlashKNN(num_nbr=k, num_down=exponent, debug=True)
                    baseline_memory = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    predicted = None
                    for _ in range(args.warmups + args.repeats):
                        if mode == "pre":
                            predicted = knn.query(grid, batch, support, memory_mode="SM", sorting_mode="PS")
                        else:
                            predicted = knn.selected_query(
                                support, grid, query_indices, batch,
                                dynamic_load=True, memory_mode="SM",
                            )
                    assert predicted is not None
                    timings = knn.time_list[args.warmups:]
                    record["flashknn"].append({
                        "alpha": alpha, "timings": timings,
                        "recall": recall_intersection(torch, exact_indices, predicted),
                        "peak_incremental_allocated_bytes": max(
                            0, torch.cuda.max_memory_allocated() - baseline_memory
                        ),
                    })
                    del predicted, knn

                if not args.skip_legacy:
                    from legacy_backends import benchmark_legacy_methods
                    legacy = benchmark_legacy_methods(
                        torch, support, query, k, args.warmups, args.repeats
                    )
                    for method in ("flann_cuda", "nanoflann"):
                        indices = legacy[method].pop("indices").to(exact_indices.device)
                        record[method] = {
                            "timings": [{
                                "construction_seconds": item["construction_s"],
                                "query_seconds": item["query_s"],
                            } for item in legacy[method]["timings"]],
                            "recall_vs_exact": recall_intersection(torch, exact_indices, indices),
                        }

                if not args.skip_faiss:
                    if faiss is None:
                        raise SystemExit("GPU FAISS is required unless --skip-faiss is set")
                    from faiss_backends import benchmark_faiss_methods
                    target = next(
                        item for item in record["flashknn"]
                        if int(item["alpha"]) == args.ivf_match_alpha
                    )
                    faiss_result = benchmark_faiss_methods(
                        support, query, k, args.warmups, args.repeats,
                        target_recall=target["recall"]["mean"],
                    )
                    record["faiss_flat"] = {
                        "timings": [{
                            "construction_seconds": item["预处理耗时"],
                            "query_seconds": item["查询耗时"],
                        } for item in faiss_result["faiss_flat_time_info"]],
                        "exact": True,
                    }
                    record["faiss_ivf"] = {
                        "timings": [{
                            "construction_seconds": item["预处理耗时"],
                            "query_seconds": item["查询耗时"],
                        } for item in faiss_result["faiss_ivf_time_info"]],
                        "training_seconds": faiss_result["faiss_ivf_training_time"],
                        "nlist": faiss_result["faiss_ivf_nlist"],
                        "nprobe": faiss_result["faiss_ivf_nprobe"],
                        "target_recall": faiss_result["faiss_ivf_target_recall"],
                        "calibration": faiss_result["faiss_ivf_calibration"],
                        "calibration_queries": faiss_result[
                            "faiss_ivf_calibration_queries"
                        ],
                        "recall_vs_exact": faiss_result["faiss_ivf_mean_recall"],
                    }
                if record_key in record_positions:
                    payload["samples"][record_positions[record_key]] = record
                else:
                    record_positions[record_key] = len(payload["samples"])
                    payload["samples"].append(record)
                completed.add(record_key)
                atomic_write(args.output, payload)
                summary = record["flashknn"]
                print(f"[{sample_number}/{len(samples)}] {entry['file']} {mode} k={k} "
                      + " ".join(f"a{x['alpha']}={x['recall']['mean']:.6f}" for x in summary), flush=True)

        del support, grid, post_indices, batch
        torch.cuda.empty_cache()
    print(f"Saved {len(payload['samples'])} sample-mode-k records to {args.output}")


if __name__ == "__main__":
    main()
