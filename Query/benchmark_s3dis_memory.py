#!/usr/bin/env python3
"""Measure incremental GPU memory for the fixed-250k S3DIS protocol.

The CUDA-ready support/query/grid tensors are prepared before each backend's
baseline.  Reported bytes therefore cover method-owned construction, index,
workspace, and output allocations, but exclude file I/O, voxelization, crop,
H2D, and the input tensors themselves.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from benchmark_s3dis import (
    atomic_json,
    co_tenant_snapshot,
    git_identity,
    gpu_info,
    load_xyz,
    prepare,
    room_paths,
    sha256,
)


METHODS = ("flashknn", "cuda_kdtree", "faiss_flat", "faiss_ivf")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--canonical-s3dis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", nargs="+", choices=("pre", "post"),
                        default=("pre", "post"))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=METHODS)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--num-down", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_allocator(torch: Any) -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def torch_peak(torch: Any, baseline_allocated: int) -> tuple[int, int]:
    return (
        max(0, int(torch.cuda.max_memory_allocated()) - baseline_allocated),
        int(torch.cuda.max_memory_reserved()),
    )


def measure_flashknn(
    torch: Any,
    FlashKNN: Any,
    support: Any,
    grid: Any,
    query_indices: Any,
    batch: Any,
    mode: str,
    k: int,
    num_down: int,
) -> dict[str, Any]:
    clean_allocator(torch)
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    model = FlashKNN(num_nbr=k, num_down=num_down, debug=False)
    if mode == "pre":
        output = model.query(
            grid, batch, support, memory_mode="SM", sorting_mode="PS"
        )
    else:
        output = model.selected_query(
            support, grid, query_indices, batch,
            dynamic_load=True, memory_mode="SM",
        )
    torch.cuda.synchronize()
    peak_allocated, peak_reserved = torch_peak(torch, baseline_allocated)
    result = {
        "peak_incremental_allocated_bytes": peak_allocated,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "accounting": "PyTorch CUDA caching allocator active-allocation high-water mark",
    }
    del output, model
    clean_allocator(torch)
    return result


def measure_cuda_kdtree(
    torch: Any,
    cukd: Any,
    support: Any,
    query: Any,
    k: int,
) -> dict[str, Any]:
    if not hasattr(cukd, "CukdKnnQueryTorchMemory"):
        raise RuntimeError(
            "Cukd extension lacks CukdKnnQueryTorchMemory; rebuild the bundled "
            "Query/ThirdParty/cudaKDTree after pulling this revision"
        )
    # The upstream spatial builder reorders support in place.  Treat this
    # private copy as the backend's CUDA-ready input and exclude it from the
    # incremental method footprint, just as the shared support input is
    # excluded for every backend.
    method_support = support.clone().contiguous()
    method_query = method_support if support.data_ptr() == query.data_ptr() else query
    clean_allocator(torch)
    indices = torch.empty((len(method_query), k), device="cuda", dtype=torch.int32)
    distances = torch.empty((len(method_query), k), device="cuda", dtype=torch.float32)
    memory_info = torch.zeros(4, dtype=torch.int64)
    cukd.CukdKnnQueryTorchMemory(
        method_support, method_query, k, indices, distances, memory_info
    )
    torch.cuda.synchronize()
    output_bytes, builder_peak, persistent_tree, total_peak = (
        int(value) for value in memory_info.tolist()
    )
    result = {
        "peak_incremental_allocated_bytes": total_peak,
        "output_bytes": output_bytes,
        "builder_and_tree_peak_bytes": builder_peak,
        "persistent_tree_bytes": persistent_tree,
        "accounting": (
            "tracked cudaMallocAsync builder/tree high-water mark plus "
            "PyTorch output tensor bytes"
        ),
    }
    del indices, distances, memory_info, method_query, method_support
    clean_allocator(torch)
    return result


def faiss_memory_snapshot(resources: Any) -> dict[str, Any]:
    raw = resources.getMemoryInfo()
    categories: dict[str, dict[str, int]] = {}
    total = 0
    for device_values in raw.values():
        for name, (count, size) in device_values.items():
            entry = categories.setdefault(name, {"count": 0, "bytes": 0})
            entry["count"] += int(count)
            entry["bytes"] += int(size)
            total += int(size)
    return {"total_bytes": total, "categories": categories}


def measure_faiss(
    torch: Any,
    support: Any,
    query: Any,
    k: int,
    method: str,
    nlist: int | None,
    nprobe: int | None,
    seed: int,
) -> dict[str, Any]:
    from faiss_backends import (
        _flat_index,
        _ivf_index,
        _search,
        import_faiss,
    )

    clean_allocator(torch)
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    faiss = import_faiss()
    resources = faiss.StandardGpuResources()
    snapshots: list[dict[str, Any]] = []
    if method == "faiss_flat":
        index = _flat_index(faiss, resources)
        index.add(support)
        torch.cuda.synchronize()
        snapshots.append({"phase": "add", **faiss_memory_snapshot(resources)})
    else:
        if nlist is None or nprobe is None:
            raise RuntimeError("FAISS IVF requires canonical nlist/nprobe")
        index = _ivf_index(faiss, resources, nlist, seed)
        index.train(support)
        torch.cuda.synchronize()
        snapshots.append({"phase": "train", **faiss_memory_snapshot(resources)})
        index.reserveMemory(len(support))
        index.nprobe = nprobe
        index.add(support)
        torch.cuda.synchronize()
        snapshots.append({"phase": "add", **faiss_memory_snapshot(resources)})
    _, output = _search(index, query, k)
    torch.cuda.synchronize()
    snapshots.append({"phase": "search", **faiss_memory_snapshot(resources)})
    peak_allocated, peak_reserved = torch_peak(torch, baseline_allocated)
    peak_faiss = max(item["total_bytes"] for item in snapshots)
    result = {
        "peak_incremental_allocated_bytes": peak_faiss + peak_allocated,
        "pytorch_peak_incremental_allocated_bytes": peak_allocated,
        "faiss_peak_resource_bytes": peak_faiss,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "resource_snapshots": snapshots,
        "accounting": (
            "FAISS StandardGpuResources allocation ledger plus PyTorch "
            "output-tensor active-allocation high-water mark"
        ),
    }
    if method == "faiss_ivf":
        result.update({"nlist": nlist, "nprobe": nprobe})
    del output, index, resources
    clean_allocator(torch)
    return result


def canonical_configuration(
    path: Path, k: int
) -> tuple[set[str], dict[tuple[str, str], dict[str, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rooms: set[str] = set()
    ivf: dict[tuple[str, str], dict[str, int]] = {}
    for record in payload.get("records", []):
        if record.get("scope") != "sample_part" or int(record.get("k", -1)) != k:
            continue
        room = str(record["room"])
        mode = str(record["mode"])
        rooms.add(room)
        faiss_ivf = record.get("methods", {}).get("faiss_ivf", {})
        if "nlist" in faiss_ivf and "nprobe" in faiss_ivf:
            ivf[(room, mode)] = {
                "nlist": int(faiss_ivf["nlist"]),
                "nprobe": int(faiss_ivf["nprobe"]),
            }
    if not rooms:
        raise RuntimeError(f"No sample_part k={k} records in {path}")
    return rooms, ivf


def main() -> None:
    args = arguments()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {args.output}; pass --overwrite")
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
    canonical_rooms, ivf_configuration = canonical_configuration(
        args.canonical_s3dis, args.k
    )
    paths = [
        path for path in room_paths(args.data_root)
        if path.relative_to(args.data_root).as_posix() in canonical_rooms
    ]
    if args.max_samples is not None:
        paths = paths[:args.max_samples]
    if not paths:
        raise RuntimeError("No canonical S3DIS rooms resolved below data root")

    repo = Path(__file__).resolve().parents[1]
    source_files = (
        "Query/benchmark_s3dis_memory.py",
        "Query/benchmark_s3dis.py",
        "Query/faiss_backends.py",
        "Query/ThirdParty/cudaKDTree/cukd/api.cu",
        "FlashKNN/functions/FlashKnnWrapper.py",
        "FlashKNN/csrc/flash_knn_query_dynamic_load.cu",
    )
    payload: dict[str, Any] = {
        "metadata": {
            "dataset": "S3DIS",
            "scope": "sample_part",
            "modes": list(args.mode),
            "k": args.k,
            "voxel_size_m": args.voxel_size,
            "crop_points": args.crop_points,
            "num_down": args.num_down,
            "seed": args.seed,
            "methods": list(args.methods),
            "gpu": gpu_info(torch, args.gpu),
            "physical_gpu_index": args.gpu,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "git": git_identity(repo),
            "canonical_s3dis": str(args.canonical_s3dis.resolve()),
            "canonical_s3dis_sha256": file_sha256(args.canonical_s3dis),
            "source_sha256": {
                relative: sha256(repo / relative) for relative in source_files
            },
            "extensions": {
                "flashknn": {
                    "path": str(Path(flash_cuda.__file__).resolve()),
                    "sha256": sha256(Path(flash_cuda.__file__).resolve()),
                },
                "cuda_kdtree": {
                    "path": str(Path(cukd.__file__).resolve()),
                    "sha256": sha256(Path(cukd.__file__).resolve()),
                    "memory_api": hasattr(cukd, "CukdKnnQueryTorchMemory"),
                },
            },
            "measurement_boundary": (
                "Peak incremental method-owned GPU allocation above CUDA-ready "
                "support/query/grid/batch tensors; includes construction/index, "
                "workspace and outputs; excludes file I/O, voxelization, crop, "
                "H2D and input tensors"
            ),
            "faiss_policy": (
                "Default StandardGpuResources scratch configuration used by the "
                "latency benchmark; IVF nlist/nprobe copied per room from the "
                "canonical matched-recall result"
            ),
            "co_tenant_start": co_tenant_snapshot(),
        },
        "records": [],
    }

    for room_number, path in enumerate(paths, 1):
        coord = load_xyz(torch, path)
        crop_center = None
        room = path.relative_to(args.data_root).as_posix()
        for mode in args.mode:
            sample = prepare(
                torch, xyz2key, coord, path, mode, "sample_part",
                args.voxel_size, args.crop_points, crop_center, crop_generator,
            )
            if sample is None:
                raise RuntimeError(f"Canonical room became ineligible: {room}")
            support, grid, query_indices, crop_center = sample
            query = support if mode == "pre" else support[query_indices].contiguous()
            batch = torch.zeros(len(support), device="cuda", dtype=torch.long)
            methods: dict[str, Any] = {}
            for method in args.methods:
                if method == "flashknn":
                    methods[method] = measure_flashknn(
                        torch, FlashKNN, support, grid, query_indices, batch,
                        mode, args.k, args.num_down,
                    )
                elif method == "cuda_kdtree":
                    methods[method] = measure_cuda_kdtree(
                        torch, cukd, support, query, args.k
                    )
                else:
                    configuration = ivf_configuration.get((room, mode), {})
                    methods[method] = measure_faiss(
                        torch, support, query, args.k, method,
                        configuration.get("nlist"),
                        configuration.get("nprobe"), args.seed,
                    )
            payload["records"].append({
                "room": room,
                "mode": mode,
                "scope": "sample_part",
                "k": args.k,
                "num_support": len(support),
                "num_query": len(query),
                "methods": methods,
            })
            atomic_json(args.output, payload)
            values = ", ".join(
                f"{name}={result['peak_incremental_allocated_bytes'] / 2**20:.1f}MiB"
                for name, result in methods.items()
            )
            print(
                f"[{room_number}/{len(paths)}] {room} {mode} k={args.k}: {values}",
                flush=True,
            )
            del support, grid, query_indices, query, batch
            clean_allocator(torch)
        del coord
        clean_allocator(torch)

    payload["metadata"]["co_tenant_end"] = co_tenant_snapshot()
    payload["metadata"]["records"] = len(payload["records"])
    atomic_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output}")


if __name__ == "__main__":
    main()
