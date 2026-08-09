#!/usr/bin/env python3
"""LiDAR end-to-end latency for DeLA/DeepLA with CPU and FlashKNN hierarchies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


VOXEL_SIZES = (0.06, 0.12, 0.24, 0.48)
NEIGHBOUR_COUNTS = (24, 24, 24, 24)
BACKENDS = ("cpu_kdtree", "flashknn")


def arguments() -> argparse.Namespace:
    """Parse the paired LiDAR network benchmark configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dela", "deepla"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--variant", default="24", choices=("default", "24", "60"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--backend", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=22)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed_cuda(torch, function) -> float:
    """Measure one CUDA operation in milliseconds with an event boundary."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def summary(values: list[float]) -> dict[str, float]:
    """Summarize repeated latency observations in milliseconds."""
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "std_ms": statistics.pstdev(values),
    }


def stratified_entries(entries: list[dict], limit: int | None) -> list[dict]:
    """Retain equal sequence coverage when a compact latency subset is used."""
    if limit is None or limit >= len(entries):
        return entries
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        groups[str(entry.get("sequence", entry["file"].split("_", 1)[0]))].append(entry)
    sequences = sorted(groups)
    if limit < len(sequences):
        sequence_ids = np.linspace(0, len(sequences) - 1, num=limit, dtype=np.int64)
        sequences = [sequences[int(index)] for index in sequence_ids]
    base, remainder = divmod(limit, len(sequences))
    selected = []
    for index, sequence in enumerate(sequences):
        count = min(len(groups[sequence]), base + (index < remainder))
        if count:
            positions = np.linspace(0, len(groups[sequence]) - 1, num=count, dtype=np.int64)
            selected.extend(groups[sequence][int(position)] for position in positions)
    return selected


def gpu_metadata(torch, physical: int) -> dict:
    """Collect the physical GPU identity used by the benchmark."""
    metadata = {"name": torch.cuda.get_device_name(0)}
    try:
        line = subprocess.check_output([
            "nvidia-smi", "-i", str(physical),
            "--query-gpu=uuid,driver_version,memory.total", "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()[0]
        uuid, driver, memory = [part.strip() for part in line.split(",")]
        metadata.update(uuid=uuid, driver=driver, memory_mib=int(memory))
    except Exception:
        pass
    return metadata


def main() -> None:
    """Run paired hierarchy-plus-network latency on stratified LiDAR frames."""
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from torch.cuda.amp import autocast
    from common import atomic_json, load_checkpoint
    from hierarchy import build_cpu_hierarchy, build_flash_hierarchy
    from model import create_model
    from utils.cutils import KDTree, grid_subsampling

    torch.set_float32_matmul_precision("high")
    np.random.seed(47)
    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    device = torch.device("cuda")
    model = create_model(args.model, args.repo, args.variant).to(device).eval()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = stratified_entries(manifest["samples"], args.max_samples)
    selected_backends = tuple(dict.fromkeys(args.backend))
    metadata = {
        "dataset": "SemanticKITTI", "model": args.model, "variant": args.variant,
        "backends": list(selected_backends), "alpha": args.alpha,
        "voxel_sizes_m": list(VOXEL_SIZES), "k": list(NEIGHBOUR_COUNTS),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "weights_note": "Random weights if checkpoint is null; operation shapes and latency are unchanged.",
        "timing_boundary": (
            "CUDA-ready voxelized scan; hierarchy/downsampling/KNN plus network forward; "
            "excludes disk I/O and voxelization. CPU backend excludes hierarchy H2D, matching "
            "the paper's CPU-KDTree timing convention."
        ),
        "warmups": args.warmups, "repeats": args.repeats,
        "gpu": gpu_metadata(torch, args.gpu), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "python": platform.python_version(),
        "manifest": manifest,
    }
    records: list[dict] = []
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        old = previous.get("metadata", {})
        fields = (
            "dataset", "model", "variant", "backends", "alpha", "voxel_sizes_m", "k",
            "checkpoint", "warmups", "repeats", "torch", "torch_cuda", "manifest",
        )
        changed = {
            field: (old.get(field), metadata.get(field))
            for field in fields if old.get(field) != metadata.get(field)
        }
        if old.get("gpu", {}).get("uuid") != metadata["gpu"].get("uuid"):
            changed["gpu.uuid"] = (old.get("gpu", {}).get("uuid"), metadata["gpu"].get("uuid"))
        if changed:
            raise SystemExit(f"Refusing to resume incompatible output {args.output}: {changed}")
        records = previous.get("samples", [])
    completed = {
        record["sample"] for record in records
        if set(selected_backends).issubset(record.get("backends", {}))
    }

    def result_payload() -> dict:
        aggregate = {}
        for backend in selected_backends:
            aggregate[backend] = {
                key: summary([record["backends"][backend][key]["mean_ms"] for record in records])
                for key in ("hierarchy", "model", "end_to_end")
            }
            aggregate[backend]["single_sample_throughput_hz"] = (
                1000.0 / aggregate[backend]["end_to_end"]["mean_ms"]
            )
        return {"metadata": metadata, "samples": records, "aggregate": aggregate}

    with torch.inference_mode():
        for number, entry in enumerate(entries, 1):
            if entry["file"] in completed:
                print(f"[{number}/{len(entries)}] skip completed {entry['file']}", flush=True)
                continue
            archive = np.load(args.data_dir / entry["file"])
            metric_xyz_cpu = torch.from_numpy(archive["support_xyz"]).float().contiguous()
            intensity_cpu = torch.from_numpy(archive["support_intensity"][:, None]).float()
            metric_xyz = metric_xyz_cpu.to(device).contiguous()
            remission = intensity_cpu.to(device)
            feature = torch.cat(((metric_xyz - metric_xyz.mean(0)) / 50.0, remission), 1)
            xyz = metric_xyz * (1.6 / float(manifest["voxel_size_m"]))
            sample_record = {
                "sample": entry["file"], "points": len(metric_xyz), "backends": {},
            }

            for backend in selected_backends:
                hierarchy_times: list[float] = []
                model_times: list[float] = []
                total_times: list[float] = []
                for iteration in range(args.warmups + args.repeats):
                    if backend == "cpu_kdtree":
                        start = time.perf_counter()
                        raw_hierarchy = build_cpu_hierarchy(
                            metric_xyz_cpu, VOXEL_SIZES, NEIGHBOUR_COUNTS,
                            grid_subsampling, KDTree,
                        )
                        hierarchy_ms = (time.perf_counter() - start) * 1000.0
                        hierarchy = [item.to(device, non_blocking=True).long() for item in raw_hierarchy[::-1]]
                        torch.cuda.synchronize()
                    else:
                        cached: list[torch.Tensor] = []

                        def flash_hierarchy() -> None:
                            nonlocal cached
                            cached = build_flash_hierarchy(
                                metric_xyz, VOXEL_SIZES, NEIGHBOUR_COUNTS, alpha=args.alpha
                            )

                        hierarchy_ms = timed_cuda(torch, flash_hierarchy)
                        hierarchy = [item.long() for item in cached[::-1]]

                    def model_forward() -> None:
                        with autocast():
                            model(xyz, feature, hierarchy)

                    model_ms = timed_cuda(torch, model_forward)
                    if iteration >= args.warmups:
                        hierarchy_times.append(hierarchy_ms)
                        model_times.append(model_ms)
                        total_times.append(hierarchy_ms + model_ms)

                sample_record["backends"][backend] = {
                    "hierarchy": summary(hierarchy_times),
                    "model": summary(model_times),
                    "end_to_end": summary(total_times),
                }

            records.append(sample_record)
            completed.add(entry["file"])
            atomic_json(args.output, result_payload())
            print(
                f"[{number}/{len(entries)}] {entry['file']} N={len(metric_xyz)} "
                + " ".join(
                    f"{backend}={sample_record['backends'][backend]['end_to_end']['mean_ms']:.3f}ms"
                    for backend in selected_backends
                ),
                flush=True,
            )

    atomic_json(args.output, result_payload())


if __name__ == "__main__":
    main()
