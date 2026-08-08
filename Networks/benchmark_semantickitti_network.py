#!/usr/bin/env python3
"""Single-scan CUDA-ready end-to-end latency for DeLA/DeepLA + FlashKNN."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dela", "deepla"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--variant", default="24", choices=("default", "24", "60"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=22)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed(torch, function) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def summary(values: list[float]) -> dict[str, float]:
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
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torch.cuda.amp import autocast
    from common import atomic_json, load_checkpoint
    from hierarchy import build_flash_hierarchy
    from model import create_model

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
    metadata = {
        "model": args.model, "variant": args.variant, "alpha": args.alpha,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "weights_note": "Random weights if checkpoint is null; operation shapes and latency are unchanged.",
        "timing_boundary": "CUDA-ready voxelized scan; hierarchy/downsampling/KNN + network forward; excludes disk I/O, voxelization and H2D",
        "warmups": args.warmups, "repeats": args.repeats,
        "gpu": gpu_metadata(torch, args.gpu), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "python": platform.python_version(),
        "manifest": manifest,
    }
    records = []
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        old = previous.get("metadata", {})
        fields = ("model", "variant", "alpha", "checkpoint", "warmups", "repeats",
                  "torch", "torch_cuda", "manifest")
        changed = {field: (old.get(field), metadata.get(field))
                   for field in fields if old.get(field) != metadata.get(field)}
        if old.get("gpu", {}).get("uuid") != metadata["gpu"].get("uuid"):
            changed["gpu.uuid"] = (old.get("gpu", {}).get("uuid"), metadata["gpu"].get("uuid"))
        if changed:
            raise SystemExit(f"Refusing to resume incompatible output {args.output}: {changed}")
        records = previous.get("samples", [])
    completed = {record["sample"] for record in records if "end_to_end" in record}

    def result_payload() -> dict:
        aggregate = {
            key: summary([record[key]["mean_ms"] for record in records])
            for key in ("hierarchy", "model", "end_to_end")
        }
        aggregate["single_sample_throughput_hz"] = 1000.0 / aggregate["end_to_end"]["mean_ms"]
        return {"metadata": metadata, "samples": records, "aggregate": aggregate}

    with torch.inference_mode():
        for number, entry in enumerate(entries, 1):
            if entry["file"] in completed:
                print(f"[{number}/{len(entries)}] skip completed {entry['file']}", flush=True)
                continue
            archive = np.load(args.data_dir / entry["file"])
            metric_xyz = torch.from_numpy(archive["support_xyz"]).to(device).contiguous()
            remission = torch.from_numpy(archive["support_intensity"][:, None]).to(device)
            feature = torch.cat(((metric_xyz - metric_xyz.mean(0)) / 50.0, remission), 1).float()
            xyz = metric_xyz * (1.6 / float(manifest["voxel_size_m"]))
            cached = build_flash_hierarchy(metric_xyz, alpha=args.alpha)

            def hierarchy_only():
                nonlocal cached
                cached = build_flash_hierarchy(metric_xyz, alpha=args.alpha)

            def model_only():
                with autocast():
                    model(xyz, feature, [item.long() for item in cached[::-1]])

            def end_to_end():
                hierarchy = build_flash_hierarchy(metric_xyz, alpha=args.alpha)
                with autocast():
                    model(xyz, feature, [item.long() for item in hierarchy[::-1]])

            for _ in range(args.warmups):
                end_to_end()
            torch.cuda.synchronize()
            baseline_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            hierarchy_times, model_times, total_times = [], [], []
            for _ in range(args.repeats):
                hierarchy_times.append(timed(torch, hierarchy_only))
                model_times.append(timed(torch, model_only))
                total_times.append(timed(torch, end_to_end))
            records.append({
                "sample": entry["file"], "points": len(metric_xyz),
                "hierarchy": summary(hierarchy_times), "model": summary(model_times),
                "end_to_end": summary(total_times),
                "peak_incremental_allocated_bytes": max(
                    0, torch.cuda.max_memory_allocated() - baseline_memory
                ),
            })
            completed.add(entry["file"])
            atomic_json(args.output, result_payload())
            print(f"[{number}/{len(entries)}] {entry['file']} N={len(metric_xyz)} "
                  f"e2e={records[-1]['end_to_end']['mean_ms']:.3f} ms", flush=True)

    atomic_json(args.output, result_payload())


if __name__ == "__main__":
    main()
