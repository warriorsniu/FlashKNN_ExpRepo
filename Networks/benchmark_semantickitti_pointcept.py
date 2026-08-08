#!/usr/bin/env python3
"""Forward latency of four Pointcept backbones on the shared LiDAR pack."""

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


CONFIGS = {
    "spunet": "configs/semantic_kitti/semseg-spunet-v1m1-0-base.py",
    "minkunet34c": "configs/semantic_kitti/semseg-minkunet34c-0-base.py",
    "ptv3": "configs/nuscenes/semseg-pt-v3m1-0-base.py",
    "octformer": "configs/s3dis/semseg-octformer-v1m1-0-base.py",
}


def arguments() -> argparse.Namespace:
    """Parse one Pointcept LiDAR latency run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--pointcept-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=22)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    """Summarize repeated CUDA latency observations in milliseconds."""
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "std_ms": statistics.pstdev(values),
    }


def stratified_entries(entries: list[dict], limit: int | None) -> list[dict]:
    """Select frames evenly across SemanticKITTI sequences."""
    if limit is None or limit >= len(entries):
        return entries
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        groups[str(entry.get("sequence", entry["file"].split("_", 1)[0]))].append(entry)
    sequences = sorted(groups)
    if limit < len(sequences):
        positions = np.linspace(0, len(sequences) - 1, num=limit, dtype=np.int64)
        sequences = [sequences[int(position)] for position in positions]
    base, remainder = divmod(limit, len(sequences))
    selected: list[dict] = []
    for index, sequence in enumerate(sequences):
        count = min(len(groups[sequence]), base + (index < remainder))
        if count:
            positions = np.linspace(0, len(groups[sequence]) - 1, num=count, dtype=np.int64)
            selected.extend(groups[sequence][int(position)] for position in positions)
    return selected


def gpu_metadata(torch, physical: int) -> dict:
    """Collect physical GPU identity and driver information."""
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


def configure_model(cfg, model_name: str) -> None:
    """Adapt input/output widths while preserving each published backbone shape.

    Args:
        cfg: Pointcept configuration loaded from the selected architecture.
        model_name: Stable benchmark identifier from :data:`CONFIGS`.
    """
    if model_name == "ptv3":
        cfg.model.num_classes = 19
        cfg.model.backbone.in_channels = 4
        # Match the repository's S3DIS PTv3 latency baseline when FlashAttention
        # is unavailable in the unified PyTorch 2.7/CUDA 12.8 environment.
        cfg.model.backbone.enable_flash = False
        cfg.model.backbone.upcast_attention = True
        cfg.model.backbone.upcast_softmax = True
        cfg.model.backbone.enc_patch_size = (128, 128, 128, 128, 128)
        cfg.model.backbone.dec_patch_size = (128, 128, 128, 128)
    elif model_name == "minkunet34c":
        cfg.model.backbone.in_channels = 4
        cfg.model.backbone.out_channels = 19
    else:
        cfg.model.backbone.in_channels = 4
        cfg.model.backbone.num_classes = 19
        if model_name == "octformer":
            # Preserve the S3DIS configuration's approximately 4 cm finest
            # octree cell while covering the roughly +/-80 m LiDAR extent.
            cfg.model.backbone.octree_scale_factor = 81.92
            cfg.model.backbone.octree_depth = 12


def input_adaptation(model_name: str) -> str:
    """Describe the minimal LiDAR input adaptation applied to one backbone."""
    base = "Four channels (XYZ plus remission) and 19 SemanticKITTI output classes."
    if model_name == "ptv3":
        return base + " Uses the same non-FlashAttention 128-point patch fallback as the S3DIS latency baseline."
    if model_name == "octformer":
        return base + " Uses zero normals because the shared LiDAR pack has no normals, and a 12-level octree scaled to cover the LiDAR extent while retaining an approximately 4 cm finest cell."
    return base


def main() -> None:
    """Benchmark CUDA-ready network forward on 22 stratified LiDAR scans."""
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from pointcept.models import build_model
    from pointcept.utils.config import Config
    from common import atomic_json

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    np.random.seed(47)
    pointcept_root = args.pointcept_root.resolve()
    config_path = CONFIGS[args.model]
    cfg = Config.fromfile(pointcept_root / config_path)
    configure_model(cfg, args.model)
    model = build_model(cfg.model).cuda().eval()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = stratified_entries(manifest["samples"], args.max_samples)
    metadata = {
        "dataset": "SemanticKITTI", "model": args.model,
        "weights": "random initialization", "config": config_path,
        "input_adaptation": input_adaptation(args.model),
        "voxel_size_m": manifest["voxel_size_m"],
        "timing_boundary": "CUDA-ready voxelized scan; network forward only; excludes file I/O, voxelization and H2D",
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
            "dataset", "model", "config", "input_adaptation", "voxel_size_m",
            "warmups", "repeats", "torch", "torch_cuda", "manifest",
        )
        changed = {
            field: (old.get(field), metadata.get(field))
            for field in fields if old.get(field) != metadata.get(field)
        }
        if old.get("gpu", {}).get("uuid") != metadata["gpu"].get("uuid"):
            changed["gpu.uuid"] = (old.get("gpu", {}).get("uuid"), metadata["gpu"].get("uuid"))
        if changed:
            raise SystemExit(f"Refusing to resume incompatible output {args.output}: {changed}")
        records = previous.get("records", [])
    completed = {record["sample"] for record in records if "network" in record}
    payload = {"metadata": metadata, "records": records}

    with torch.inference_mode():
        for number, entry in enumerate(entries, 1):
            if entry["file"] in completed:
                print(f"[{number}/{len(entries)}] skip completed {entry['file']}", flush=True)
                continue
            archive = np.load(args.data_dir / entry["file"])
            coord = torch.from_numpy(archive["support_xyz"]).float().cuda().contiguous()
            remission = torch.from_numpy(archive["support_intensity"][:, None]).float().cuda()
            grid_coord = torch.from_numpy(archive["grid_coord"]).long().cuda().contiguous()
            grid_coord = grid_coord - grid_coord.amin(dim=0, keepdim=True)
            feature = torch.cat((coord, remission), dim=1).contiguous()
            normal = torch.zeros_like(coord)
            offset = torch.tensor([len(coord)], dtype=torch.long, device="cuda")
            batch = torch.zeros(len(coord), dtype=torch.long, device="cuda")

            def forward() -> None:
                model({
                    "coord": coord, "grid_coord": grid_coord, "feat": feature,
                    "normal": normal, "offset": offset, "batch": batch,
                })

            timings: list[float] = []
            for iteration in range(args.warmups + args.repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                forward()
                end.record()
                end.synchronize()
                if iteration >= args.warmups:
                    timings.append(float(start.elapsed_time(end)))
            records.append({
                "sample": entry["file"], "points": len(coord),
                "network": summary(timings),
            })
            completed.add(entry["file"])
            atomic_json(args.output, payload)
            print(
                f"[{number}/{len(entries)}] {entry['file']} N={len(coord)} "
                f"network={records[-1]['network']['mean_ms']:.3f}ms",
                flush=True,
            )
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
