#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


p = argparse.ArgumentParser()
p.add_argument("--s3dis", type=Path, required=True)
p.add_argument("--semantickitti", type=Path, required=True)
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--data-only", action="store_true")
p.add_argument(
    "--quick", action="store_true",
    help="Check layout/counts without re-reading and hashing every prepared sample",
)
args = p.parse_args()

rooms = sorted(args.s3dis.glob("Area_*/*.pth"))
layout = "legacy_pth"
if not rooms:
    area_dirs = sorted([*args.s3dis.glob("Area_*"), *args.s3dis.glob("area_*")])
    rooms = sorted(
        room for area in area_dirs for room in area.iterdir()
        if room.is_dir() and (room / "coord.npy").is_file()
    )
    layout = "per_field_npy"
if len(rooms) != 272:
    raise SystemExit(f"Expected 272 S3DIS rooms, found {len(rooms)} below {args.s3dis}")
if layout == "per_field_npy" and not args.quick:
    for room in rooms:
        required = [room / "coord.npy", room / "color.npy"]
        if not any((room / name).is_file() for name in ("segment.npy", "semantic_gt.npy")):
            required.append(room / "segment.npy")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"Incomplete S3DIS room {room}: missing {missing}")
        coord = np.load(room / "coord.npy", mmap_mode="r")
        color = np.load(room / "color.npy", mmap_mode="r")
        segment_path = next(
            room / name for name in ("segment.npy", "semantic_gt.npy")
            if (room / name).is_file()
        )
        segment = np.load(segment_path, mmap_mode="r")
        if coord.ndim != 2 or coord.shape[1] != 3 or coord.shape[0] == 0:
            raise SystemExit(f"Invalid coord.npy shape in {room}: {coord.shape}")
        if color.shape != coord.shape or segment.reshape(-1).shape[0] != coord.shape[0]:
            raise SystemExit(
                f"Inconsistent Pointcept fields in {room}: "
                f"coord={coord.shape}, color={color.shape}, segment={segment.shape}"
            )
        for optional, width in (("normal.npy", 3), ("instance.npy", None)):
            path = room / optional
            if not path.is_file():
                continue
            value = np.load(path, mmap_mode="r")
            valid = value.shape[0] == coord.shape[0]
            if width is not None:
                valid = valid and value.ndim == 2 and value.shape[1] == width
            if not valid:
                raise SystemExit(
                    f"Inconsistent {optional} in {room}: {value.shape} vs {coord.shape}"
                )
manifest_path = args.semantickitti / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Missing SemanticKITTI manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not manifest.get("samples"):
    raise SystemExit("SemanticKITTI manifest contains no samples")
expected_sequences = [f"{i:02d}" for i in range(22)]
if manifest.get("sequences") != expected_sequences:
    raise SystemExit(
        "SemanticKITTI pack is obsolete: expected equal sampling from sequences 00--21"
    )
per_sequence = {sequence: 0 for sequence in expected_sequences}
seen_files = set()
for sample in manifest["samples"]:
    sample_path = args.semantickitti / sample["file"]
    if not sample_path.is_file():
        raise SystemExit(f"Missing SemanticKITTI sample: {sample['file']}")
    if sample["file"] in seen_files:
        raise SystemExit(f"Duplicate SemanticKITTI sample: {sample['file']}")
    seen_files.add(sample["file"])
    if not args.quick:
        digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
        if digest != sample.get("sha256"):
            raise SystemExit(f"SHA-256 mismatch for SemanticKITTI sample: {sample['file']}")
        archive = np.load(sample_path)
        required_arrays = {
            "support_xyz", "support_intensity", "grid_coord", "post_query_indices"
        }
        missing_arrays = required_arrays - set(archive.files)
        if missing_arrays:
            raise SystemExit(
                f"SemanticKITTI {sample['file']} misses arrays: {sorted(missing_arrays)}"
            )
        support = archive["support_xyz"]
        intensity = archive["support_intensity"]
        grid = archive["grid_coord"]
        post = archive["post_query_indices"]
        if support.ndim != 2 or support.shape[1] != 3 or grid.shape != support.shape:
            raise SystemExit(
                f"Invalid support/grid shape in {sample['file']}: "
                f"{support.shape}, {grid.shape}"
            )
        if intensity.reshape(-1).shape[0] != support.shape[0] or post.ndim != 1:
            raise SystemExit(f"Invalid intensity/post indices in {sample['file']}")
        if len(post) and (post.min() < 0 or post.max() >= support.shape[0]):
            raise SystemExit(f"Out-of-range post indices in {sample['file']}")
        archive.close()
    per_sequence[sample.get("sequence", "")] = per_sequence.get(sample.get("sequence", ""), 0) + 1
if set(per_sequence.values()) != {5}:
    raise SystemExit(f"Expected exactly 5 frames per sequence, got {per_sequence}")

if not args.data_only:
    import torch
    import faiss
    import Cukd.CuFun
    import PyTorchCudaFlann
    import PyTorchNanoFlann
    from FlashKNN import FlashKNN
    if not torch.__version__.startswith("2.7.1+") or torch.version.cuda != "12.8":
        raise SystemExit(
            f"Expected unified PyTorch 2.7.1+cu128, got {torch.__version__} "
            f"with CUDA {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.gpu >= torch.cuda.device_count():
        raise SystemExit(f"GPU {args.gpu} is outside visible device count {torch.cuda.device_count()}")
    if not hasattr(faiss, "StandardGpuResources"):
        raise SystemExit("FAISS was installed without GPU support")
    print(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    print(f"Torch {torch.__version__}, CUDA {torch.version.cuda}, FlashKNN={FlashKNN.__name__}")
suffix = ", quick layout check" if args.quick else ", full content check"
print(f"S3DIS rooms: {len(rooms)} ({layout}{suffix})")
print(f"SemanticKITTI samples: {len(manifest['samples'])}{suffix}")
