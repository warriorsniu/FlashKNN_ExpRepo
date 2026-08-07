#!/usr/bin/env python3
"""Create a compact deterministic SemanticKITTI pack for cross-GPU timing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("/data2/Dataset/Lidar/SemanticKITTI/dataset"),
    )
    parser.add_argument("--sequence", help="Legacy single-sequence mode")
    parser.add_argument("--sequences", nargs="+",
                        help="Sequences to sample; defaults to all 00--21")
    parser.add_argument("--samples-per-sequence", type=int, default=5,
                        help="Evenly spaced frames retained from every sequence")
    parser.add_argument("--voxel-size", type=float, default=0.06)
    parser.add_argument("--stride", type=int, default=100,
                        help="Legacy single-sequence stride")
    parser.add_argument("--max-samples", type=int, default=32,
                        help="Legacy single-sequence sample limit")
    parser.add_argument("--clip-range", type=float, default=None,
                        help="Optional symmetric x/y range in meters; omit for full scans")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/lidar/data/semantickitti_allseq_v006_5each"),
    )
    return parser.parse_args()


def voxelize_first(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, first = np.unique(grid, axis=0, return_index=True)
    first.sort()
    return points[first], grid[first]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.sequence and args.sequences:
        raise SystemExit("Use either --sequence or --sequences, not both")
    sequences = args.sequences or ([args.sequence] if args.sequence else [f"{i:02d}" for i in range(22)])
    selected: list[tuple[str, Path]] = []
    sequence_counts = {}
    for sequence in sequences:
        scan_dir = args.root / "sequences" / sequence / "velodyne"
        paths = sorted(scan_dir.glob("*.bin"))
        if not paths:
            raise SystemExit(f"No scans found under {scan_dir}")
        sequence_counts[sequence] = len(paths)
        if args.sequence:
            chosen = paths[::args.stride][:args.max_samples]
        else:
            count = min(args.samples_per_sequence, len(paths))
            positions = np.linspace(0, len(paths) - 1, num=count, dtype=np.int64)
            chosen = [paths[int(position)] for position in positions]
        selected.extend((sequence, path) for path in chosen)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for sequence, path in selected:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        raw_points = len(points)
        if args.clip_range is not None:
            r = args.clip_range
            mask = ((points[:, 0] >= -r) & (points[:, 0] < r) &
                    (points[:, 1] >= -r) & (points[:, 1] < r))
            points = points[mask]
        support, grid = voxelize_first(points, args.voxel_size)
        query_grid = np.floor(support[:, :3] / (2 * args.voxel_size)).astype(np.int64)
        _, query_indices = np.unique(query_grid, axis=0, return_index=True)
        query_indices.sort()
        output = args.output_dir / f"{sequence}_{path.stem}.npz"
        np.savez_compressed(
            output,
            support_xyz=support[:, :3].astype(np.float32),
            support_intensity=support[:, 3].astype(np.float32),
            grid_coord=grid.astype(np.int64),
            post_query_indices=query_indices.astype(np.int64),
        )
        samples.append({
            "file": output.name, "sha256": sha256(output), "source": str(path),
            "sequence": sequence, "frame": path.stem,
            "raw_points": raw_points, "clipped_points": len(points),
            "support_points": len(support), "post_query_points": len(query_indices),
        })
        print(f"{sequence}/{path.name}: {raw_points} -> {len(support)} support, {len(query_indices)} post")
    manifest = {
        "format_version": 2,
        "dataset": "SemanticKITTI",
        "sequences": sequences,
        "source_frame_counts": sequence_counts,
        "voxel_size_m": args.voxel_size,
        "post_voxel_size_m": 2 * args.voxel_size,
        "clip_range_xy_m": args.clip_range,
        "selection": (
            {"strategy": "legacy_stride", "stride": args.stride, "max_samples": args.max_samples}
            if args.sequence else
            {"strategy": "per_sequence_linspace", "samples_per_sequence": args.samples_per_sequence}
        ),
        "voxel_representative": "first point in raw frame order",
        "samples": samples,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
