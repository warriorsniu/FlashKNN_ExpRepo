#!/usr/bin/env python3
"""Validate final-kernel NCU reports and record compact L20 provenance.

This utility reads the raw CSV exported from each `.ncu-rep`, retains kernel,
launch and paper-reported metric fields, and writes one machine-readable
provenance file beside the reports.  It does not reinterpret or aggregate NCU
metrics across kernels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BACKENDS = ("cukd", "flash-smps", "flash-gmss")
NCU_SOURCE_FILES = (
    "FlashKNN/csrc/flash_knn_query_dynamic_load.cu",
    "FlashKNN/csrc/flash_knn_query.h",
    "FlashKNN/functions/FlashKnnWrapper.py",
    "FlashKNN/csrc/api.cpp",
    "FlashKNN/csrc/flash_knn_bitonic_top_p.cuh",
    "Query/benchmark_s3dis.py",
    "scripts/profile_knn_threads.py",
)
FIELDS = (
    "Kernel Name",
    "Block Size",
    "Grid Size",
    "Device",
    "CC",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__block_size",
    "launch__grid_size",
    "launch__waves_per_multiprocessor",
)


def arguments() -> argparse.Namespace:
    """Parse the profile directory and immutable source/GPU identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    return parser.parse_args()


def raw_rows(path: Path) -> tuple[list[str], list[str], list[list[str]]]:
    """Read NCU raw CSV header, units and kernel rows.

    Args:
        path: Raw wide CSV produced by `ncu --page raw --csv`.

    Returns:
        A tuple containing column names, units, and one or more kernel rows.
    """
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("==")
    ]
    parsed = list(csv.reader(lines))
    if len(parsed) < 3:
        raise SystemExit(f"NCU raw CSV has no kernel row: {path}")
    header, units, *rows = parsed
    if len(header) != len(units) or any(len(row) != len(header) for row in rows):
        raise SystemExit(f"NCU raw CSV has inconsistent columns: {path}")
    return header, units, rows


def source_hashes(repo: Path) -> dict[str, str]:
    """Hash the production kernel and exact NCU profile entry dependencies."""
    hashes = {}
    for relative in NCU_SOURCE_FILES:
        digest = hashlib.sha256()
        with (repo / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative] = digest.hexdigest()
    return hashes


def profile_summary(path: Path) -> list[dict[str, object]]:
    """Extract required launch and metric values from every profiled kernel row."""
    header, units, rows = raw_rows(path)
    positions = {name: position for position, name in enumerate(header)}
    required = set(FIELDS)
    missing = required - set(positions)
    if missing:
        raise SystemExit(f"NCU CSV {path.name} misses columns: {sorted(missing)}")
    summaries = []
    for row in rows:
        values: dict[str, object] = {}
        for name in FIELDS:
            if name not in positions:
                continue
            position = positions[name]
            value = row[position]
            if name in required and value == "":
                raise SystemExit(f"NCU CSV {path.name} has empty required metric {name}")
            values[name] = {
                "value": value,
                "unit": units[position],
            } if units[position] else value
        summaries.append(values)
    return summaries


def main() -> None:
    """Validate all three reports and write their compact provenance JSON."""
    args = arguments()
    profile_dir = args.profile_dir.resolve()
    repo = args.repo.resolve()
    profiles: dict[str, object] = {}
    for backend in BACKENDS:
        stem = f"{backend}_k32"
        report = profile_dir / f"{stem}.ncu-rep"
        summary_csv = profile_dir / f"{stem}.csv"
        raw_csv = profile_dir / f"{stem}_raw.csv"
        for path in (report, summary_csv, raw_csv):
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"Missing or empty NCU artifact: {path}")
        profiles[backend] = {
            "report": report.name,
            "summary_csv": summary_csv.name,
            "raw_csv": raw_csv.name,
            "kernels": profile_summary(raw_csv),
        }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "production_source_sha256": source_hashes(repo),
        "gpu": {
            "physical_index": args.physical_gpu,
            "uuid": args.gpu_uuid,
            "name": "NVIDIA L20",
            "compute_capability": "8.9",
        },
        "dataset": "S3DIS",
        "crop_points": 250000,
        "voxel_size_m": 0.02,
        "seed": 47,
        "k": 32,
        "profiles": profiles,
    }
    output = profile_dir / "provenance.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
