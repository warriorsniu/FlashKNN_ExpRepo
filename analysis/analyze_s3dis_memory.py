#!/usr/bin/env python3
"""Validate and summarize the fixed-250k S3DIS memory benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-rooms", type=int, default=81)
    return parser.parse_args()


def t_critical_95(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        # For the formal 81-room experiment this differs from Student-t by
        # less than 1.6%; environments used for the paper include SciPy.
        return 1.959963984540054


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = t_critical_95(len(values) - 1) * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean_mib": mean,
        "sample_sd_mib": sd,
        "ci95_low_mib": mean - half,
        "ci95_high_mib": mean + half,
        "min_mib": min(values),
        "max_mib": max(values),
    }


def main() -> None:
    args = arguments()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("dataset") != "S3DIS" or metadata.get("scope") != "sample_part":
        raise SystemExit("Expected S3DIS sample_part memory result")
    if int(metadata.get("k", -1)) != 32:
        raise SystemExit("Formal memory result must use k=32")
    expected_methods = set(metadata.get("methods", []))
    expected_modes = set(metadata.get("modes", []))
    grouped: dict[str, dict[str, list[float]]] = {
        mode: {method: [] for method in sorted(expected_methods)}
        for mode in sorted(expected_modes)
    }
    room_sets: dict[str, set[str]] = {mode: set() for mode in expected_modes}
    seen: set[tuple[str, str]] = set()
    for record in payload.get("records", []):
        room, mode = str(record["room"]), str(record["mode"])
        key = (room, mode)
        if key in seen:
            raise SystemExit(f"Duplicate memory record: {key}")
        seen.add(key)
        if mode not in grouped:
            raise SystemExit(f"Unexpected mode: {mode}")
        if int(record.get("num_support", -1)) != 250_000:
            raise SystemExit(f"{key} does not use a 250k support crop")
        methods = record.get("methods", {})
        missing = expected_methods - set(methods)
        if missing:
            raise SystemExit(f"{key} is missing methods: {sorted(missing)}")
        room_sets[mode].add(room)
        for method in expected_methods:
            value = int(methods[method].get("peak_incremental_allocated_bytes", -1))
            if value <= 0:
                raise SystemExit(f"{key}/{method} has invalid peak allocation {value}")
            grouped[mode][method].append(value / (1024.0 * 1024.0))

    for mode, rooms in room_sets.items():
        if len(rooms) != args.expected_rooms:
            raise SystemExit(
                f"{mode}: expected {args.expected_rooms} rooms, found {len(rooms)}"
            )
    if len({frozenset(rooms) for rooms in room_sets.values()}) != 1:
        raise SystemExit("pre/post room sets differ")

    summary: dict[str, Any] = {
        "source": str(args.input),
        "measurement_boundary": metadata.get("measurement_boundary"),
        "gpu": metadata.get("gpu"),
        "k": metadata.get("k"),
        "crop_points": metadata.get("crop_points"),
        "modes": {
            mode: {
                method: stats(values)
                for method, values in methods.items()
            }
            for mode, methods in grouped.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# S3DIS fixed-250k GPU memory summary",
        "",
        f"GPU: `{metadata.get('gpu', {}).get('name', 'unknown')}`; "
        f"$k={metadata.get('k')}$; {args.expected_rooms} rooms; pre/post query.",
        "",
        "Metric: peak incremental method-owned GPU allocation above CUDA-ready "
        "inputs. It includes construction/index, workspace and outputs, and "
        "excludes file I/O, voxelization, crop, H2D and input tensors.",
        "",
        "| Mode | Method | Mean (MiB) | Room SD | 95% CI | Min--max (MiB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode in sorted(summary["modes"]):
        for method, values in summary["modes"][mode].items():
            lines.append(
                f"| {mode} | {method} | {values['mean_mib']:.2f} | "
                f"{values['sample_sd_mib']:.2f} | "
                f"[{values['ci95_low_mib']:.2f}, {values['ci95_high_mib']:.2f}] | "
                f"{values['min_mib']:.2f}--{values['max_mib']:.2f} |"
            )
    lines.extend([
        "",
        "FAISS uses the same default `StandardGpuResources` scratch policy as "
        "the latency benchmark. cudaKDTree memory includes PyTorch output "
        "tensors and all allocations made by its native spatial-tree builder.",
        "",
    ])
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Validated {len(seen)} records; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
