#!/usr/bin/env python3
"""Validate and summarize paired SemanticKITTI network-latency results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t as student_t


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dela", type=Path, required=True)
    parser.add_argument("--deepla", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = float(student_t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return {"n": len(values), "mean_ms": mean, "sample_sd_ms": sd,
            "ci95_low_ms": mean - half, "ci95_high_ms": mean + half}


def main() -> None:
    args = arguments()
    rows = []
    protocol = None
    for model, path in (("dela", args.dela), ("deepla", args.deepla)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload["metadata"]
        current = (meta["alpha"], meta["warmups"], meta["repeats"],
                   meta["gpu"].get("uuid"), meta["torch"], meta["torch_cuda"])
        if protocol is None:
            protocol = current
        if current != protocol or current[:3] != (8, 10, 30):
            raise SystemExit(f"Incompatible protocol in {path}: {current}")
        samples = payload["samples"]
        if len(samples) != 22:
            raise SystemExit(f"Expected 22 frames in {path}, found {len(samples)}")
        for backend in ("cpu_kdtree", "flashknn"):
            row = {"model_name": model, "backend": backend}
            for metric in ("hierarchy", "model", "end_to_end"):
                values = [float(sample["backends"][backend][metric]["mean_ms"])
                          for sample in samples]
                row[metric] = summary(values)
            rows.append(row)
    result = {
        "protocol": {"alpha": protocol[0], "warmups": protocol[1], "repeats": protocol[2],
                     "gpu_uuid": protocol[3], "torch": protocol[4], "torch_cuda": protocol[5],
                     "aggregation": "per-frame repeat mean, then mean/sample SD/t-interval over 22 frames"},
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SemanticKITTI alpha=8 network latency",
        "",
        "| Model | Backend | Hierarchy ms | Model ms | End-to-end ms | Speedup vs CPU |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        cpu = next(item for item in rows if item["model_name"] == row["model_name"] and item["backend"] == "cpu_kdtree")
        speedup = cpu["end_to_end"]["mean_ms"] / row["end_to_end"]["mean_ms"]
        lines.append(
            f"| {row['model_name']} | {row['backend']} | {row['hierarchy']['mean_ms']:.3f} | "
            f"{row['model']['mean_ms']:.3f} | {row['end_to_end']['mean_ms']:.3f} | {speedup:.3f}x |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
