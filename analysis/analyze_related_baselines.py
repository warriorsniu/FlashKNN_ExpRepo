#!/usr/bin/env python3
"""Summarize matched FlashKNN, cudaKDTree, ball-query, and Arkade results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    """Parse result files and the generated Markdown destination."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn", type=Path, required=True)
    parser.add_argument("--ball", type=Path, required=True)
    parser.add_argument("--ball-sweep", type=Path)
    parser.add_argument("--arkade", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    """Load one benchmark result and reject incomplete temporary files."""
    return json.loads(path.read_text(encoding="utf-8"))


def median_timing(record: dict[str, Any], method: str, field: str) -> float:
    """Return the per-room median in milliseconds from standard kNN records."""
    values = [item[field] for item in record["methods"][method]["timings"]]
    return statistics.median(values) * 1000.0


def grouped(records: list[dict[str, Any]], mode: str, k: int) -> list[dict[str, Any]]:
    """Select one query mode and neighborhood size."""
    return [record for record in records if record["mode"] == mode and int(record["k"]) == k]


def main() -> None:
    """Generate compact paper-facing tables with explicit semantic caveats."""
    args = arguments()
    knn = load(args.knn)
    ball = load(args.ball)
    arkade = load(args.arkade)
    ball_sweep = load(args.ball_sweep) if args.ball_sweep else None
    lines = [
        "# L20 ball-query and Arkade comparison",
        "",
        f"All values use the same 250,000-point S3DIS crops on NVIDIA L20 ({len({record['room'] for record in ball['records']})} rooms). The tables report the median of per-room median times. Pointcept ball query uses a global 90th-percentile radius and `nsample=k`; Arkade uses the official TrueKNN radius-doubling path starting from 0.02 m. These operators have different semantics, so latency must be read together with recall and coverage.",
        "",
        "| Mode | k | FlashKNN ms | cudaKDTree ms | Ball query ms | Ball recall | Ball valid | Arkade ms | Arkade recall | Arkade rounds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("pre", "post"):
        for k in (24, 32, 48):
            knn_records = grouped(knn["records"], mode, k)
            ball_records = grouped(ball["records"], mode, k)
            arkade_records = grouped(arkade["records"], mode, k)
            if not knn_records or not ball_records or not arkade_records:
                raise RuntimeError(f"incomplete records for {mode} k={k}")
            flash_ms = statistics.median(
                median_timing(record, "flashknn", "query_s") for record in knn_records
            )
            cukd_ms = statistics.median(
                median_timing(record, "cuda_kdtree", "query_s") for record in knn_records
            )
            ball_ms = statistics.median(
                statistics.median(record["query_timings_s"]) * 1000.0
                for record in ball_records
            )
            ball_recall = statistics.mean(
                record["recall_vs_cukd"]["mean"] for record in ball_records
            )
            ball_valid = statistics.mean(
                record["valid_neighbor_ratio"] for record in ball_records
            )
            arkade_ms = statistics.median(
                statistics.median(record["query_timings_s"]) * 1000.0
                for record in arkade_records
            )
            arkade_recall = statistics.mean(
                record["recall_vs_cukd"]["mean"] for record in arkade_records
            )
            arkade_rounds = statistics.median(
                statistics.median(record["rounds"]) for record in arkade_records
            )
            lines.append(
                f"| {mode} | {k} | {flash_ms:.4f} | {cukd_ms:.4f} | "
                f"{ball_ms:.4f} | {ball_recall:.6f} | {ball_valid:.6f} | "
                f"{arkade_ms:.4f} | {arkade_recall:.6f} | {arkade_rounds:g} |"
            )
    lines.extend([
        "",
        "## Ball-query coverage",
        "",
        "`Insufficient` is the fraction of queries with fewer than k points in the radius; `truncated` is the fraction with more than k and therefore subject to `nsample` truncation.",
        "",
        "| Mode | k | Radius m | Valid slots | Insufficient | Truncated |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in ("pre", "post"):
        for k in (24, 32, 48):
            records = grouped(ball["records"], mode, k)
            lines.append(
                f"| {mode} | {k} | {statistics.median(record['radius_m'] for record in records):.6f} | "
                f"{statistics.mean(record['valid_neighbor_ratio'] for record in records):.6f} | "
                f"{statistics.mean(record['insufficient_query_ratio'] for record in records):.6f} | "
                f"{statistics.mean(record['truncated_query_ratio'] for record in records):.6f} |"
            )
    if ball_sweep is not None:
        lines.extend([
            "",
            "### One-room radius sensitivity",
            "",
            "This diagnostic uses Area_1/WC_1 and locally calibrated distance quantiles. A larger radius does not improve kNN recall in this operator once more than `nsample=k` candidates are present, because the fixed-radius operator truncates candidates rather than selecting the nearest k.",
            "",
            "| Mode | k | Quantile | Radius m | Query ms | Recall | Insufficient | Truncated |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for record in ball_sweep["records"]:
            lines.append(
                f"| {record['mode']} | {record['k']} | {record['percentile']:.2f} | "
                f"{record['radius_m']:.6f} | "
                f"{statistics.median(record['query_timings_s']) * 1000.0:.4f} | "
                f"{record['recall_vs_cukd']['mean']:.6f} | "
                f"{record['insufficient_query_ratio']:.6f} | "
                f"{record['truncated_query_ratio']:.6f} |"
            )
    lines.extend([
        "",
        "## Index construction",
        "",
        "Ball query has no persistent index construction in this implementation. Arkade's value is its OptiX BVH build; FlashKNN and cudaKDTree use their respective index-build timing boundaries.",
        "",
        "| Mode | k | FlashKNN ms | cudaKDTree ms | Arkade BVH ms |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode in ("pre", "post"):
        for k in (24, 32, 48):
            knn_records = grouped(knn["records"], mode, k)
            arkade_records = grouped(arkade["records"], mode, k)
            flash_build = statistics.median(
                median_timing(record, "flashknn", "construction_s")
                for record in knn_records
            )
            cukd_build = statistics.median(
                median_timing(record, "cuda_kdtree", "construction_s")
                for record in knn_records
            )
            arkade_build = statistics.median(
                record["construction_s"] * 1000.0 for record in arkade_records
            )
            lines.append(
                f"| {mode} | {k} | {flash_build:.4f} | {cukd_build:.4f} | "
                f"{arkade_build:.4f} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The Pointcept operator is a representative pipeline implementation, but it assigns one thread to each query and scans the full support set. Its latency therefore characterizes this public operator rather than all possible grid- or tree-accelerated radius searches. Its lower kNN recall is expected because fixed-radius sampling is not required to return the nearest k points.",
        "",
        "Arkade exercises a genuinely different hardware path through OptiX RT cores. The benchmark retains its pinned-host output, host-side completion scan, iterative radius expansion, and acceleration-structure refits. Its measured recall, rather than an assumed exact label, is reported because the public AABB implementation and voxel-distance ties do not always return the same index set as cudaKDTree.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
