#!/usr/bin/env python3
"""Summarize matched FlashKNN, Pointcept, and PyTorch3D operator measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_room_timing(record: dict, field: str = "query_timings_s") -> float:
    return statistics.mean(record[field]) * 1000.0


def selected(records: list[dict], mode: str, k: int) -> list[dict]:
    return [record for record in records if record["mode"] == mode and int(record["k"]) == k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn", type=Path, required=True)
    parser.add_argument("--pointcept", type=Path, required=True)
    parser.add_argument("--pytorch3d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    knn, pointcept, pytorch3d = load(args.knn), load(args.pointcept), load(args.pytorch3d)
    lines = [
        "# RTX 3090 representative ball-query operators",
        "",
        "All values are means of 81 per-room means on the same RTX 3090 and deterministic S3DIS fixed-250k crops. Radius operators use the same global p90 exact-kth-distance radius. Their fixed-radius semantics differ from fixed-k nearest-neighbor selection.",
        "",
        "| Mode | k | Operator | Query ms | Set recall | Valid slots | Insufficient | Truncated |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("pre", "post"):
        for k in (24, 32, 48):
            knn_rows = selected(knn["records"], mode, k)
            flash_ms = statistics.mean(
                statistics.mean(item["query_s"] for item in record["methods"]["flashknn"]["timings"]) * 1000.0
                for record in knn_rows
            )
            flash_recall = statistics.mean(
                record["methods"]["flashknn"]["recall_vs_cukd"]["mean"] for record in knn_rows
            )
            lines.append(
                f"| {mode} | {k} | FlashKNN | {flash_ms:.3f} | {flash_recall:.6f} | -- | -- | -- |"
            )
            for label, payload in (("Pointcept", pointcept), ("PyTorch3D", pytorch3d)):
                rows = selected(payload["records"], mode, k)
                lines.append(
                    f"| {mode} | {k} | {label} | "
                    f"{statistics.mean(mean_room_timing(record) for record in rows):.3f} | "
                    f"{statistics.mean(record['recall_vs_cukd']['mean'] for record in rows):.6f} | "
                    f"{statistics.mean(record['valid_neighbor_ratio'] for record in rows):.6f} | "
                    f"{statistics.mean(record['insufficient_query_ratio'] for record in rows):.6f} | "
                    f"{statistics.mean(record['truncated_query_ratio'] for record in rows):.6f} |"
                )
    lines.extend([
        "",
        "The comparison is limited to the measured public operators and does not rank specialized grid-, cell-list-, tree-, or BVH-accelerated radius-search systems.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
