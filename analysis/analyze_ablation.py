#!/usr/bin/env python3
"""Summarize and plot final-kernel FlashKNN design ablations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "smps": "SMPS",
    "smss": "SMSS",
    "gmps": "GMPS",
    "candidate_shared": "CandidateSM",
    "no_skip": "NoSkip",
    "candidate_shared_no_skip": "CandidateSM+NoSkip",
}


def aggregate(payload: dict) -> list[dict]:
    grouped: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: {
            "query": [], "room_query_mean": [], "construction": [],
            "total": [], "recall": [],
        }
    )
    rooms: dict[tuple[int, str], set[str]] = defaultdict(set)
    for record in payload.get("records", []):
        for variant, result in record.get("variants", {}).items():
            key = (int(record["k"]), variant)
            rooms[key].add(record["room"])
            grouped[key]["recall"].append(result["recall_vs_cukd"]["mean"])
            room_query = [
                timing["query_s"] * 1000.0
                for timing in result["timings"]
            ]
            grouped[key]["room_query_mean"].append(float(np.mean(room_query)))
            for timing in result["timings"]:
                grouped[key]["query"].append(timing["query_s"] * 1000.0)
                grouped[key]["construction"].append(
                    timing["construction_s"] * 1000.0
                )
                grouped[key]["total"].append(timing["total_s"] * 1000.0)

    rows = []
    for (k, variant), values in sorted(grouped.items()):
        query = np.asarray(values["query"], dtype=float)
        room_query_mean = np.asarray(values["room_query_mean"], dtype=float)
        construction = np.asarray(values["construction"], dtype=float)
        total = np.asarray(values["total"], dtype=float)
        recall = np.asarray(values["recall"], dtype=float)
        rows.append({
            "k": k, "variant": variant, "label": LABELS.get(variant, variant),
            "rooms": len(rooms[(k, variant)]), "timings": len(query),
            "query_mean_ms": query.mean(), "query_median_ms": np.median(query),
            "run_std_ms": query.std(ddof=1) if len(query) > 1 else 0.0,
            "query_p95_ms": np.percentile(query, 95),
            "room_mean_std_ms": (
                room_query_mean.std(ddof=1) if len(room_query_mean) > 1 else 0.0
            ),
            "room_mean_ci95_ms": (
                1.96 * room_query_mean.std(ddof=1) / np.sqrt(len(room_query_mean))
                if len(room_query_mean) > 1 else 0.0
            ),
            "construction_mean_ms": construction.mean(),
            "total_mean_ms": total.mean(),
            "recall_mean": recall.mean(), "recall_min": recall.min(),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict, rows: list[dict]) -> None:
    metadata = payload["metadata"]
    lines = [
        "# FlashKNN final-kernel ablation summary", "",
        f"- GPU: {metadata['gpu'].get('name')} ({metadata['gpu'].get('uuid')})",
        f"- Torch/CUDA: {metadata['torch']} / {metadata['torch_cuda']}",
        f"- Sorting: `{metadata['sorting_revision']}`",
        f"- Protocol: {metadata['warmups']} warm-ups, {metadata['repeats']} repeats",
        "", "Statistical unit for SD and 95% CI: per-room timing mean. "
        "Median and p95 pool the recorded repetitions.",
        "", "| k | Variant | Rooms | Query mean (ms) | Median | Room SD | 95% CI | p95 | Recall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['label']} | {row['rooms']} | "
            f"{row['query_mean_ms']:.4f} | {row['query_median_ms']:.4f} | "
            f"{row['room_mean_std_ms']:.4f} | ±{row['room_mean_ci95_ms']:.4f} | "
            f"{row['query_p95_ms']:.4f} | "
            f"{row['recall_mean']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict], variants: tuple[str, ...], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5), dpi=300)
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        selected.sort(key=lambda row: row["k"])
        if not selected:
            continue
        axis.plot(
            [row["k"] for row in selected],
            [row["query_mean_ms"] for row in selected],
            marker="o", linewidth=2, label=LABELS[variant],
        )
    axis.set_xlabel("Number of Neighbors (k)")
    axis.set_ylabel("KNN Query Time (ms)")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    rows = aggregate(payload)
    if not rows:
        raise SystemExit("No ablation records found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    write_markdown(args.output_dir / "summary.md", payload, rows)
    plot(rows, ("smps", "smss", "gmps"), args.output_dir / "memory_sorting")
    plot(
        rows,
        ("smps", "candidate_shared", "no_skip", "candidate_shared_no_skip"),
        args.output_dir / "candidate_skip",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
