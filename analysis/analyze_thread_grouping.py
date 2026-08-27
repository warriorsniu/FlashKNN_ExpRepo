#!/usr/bin/env python3
"""Summarize and plot fixed/adaptive thread-grouping ablations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


LABELS = {
    "adaptive": "Adaptive",
    "fixed_8": "Fixed-8",
    "fixed_16": "Fixed-16",
    "fixed_32": "Fixed-32",
}


def aggregate(payload: dict) -> list[dict]:
    grouped = defaultdict(
        lambda: {
            "query": [], "room_query_mean": [], "recall": [],
            "distance_delta": [], "distance_equivalent": [],
        }
    )
    rooms = defaultdict(set)
    for record in payload.get("records", []):
        for variant, result in record.get("variants", {}).items():
            key = (int(record["k"]), variant)
            rooms[key].add(record["room"])
            query = [
                timing["query_s"] * 1000.0 for timing in result["timings"]
            ]
            grouped[key]["query"].extend(query)
            grouped[key]["room_query_mean"].append(float(np.mean(query)))
            grouped[key]["recall"].append(result["recall_vs_cukd"]["mean"])
            equivalence = result["equivalence_vs_adaptive"]
            grouped[key]["distance_delta"].append(
                equivalence["squared_distance_max_abs_diff"]
            )
            grouped[key]["distance_equivalent"].append(
                equivalence["squared_distance_allclose"]
            )

    rows = []
    for (k, variant), values in sorted(grouped.items()):
        query = np.asarray(values["query"], dtype=float)
        room_means = np.asarray(values["room_query_mean"], dtype=float)
        recall = np.asarray(values["recall"], dtype=float)
        room_sd = room_means.std(ddof=1) if len(room_means) > 1 else 0.0
        ci_multiplier = (
            float(student_t.ppf(0.975, len(room_means) - 1))
            if len(room_means) > 1 else 0.0
        )
        rows.append({
            "k": k,
            "variant": variant,
            "label": LABELS[variant],
            "rooms": len(rooms[(k, variant)]),
            "timings": len(query),
            "query_mean_ms": query.mean(),
            "query_median_ms": np.median(query),
            "room_mean_std_ms": room_sd,
            "room_mean_ci95_ms": (
                ci_multiplier * room_sd / np.sqrt(len(room_means))
                if len(room_means) > 1 else 0.0
            ),
            "query_p95_ms": np.percentile(query, 95),
            "recall_mean": recall.mean(),
            "recall_min": recall.min(),
            "adaptive_distance_max_abs_diff": max(values["distance_delta"]),
            "adaptive_distance_equivalent": all(
                values["distance_equivalent"]
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    font_root = Path("/data/nyc/fonts")
    font_paths = [font_root / name for name in (
        "TIMES.TTF", "TIMESBD.TTF", "TIMESI.TTF", "TIMESBI.TTF"
    )]
    missing = [path for path in font_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Times New Roman fonts: {missing}")
    for font_path in font_paths:
        fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_paths[0]).get_name()
    plt.rcParams.update({
        "font.family": font_name,
        "font.serif": [font_name],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    })
    rows = aggregate(payload)
    if not rows:
        raise SystemExit("No thread-grouping records found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = payload["metadata"]
    lines = [
        "# FlashKNN thread-grouping ablation", "",
        f"- GPU: {metadata['gpu'].get('name')} ({metadata['gpu'].get('uuid')})",
        f"- Torch/CUDA: {metadata['torch']} / {metadata['torch_cuda']}",
        f"- Protocol: {metadata['warmups']} warm-ups, {metadata['repeats']} repeats",
        f"- Adaptive rule: {metadata['adaptive_rule']}", "",
        (
            "Statistical unit for SD and two-sided Student-t 95% CI: "
            "per-room timing mean."
        ), "",
        (
            "Recall is index-set recall versus CUKD. Distance-equivalent "
            "means every query has the same sorted neighbor-distance vector "
            "as Adaptive; equal-distance index substitutions are permitted."
        ), "",
        "| k | Strategy | Rooms | Query mean (ms) | Room SD | 95% CI | p95 | Recall | Distance-equivalent |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['label']} | {row['rooms']} | "
            f"{row['query_mean_ms']:.4f} | {row['room_mean_std_ms']:.4f} | "
            f"±{row['room_mean_ci95_ms']:.4f} | {row['query_p95_ms']:.4f} | "
            f"{row['recall_mean']:.6f} | "
            f"{'yes' if row['adaptive_distance_equivalent'] else 'no'} |"
        )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(8, 4.5), dpi=300)
    for variant in LABELS:
        selected = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: row["k"],
        )
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
    fig.savefig(args.output_dir / "thread_grouping.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "thread_grouping.svg", bbox_inches="tight")
    plt.close(fig)
    print(args.output_dir)


if __name__ == "__main__":
    main()
