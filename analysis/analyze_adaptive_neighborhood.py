#!/usr/bin/env python3
"""Summarize the fixed/adaptive/cudaKDTree neighborhood ablation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm


T95_DF80 = 1.990063


def room_stat(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = T95_DF80 * sd / math.sqrt(len(values))
    return {"mean": mean, "room_sd": sd, "ci95_low": mean - half,
            "ci95_high": mean + half}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sparse-bins", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    variants = payload["metadata"]["variants"]
    adaptive = next(name for name in variants if name.startswith("adaptive_"))
    by_k: dict[int, list[dict[str, Any]]] = {}
    for record in payload["records"]:
        by_k.setdefault(int(record["k"]), []).append(record)

    summary: dict[str, Any] = {"metadata": payload["metadata"], "by_k": {}}
    lines = [
        "# Adaptive octree-neighborhood ablation",
        "",
        "Means are computed over the per-room timing means; uncertainty is the "
        "room SD and Student-t 95% CI (n=81).",
        "",
        "| k | Fixed total (ms) | Adaptive total (ms) | cudaKDTree total (ms) | "
        "Adaptive / Fixed | Adaptive / cudaKDTree | Fixed recall | Adaptive recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in sorted(by_k):
        records = by_k[k]
        item: dict[str, Any] = {"rooms": len(records), "variants": {}}
        for name in variants:
            results = [record["variants"][name] for record in records]
            variant: dict[str, Any] = {
                "recall": room_stat([
                    float(result["recall_vs_cukd"]["mean"])
                    for result in results
                ]),
                "peak_incremental_allocated_mib": room_stat([
                    float(result["peak_incremental_allocated_mib"])
                    for result in results
                ]),
                "timings_ms": {},
            }
            for field in (
                "construction_ms", "selection_ms", "compatibility_ms",
                "query_ms", "total_ms",
            ):
                variant["timings_ms"][field] = room_stat([
                    float(result["mean_ms"][field]) for result in results
                ])
            if name == adaptive:
                octrees = [result["octree"] for result in results]
                band = {
                    key: sum(
                        int(octree["selection_band_points"][key])
                        for octree in octrees
                    )
                    for key in (
                        "below_band_points", "within_band_points",
                        "above_band_points",
                    )
                }
                band_total = sum(band.values())
                variant["selection_band_points"] = band
                variant["selection_band_fraction"] = {
                    key: value / band_total for key, value in band.items()
                }
                variant["compatible_point_ratio"] = room_stat([
                    float(octree["compatible_point_ratio"])
                    for octree in octrees
                ])
            item["variants"][name] = variant
        fixed_total = item["variants"]["fixed_3x3x3"]["timings_ms"]["total_ms"]["mean"]
        adaptive_total = item["variants"][adaptive]["timings_ms"]["total_ms"]["mean"]
        cukd_total = item["variants"]["cuda_kdtree_exact"]["timings_ms"]["total_ms"]["mean"]
        item["adaptive_over_fixed"] = adaptive_total / fixed_total
        item["adaptive_over_cuda_kdtree"] = adaptive_total / cukd_total
        summary["by_k"][str(k)] = item
        lines.append(
            f"| {k} | {fixed_total:.3f} | {adaptive_total:.3f} | "
            f"{cukd_total:.3f} | {adaptive_total / fixed_total:.2f}x | "
            f"{adaptive_total / cukd_total:.2f}x | "
            f"{item['variants']['fixed_3x3x3']['recall']['mean']:.6f} | "
            f"{item['variants'][adaptive]['recall']['mean']:.6f} |"
        )

    lines.extend([
        "",
        "| k | Adaptive construction | selection | compatible input | query/map | "
        "total 95% CI | compatible points | peak allocated | within [2k,8k] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for k in sorted(by_k):
        variant = summary["by_k"][str(k)]["variants"][adaptive]
        timing = variant["timings_ms"]
        total = timing["total_ms"]
        lines.append(
            f"| {k} | {timing['construction_ms']['mean']:.3f} ms | "
            f"{timing['selection_ms']['mean']:.3f} ms | "
            f"{timing['compatibility_ms']['mean']:.3f} ms | "
            f"{timing['query_ms']['mean']:.3f} ms | "
            f"[{total['ci95_low']:.3f}, {total['ci95_high']:.3f}] ms | "
            f"{variant['compatible_point_ratio']['mean']:.2f}x | "
            f"{variant['peak_incremental_allocated_mib']['mean']:.1f} MiB | "
            f"{100 * variant['selection_band_fraction']['within_band_points']:.2f}% |"
        )

    if args.sparse_bins:
        sparse = json.loads(
            args.sparse_bins.read_text(encoding="utf-8")
        )["summary"]
        summary["fixed_candidate_recall_bins"] = sparse
        lines.extend([
            "",
            "Recall by the fixed-alpha candidate count. Delta is Adaptive minus Fixed.",
            "",
            "| k | queries below 2k | Fixed recall | Adaptive recall | delta | "
            "delta within [2k,8k] | delta above 8k |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for k in sorted(by_k):
            bins = sparse[str(k)]
            below = bins["below_2k"]
            lines.append(
                f"| {k} | {100 * below['query_fraction']:.4f}% | "
                f"{below['fixed_recall']:.6f} | "
                f"{below['adaptive_recall']:.6f} | "
                f"{below['adaptive_minus_fixed']:+.6f} | "
                f"{bins['within_2k_8k']['adaptive_minus_fixed']:+.6f} | "
                f"{bins['above_8k']['adaptive_minus_fixed']:+.6f} |"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    import matplotlib.pyplot as plt

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

    ks = sorted(by_k)
    labels = {
        "fixed_3x3x3": "Fixed 3x3x3",
        adaptive: "Adaptive [2k,8k]",
        "cuda_kdtree_exact": "Exact cudaKDTree",
    }
    colors = {
        "fixed_3x3x3": "#0072B2",
        adaptive: "#D55E00",
        "cuda_kdtree_exact": "#009E73",
    }
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for name in variants:
        total_values = [
            summary["by_k"][str(k)]["variants"][name]
            ["timings_ms"]["total_ms"]["mean"]
            for k in ks
        ]
        recall_values = [
            summary["by_k"][str(k)]["variants"][name]
            ["recall"]["mean"]
            for k in ks
        ]
        axes[0].plot(
            ks, total_values,
            marker="o", label=labels[name], color=colors[name],
        )
        axes[1].plot(
            ks, recall_values,
            marker="o", label=labels[name], color=colors[name],
        )
    axes[0].set(xlabel="k", ylabel="Total latency (ms)")
    axes[1].set(xlabel="k", ylabel="Recall vs. cudaKDTree", ylim=(0.99, 1.0005))
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xticks(ks)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(
            args.output_dir / f"adaptive_neighborhood_latency_recall.{suffix}",
            dpi=240, bbox_inches="tight",
        )
    plt.close(figure)
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
