#!/usr/bin/env python3
"""Summarize and plot final-kernel FlashKNN design ablations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from scipy.stats import t as student_t


LABELS = {
    "smps": "SMPS",
    "smss": "SMSS",
    "gmps": "GMPS",
    "gmss": "GMSS",
    "candidate_shared": "CandidateSM",
    "no_skip": "NoSkip",
    "candidate_shared_no_skip": "CandidateSM+NoSkip",
}

COLORS = {
    "smps": "#0077BB",
    "smss": "#EE7733",
    "gmps": "#009988",
    "gmss": "#CC3311",
    "candidate_shared": "#33BBEE",
    "no_skip": "#EE3377",
    "candidate_shared_no_skip": "#BBBBBB",
}
MARKERS = {
    "smps": "o", "smss": "s", "gmps": "^", "gmss": "D",
    "candidate_shared": "s", "no_skip": "^",
    "candidate_shared_no_skip": "D",
}


def configure_font(font_dir: Path) -> str:
    files = (
        font_dir / "TIMES.TTF",
        font_dir / "TIMESBD.TTF",
        font_dir / "TIMESI.TTF",
        font_dir / "TIMESBI.TTF",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing Times New Roman font files: {missing}")
    for path in files:
        font_manager.fontManager.addfont(path)
    family = font_manager.FontProperties(fname=files[0]).get_name()
    plt.rcParams.update({
        "font.family": family,
        "font.serif": [family],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "savefig.dpi": 450,
    })
    return family


def merge_payloads(primary: dict, extras: list[dict]) -> dict:
    identity_fields = (
        "dataset", "scope", "mode", "voxel_size_m", "crop_points",
        "num_down", "warmups", "repeats", "seed", "k",
    )
    primary_metadata = primary["metadata"]
    merged = {
        "metadata": dict(primary_metadata),
        "records": list(primary.get("records", [])),
    }
    sources = [primary_metadata.get("git", {})]
    seen = {
        (record["room"], int(record["k"]), variant)
        for record in merged["records"]
        for variant in record.get("variants", {})
    }
    by_record = {
        (record["room"], int(record["k"])): record
        for record in merged["records"]
    }
    for extra in extras:
        metadata = extra["metadata"]
        changed = {
            field: (primary_metadata.get(field), metadata.get(field))
            for field in identity_fields
            if primary_metadata.get(field) != metadata.get(field)
        }
        old_uuid = primary_metadata.get("gpu", {}).get("uuid")
        new_uuid = metadata.get("gpu", {}).get("uuid")
        if old_uuid != new_uuid:
            changed["gpu.uuid"] = (old_uuid, new_uuid)
        if changed:
            raise SystemExit(f"Cannot merge incompatible ablation result: {changed}")
        sources.append(metadata.get("git", {}))
        for record in extra.get("records", []):
            record_key = (record["room"], int(record["k"]))
            destination = by_record.get(record_key)
            if destination is None:
                destination = {
                    key: value for key, value in record.items()
                    if key != "variants"
                }
                destination["variants"] = {}
                by_record[record_key] = destination
                merged["records"].append(destination)
            for variant, result in record.get("variants", {}).items():
                key = (*record_key, variant)
                if key in seen:
                    raise SystemExit(f"Duplicate merged ablation entry: {key}")
                destination["variants"][variant] = result
                seen.add(key)
    merged["metadata"]["merged_source_git"] = sources
    return merged


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
                student_t.ppf(0.975, len(room_query_mean) - 1)
                * room_query_mean.std(ddof=1) / np.sqrt(len(room_query_mean))
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


def save_figure(fig, path: Path, paper_asset: Path | None = None) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        path.with_suffix(".png"), dpi=450,
        bbox_inches="tight", pad_inches=0.02,
    )
    if paper_asset is not None:
        paper_asset.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            paper_asset.with_suffix(".pdf"),
            bbox_inches="tight", pad_inches=0.02,
        )


def plot(
    rows: list[dict], variants: tuple[str, ...], path: Path,
    paper_asset: Path | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.3), dpi=450)
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        selected.sort(key=lambda row: row["k"])
        if not selected:
            continue
        axis.plot(
            [row["k"] for row in selected],
            [row["query_mean_ms"] for row in selected],
            color=COLORS[variant], marker=MARKERS[variant],
            markersize=5, linewidth=1.8, label=LABELS[variant],
        )
        x = np.asarray([row["k"] for row in selected])
        y = np.asarray([row["query_mean_ms"] for row in selected])
        ci = np.asarray([row["room_mean_ci95_ms"] for row in selected])
        axis.fill_between(x, y - ci, y + ci, color=COLORS[variant], alpha=0.10)
    axis.set_xlabel("Number of neighbors ($k$)")
    axis.set_ylabel("kNN query time (ms)")
    axis.set_ylim(bottom=0)
    axis.set_xticks(sorted({int(row["k"]) for row in rows}))
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, path, paper_asset)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--extra-result", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("/data/nyc/fonts"))
    parser.add_argument("--memory-sorting-asset", type=Path)
    parser.add_argument("--candidate-skip-asset", type=Path)
    args = parser.parse_args()
    configure_font(args.font_dir)
    primary = json.loads(args.result.read_text(encoding="utf-8"))
    extras = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.extra_result
    ]
    payload = merge_payloads(primary, extras)
    rows = aggregate(payload)
    if not rows:
        raise SystemExit("No ablation records found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    write_markdown(args.output_dir / "summary.md", payload, rows)
    plot(
        rows, ("smps", "smss", "gmps", "gmss"),
        args.output_dir / "memory_sorting", args.memory_sorting_asset,
    )
    plot(
        rows,
        ("smps", "candidate_shared", "no_skip", "candidate_shared_no_skip"),
        args.output_dir / "candidate_skip", args.candidate_skip_asset,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
