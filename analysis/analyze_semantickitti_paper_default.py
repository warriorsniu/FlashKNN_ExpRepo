#!/usr/bin/env python3
"""Validate and summarize the paper-default SemanticKITTI query sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t as student_t


MODES = ("pre", "post")
KS = (8, 16, 24, 32, 48, 64)
PAPER_ALPHA = 8
EXPECTED_SAMPLES = 110


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timing_mean_ms(timings: list[dict], first: str, second: str) -> float:
    return statistics.mean(float(row[first]) + float(row[second]) for row in timings) * 1000.0


def summarize(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half_width = float(student_t.ppf(0.975, count - 1)) * sd / math.sqrt(count)
    return {
        "n": count,
        "mean": mean,
        "sample_sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def main() -> None:
    args = arguments()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    records = payload["samples"]
    if metadata.get("dataset") != "SemanticKITTI":
        raise SystemExit("Input is not a SemanticKITTI result")
    if int(metadata.get("faiss_ivf_match_alpha", -1)) != PAPER_ALPHA:
        raise SystemExit("FAISS IVF is not calibrated to paper-default alpha=8")
    if int(metadata.get("repeats", -1)) != 10 or int(metadata.get("warmups", -1)) != 3:
        raise SystemExit("Expected the formal 3-warmup/10-repeat protocol")
    expected = EXPECTED_SAMPLES * len(MODES) * len(KS)
    if len(records) != expected:
        raise SystemExit(f"Expected {expected} records, found {len(records)}")
    keys = {(row["sample"], row["mode"], int(row["k"])) for row in records}
    if len(keys) != expected:
        raise SystemExit("Duplicate sample/mode/k records found")

    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        mode, k = str(row["mode"]), int(row["k"])
        if mode not in MODES or k not in KS:
            raise SystemExit(f"Unexpected configuration: {mode}, k={k}")
        flash_entries = row.get("flashknn", [])
        if len(flash_entries) != 1 or int(flash_entries[0]["alpha"]) != PAPER_ALPHA:
            raise SystemExit(f"{row['sample']} {mode} k={k} is not an alpha=8-only refresh")
        flash = flash_entries[0]
        exact = row["exact"]
        ivf = row["faiss_ivf"]
        for name, value in (("flash", flash), ("exact", exact), ("ivf", ivf)):
            if len(value["timings"]) != 10:
                raise SystemExit(f"{row['sample']} {mode} k={k} has incomplete {name} timings")
        target = float(ivf["target_recall"])
        flash_recall = float(flash["recall"]["mean"])
        if abs(target - flash_recall) > 1e-8:
            raise SystemExit(f"{row['sample']} {mode} k={k} has a mismatched IVF target")

        group = grouped[(mode, k)]
        group["flash_total_ms"].append(
            timing_mean_ms(flash["timings"], "预处理耗时", "查询耗时")
        )
        group["flash_recall"].append(flash_recall * 100.0)
        group["exact_total_ms"].append(
            timing_mean_ms(exact["timings"], "construction_seconds", "query_seconds")
        )
        group["ivf_total_ms"].append(
            float(ivf["training_seconds"]) * 1000.0
            + timing_mean_ms(ivf["timings"], "construction_seconds", "query_seconds")
        )
        group["ivf_recall"].append(float(ivf["recall_vs_exact"]) * 100.0)

    rows = []
    for mode in MODES:
        for k in KS:
            values = grouped[(mode, k)]
            row = {"mode": mode, "k": k}
            for metric in (
                "flash_total_ms", "flash_recall", "exact_total_ms",
                "ivf_total_ms", "ivf_recall",
            ):
                row[metric] = summarize(values[metric])
            row["speedup_vs_exact"] = (
                row["exact_total_ms"]["mean"] / row["flash_total_ms"]["mean"]
            )
            rows.append(row)

    result = {
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "paper_alpha": PAPER_ALPHA,
        "protocol": {
            "samples": EXPECTED_SAMPLES,
            "modes": list(MODES),
            "k": list(KS),
            "warmups": metadata["warmups"],
            "repeats": metadata["repeats"],
            "gpu": metadata["gpu"],
            "torch": metadata["torch"],
            "torch_cuda": metadata["torch_cuda"],
            "aggregation": "per-frame repeat mean, then mean/sample SD/t-interval over 110 frames",
        },
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# SemanticKITTI paper-default query summary",
        "",
        "Protocol: alpha=8, 110 frames, 3 warmups, 10 repeats. Values are frame-level means aggregated across frames.",
        "",
        "| Mode | k | Flash ms | Flash recall % | cudaKDTree ms | Speedup | IVF ms | IVF recall % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['k']} | {row['flash_total_ms']['mean']:.3f} | "
            f"{row['flash_recall']['mean']:.3f} | {row['exact_total_ms']['mean']:.3f} | "
            f"{row['speedup_vs_exact']:.3f}x | {row['ivf_total_ms']['mean']:.3f} | "
            f"{row['ivf_recall']['mean']:.3f} |"
        )
    lines += [
        "",
        "The JSON companion records sample SD and two-sided 95% Student-t confidence intervals for every reported metric.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
