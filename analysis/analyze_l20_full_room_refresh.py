#!/usr/bin/env python3
"""Summarize final-kernel L20 full-room throughput and historical deltas.

The script covers only the S3DIS full/pre/k=32 scaling result.  It reports
per-bin latency, throughput and query speedup, plus rooms nearest the paper's
representative point counts.  Historical data is read from Git and is never
mixed into current figures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


JsonObject = dict[str, object]
BIN_EDGES = (
    ("<=250k", 0, 250_000),
    ("250k-500k", 250_001, 500_000),
    ("500k-1m", 500_001, 1_000_000),
    ("1m-2m", 1_000_001, 2_000_000),
    (">2m", 2_000_001, 10**12),
)
TARGETS = (250_000, 500_000, 1_000_000, 2_425_000)


def arguments() -> argparse.Namespace:
    """Parse the current canonical file, historical Git ref, and output path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--old-ref", required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_file(path: Path) -> JsonObject:
    """Load the current canonical JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def load_git(repo: Path, revision: str, relative_path: Path) -> JsonObject:
    """Load a historical canonical JSON object directly from a Git revision."""
    content = subprocess.check_output(
        ["git", "show", f"{revision}:{relative_path.as_posix()}"],
        cwd=repo,
        text=True,
    )
    value = json.loads(content)
    if not isinstance(value, dict):
        raise SystemExit("Historical full-room payload is not a JSON object")
    return value


def mean_query_seconds(record: JsonObject, method: str) -> float:
    """Return mean query seconds for one room and one backend."""
    methods = record.get("methods")
    if not isinstance(methods, dict) or not isinstance(methods.get(method), dict):
        raise SystemExit(f"Missing {method} in {record.get('room')}")
    timings = methods[method].get("timings")
    if not isinstance(timings, list) or not timings:
        raise SystemExit(f"Missing {method} timings in {record.get('room')}")
    return statistics.mean(float(timing["query_s"]) for timing in timings)


def row(record: JsonObject) -> JsonObject:
    """Compute latency, throughput and paired query speedup for one room."""
    points = int(record["num_support"])
    queries = int(record["num_query"])
    flash_s = mean_query_seconds(record, "flashknn")
    cukd_s = mean_query_seconds(record, "cuda_kdtree")
    return {
        "room": str(record["room"]),
        "support_points": points,
        "query_points": queries,
        "flashknn_query_ms": flash_s * 1000.0,
        "flashknn_mquery_s": queries / flash_s / 1_000_000.0,
        "cuda_kdtree_query_ms": cukd_s * 1000.0,
        "cuda_kdtree_mquery_s": queries / cukd_s / 1_000_000.0,
        "query_speedup": cukd_s / flash_s,
    }


def aggregate(items: list[JsonObject], label: str) -> JsonObject:
    """Average per-room metrics for one support-point bin."""
    return {
        "bin": label,
        "rooms": len(items),
        "mean_support_points": statistics.mean(
            int(item["support_points"]) for item in items
        ),
        "flashknn_query_ms": statistics.mean(
            float(item["flashknn_query_ms"]) for item in items
        ),
        "flashknn_mquery_s": statistics.mean(
            float(item["flashknn_mquery_s"]) for item in items
        ),
        "cuda_kdtree_query_ms": statistics.mean(
            float(item["cuda_kdtree_query_ms"]) for item in items
        ),
        "cuda_kdtree_mquery_s": statistics.mean(
            float(item["cuda_kdtree_mquery_s"]) for item in items
        ),
        "query_speedup": statistics.mean(
            float(item["query_speedup"]) for item in items
        ),
    }


def index(payload: JsonObject) -> dict[str, JsonObject]:
    """Index full/pre/k=32 records by room and reject duplicate rooms."""
    records = payload.get("records")
    if not isinstance(records, list):
        raise SystemExit("Full-room payload has no records")
    selected = [
        record for record in records
        if isinstance(record, dict)
        and record.get("scope") == "full"
        and record.get("mode") == "pre"
        and int(record.get("k", -1)) == 32
    ]
    result = {str(record["room"]): record for record in selected}
    if len(selected) != 272 or len(result) != 272:
        raise SystemExit(f"Expected 272 unique full-room records, got {len(result)}")
    return result


def markdown_table(items: list[JsonObject], columns: tuple[str, ...]) -> list[str]:
    """Render compact numeric JSON rows as a GitHub-compatible Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for item in items:
        values = []
        for column in columns:
            value = item[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    """Build JSON and Markdown summaries from current and historical results."""
    args = arguments()
    repo = args.repo.resolve()
    current_path = args.current.resolve()
    relative_path = current_path.relative_to(repo)
    current = index(load_file(current_path))
    old = index(load_git(repo, args.old_ref, relative_path))
    current_rows = [row(record) for record in current.values()]
    old_rows = {room: row(record) for room, record in old.items()}
    bins = []
    for label, lower, upper in BIN_EDGES:
        matches = [
            item for item in current_rows
            if lower <= int(item["support_points"]) <= upper
        ]
        if matches:
            bins.append(aggregate(matches, label))
    representatives = []
    for target in TARGETS:
        current_item = min(
            current_rows,
            key=lambda item: abs(int(item["support_points"]) - target),
        )
        old_item = old_rows[str(current_item["room"])]
        representatives.append({
            **current_item,
            "target_points": target,
            "old_flashknn_query_ms": old_item["flashknn_query_ms"],
            "old_cuda_kdtree_query_ms": old_item["cuda_kdtree_query_ms"],
            "old_query_speedup": old_item["query_speedup"],
        })
    output = {
        "current_source": str(relative_path),
        "historical_source": f"{args.old_ref}:{relative_path.as_posix()}",
        "bins": bins,
        "representative_rooms": representatives,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "full_room_throughput.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# L20 final-kernel full-room throughput",
        "",
        "Historical values are comparison-only and are not used by the rebuilt paper figures.",
        "",
        "## Point-count bins",
        "",
        *markdown_table(bins, (
            "bin", "rooms", "mean_support_points", "flashknn_query_ms",
            "flashknn_mquery_s", "cuda_kdtree_query_ms",
            "cuda_kdtree_mquery_s", "query_speedup",
        )),
        "",
        "## Representative rooms",
        "",
        *markdown_table(representatives, (
            "target_points", "room", "support_points", "flashknn_query_ms",
            "cuda_kdtree_query_ms", "query_speedup", "old_flashknn_query_ms",
            "old_cuda_kdtree_query_ms", "old_query_speedup",
        )),
        "",
    ]
    (args.output_dir / "full_room_throughput.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    print(json_path)


if __name__ == "__main__":
    main()
