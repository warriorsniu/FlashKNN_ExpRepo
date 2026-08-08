#!/usr/bin/env python3
"""Merge compatible historical result JSON files without changing their schema."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


RESULT_FILES = (
    "system.json",
    "query/s3dis_sample_part.json",
    "query/s3dis_full_k32.json",
    "query/ball_query_s3dis_sample_part.json",
    "query/semantickitti.json",
    "network/dela_s3dis.json",
    "network/ptv3_s3dis.json",
    "network/octformer_s3dis.json",
    "network/spunet_s3dis.json",
    "network/minkunet34c_s3dis.json",
    "network/dela_semantickitti.json",
    "network/deepla_semantickitti.json",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    return parser.parse_args()


def record_key(relative: str, record: dict[str, Any]) -> tuple[Any, ...]:
    """Return the historical schema's stable identity fields."""
    if relative.startswith("network/"):
        return (record.get("room", record.get("sample")),)
    if relative.endswith("semantickitti.json"):
        return (record["sample"], record["mode"], int(record["k"]))
    if "ball_query" in relative:
        return (record["room"], record["mode"], int(record["k"]),
                float(record["percentile"]))
    return (record["room"], record["mode"], record["scope"], int(record["k"]))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge_file(source: Path, destination: Path, relative: str) -> None:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"seeded {relative} from {source.parent.parent}")
        return
    if relative == "system.json":
        return
    incoming = json.loads(source.read_text(encoding="utf-8"))
    current = json.loads(destination.read_text(encoding="utf-8"))
    if incoming.get("metadata") != current.get("metadata"):
        print(f"skip incompatible {source}: metadata differs")
        return
    collection = "records" if "records" in current else "samples"
    if collection not in incoming:
        raise SystemExit(f"Schema mismatch between {source} and {destination}")
    positions = {
        record_key(relative, record): position
        for position, record in enumerate(current[collection])
    }
    added = 0
    for record in incoming[collection]:
        key = record_key(relative, record)
        if key in positions:
            if current[collection][positions[key]] != record:
                raise SystemExit(f"Conflicting record {key} in {source} and {destination}")
            continue
        positions[key] = len(current[collection])
        current[collection].append(record)
        added += 1
    atomic_json(destination, current)
    print(f"merged {added} records into {relative} from {source.parent.parent}")


def main() -> None:
    args = arguments()
    destination = args.destination.resolve()
    for source_root in args.source:
        source_root = source_root.resolve()
        if not source_root.is_dir():
            raise SystemExit(f"Historical run directory does not exist: {source_root}")
        for relative in RESULT_FILES:
            source = source_root / relative
            if source.is_file():
                merge_file(source, destination / relative, relative)


if __name__ == "__main__":
    main()
