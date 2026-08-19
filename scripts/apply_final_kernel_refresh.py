#!/usr/bin/env python3
"""Merge paired production-kernel refreshes into a canonical result pack."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = ("flashknn", "cuda_kdtree")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--refresh", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def query_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record["room"]), str(record["mode"]),
        str(record["scope"]), int(record["k"]),
    )


def merge_query(canonical_path: Path, refresh_path: Path) -> None:
    canonical = load(canonical_path)
    refresh = load(refresh_path)
    canonical_records = canonical.get("records", [])
    refresh_records = refresh.get("records", [])
    canonical_map = {query_key(record): record for record in canonical_records}
    refresh_map = {query_key(record): record for record in refresh_records}
    if len(canonical_map) != len(canonical_records):
        raise SystemExit(f"duplicate canonical query key: {canonical_path}")
    if len(refresh_map) != len(refresh_records):
        raise SystemExit(f"duplicate refresh query key: {refresh_path}")
    if canonical_map.keys() != refresh_map.keys():
        missing = sorted(canonical_map.keys() - refresh_map.keys())[:5]
        extra = sorted(refresh_map.keys() - canonical_map.keys())[:5]
        raise SystemExit(
            f"query key mismatch for {canonical_path}: missing={missing}, extra={extra}"
        )

    unaffected_before = [
        (key, {
            name: copy.deepcopy(value)
            for name, value in record.get("methods", {}).items()
            if name not in METHODS
        })
        for key, record in sorted(canonical_map.items())
    ]
    for key, target in canonical_map.items():
        source = refresh_map[key]
        if (target.get("num_support"), target.get("num_query")) != (
            source.get("num_support"), source.get("num_query")
        ):
            raise SystemExit(f"point-count mismatch at {key}")
        if set(source.get("methods", {})) != set(METHODS):
            raise SystemExit(
                f"refresh methods at {key} are {set(source.get('methods', {}))}"
            )
        for method in METHODS:
            timings = source["methods"][method].get("timings", [])
            expected = int(refresh["metadata"]["repeats"])
            if len(timings) != expected:
                raise SystemExit(
                    f"{key} {method} has {len(timings)} timings, expected {expected}"
                )
            target.setdefault("methods", {})[method] = copy.deepcopy(
                source["methods"][method]
            )

    unaffected_after = [
        (key, {
            name: value for name, value in record.get("methods", {}).items()
            if name not in METHODS
        })
        for key, record in sorted(canonical_map.items())
    ]
    if sha256_json(unaffected_before) != sha256_json(unaffected_after):
        raise RuntimeError("an unaffected baseline changed during query merge")

    metadata = canonical.setdefault("metadata", {})
    old_overrides = metadata.get("timing_overrides", [])
    metadata["timing_overrides"] = [
        override for override in old_overrides
        if not set(override.get("methods", ())) & set(METHODS)
    ]
    refresh_metadata = refresh.get("metadata", {})
    metadata["timing_overrides"].append({
        "source": str(refresh_path),
        "reason": (
            "Replace pre-final or mixed-protocol FlashKNN timings and the "
            "paired cudaKDTree control with the production generated-bitonic "
            "kernel under the canonical timing protocol."
        ),
        "methods": list(METHODS),
        "modes": sorted({record["mode"] for record in refresh_records}),
        "k": sorted({int(record["k"]) for record in refresh_records}),
        "records": len(refresh_records),
        "warmups": refresh_metadata.get("warmups"),
        "repeats": refresh_metadata.get("repeats"),
        "gpu": refresh_metadata.get("gpu"),
        "git": refresh_metadata.get("git"),
        "source_sha256": refresh_metadata.get("source_sha256"),
        "extension": refresh_metadata.get("extension"),
        "flashknn_configuration": refresh_metadata.get(
            "flashknn_configuration"
        ),
        "unaffected_baselines_sha256": sha256_json(unaffected_before),
    })
    atomic_json(canonical_path, canonical)


def replace_network(canonical_path: Path, refresh_path: Path) -> None:
    refresh = load(refresh_path)
    records = refresh.get("records", [])
    if len(records) != 68:
        raise SystemExit(f"expected 68 DeLA records, found {len(records)}")
    for record in records:
        if set(record.get("backends", {})) != {"cpu_kdtree", "flashknn"}:
            raise SystemExit(f"incomplete DeLA backends in {record.get('room')}")
    refresh.setdefault("metadata", {})["canonical_source"] = str(refresh_path)
    atomic_json(canonical_path, refresh)


def main() -> None:
    args = arguments()
    canonical_query = args.canonical / "query"
    refresh_query = args.refresh / "query"
    merge_query(
        canonical_query / "s3dis_sample_part.json",
        refresh_query / "s3dis_sample_part.json",
    )
    merge_query(
        canonical_query / "s3dis_full_k32.json",
        refresh_query / "s3dis_full_k32.json",
    )
    replace_network(
        args.canonical / "network/dela_s3dis.json",
        args.refresh / "network/dela_s3dis.json",
    )
    print(f"Applied production-kernel refresh from {args.refresh} to {args.canonical}")


if __name__ == "__main__":
    main()
