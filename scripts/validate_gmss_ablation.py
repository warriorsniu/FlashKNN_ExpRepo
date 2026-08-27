#!/usr/bin/env python3
"""Validate the directed full-k GMSS completion of the design ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_K = (8, 16, 24, 32, 40, 48, 56, 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected-rooms", type=int, default=81)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    expected_variant = {
        "memory_mode": "GM", "sorting_mode": "SS",
        "candidate_mode": "register", "enable_skip": True,
    }
    if metadata.get("variants") != {"gmss": expected_variant}:
        raise SystemExit(f"Unexpected variant configuration: {metadata.get('variants')}")
    if metadata.get("selection_revision_by_variant") != {"gmss": "serial_max_heap"}:
        raise SystemExit("GMSS must record serial_max_heap selection")
    expected_metadata = {
        "dataset": "S3DIS", "scope": "sample_part", "mode": "pre",
        "voxel_size_m": 0.02, "crop_points": 250000, "num_down": 2,
        "warmups": 5, "repeats": 20, "k": list(EXPECTED_K),
    }
    changed = {
        key: (value, metadata.get(key))
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if changed:
        raise SystemExit(f"GMSS protocol mismatch: {changed}")
    if str(metadata.get("torch_cuda")) != "11.8":
        raise SystemExit("RTX 3090 GMSS ablation must use CUDA 11.8")
    if not metadata.get("source_sha256") or not metadata.get("extension", {}).get("sha256"):
        raise SystemExit("Missing source/extension provenance")

    records = payload.get("records", [])
    rooms: set[str] = set()
    seen: set[tuple[str, int]] = set()
    for record in records:
        room = str(record.get("room"))
        k = int(record.get("k", -1))
        key = (room, k)
        if key in seen:
            raise SystemExit(f"Duplicate GMSS record: {key}")
        seen.add(key)
        rooms.add(room)
        if set(record.get("variants", {})) != {"gmss"}:
            raise SystemExit(f"{key} does not contain exactly GMSS")
        result = record["variants"]["gmss"]
        if result.get("configuration") != expected_variant:
            raise SystemExit(f"{key} has an incorrect GMSS configuration")
        if len(result.get("timings", [])) != 20:
            raise SystemExit(f"{key} does not contain 20 timed repeats")
        recall = float(result.get("recall_vs_cukd", {}).get("mean", -1))
        if not 0 <= recall <= 1:
            raise SystemExit(f"{key} has invalid recall {recall}")

    if len(rooms) != args.expected_rooms:
        raise SystemExit(f"Expected {args.expected_rooms} rooms, found {len(rooms)}")
    expected = {(room, k) for room in rooms for k in EXPECTED_K}
    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        raise SystemExit(f"GMSS coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    start = metadata.get("co_tenant_start", {}).get("compute_processes", [])
    if start:
        raise SystemExit(f"GMSS run started with compute co-tenants: {start}")
    print(
        f"GMSS ablation coverage OK: {len(rooms)} rooms, {len(records)} records, "
        "8 k values, 5/20 protocol"
    )


if __name__ == "__main__":
    main()
