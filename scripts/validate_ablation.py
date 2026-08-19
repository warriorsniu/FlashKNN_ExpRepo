#!/usr/bin/env python3
"""Validate final-kernel FlashKNN design-ablation coverage and provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_K = (8, 16, 24, 32, 40, 48, 56, 64)
EXPECTED_VARIANTS = {
    "smps", "smss", "gmps", "candidate_shared", "no_skip",
    "candidate_shared_no_skip",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected-rooms", type=int, default=81)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})

    if metadata.get("sorting_revision") != "generated_bitonic_top_p":
        raise SystemExit("Ablation does not record the current generated top-P revision")
    if not metadata.get("source_sha256") or not metadata.get("extension", {}).get("sha256"):
        raise SystemExit("Ablation is missing source or extension build hashes")
    if str(metadata.get("torch_cuda")) != "11.8":
        raise SystemExit("Local RTX 3090 ablation must use CUDA 11.8")

    records = payload.get("records", [])
    seen: set[tuple[str, int]] = set()
    rooms: set[str] = set()
    for record in records:
        room = str(record.get("room"))
        k = int(record.get("k", -1))
        key = (room, k)
        if key in seen:
            raise SystemExit(f"Duplicate ablation record: {key}")
        seen.add(key)
        rooms.add(room)
        variants = record.get("variants", {})
        missing = EXPECTED_VARIANTS - set(variants)
        if missing and not args.allow_partial:
            raise SystemExit(f"{key} misses variants: {sorted(missing)}")
        for name, result in variants.items():
            timings = result.get("timings", [])
            if len(timings) != int(metadata.get("repeats", -1)):
                raise SystemExit(f"{key} {name} has {len(timings)} timings")
            recall = float(result.get("recall_vs_cukd", {}).get("mean", -1))
            if recall < 0 or recall > 1:
                raise SystemExit(f"{key} {name} has invalid recall {recall}")

    if not args.allow_partial:
        expected = {
            (room, k) for room in rooms for k in EXPECTED_K
        }
        if len(rooms) != args.expected_rooms:
            raise SystemExit(
                f"Expected {args.expected_rooms} rooms, found {len(rooms)}"
            )
        missing_records = expected - seen
        if missing_records:
            raise SystemExit(f"Missing {len(missing_records)} room/k records")
    print(
        f"Ablation coverage OK: {len(rooms)} rooms, {len(records)} records, "
        f"sorting={metadata['sorting_revision']}"
    )


if __name__ == "__main__":
    main()
