#!/usr/bin/env python3
"""Validate thread-grouping ablation coverage, provenance, and recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_K = (8, 16, 24, 32, 48, 64)
EXPECTED_VARIANTS = {"adaptive", "fixed_8", "fixed_16", "fixed_32"}
EXPECTED_GROUPS = {
    "adaptive": None, "fixed_8": 8, "fixed_16": 16, "fixed_32": 32,
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
        raise SystemExit("Thread grouping does not use generated top-P")
    if not metadata.get("source_sha256") or not metadata.get("extension", {}).get("sha256"):
        raise SystemExit("Missing source or extension hash")
    if str(metadata.get("torch_cuda")) != "11.8":
        raise SystemExit("Local RTX 3090 thread grouping must use CUDA 11.8")
    equivalence_policy = metadata.get("strategy_equivalence", {})
    if equivalence_policy.get("reference") != "adaptive":
        raise SystemExit("Missing adaptive strategy-equivalence policy")
    if "counted at most once" not in metadata.get("recall_definition", ""):
        raise SystemExit("Missing strict row-wise set-recall definition")
    if not str(metadata.get("variant_ordering", "")).startswith(
        "balanced cyclic rotation"
    ):
        raise SystemExit("Missing balanced strategy measurement ordering")
    if tuple(metadata.get("k", ())) != EXPECTED_K:
        raise SystemExit(f"Expected k={EXPECTED_K}, found {metadata.get('k')}")
    if metadata.get("scope") != "sample_part" or metadata.get("mode") != "pre":
        raise SystemExit("Thread grouping must use S3DIS sample_part pre-query")
    if int(metadata.get("crop_points", -1)) != 250_000:
        raise SystemExit("Thread grouping must use 250,000-point crops")
    target_uuid = str(metadata.get("gpu", {}).get("uuid", ""))
    benchmark_pids = {
        str(pid) for pid in metadata.get(
            "benchmark_pids", [metadata.get("pid")]
        ) if pid is not None
    }
    if not target_uuid or not benchmark_pids:
        raise SystemExit("Missing target GPU UUID or benchmark PID")
    for snapshot_name in ("co_tenant_start", "co_tenant_end"):
        process_lines = metadata.get(snapshot_name, {}).get(
            "compute_processes", []
        )
        for line in process_lines:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 2 and fields[0] == target_uuid:
                if fields[1] not in benchmark_pids:
                    raise SystemExit(
                        f"Foreign target-GPU process in {snapshot_name}: {line}"
                    )

    records = payload.get("records", [])
    seen: set[tuple[str, int]] = set()
    rooms: set[str] = set()
    for record in records:
        room = str(record.get("room"))
        k = int(record.get("k", -1))
        if k not in EXPECTED_K:
            raise SystemExit(f"{room} has unexpected k={k}")
        key = (room, k)
        if key in seen:
            raise SystemExit(f"Duplicate thread-group record: {key}")
        seen.add(key)
        rooms.add(room)
        if int(record.get("num_support", -1)) != 250_000:
            raise SystemExit(f"{key} is not a 250,000-point crop")
        if record.get("crop_center") is None:
            raise SystemExit(f"{key} is missing its deterministic crop center")
        variants = record.get("variants", {})
        measurement_order = record.get("measurement_order", [])
        if len(measurement_order) != len(set(measurement_order)) or set(
            measurement_order
        ) != EXPECTED_VARIANTS:
            raise SystemExit(f"{key} has invalid measurement order")
        missing = EXPECTED_VARIANTS - set(variants)
        if missing and not args.allow_partial:
            raise SystemExit(f"{key} misses variants: {sorted(missing)}")
        for name, result in variants.items():
            if name not in EXPECTED_VARIANTS:
                raise SystemExit(f"{key} has unexpected variant {name}")
            actual_group = result.get("configuration", {}).get(
                "thread_group_size"
            )
            if actual_group != EXPECTED_GROUPS[name]:
                raise SystemExit(
                    f"{key} {name} has thread_group_size={actual_group}"
                )
            timings = result.get("timings", [])
            if len(timings) != int(metadata.get("repeats", -1)):
                raise SystemExit(f"{key} {name} has {len(timings)} timings")
            value = float(result.get("recall_vs_cukd", {}).get("mean", -1))
            if value < 0 or value > 1:
                raise SystemExit(f"{key} {name} has invalid recall {value}")
            equivalence = result.get("equivalence_vs_adaptive", {})
            if not equivalence.get("squared_distance_allclose", False):
                raise SystemExit(
                    f"{key} {name} neighbor distances differ from adaptive: "
                    f"{equivalence}"
                )
            if int(equivalence.get("squared_distance_differing_queries", -1)) != 0:
                raise SystemExit(
                    f"{key} {name} has distance-differing queries"
                )

    if not args.allow_partial:
        if int(metadata.get("warmups", -1)) != 5:
            raise SystemExit("Formal thread grouping requires 5 warmups")
        if int(metadata.get("repeats", -1)) != 20:
            raise SystemExit("Formal thread grouping requires 20 repeats")
        if len(rooms) != args.expected_rooms:
            raise SystemExit(
                f"Expected {args.expected_rooms} rooms, found {len(rooms)}"
            )
        expected = {(room, k) for room in rooms for k in EXPECTED_K}
        missing_records = expected - seen
        if missing_records:
            raise SystemExit(f"Missing {len(missing_records)} room/k records")
        if len(records) != args.expected_rooms * len(EXPECTED_K):
            raise SystemExit(
                f"Expected {args.expected_rooms * len(EXPECTED_K)} records, "
                f"found {len(records)}"
            )
        crop_centers: dict[str, set[int]] = {}
        for record in records:
            crop_centers.setdefault(record["room"], set()).add(
                int(record["crop_center"])
            )
        inconsistent = {
            room: centers for room, centers in crop_centers.items()
            if len(centers) != 1
        }
        if inconsistent:
            raise SystemExit(f"Crop center changes across k: {inconsistent}")
        for k in EXPECTED_K:
            first_counts = {name: 0 for name in EXPECTED_VARIANTS}
            for record in records:
                if int(record["k"]) == k:
                    first_counts[record["measurement_order"][0]] += 1
            if max(first_counts.values()) - min(first_counts.values()) > 1:
                raise SystemExit(
                    f"k={k} measurement order is not balanced: {first_counts}"
                )
    print(
        f"Thread-group coverage OK: {len(rooms)} rooms, {len(records)} records, "
        "all strategies have per-query neighbor-distance equivalence"
    )


if __name__ == "__main__":
    main()
