#!/usr/bin/env python3
"""Validate adaptive-neighborhood coverage, provenance, and timing fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_K = (8, 16, 24, 32, 48, 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected-rooms", type=int, default=81)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})

    if str(metadata.get("torch_cuda")) != "11.8":
        raise SystemExit("RTX 3090 adaptive-neighborhood runs require CUDA 11.8")
    if not str(metadata.get("torch", "")).startswith("2.7.1+cu118"):
        raise SystemExit("Adaptive-neighborhood run did not use PyTorch 2.7.1+cu118")
    if not metadata.get("source_sha256") or not metadata.get("extension", {}).get("sha256"):
        raise SystemExit("Missing source or extension hash")
    if tuple(metadata.get("k", ())) != EXPECTED_K:
        raise SystemExit(f"Expected k={EXPECTED_K}, found {metadata.get('k')}")
    if metadata.get("scope") != "sample_part" or metadata.get("mode") != "pre":
        raise SystemExit("Expected S3DIS sample_part pre-query protocol")
    if int(metadata.get("crop_points", -1)) != 250_000:
        raise SystemExit("Expected 250,000-point crops")
    if int(metadata.get("num_down_fixed", -1)) != 2:
        raise SystemExit("Fixed baseline must use num_down=2")
    timing_boundary = str(metadata.get("timing_boundary", ""))
    if "selection" not in timing_boundary:
        raise SystemExit("Timing boundary does not include adaptive selection")
    if "compatible-input" not in timing_boundary:
        raise SystemExit("Timing boundary excludes compatible-input construction")
    if int(metadata.get("adaptive_min_candidates_factor", -1)) != 2:
        raise SystemExit("Adaptive lower candidate threshold must be 2k")
    if int(metadata.get("adaptive_max_candidates_factor", -1)) != 8:
        raise SystemExit("Adaptive upper candidate threshold must be 8k")
    if "unchanged" not in str(metadata.get("kernel_abi", "")):
        raise SystemExit("Missing unchanged production-kernel ABI declaration")
    variants = tuple(metadata.get("variants", ()))
    if (
        len(variants) != 3
        or variants[0] != "fixed_3x3x3"
        or not variants[1].startswith("adaptive_")
        or variants[2] != "cuda_kdtree_exact"
    ):
        raise SystemExit(f"Unexpected variants: {variants}")
    expected_variants = set(variants)

    target_uuid = str(metadata.get("gpu", {}).get("uuid", ""))
    benchmark_pids = {
        str(pid) for pid in metadata.get("benchmark_pids", [metadata.get("pid")])
        if pid is not None
    }
    for snapshot_name in ("co_tenant_start", "co_tenant_end"):
        for line in metadata.get(snapshot_name, {}).get("compute_processes", []):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 2 and fields[0] == target_uuid:
                if fields[1] not in benchmark_pids:
                    raise SystemExit(
                        f"Foreign target-GPU process in {snapshot_name}: {line}"
                    )

    records = payload.get("records", [])
    rooms: set[str] = set()
    seen: set[tuple[str, int]] = set()
    for record in records:
        room = str(record.get("room"))
        k = int(record.get("k", -1))
        key = (room, k)
        if key in seen:
            raise SystemExit(f"Duplicate record: {key}")
        if k not in EXPECTED_K:
            raise SystemExit(f"Unexpected k in {key}")
        seen.add(key)
        rooms.add(room)
        if int(record.get("num_support", -1)) != 250_000:
            raise SystemExit(f"{key} is not a 250,000-point crop")
        order = record.get("measurement_order", [])
        if len(order) != 3 or set(order) != expected_variants:
            raise SystemExit(f"{key} has an invalid measurement order")
        results = record.get("variants", {})
        if set(results) != expected_variants:
            raise SystemExit(f"{key} has variants {sorted(results)}")
        for name, result in results.items():
            timings = result.get("timings", [])
            if len(timings) != int(metadata.get("repeats", -1)):
                raise SystemExit(f"{key} {name} has {len(timings)} timings")
            recall = float(result.get("recall_vs_cukd", {}).get("mean", -1))
            if recall < 0 or recall > 1:
                raise SystemExit(f"{key} {name} has invalid recall {recall}")
            for timing in timings:
                fields = (
                    float(timing["construction_ms"]),
                    float(timing["selection_ms"]),
                    float(timing["compatibility_ms"]),
                    float(timing["query_ms"]),
                    float(timing["total_ms"]),
                )
                if not all(math.isfinite(value) and value >= 0 for value in fields):
                    raise SystemExit(f"{key} {name} has invalid timing {timing}")
                if not math.isclose(sum(fields[:4]), fields[4], abs_tol=1e-5):
                    raise SystemExit(f"{key} {name} total excludes a timing component")
            if name.startswith("adaptive_"):
                if not result.get("distance_check", {}).get("allclose", False):
                    raise SystemExit(f"{key} adaptive output distances are invalid")
                octree = result.get("octree", {})
                if int(octree.get("octree_levels", 0)) < 2:
                    raise SystemExit(f"{key} did not construct a multilevel octree")
                if not octree.get("selected_levels"):
                    raise SystemExit(f"{key} has no adaptive level-selection record")
                if int(octree.get("min_candidates_factor", -1)) != 2:
                    raise SystemExit(f"{key} adaptive lower threshold is not 2k")
                if int(octree.get("max_candidates_factor", -1)) != 8:
                    raise SystemExit(f"{key} adaptive upper threshold is not 8k")
                if int(octree.get("query_kernel_launches", -1)) != 1:
                    raise SystemExit(f"{key} adaptive query used multiple launches")
                band = octree.get("selection_band_points", {})
                band_total = sum(int(value) for value in band.values())
                if band_total != int(record["num_support"]):
                    raise SystemExit(f"{key} selection-band accounting is incomplete")
                if int(octree.get("compatible_group_count", 0)) <= 0:
                    raise SystemExit(f"{key} has no compatible query groups")
                if int(octree.get("compatible_support_descriptor_count", 0)) <= 0:
                    raise SystemExit(f"{key} has no compatible support descriptors")
                if int(octree.get("compatible_point_count", 0)) < int(record["num_support"]):
                    raise SystemExit(f"{key} compatible coordinate array is incomplete")
                if float(octree.get("compatible_point_ratio", 0)) < 1.0:
                    raise SystemExit(f"{key} has an invalid compatible point ratio")
            elif name == "cuda_kdtree_exact":
                if not result.get("exact", False):
                    raise SystemExit(f"{key} cudaKDTree result is not marked exact")
                if not math.isclose(recall, 1.0, abs_tol=1e-8):
                    raise SystemExit(f"{key} cudaKDTree self-recall is not one")

    if not args.allow_partial:
        if int(metadata.get("warmups", -1)) != 5:
            raise SystemExit("Formal run requires 5 warmups")
        if int(metadata.get("repeats", -1)) != 20:
            raise SystemExit("Formal run requires 20 repeats")
        if len(rooms) != args.expected_rooms:
            raise SystemExit(f"Expected {args.expected_rooms} rooms, found {len(rooms)}")
        expected = {(room, k) for room in rooms for k in EXPECTED_K}
        if seen != expected:
            raise SystemExit(f"Missing {len(expected - seen)} room/k records")
        for k in EXPECTED_K:
            first = {name: 0 for name in expected_variants}
            for record in records:
                if int(record["k"]) == k:
                    first[record["measurement_order"][0]] += 1
            if max(first.values()) - min(first.values()) > 1:
                raise SystemExit(f"k={k} ordering is not balanced: {first}")
    print(
        f"Adaptive-neighborhood coverage OK: {len(rooms)} rooms, "
        f"{len(records)} records"
    )


if __name__ == "__main__":
    main()
