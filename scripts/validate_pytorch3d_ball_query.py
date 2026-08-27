#!/usr/bin/env python3
"""Validate matched Pointcept/PyTorch3D ball-query result coverage and provenance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcept", type=Path, required=True)
    parser.add_argument("--pytorch3d", type=Path, required=True)
    args = parser.parse_args()
    pointcept, pytorch3d = load(args.pointcept), load(args.pytorch3d)
    left, right = pointcept["metadata"], pytorch3d["metadata"]
    for field in ("dataset", "torch", "torch_cuda", "voxel_size_m", "crop_points",
                  "warmups", "repeats", "seed", "eligible_rooms"):
        if left.get(field) != right.get(field):
            raise SystemExit(f"Protocol mismatch for {field}: {left.get(field)!r} != {right.get(field)!r}")
    if left["gpu"]["uuid"] != right["gpu"]["uuid"]:
        raise SystemExit("Pointcept and PyTorch3D were not measured on the same GPU")
    if pointcept["radii"] != pytorch3d["radii"]:
        raise SystemExit("PyTorch3D did not reuse the exact Pointcept radii")
    expected = 81 * 2 * 3
    expected_keys = {
        (record["room"], record["mode"], int(record["k"]))
        for record in pointcept["records"]
    }
    actual_keys = {
        (record["room"], record["mode"], int(record["k"]))
        for record in pytorch3d["records"]
    }
    if len(pointcept["records"]) != expected or len(expected_keys) != expected:
        raise SystemExit("Pointcept reference is incomplete or contains duplicate keys")
    if len(pytorch3d["records"]) != expected or len(actual_keys) != expected:
        raise SystemExit("PyTorch3D result is incomplete or contains duplicate keys")
    if actual_keys != expected_keys:
        raise SystemExit("Pointcept/PyTorch3D record identities differ")
    reference = {
        (record["room"], record["mode"], int(record["k"])): record
        for record in pointcept["records"]
    }
    repeats = int(right["repeats"])
    for record in pytorch3d["records"]:
        key = (record["room"], record["mode"], int(record["k"]))
        matched = reference[key]
        if (record["num_support"], record["num_query"], record["radius_m"]) != (
            matched["num_support"], matched["num_query"], matched["radius_m"]
        ):
            raise SystemExit(f"Input/radius mismatch at {key}")
        if len(record["query_timings_s"]) != repeats or not all(
            math.isfinite(value) and value > 0 for value in record["query_timings_s"]
        ):
            raise SystemExit(f"Invalid timing samples at {key}")
        for field in ("valid_neighbor_ratio", "insufficient_query_ratio", "truncated_query_ratio"):
            value = float(record[field])
            if not 0.0 <= value <= 1.0:
                raise SystemExit(f"Invalid {field} at {key}: {value}")
        if abs(record["insufficient_query_ratio"] - matched["insufficient_query_ratio"]) > 1e-7:
            raise SystemExit(f"Exact-reference insufficient fraction mismatch at {key}")
        if abs(record["truncated_query_ratio"] - matched["truncated_query_ratio"]) > 1e-7:
            raise SystemExit(f"Exact-reference truncation fraction mismatch at {key}")
    print("Coverage OK: 81 rooms x pre/post x k=24/32/48; same GPU, crops and radii")


if __name__ == "__main__":
    main()
