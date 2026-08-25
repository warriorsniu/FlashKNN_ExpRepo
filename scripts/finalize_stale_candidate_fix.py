#!/usr/bin/env python3
"""Finalize the stale-candidate-fix L20 result pack without changing baselines.

This utility creates canonical query files beside the independent refresh
files. It replaces only methods measured with the current production build,
copies the unaffected ball-query result byte-for-byte, validates the complete
network matrix, and records hashes that make every preserved field auditable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


S3DIS_REFRESH_METHODS = ("flashknn", "cuda_kdtree")
LIDAR_REFRESH_FIELDS = ("flashknn", "exact", "faiss_flat", "faiss_ivf")
LIDAR_PRESERVED_FIELDS = ("flann_cuda", "nanoflann")
NETWORK_FILES = (
    "dela_s3dis.json",
    "ptv3_s3dis.json",
    "octformer_s3dis.json",
    "spunet_s3dis.json",
    "minkunet34c_s3dis.json",
    "dela_semantickitti_backends.json",
    "deepla_semantickitti_backends.json",
    "ptv3_semantickitti.json",
    "octformer_semantickitti.json",
    "spunet_semantickitti.json",
    "minkunet34c_semantickitti.json",
)
SOURCE_FILES = (
    "FlashKNN/csrc/flash_knn_query_dynamic_load.cu",
    "FlashKNN/csrc/flash_knn_query_GMPS.cu",
    "FlashKNN/csrc/flash_knn_bitonic_top_p.cuh",
    "FlashKNN/functions/FlashKnnWrapper.py",
    "Query/benchmark_s3dis.py",
    "Query/benchmark_semantickitti.py",
    "Query/faiss_backends.py",
)


def arguments() -> argparse.Namespace:
    """Parse the historical canonical pack and independent refresh directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    """Load one JSON object from ``path``."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return value


def save(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    """Return a stable SHA256 digest for a JSON-compatible value."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def keyed_object_sha256(value: dict[tuple[Any, ...], Any]) -> str:
    """Hash a tuple-keyed mapping through a deterministic JSON representation."""
    serializable = [
        {"key": list(key), "value": item}
        for key, item in sorted(value.items())
    ]
    return object_sha256(serializable)


def finite_timings(timings: Any, expected: int) -> bool:
    """Check timing count and all numeric values while allowing string labels."""
    return (
        isinstance(timings, list)
        and len(timings) == expected
        and all(
            math.isfinite(float(value))
            for timing in timings
            if isinstance(timing, dict)
            for value in timing.values()
            if isinstance(value, (int, float))
        )
    )


def query_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    """Return the S3DIS room/mode/scope/k identity."""
    return (
        str(record["room"]), str(record["mode"]),
        str(record["scope"]), int(record["k"]),
    )


def lidar_key(record: dict[str, Any]) -> tuple[str, str, int]:
    """Return the SemanticKITTI sample/mode/k identity."""
    return str(record["sample"]), str(record["mode"]), int(record["k"])


def query_mean_ms(method: dict[str, Any]) -> float:
    """Return a method's mean query latency in milliseconds."""
    return statistics.mean(float(timing["query_s"]) for timing in method["timings"]) * 1000


def lidar_flash_mean_ms(item: dict[str, Any]) -> float:
    """Return one SemanticKITTI FlashKNN entry's mean query latency."""
    return statistics.mean(float(timing["查询耗时"]) for timing in item["timings"]) * 1000


def protocol_difference(
    identities: list[tuple[str, float, float]],
) -> dict[str, Any]:
    """Summarize paired new/old timings and the largest absolute row delta."""
    old_mean = statistics.mean(old for _, old, _ in identities)
    new_mean = statistics.mean(new for _, _, new in identities)
    deltas = [
        (identity, (new / old - 1.0) * 100.0)
        for identity, old, new in identities
    ]
    worst_identity, worst_delta = max(deltas, key=lambda value: abs(value[1]))
    return {
        "records": len(identities),
        "old_mean_ms": old_mean,
        "new_mean_ms": new_mean,
        "mean_delta_percent": (new_mean / old_mean - 1.0) * 100.0,
        "largest_absolute_row_delta": {
            "identity": worst_identity,
            "delta_percent": worst_delta,
        },
    }


def compare_s3dis(base_path: Path, refresh_path: Path) -> list[dict[str, Any]]:
    """Compare every S3DIS mode/k protocol and inspect all room-level deltas."""
    base, refresh = load(base_path), load(refresh_path)
    base_map = {query_key(record): record for record in base["records"]}
    refresh_map = {query_key(record): record for record in refresh["records"]}
    protocols = sorted({(key[1], key[3]) for key in refresh_map})
    output = []
    for mode, k in protocols:
        keys = [key for key in refresh_map if key[1] == mode and key[3] == k]
        values = [
            (
                key[0],
                query_mean_ms(base_map[key]["methods"]["flashknn"]),
                query_mean_ms(refresh_map[key]["methods"]["flashknn"]),
            )
            for key in keys
        ]
        output.append({"mode": mode, "k": k, **protocol_difference(values)})
    return output


def compare_lidar(base_path: Path, refresh_path: Path) -> list[dict[str, Any]]:
    """Compare alpha=4 SemanticKITTI FlashKNN latency for every mode/k."""
    base, refresh = load(base_path), load(refresh_path)
    base_map = {lidar_key(record): record for record in base["samples"]}
    refresh_map = {lidar_key(record): record for record in refresh["samples"]}
    protocols = sorted({(key[1], key[2]) for key in refresh_map})
    output = []
    for mode, k in protocols:
        keys = [key for key in refresh_map if key[1] == mode and key[2] == k]
        values = []
        for key in keys:
            old = next(item for item in base_map[key]["flashknn"] if int(item["alpha"]) == 4)
            new = next(item for item in refresh_map[key]["flashknn"] if int(item["alpha"]) == 4)
            values.append((key[0], lidar_flash_mean_ms(old), lidar_flash_mean_ms(new)))
        output.append({"mode": mode, "k": k, **protocol_difference(values)})
    return output


def compare_network(base_path: Path, refresh_path: Path) -> dict[str, Any]:
    """Compare paired FlashKNN end-to-end latency for every network sample."""
    base, refresh = load(base_path), load(refresh_path)
    collection = "samples" if "samples" in refresh else "records"
    identity_name = "sample" if collection == "samples" else "room"
    base_map = {str(record[identity_name]): record for record in base[collection]}
    refresh_map = {str(record[identity_name]): record for record in refresh[collection]}
    values = []
    for identity, record in refresh_map.items():
        old_summary = base_map[identity]["backends"]["flashknn"]["end_to_end"]
        new_summary = record["backends"]["flashknn"]["end_to_end"]
        values.append((identity, float(old_summary["mean_ms"]), float(new_summary["mean_ms"])))
    return protocol_difference(values)


def merge_s3dis(
    base_path: Path,
    refresh_path: Path,
    output_path: Path,
    expected_records: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Replace paired FlashKNN/cudaKDTree fields and preserve other methods."""
    base, refresh = load(base_path), load(refresh_path)
    base_records, refresh_records = base["records"], refresh["records"]
    base_map = {query_key(record): record for record in base_records}
    refresh_map = {query_key(record): record for record in refresh_records}
    if len(base_map) != expected_records or set(base_map) != set(refresh_map):
        raise SystemExit(f"S3DIS key mismatch: {refresh_path}")
    preserved_before = {
        key: {
            name: copy.deepcopy(value)
            for name, value in record["methods"].items()
            if name not in S3DIS_REFRESH_METHODS
        }
        for key, record in base_map.items()
    }
    for key, target in base_map.items():
        source = refresh_map[key]
        if (target["num_support"], target["num_query"]) != (
            source["num_support"], source["num_query"],
        ):
            raise SystemExit(f"S3DIS point-count mismatch: {key}")
        if set(source["methods"]) != set(S3DIS_REFRESH_METHODS):
            raise SystemExit(f"Unexpected refresh methods: {key}")
        for method in S3DIS_REFRESH_METHODS:
            if not finite_timings(source["methods"][method]["timings"], 10):
                raise SystemExit(f"Invalid S3DIS timings: {key} {method}")
            target["methods"][method] = copy.deepcopy(source["methods"][method])
    preserved_after = {
        key: {
            name: value
            for name, value in record["methods"].items()
            if name not in S3DIS_REFRESH_METHODS
        }
        for key, record in base_map.items()
    }
    if preserved_before != preserved_after:
        raise SystemExit("An unaffected S3DIS baseline changed")
    base["metadata"]["timing_overrides"] = [{
        "source": refresh_path.name,
        "reason": "Refresh stale-candidate-fix production FlashKNN and paired cudaKDTree.",
        "methods": list(S3DIS_REFRESH_METHODS),
        "modes": sorted({str(record["mode"]) for record in refresh_records}),
        "k": sorted({int(record["k"]) for record in refresh_records}),
        "records": expected_records,
        "warmups": 3,
        "repeats": 10,
        "gpu": copy.deepcopy(refresh["metadata"]["gpu"]),
        "provenance": provenance,
        "preserved_baselines_sha256": keyed_object_sha256(preserved_before),
    }]
    save(output_path, base)
    return {
        "records": expected_records,
        "preserved_baselines_sha256": keyed_object_sha256(preserved_before),
        "output_sha256": file_sha256(output_path),
    }


def merge_lidar(
    base_path: Path,
    refresh_path: Path,
    output_path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Install refreshed GPU fields and preserve only unchanged legacy backends."""
    base, refresh = load(base_path), load(refresh_path)
    base_map = {lidar_key(record): record for record in base["samples"]}
    refresh_map = {lidar_key(record): record for record in refresh["samples"]}
    if len(base_map) != 1320 or set(base_map) != set(refresh_map):
        raise SystemExit("SemanticKITTI key mismatch")
    preserved = {
        key: {name: copy.deepcopy(base_map[key][name]) for name in LIDAR_PRESERVED_FIELDS}
        for key in base_map
    }
    for key, record in refresh_map.items():
        if (record["num_support"], record["num_query"]) != (
            base_map[key]["num_support"], base_map[key]["num_query"],
        ):
            raise SystemExit(f"SemanticKITTI point-count mismatch: {key}")
        for name in LIDAR_REFRESH_FIELDS:
            value = record[name]
            entries = value if name == "flashknn" else (value,)
            if any(not finite_timings(entry["timings"], 10) for entry in entries):
                raise SystemExit(f"Invalid SemanticKITTI timings: {key} {name}")
        for name in LIDAR_PRESERVED_FIELDS:
            record[name] = copy.deepcopy(preserved[key][name])
    refresh["metadata"]["provenance"] = {
        **provenance,
        "refreshed_fields": list(LIDAR_REFRESH_FIELDS),
        "preserved_fields": list(LIDAR_PRESERVED_FIELDS),
        "preserved_baselines_sha256": keyed_object_sha256(preserved),
    }
    save(output_path, refresh)
    merged = load(output_path)
    after = {
        lidar_key(record): {name: record[name] for name in LIDAR_PRESERVED_FIELDS}
        for record in merged["samples"]
    }
    if preserved != after:
        raise SystemExit("An unaffected SemanticKITTI baseline changed")
    return {
        "records": 1320,
        "preserved_baselines_sha256": keyed_object_sha256(preserved),
        "output_sha256": file_sha256(output_path),
    }


def validate_networks(run_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Validate all refreshed S3DIS and SemanticKITTI network files."""
    identities: dict[str, Any] = {}
    for name in NETWORK_FILES:
        path = run_dir / "network" / name
        payload = load(path)
        records = payload.get("records", payload.get("samples", []))
        expected = 22 if "semantickitti" in name else 68
        if len(records) != expected:
            raise SystemExit(f"Network coverage mismatch: {name}")
        if payload.get("metadata", {}).get("gpu", {}).get("uuid") != provenance["gpu_uuid"]:
            raise SystemExit(f"Network GPU mismatch: {name}")
        identities[name] = {"records": expected, "sha256": file_sha256(path)}
    return identities


def main() -> None:
    """Create canonical files and one machine-readable merge audit."""
    args = arguments()
    repo, base, run_dir = args.repo.resolve(), args.base.resolve(), args.run_dir.resolve()
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    if actual_commit != args.source_commit:
        raise SystemExit(f"HEAD {actual_commit} != requested {args.source_commit}")
    source_hashes = {name: file_sha256(repo / name) for name in SOURCE_FILES}
    extension = next((repo / "FlashKNN/functions").glob("CuFun*.so"))
    provenance = {
        "source_commit": args.source_commit,
        "gpu_uuid": "GPU-78990023-5606-bb80-49bf-8ddfe8683461",
        "source_sha256": source_hashes,
        "extension_sha256": file_sha256(extension),
        "torch_cuda_arch_list": "8.9",
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    query_dir = run_dir / "query"
    audit: dict[str, Any] = {"provenance": provenance, "merged": {}}
    audit["merged"]["s3dis_sample_part"] = merge_s3dis(
        base / "query/s3dis_sample_part.json",
        query_dir / "s3dis_sample_part_refresh.json",
        query_dir / "s3dis_sample_part.json", 972, provenance,
    )
    audit["merged"]["s3dis_full_k32"] = merge_s3dis(
        base / "query/s3dis_full_k32.json",
        query_dir / "s3dis_full_k32_refresh.json",
        query_dir / "s3dis_full_k32.json", 272, provenance,
    )
    audit["merged"]["semantickitti"] = merge_lidar(
        base / "query/semantickitti.json",
        query_dir / "semantickitti_refresh.json",
        query_dir / "semantickitti.json", provenance,
    )
    ball_source = base / "query/ball_query_s3dis_sample_part.json"
    ball_target = query_dir / "ball_query_s3dis_sample_part.json"
    ball_target.write_bytes(ball_source.read_bytes())
    audit["merged"]["ball_query"] = {
        "source": str(ball_source.relative_to(repo)),
        "sha256": file_sha256(ball_target),
        "byte_identical": file_sha256(ball_source) == file_sha256(ball_target),
    }
    audit["networks"] = validate_networks(run_dir, provenance)
    audit["latency_difference"] = {
        "s3dis_sample_part": compare_s3dis(
            base / "query/s3dis_sample_part.json",
            query_dir / "s3dis_sample_part_refresh.json",
        ),
        "s3dis_full_k32": compare_s3dis(
            base / "query/s3dis_full_k32.json",
            query_dir / "s3dis_full_k32_refresh.json",
        ),
        "semantickitti_alpha4": compare_lidar(
            base / "query/semantickitti.json",
            query_dir / "semantickitti_refresh.json",
        ),
        "network": {
            name: compare_network(base / "network" / name, run_dir / "network" / name)
            for name in (
                "dela_s3dis.json",
                "dela_semantickitti_backends.json",
                "deepla_semantickitti_backends.json",
            )
        },
    }
    save(run_dir / "audit/finalization.json", audit)
    print(run_dir / "audit/finalization.json")


if __name__ == "__main__":
    main()
