#!/usr/bin/env python3
"""Attach final-kernel provenance and merge refreshed L20 S3DIS measurements.

This file is limited to the final-revision refresh protocol: it validates the
independent raw query/network files, enriches their metadata, replaces only
FlashKNN and cudaKDTree query fields in the canonical result, and installs the
paired DeLA result.  It never runs a benchmark or changes a timing boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


JsonObject = dict[str, object]
SOURCE_FILES = (
    "FlashKNN/csrc/flash_knn_query_dynamic_load.cu",
    "FlashKNN/csrc/flash_knn_query.h",
    "FlashKNN/functions/FlashKnnWrapper.py",
    "FlashKNN/csrc/api.cpp",
    "FlashKNN/csrc/flash_knn_bitonic_top_p.cuh",
    "Query/benchmark_s3dis.py",
    "DeLA/S3DIS/benchmark_latency.py",
)
EXPECTED_SOURCE_SHA256 = {
    "FlashKNN/csrc/flash_knn_query_dynamic_load.cu": "d09b091ee345a509b55742f058a0bab0bf0b0ae01344b0a14258d6e5fe7a51ac",
    "FlashKNN/csrc/flash_knn_query.h": "20ef7a84e9f5893b67a2aac60a9e43c593a17bf9099b57d59a3f5c1603c73406",
    "FlashKNN/functions/FlashKnnWrapper.py": "080865079cf9e8bc21eab360292527158d2328c196f8f5d9b16485075488df30",
    "FlashKNN/csrc/api.cpp": "f74a62904aab4aa6aaaa507e3e0d93accce5e9ceffe529a88ae1486c577848c6",
    "FlashKNN/csrc/flash_knn_bitonic_top_p.cuh": "323bf535be8078635cf5e8f0d83a24a189c9b14dee088cfa94dcf7afbd3b28aa",
    "Query/benchmark_s3dis.py": "1cfccf90bec663ac9f214fbf3f7029ed839b86e6f7708ddf0a6d1d7f32b35fa4",
    "DeLA/S3DIS/benchmark_latency.py": "88c7d4a0097bf213f267968f4cebd6674d9c3ca3ec57579a4695e1f9832e7d61",
}
QUERY_METHODS = ("flashknn", "cuda_kdtree")


def arguments() -> argparse.Namespace:
    """Parse paths and immutable run identity supplied by the operator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> JsonObject:
    """Load one JSON object and reject missing or non-object payloads."""
    if not path.is_file():
        raise SystemExit(f"Missing required result: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def save(path: Path, value: JsonObject) -> None:
    """Atomically save a JSON object so canonical files are never partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a source file or room manifest input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(payload: JsonObject) -> list[JsonObject]:
    """Return a benchmark's record list after validating its basic shape."""
    value = payload.get("records")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SystemExit("Benchmark payload has no valid records list")
    return value


def metadata(payload: JsonObject) -> JsonObject:
    """Return a benchmark's metadata mapping after validating its shape."""
    value = payload.get("metadata")
    if not isinstance(value, dict):
        raise SystemExit("Benchmark payload has no valid metadata object")
    return value


def query_key(record: JsonObject) -> tuple[str, str, str, int]:
    """Build the canonical room/mode/scope/k identity of a query record."""
    return (
        str(record["room"]), str(record["mode"]),
        str(record["scope"]), int(record["k"]),
    )


def source_hashes(repo: Path) -> dict[str, str]:
    """Hash every production source and benchmark entry named by the protocol."""
    actual = {relative: sha256(repo / relative) for relative in SOURCE_FILES}
    if actual != EXPECTED_SOURCE_SHA256:
        differences = {
            path: (EXPECTED_SOURCE_SHA256.get(path), digest)
            for path, digest in actual.items()
            if EXPECTED_SOURCE_SHA256.get(path) != digest
        }
        raise SystemExit(f"Production source identity changed: {differences}")
    return actual


def room_manifest(payload: JsonObject) -> JsonObject:
    """Summarize ordered room names without duplicating the full list in metadata."""
    names = sorted({str(record["room"]) for record in records(payload)})
    encoded = "\n".join(names).encode("utf-8")
    return {
        "room_count": len(names),
        "room_names_sha256": hashlib.sha256(encoded).hexdigest(),
        "first_room": names[0],
        "last_room": names[-1],
    }


def git_dirty(repo: Path) -> bool:
    """Report whether finalization occurs with tracked or untracked changes."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
    )
    return bool(status.strip())


def common_provenance(
    run_dir: Path,
    repo: Path,
    source_commit: str,
    physical_gpu: int,
    data_root: Path,
    hashes: dict[str, str],
) -> JsonObject:
    """Create provenance shared by raw query and network payloads.

    Args:
        run_dir: Independent raw result directory for this refresh.
        repo: Repository containing the verified production sources.
        source_commit: Exact Git commit compiled for the measurements.
        physical_gpu: Physical nvidia-smi index used for all measurements.
        data_root: Pointcept-format S3DIS root used by the benchmarks.
        hashes: SHA256 mapping for all protocol-critical source files.
    """
    extension_candidates = sorted(
        repo.glob(".venv/lib/python*/site-packages/FlashKNN/CuFun*.so")
    )
    if len(extension_candidates) != 1:
        raise SystemExit(
            f"Expected one installed FlashKNN extension, got {extension_candidates}"
        )
    extension = extension_candidates[0]
    return {
        "run_id": run_dir.name,
        "source_commit": source_commit,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "source_commit": source_commit,
            "worktree_at_source_verification": "clean",
            "dirty_at_finalization": git_dirty(repo),
        },
        "production_source_sha256": hashes,
        "installed_extension": {
            "path": str(extension.relative_to(repo)),
            "sha256": sha256(extension),
            "torch_cuda_arch_list": "8.9",
        },
        "physical_gpu_index": physical_gpu,
        "effective_flashknn": {
            "memory_mode": "SM",
            "sorting_mode": "PS",
            "top_p": "generated_bitonic_top_p",
            "candidate_storage": "registers",
            "skip": True,
            "thread_grouping": "adaptive",
            "alpha": 4,
            "num_down": 2,
        },
        "co_tenant_snapshots": {
            "start_gpu": "provenance/gpu_start.csv",
            "start_compute_processes": "provenance/compute_processes_start.csv",
            "end_gpu": "provenance/gpu_end.csv",
            "end_compute_processes": "provenance/compute_processes_end.csv",
        },
        "data_root": str(data_root.resolve()),
    }


def validate_query(
    raw: JsonObject,
    canonical: JsonObject,
    expected_records: int,
) -> None:
    """Validate raw coverage, timing counts, recall, and canonical identities."""
    raw_records = records(raw)
    canonical_records = records(canonical)
    raw_map = {query_key(record): record for record in raw_records}
    canonical_map = {query_key(record): record for record in canonical_records}
    if len(raw_records) != expected_records or len(raw_map) != expected_records:
        raise SystemExit(
            f"Expected {expected_records} unique raw records, got "
            f"{len(raw_records)} records and {len(raw_map)} keys"
        )
    if set(raw_map) != set(canonical_map):
        raise SystemExit("Raw and canonical query keys differ")
    for key, record in raw_map.items():
        base = canonical_map[key]
        if (record["num_support"], record["num_query"]) != (
            base["num_support"], base["num_query"],
        ):
            raise SystemExit(f"Point counts differ for {key}")
        methods = record.get("methods")
        if not isinstance(methods, dict) or set(methods) != set(QUERY_METHODS):
            raise SystemExit(f"Raw methods differ for {key}")
        for method_name in QUERY_METHODS:
            method = methods[method_name]
            if not isinstance(method, dict):
                raise SystemExit(f"Invalid method payload for {key} {method_name}")
            timings = method.get("timings")
            if not isinstance(timings, list) or len(timings) != 10:
                raise SystemExit(f"Expected 10 timings for {key} {method_name}")
            for timing in timings:
                if not isinstance(timing, dict) or any(
                    not math.isfinite(float(value)) for value in timing.values()
                ):
                    raise SystemExit(f"Non-finite timing for {key} {method_name}")
        flash = methods["flashknn"]
        recall = flash.get("recall_vs_cukd")
        if not isinstance(recall, dict):
            raise SystemExit(f"Missing FlashKNN recall for {key}")
        mean = float(recall["mean"])
        minimum = float(recall["minimum"])
        if not 0.0 <= minimum <= mean <= 1.0:
            raise SystemExit(f"Invalid FlashKNN recall for {key}")


def merge_query(
    raw_path: Path,
    canonical_path: Path,
    expected_records: int,
    provenance: JsonObject,
    hashes: dict[str, str],
) -> None:
    """Replace only final-kernel query methods and preserve historical baselines."""
    raw = load(raw_path)
    canonical = load(canonical_path)
    validate_query(raw, canonical, expected_records)
    raw_meta = metadata(raw)
    raw_meta["provenance"] = {**provenance, "room_manifest": room_manifest(raw)}
    raw_meta["seed"] = 47
    save(raw_path, raw)

    raw_map = {query_key(record): record for record in records(raw)}
    baseline_names = (
        ("flann_cuda", "nanoflann", "faiss_flat", "faiss_ivf")
        if expected_records == 972 else ("flann_cuda", "nanoflann")
    )
    before = {
        query_key(record): {
            name: copy.deepcopy(record.get("methods", {}).get(name))
            for name in baseline_names
        }
        for record in records(canonical)
    }
    for record in records(canonical):
        key = query_key(record)
        base_methods = record.get("methods")
        new_methods = raw_map[key].get("methods")
        if not isinstance(base_methods, dict) or not isinstance(new_methods, dict):
            raise SystemExit(f"Invalid canonical method mapping for {key}")
        for name in QUERY_METHODS:
            base_methods[name] = copy.deepcopy(new_methods[name])
    after = {
        query_key(record): {
            name: record.get("methods", {}).get(name) for name in baseline_names
        }
        for record in records(canonical)
    }
    if before != after:
        raise SystemExit("Historical baseline fields changed during query merge")

    canonical_meta = metadata(canonical)
    source = os.path.relpath(raw_path, canonical_path.parent)
    previous_overrides = canonical_meta.pop("timing_overrides", [])
    if (
        isinstance(previous_overrides, list)
        and len(previous_overrides) == 1
        and isinstance(previous_overrides[0], dict)
        and previous_overrides[0].get("source") == source
    ):
        superseded = previous_overrides[0].get("supersedes", [])
    else:
        superseded = previous_overrides
    raw_meta_gpu = raw_meta.get("gpu")
    canonical_meta["timing_overrides"] = [{
        "source": source,
        "reason": (
            "Refresh FlashKNN and paired cudaKDTree with the verified final "
            "production generated-bitonic SMPS kernel and unified 3/10 protocol."
        ),
        "methods": list(QUERY_METHODS),
        "modes": ["pre", "post"] if expected_records == 972 else ["pre"],
        "k": [8, 16, 24, 32, 48, 64] if expected_records == 972 else [32],
        "records": expected_records,
        "gpu": raw_meta_gpu,
        "physical_gpu_index": provenance["physical_gpu_index"],
        "warmups": 3,
        "repeats": 10,
        "source_commit": provenance["source_commit"],
        "production_source_sha256": hashes,
        "supersedes": superseded,
    }]
    save(canonical_path, canonical)


def install_network(
    raw_path: Path,
    canonical_path: Path,
    provenance: JsonObject,
) -> None:
    """Validate and install the complete paired 68-room DeLA result."""
    raw = load(raw_path)
    raw_records = records(raw)
    if len(raw_records) != 68 or len({str(record["room"]) for record in raw_records}) != 68:
        raise SystemExit("DeLA refresh must contain 68 unique rooms")
    for record in raw_records:
        backends = record.get("backends")
        if not isinstance(backends, dict) or set(backends) != {"cpu_kdtree", "flashknn"}:
            raise SystemExit(f"Incomplete DeLA backends for {record.get('room')}")
        for backend in backends.values():
            if not isinstance(backend, dict):
                raise SystemExit("Invalid DeLA backend payload")
            for field in ("preprocessing", "network", "end_to_end"):
                summary = backend.get(field)
                if not isinstance(summary, dict) or any(
                    not math.isfinite(float(value)) for value in summary.values()
                ):
                    raise SystemExit(f"Invalid DeLA {field} summary")
    raw_meta = metadata(raw)
    raw_meta["provenance"] = {**provenance, "room_manifest": room_manifest(raw)}
    save(raw_path, raw)
    save(canonical_path, copy.deepcopy(raw))


def main() -> None:
    """Finalize raw provenance and update the three canonical result files."""
    args = arguments()
    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    canonical_dir = args.canonical_dir.resolve()
    hashes = source_hashes(repo)
    provenance = common_provenance(
        run_dir, repo, args.source_commit, args.physical_gpu,
        args.data_root.resolve(), hashes,
    )
    merge_query(
        run_dir / "query/s3dis_sample_part.json",
        canonical_dir / "query/s3dis_sample_part.json",
        972, provenance, hashes,
    )
    merge_query(
        run_dir / "query/s3dis_full_k32.json",
        canonical_dir / "query/s3dis_full_k32.json",
        272, provenance, hashes,
    )
    install_network(
        run_dir / "network/dela_s3dis.json",
        canonical_dir / "network/dela_s3dis.json",
        provenance,
    )
    print("Final-kernel raw provenance and canonical L20 S3DIS merge complete")


if __name__ == "__main__":
    main()
