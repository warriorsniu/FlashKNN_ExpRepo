#!/usr/bin/env python3
"""Fail a run when it cannot reproduce every method in the paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


QUERY_METHODS = {
    "flashknn", "cuda_kdtree", "flann_cuda", "nanoflann", "faiss_flat", "faiss_ivf"
}
LIDAR_FIELDS = {
    "flashknn", "exact", "flann_cuda", "nanoflann", "faiss_flat", "faiss_ivf"
}
NETWORK_FILES = {
    "dela_s3dis.json", "ptv3_s3dis.json", "octformer_s3dis.json",
    "spunet_s3dis.json", "minkunet34c_s3dis.json",
    "dela_semantickitti.json", "deepla_semantickitti.json",
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_unified_environment(path: Path, payload: dict) -> str:
    metadata = payload.get("metadata", {})
    torch_version = str(metadata.get("torch", ""))
    torch_cuda = str(metadata.get("torch_cuda", ""))
    if not torch_version.startswith("2.7.1+") or torch_cuda != "12.8":
        raise SystemExit(
            f"{path.name} was not produced by unified PyTorch 2.7.1+cu128: "
            f"torch={torch_version!r}, CUDA={torch_cuda!r}"
        )
    gpu_uuid = str(metadata.get("gpu", {}).get("uuid", ""))
    if not gpu_uuid:
        raise SystemExit(f"{path.name} does not record a GPU UUID")
    return gpu_uuid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    query_dir, network_dir = args.run_dir / "query", args.run_dir / "network"
    gpu_uuids: set[str] = set()
    system = load(args.run_dir / "system.json")
    system_torch = system.get("torch", {})
    if not str(system_torch.get("version", "")).startswith("2.7.1+") or \
            str(system_torch.get("cuda", "")) != "12.8":
        raise SystemExit(
            "system.json does not record the unified PyTorch 2.7.1+cu128 environment"
        )

    expected_s3dis = {
        "s3dis_sample_part.json": ({"pre", "post"}, {8, 16, 24, 32, 48, 64}, 81),
        "s3dis_full_k32.json": ({"pre", "post"}, {32}, 272),
    }
    expected_repeats = 1 if args.smoke else 10
    for filename, (modes, ks, room_count) in expected_s3dis.items():
        path = query_dir / filename
        payload = load(path)
        gpu_uuids.add(require_unified_environment(path, payload))
        records = payload.get("records", [])
        if not records:
            raise SystemExit(f"No records in {filename}")
        expected_records = len(modes) * len(ks) * (1 if args.smoke else room_count)
        if len(records) != expected_records:
            raise SystemExit(f"{filename} expected {expected_records} records, found {len(records)}")
        for record in records:
            missing = QUERY_METHODS - set(record.get("methods", {}))
            if missing:
                raise SystemExit(f"{filename} {record.get('room')} misses methods: {sorted(missing)}")
            for method in QUERY_METHODS:
                timings = record["methods"][method].get("timings", [])
                if len(timings) != expected_repeats:
                    raise SystemExit(
                        f"{filename} {record.get('room')} {method} expected "
                        f"{expected_repeats} timings, found {len(timings)}"
                    )
        if {record["mode"] for record in records} != modes:
            raise SystemExit(f"{filename} does not cover modes {sorted(modes)}")
        if {int(record["k"]) for record in records} != ks:
            raise SystemExit(f"{filename} does not cover k={sorted(ks)}")

    lidar_path = query_dir / "semantickitti.json"
    lidar_payload = load(lidar_path)
    gpu_uuids.add(require_unified_environment(lidar_path, lidar_payload))
    lidar = lidar_payload.get("samples", [])
    expected_lidar_records = 6 if args.smoke else 110 * 2 * 3
    if len(lidar) != expected_lidar_records:
        raise SystemExit(
            f"SemanticKITTI expected {expected_lidar_records} records, found {len(lidar)}"
        )
    for record in lidar:
        missing = LIDAR_FIELDS - set(record)
        if missing:
            raise SystemExit(f"SemanticKITTI {record.get('sample')} misses: {sorted(missing)}")
        if {int(item["alpha"]) for item in record["flashknn"]} != {4, 8, 16, 32}:
            raise SystemExit(f"SemanticKITTI {record.get('sample')} misses FlashKNN alpha sweep")
        for value in [record["exact"], *record["flashknn"],
                      *(record[name] for name in LIDAR_FIELDS - {"exact", "flashknn"})]:
            if len(value.get("timings", [])) != expected_repeats:
                raise SystemExit(
                    f"SemanticKITTI {record.get('sample')} has incomplete timing repetitions"
                )

    found_network = {path.name for path in network_dir.glob("*.json")}
    missing_files = NETWORK_FILES - found_network
    if missing_files:
        raise SystemExit(f"Network figure misses result files: {sorted(missing_files)}")
    expected_s3dis_samples = 1 if args.smoke else 68
    expected_lidar_samples = 1 if args.smoke else 22
    for filename in sorted(NETWORK_FILES):
        path = network_dir / filename
        payload = load(path)
        gpu_uuids.add(require_unified_environment(path, payload))
        records = payload.get("records", payload.get("samples", []))
        expected = expected_lidar_samples if "semantickitti" in filename else expected_s3dis_samples
        if len(records) != expected:
            raise SystemExit(f"{filename} expected {expected} samples, found {len(records)}")
    dela_payload = load(network_dir / "dela_s3dis.json")
    if int(dela_payload.get("metadata", {}).get("flashknn_alpha", -1)) != 4:
        raise SystemExit("DeLA S3DIS must use the paper default FlashKNN alpha=4")
    dela = dela_payload.get("records", [])
    if not dela or any(set(record.get("backends", {})) != {"cpu_kdtree", "flashknn"}
                       for record in dela):
        raise SystemExit("DeLA result must contain both CPU nanoflann and FlashKNN backends")
    for model in ("dela", "deepla"):
        payload = load(network_dir / f"{model}_semantickitti.json")
        if int(payload.get("metadata", {}).get("alpha", -1)) != 8:
            raise SystemExit(f"{model} SemanticKITTI must use the selected alpha=8 operating point")
    if len(gpu_uuids) != 1:
        raise SystemExit(f"A run mixes physical GPUs: {sorted(gpu_uuids)}")
    print("Coverage OK: paper query table, point-scaling curves, and network figure are complete")


if __name__ == "__main__":
    main()
