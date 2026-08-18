#!/usr/bin/env python3
"""Fail a run when it cannot reproduce every method in the paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


QUERY_METHODS = {
    "flashknn", "cuda_kdtree", "flann_cuda", "nanoflann", "faiss_flat", "faiss_ivf"
}
FULL_QUERY_METHODS = {"flashknn", "cuda_kdtree", "flann_cuda", "nanoflann"}
LIDAR_FIELDS = {
    "flashknn", "exact", "flann_cuda", "nanoflann", "faiss_flat", "faiss_ivf"
}
NETWORK_FILES = {
    "dela_s3dis.json", "ptv3_s3dis.json", "octformer_s3dis.json",
    "spunet_s3dis.json", "minkunet34c_s3dis.json",
    "dela_semantickitti_backends.json", "deepla_semantickitti_backends.json",
    "ptv3_semantickitti.json", "octformer_semantickitti.json",
    "spunet_semantickitti.json", "minkunet34c_semantickitti.json",
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_unified_environment(path: Path, payload: dict) -> tuple[str, str, int]:
    metadata = payload.get("metadata", {})
    torch_version = str(metadata.get("torch", ""))
    torch_cuda = str(metadata.get("torch_cuda", ""))
    if not torch_version.startswith("2.7.1+") or torch_cuda != "12.8":
        raise SystemExit(
            f"{path.name} was not produced by unified PyTorch 2.7.1+cu128: "
            f"torch={torch_version!r}, CUDA={torch_cuda!r}"
        )
    gpu = metadata.get("gpu", {})
    gpu_uuid = str(gpu.get("uuid", ""))
    if not gpu_uuid:
        raise SystemExit(f"{path.name} does not record a GPU UUID")
    name = str(gpu.get("name", ""))
    driver = str(gpu.get("driver", ""))
    memory_mib = int(gpu.get("memory_mib", 0))
    if not name or not driver or not memory_mib:
        raise SystemExit(f"{path.name} has incomplete GPU platform metadata")
    return name, driver, memory_mib


def expected_method_repeats(
    metadata: dict,
    record: dict,
    method: str,
    default_repeats: int,
) -> int:
    """Resolve a method's repeat count when a documented timing batch overrides it.

    Args:
        metadata: Top-level result metadata containing optional timing overrides.
        record: Query record whose mode and k identify the measured configuration.
        method: Backend name stored under the record's ``methods`` mapping.
        default_repeats: Repeat count declared by the base result file.

    Returns:
        The override repeat count for a matching batch, otherwise the base count.
    """
    matches = [
        override for override in metadata.get("timing_overrides", [])
        if method in override.get("methods", [])
        and record.get("mode") in override.get("modes", [])
        and int(record.get("k", -1)) in {int(k) for k in override.get("k", [])}
    ]
    if len(matches) > 1:
        raise SystemExit(
            f"Multiple timing overrides match {record.get('room')} "
            f"{record.get('mode')} k={record.get('k')} {method}"
        )
    return int(matches[0]["repeats"]) if matches else default_repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    query_dir, network_dir = args.run_dir / "query", args.run_dir / "network"
    gpu_platforms: set[tuple[str, str, int]] = set()
    system = load(args.run_dir / "system.json")
    system_torch = system.get("torch", {})
    if not str(system_torch.get("version", "")).startswith("2.7.1+") or \
            str(system_torch.get("cuda", "")) != "12.8":
        raise SystemExit(
            "system.json does not record the unified PyTorch 2.7.1+cu128 environment"
        )

    expected_s3dis = {
        "s3dis_sample_part.json": (
            {"pre", "post"}, {8, 16, 24, 32, 48, 64}, 81, QUERY_METHODS, False
        ),
        # The paper's point-count scaling experiment uses only the complete
        # 0.02 m voxelized cloud before downsampling.  Older result files may
        # contain partial post-query rows, which remain valid historical data
        # but are not required for this figure.
        "s3dis_full_k32.json": (
            {"pre"}, {32}, 272, FULL_QUERY_METHODS, True
        ),
    }
    expected_repeats = 1 if args.smoke else 10
    for filename, (modes, ks, room_count, methods, allow_extra_records) in expected_s3dis.items():
        path = query_dir / filename
        payload = load(path)
        gpu_platforms.add(require_unified_environment(path, payload))
        metadata = payload.get("metadata", {})
        for override in metadata.get("timing_overrides", []):
            gpu = override.get("gpu", {})
            platform = (
                str(gpu.get("name", "")),
                str(gpu.get("driver", "")),
                int(gpu.get("memory_mib", 0)),
            )
            if not all(platform):
                raise SystemExit(f"{filename} has incomplete timing override GPU metadata")
            gpu_platforms.add(platform)
        records = payload.get("records", [])
        if not records:
            raise SystemExit(f"No records in {filename}")
        required_records = [
            record for record in records
            if record.get("mode") in modes and int(record.get("k", -1)) in ks
        ]
        expected_records = len(modes) * len(ks) * (1 if args.smoke else room_count)
        if len(required_records) != expected_records:
            raise SystemExit(
                f"{filename} expected {expected_records} required records, "
                f"found {len(required_records)}"
            )
        if not allow_extra_records and len(records) != expected_records:
            raise SystemExit(f"{filename} expected {expected_records} records, found {len(records)}")
        for record in required_records:
            missing = methods - set(record.get("methods", {}))
            if missing:
                raise SystemExit(f"{filename} {record.get('room')} misses methods: {sorted(missing)}")
            for method in methods:
                timings = record["methods"][method].get("timings", [])
                method_repeats = expected_method_repeats(
                    metadata, record, method, expected_repeats
                )
                if len(timings) != method_repeats:
                    raise SystemExit(
                        f"{filename} {record.get('room')} {method} expected "
                        f"{method_repeats} timings, found {len(timings)}"
                    )
        if {record["mode"] for record in required_records} != modes:
            raise SystemExit(f"{filename} does not cover modes {sorted(modes)}")
        if {int(record["k"]) for record in required_records} != ks:
            raise SystemExit(f"{filename} does not cover k={sorted(ks)}")

    ball_path = query_dir / "ball_query_s3dis_sample_part.json"
    ball_payload = load(ball_path)
    gpu_platforms.add(require_unified_environment(ball_path, ball_payload))
    ball_records = ball_payload.get("records", [])
    expected_ball_records = 6 if args.smoke else 81 * 2 * 3
    if len(ball_records) != expected_ball_records:
        raise SystemExit(
            f"S3DIS ball query expected {expected_ball_records} records, "
            f"found {len(ball_records)}"
        )
    if {record["mode"] for record in ball_records} != {"pre", "post"}:
        raise SystemExit("S3DIS ball query does not cover pre/post")
    if {int(record["k"]) for record in ball_records} != {24, 32, 48}:
        raise SystemExit("S3DIS ball query does not cover k=24/32/48")
    for record in ball_records:
        if float(record.get("percentile", -1)) != 0.9:
            raise SystemExit("S3DIS ball query must use the global p90 radius")
        if len(record.get("query_timings_s", [])) != expected_repeats:
            raise SystemExit(
                f"S3DIS ball query {record.get('room')} expected "
                f"{expected_repeats} timings"
            )

    lidar_path = query_dir / "semantickitti.json"
    lidar_payload = load(lidar_path)
    gpu_platforms.add(require_unified_environment(lidar_path, lidar_payload))
    lidar = lidar_payload.get("samples", [])
    expected_lidar_records = 12 if args.smoke else 110 * 2 * 6
    if len(lidar) != expected_lidar_records:
        raise SystemExit(
            f"SemanticKITTI expected {expected_lidar_records} records, found {len(lidar)}"
        )
    if {record.get("mode") for record in lidar} != {"pre", "post"}:
        raise SystemExit("SemanticKITTI does not cover pre/post query modes")
    if {int(record.get("k", -1)) for record in lidar} != {8, 16, 24, 32, 48, 64}:
        raise SystemExit("SemanticKITTI does not cover k=8/16/24/32/48/64")
    lidar_keys = {
        (record.get("sample"), record.get("mode"), int(record.get("k", -1)))
        for record in lidar
    }
    if len(lidar_keys) != expected_lidar_records:
        raise SystemExit("SemanticKITTI contains duplicate sample/mode/k records")
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
        gpu_platforms.add(require_unified_environment(path, payload))
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
        payload = load(network_dir / f"{model}_semantickitti_backends.json")
        if int(payload.get("metadata", {}).get("alpha", -1)) != 4:
            raise SystemExit(f"{model} SemanticKITTI must use the paper-default alpha=4 operating point")
        samples = payload.get("samples", [])
        if not samples or any(
            set(sample.get("backends", {})) != {"cpu_kdtree", "flashknn"}
            for sample in samples
        ):
            raise SystemExit(
                f"{model} SemanticKITTI must contain paired CPU-KDTree and FlashKNN backends"
            )
    if len(gpu_platforms) != 1:
        raise SystemExit(f"A run mixes GPU platforms: {sorted(gpu_platforms)}")
    print("Coverage OK: paper query table, point-scaling curves, and network figure are complete")


if __name__ == "__main__":
    main()
