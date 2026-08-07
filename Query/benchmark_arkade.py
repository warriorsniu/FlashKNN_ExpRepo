#!/usr/bin/env python3
"""Benchmark Arkade/TrueKNN RT-core queries on matched S3DIS point sets."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def arguments() -> argparse.Namespace:
    """Parse paths and the matched S3DIS timing protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", nargs="+", choices=("pre", "post"), default=("pre", "post"))
    parser.add_argument("--k", nargs="+", type=int, default=(24, 32, 48))
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--crop-points", type=int, default=250_000)
    parser.add_argument("--initial-radius", type=float, default=0.02)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist each completed Arkade record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_info(physical_gpu: int) -> dict[str, Any]:
    """Read GPU identity for cross-architecture provenance."""
    fields = "name,uuid,driver_version,memory.total"
    line = subprocess.check_output(
        ["nvidia-smi", "-i", str(physical_gpu), f"--query-gpu={fields}",
         "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()[0]
    name, uuid, driver, memory = [part.strip() for part in line.split(",")]
    return {"name": name, "uuid": uuid, "driver": driver, "memory_mib": int(memory)}


def exact_knn(torch: Any, cukd: Any, support: Any, query: Any, k: int) -> Any:
    """Return exact distance-sorted indices while preserving support ordering."""
    indices = torch.empty((len(query), k), device="cuda", dtype=torch.int32)
    distances = torch.empty((len(query), k), device="cuda", dtype=torch.float32)
    cukd.CukdKnnQueryTorch(
        support.clone(), query, k, indices, distances, torch.zeros(2), True
    )
    return indices.gather(1, distances.argsort(dim=1))


def set_recall(torch: Any, exact: Any, predicted: Any, chunk_size: int = 16_384) -> dict[str, float]:
    """Compute order-independent kNN recall for unsorted Arkade output."""
    values = []
    for start in range(0, len(exact), chunk_size):
        reference = exact[start:start + chunk_size].long()
        candidate = predicted[start:start + chunk_size].long()
        matches = (reference[:, :, None] == candidate[:, None, :]) & (candidate[:, None, :] >= 0)
        values.append(matches.any(dim=2).float().mean(dim=1).cpu())
    per_query = torch.cat(values)
    return {"mean": float(per_query.mean()), "minimum": float(per_query.min())}


def executable(repo: Path, k: int) -> Path:
    """Resolve the separately compiled Arkade executable for one static k."""
    path = repo / "third_party" / "Arkade" / f"build-k{k}-l2" / "arkade-benchmark"
    if not path.is_file():
        raise FileNotFoundError(f"missing Arkade k={k} executable: {path}")
    return path


def parse_result(output: str) -> dict[str, Any]:
    """Extract the benchmark JSON object despite possible OWL log messages."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"Arkade did not emit a JSON result:\n{output[-2000:]}")


def run_arkade(command: list[str], output_path: Path, timeout: float,
               retries: int) -> tuple[subprocess.CompletedProcess[str], int]:
    """Run Arkade with bounded retries for intermittent OptiX process faults."""
    last_error: subprocess.SubprocessError | None = None
    for attempt in range(retries + 1):
        output_path.unlink(missing_ok=True)
        try:
            return (
                subprocess.run(
                    command, check=True, capture_output=True, text=True,
                    timeout=timeout,
                ),
                attempt + 1,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            last_error = error
            print(
                f"Arkade attempt {attempt + 1}/{retries + 1} failed: {error}",
                flush=True,
            )
    assert last_error is not None
    raise last_error


def main() -> None:
    """Export matched point sets, launch Arkade, and validate exact recall."""
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo = args.repo.resolve()

    import torch
    import Cukd.CuFun as cukd
    from FlashKNN import xyz2key
    from benchmark_s3dis import load_xyz, prepare, room_paths

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    crop_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    new_payload: dict[str, Any] = {
        "metadata": {
            "dataset": "S3DIS", "operator": "Arkade/TrueKNN",
            "distance": "squared L2 (NORM=2)", "gpu": gpu_info(args.gpu),
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "optix": "8.1.0",
            "voxel_size_m": args.voxel_size, "crop_points": args.crop_points,
            "initial_radius_m": args.initial_radius, "warmups": args.warmups,
            "repeats": args.repeats, "seed": args.seed,
            "timing_boundary": "BVH build and synchronous TrueKNN query/refit measured inside Arkade; excludes file I/O, OptiX context/pipeline creation, voxelization, cropping and H2D",
        },
        "records": [],
    }
    if args.resume and args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = new_payload
        atomic_json(args.output, payload)
    completed = {
        (record["room"], record["mode"], int(record["k"]))
        for record in payload["records"]
    }
    completed_rooms = 0
    for path in room_paths(args.data_root.resolve()):
        coord = load_xyz(torch, path)
        relative = path.relative_to(args.data_root.resolve()).as_posix()
        crop_center = None
        room_records = []
        for mode in args.mode:
            sample = prepare(
                torch, xyz2key, coord, path, mode, "sample_part", args.voxel_size,
                args.crop_points, crop_center, crop_generator,
            )
            if sample is None:
                continue
            support, _, query_indices, crop_center = sample
            query = support if mode == "pre" else support[query_indices].contiguous()
            max_k = max(args.k)
            exact = exact_knn(torch, cukd, support, query, max_k)
            with tempfile.TemporaryDirectory(prefix="arkade-s3dis-") as temporary:
                temporary_path = Path(temporary)
                input_path = temporary_path / "points.bin"
                output_path = temporary_path / "indices.bin"
                points = torch.cat((support, query), dim=0).cpu().numpy().astype(
                    np.float32, copy=False
                )
                points.tofile(input_path)
                for k in args.k:
                    record_key = (relative, mode, k)
                    if record_key in completed:
                        continue
                    command = [
                        os.fspath(executable(repo, k)), os.fspath(input_path),
                        str(len(support)), str(len(query)), str(args.initial_radius),
                        str(args.warmups), str(args.repeats), os.fspath(output_path),
                    ]
                    process, subprocess_attempts = run_arkade(
                        command, output_path, args.timeout, args.retries,
                    )
                    result = parse_result(process.stdout)
                    predicted_cpu = np.fromfile(output_path, dtype=np.int32)
                    expected_size = len(query) * k
                    if predicted_cpu.size != expected_size:
                        raise RuntimeError(
                            f"Arkade wrote {predicted_cpu.size} indices; expected {expected_size}"
                        )
                    predicted = torch.from_numpy(
                        predicted_cpu.reshape(len(query), k)
                    ).to(device="cuda")
                    record = {
                        "room": relative, "mode": mode, "scope": "sample_part",
                        "k": k, "num_support": len(support), "num_query": len(query),
                        "construction_s": float(result["build_s"]),
                        "query_timings_s": [float(value) for value in result["query_s"]],
                        "rounds": [int(value) for value in result["rounds"]],
                        "subprocess_attempts": subprocess_attempts,
                        "recall_vs_cukd": set_recall(torch, exact[:, :k], predicted),
                    }
                    room_records.append(record)
                    payload["records"].append(record)
                    completed.add(record_key)
                    atomic_json(args.output, payload)
                    print(
                        f"{relative} {mode} k={k} "
                        f"query={np.median(record['query_timings_s']) * 1000:.3f}ms "
                        f"rounds={int(np.median(record['rounds']))} "
                        f"recall={record['recall_vs_cukd']['mean']:.6f}",
                        flush=True,
                    )
            del support, query_indices, query, exact
        if room_records:
            completed_rooms += 1
            if args.max_samples is not None and completed_rooms >= args.max_samples:
                break
        del coord


if __name__ == "__main__":
    main()
