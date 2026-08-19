#!/usr/bin/env python3
"""DeLA S3DIS latency with random weights and CPU/FlashKNN preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--max-samples", type=int)
    return p.parse_args()


def summarize(values):
    values = [float(v) for v in values]
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
        "p95_ms": ordered[round(.95 * (len(ordered) - 1))],
        "std_ms": statistics.pstdev(values),
    }


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_info(torch, physical):
    result = {"name": torch.cuda.get_device_name(0)}
    try:
        line = subprocess.check_output([
            "nvidia-smi", "-i", str(physical),
            "--query-gpu=uuid,driver_version,memory.total", "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()[0]
        uuid, driver, memory = (x.strip() for x in line.split(","))
        result.update(uuid=uuid, driver=driver, memory_mib=int(memory))
    except Exception:
        pass
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(repo):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo, text=True,
    ).strip())
    return {"commit": commit, "dirty": dirty}


def co_tenant_snapshot():
    snapshot = {}
    for name, command in {
        "gpus": [
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.used,"
            "utilization.gpu,pstate", "--format=csv,noheader,nounits",
        ],
        "compute_processes": [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,"
            "used_memory", "--format=csv,noheader,nounits",
        ],
    }.items():
        try:
            snapshot[name] = subprocess.check_output(
                command, text=True,
            ).splitlines()
        except (OSError, subprocess.CalledProcessError) as error:
            snapshot[name] = [f"unavailable: {error}"]
    try:
        processes = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,args="], text=True,
        ).splitlines()
        snapshot["training_processes"] = [
            line.strip() for line in processes if "tools/train.py" in line
        ]
    except (OSError, subprocess.CalledProcessError) as error:
        snapshot["training_processes"] = [f"unavailable: {error}"]
    return snapshot


def load_room(torch, path):
    if path.is_dir():
        xyz = torch.as_tensor(np.load(path / "coord.npy"), dtype=torch.float32)
        xyz -= xyz.amin(0, keepdim=True)
        color = torch.as_tensor(np.load(path / "color.npy"), dtype=torch.float32)
        return xyz.contiguous(), color.contiguous()
    try:
        room = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        room = torch.load(path, map_location="cpu")
    xyz = torch.as_tensor(np.asarray(room["coord"]), dtype=torch.float32)
    xyz -= xyz.amin(0, keepdim=True)
    color = torch.as_tensor(np.asarray(room["color"]), dtype=torch.float32)
    return xyz.contiguous(), color.contiguous()


def cpu_hierarchy(torch, xyz, grid_sizes, ks, grid_subsampling, KDTree):
    indices, levels, full_xyz = [], [], xyz

    def recurse(current, depth):
        tree = KDTree(current)
        indices.append(tree.knn(current, ks[depth], False)[0])
        levels.append(current)
        if depth + 1 < len(ks):
            local = grid_subsampling(current, grid_sizes[depth + 1])
            indices.append(local)
            recurse(current[local], depth + 1)
            indices.append(KDTree(current[local]).knn(full_xyz, 1, False)[0].squeeze(-1))

    recurse(xyz, 0)
    return indices


def main():
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    here = Path(__file__).resolve()
    repo = here.parents[2]
    sys.path.insert(0, str(repo / "Networks"))
    sys.path.insert(0, str(repo / "DeLA"))
    sys.path.insert(0, str(here.parent))
    import torch
    from torch.cuda.amp import autocast
    from config import dela_args, s3dis_args
    from delasemseg import DelaSemSeg
    from utils.cutils import grid_subsampling, grid_subsampling_test, KDTree
    from hierarchy import _voxel_first_cuda, build_flash_hierarchy
    try:
        from FlashKNN import FlashKNN
        import FlashKNN.CuFun as flash_cuda
    except ImportError:
        from functions import FlashKNN
        from functions import CuFun as flash_cuda

    np.random.seed(47)
    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    model = DelaSemSeg(dela_args).cuda().eval()
    area = args.data_root / "Area_5"
    if not area.is_dir():
        area = args.data_root / "area_5"
    paths = sorted(area.glob("*.pth"))
    if not paths and area.is_dir():
        paths = sorted(
            room for room in area.iterdir()
            if room.is_dir() and (room / "coord.npy").is_file()
        )
    if args.max_samples is not None:
        paths = paths[:args.max_samples]
    if not paths:
        raise SystemExit(f"No Pointcept PTH/per-field-NPY S3DIS rooms found below {args.data_root}")
    source_files = (
        "DeLA/S3DIS/benchmark_latency.py",
        "Networks/hierarchy.py",
        "FlashKNN/csrc/api.cpp",
        "FlashKNN/csrc/flash_knn_query.h",
        "FlashKNN/csrc/flash_knn_query_dynamic_load.cu",
        "FlashKNN/functions/FlashKnnWrapper.py",
    )
    metadata = {
        "dataset": "S3DIS", "split": "Area_5", "model": "DeLA",
        "weights": "random initialization",
        "flashknn_alpha": 4,
        "voxel_sizes_m": s3dis_args.grid_size, "k": s3dis_args.k,
        "timing_boundary": "preprocessing plus network; excludes disk I/O and H2D",
        "warmups": args.warmups, "repeats": args.repeats,
        "gpu": gpu_info(torch, args.gpu), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "python": platform.python_version(),
        "physical_gpu_index": args.gpu,
        "git": git_identity(repo),
        "source_sha256": {
            relative: sha256(repo / relative) for relative in source_files
        },
        "extension": {
            "path": str(Path(flash_cuda.__file__).resolve()),
            "sha256": sha256(Path(flash_cuda.__file__).resolve()),
        },
        "flashknn_configuration": {
            "memory_mode": "SM", "sorting_mode": "PS",
            "candidate_mode": "register", "enable_skip": True,
            "thread_group_size": "adaptive", "alpha": 4,
            "sorting_revision": "generated_bitonic_top_p",
        },
        "co_tenant_start": co_tenant_snapshot(),
    }
    records = []
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        old = previous.get("metadata", {})
        fields = (
            "dataset", "split", "model", "warmups", "repeats", "torch",
            "torch_cuda", "source_sha256", "extension",
            "flashknn_configuration",
        )
        changed = {field: (old.get(field), metadata.get(field))
                   for field in fields if old.get(field) != metadata.get(field)}
        if old.get("gpu", {}).get("uuid") != metadata["gpu"].get("uuid"):
            changed["gpu.uuid"] = (old.get("gpu", {}).get("uuid"), metadata["gpu"].get("uuid"))
        if changed:
            raise SystemExit(f"Refusing to resume incompatible output {args.output}: {changed}")
        records = previous.get("records", [])
        metadata["co_tenant_resume"] = co_tenant_snapshot()
    completed = {
        record["room"] for record in records
        if set(record.get("backends", {})) == {"cpu_kdtree", "flashknn"}
    }
    payload = {"metadata": metadata, "records": records}
    with torch.inference_mode():
        for room_path in paths:
            relative_room = room_path.relative_to(args.data_root).as_posix()
            if relative_room in completed:
                print(f"skip completed {relative_room}", flush=True)
                continue
            full_cpu, color_cpu = load_room(torch, room_path)
            room_record = {"room": relative_room,
                           "num_full": len(full_cpu), "backends": {}}
            for backend in ("cpu_kdtree", "flashknn"):
                prep_ms, forward_ms, total_ms, down_counts = [], [], [], []
                for iteration in range(args.warmups + args.repeats):
                    if backend == "cpu_kdtree":
                        start = time.perf_counter()
                        selected = grid_subsampling_test(full_cpu, s3dis_args.grid_size[0], 2.5 / 14)
                        xyz_cpu = full_cpu[selected].clone()
                        feature_cpu = torch.cat((color_cpu[selected] / 250.0, xyz_cpu[:, 2:]), 1)
                        full_nn_cpu = KDTree(xyz_cpu).knn(full_cpu, 1)[0].squeeze(-1)
                        hierarchy_cpu = cpu_hierarchy(
                            torch, xyz_cpu, s3dis_args.grid_size, s3dis_args.k,
                            grid_subsampling, KDTree,
                        )
                        this_prep = (time.perf_counter() - start) * 1000
                        xyz = (xyz_cpu * 40).cuda(non_blocking=True)
                        feature = feature_cpu.cuda(non_blocking=True)
                        hierarchy = [x.cuda(non_blocking=True).long() for x in hierarchy_cpu[::-1]]
                        full_nn = full_nn_cpu.cuda(non_blocking=True).long()
                        torch.cuda.synchronize()
                    else:
                        full = full_cpu.cuda(non_blocking=True)
                        color = color_cpu.cuda(non_blocking=True)
                        torch.cuda.synchronize()
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        selected = _voxel_first_cuda(full, s3dis_args.grid_size[0])
                        metric_xyz = full[selected].contiguous()
                        full_nn = FlashKNN(num_nbr=1).back_query(
                            full, selected, query_grid_size=.08, down_grid_size=.04,
                            batch_idx=None,
                        ).long()
                        raw_hierarchy = build_flash_hierarchy(
                            metric_xyz, tuple(s3dis_args.grid_size), tuple(s3dis_args.k), alpha=4,
                        )
                        hierarchy = [x.long() for x in raw_hierarchy[::-1]]
                        xyz = metric_xyz * 40
                        feature = torch.cat((color[selected] / 250.0, metric_xyz[:, 2:]), 1)
                        end_event.record(); end_event.synchronize()
                        this_prep = start_event.elapsed_time(end_event)
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with autocast():
                        prediction = model(xyz, feature, hierarchy)
                        _ = prediction[full_nn]
                    end_event.record(); end_event.synchronize()
                    this_forward = start_event.elapsed_time(end_event)
                    if iteration >= args.warmups:
                        prep_ms.append(this_prep); forward_ms.append(this_forward)
                        total_ms.append(this_prep + this_forward); down_counts.append(len(xyz))
                room_record["backends"][backend] = {
                    "num_down": round(statistics.mean(down_counts)),
                    "preprocessing": summarize(prep_ms), "network": summarize(forward_ms),
                    "end_to_end": summarize(total_ms),
                }
            records.append(room_record)
            completed.add(relative_room)
            save(args.output, payload)
            print(room_record["room"], {
                k: round(v["end_to_end"]["mean_ms"], 3)
                for k, v in room_record["backends"].items()
            }, flush=True)
    payload["metadata"]["co_tenant_end"] = co_tenant_snapshot()
    save(args.output, payload)


if __name__ == "__main__":
    main()
