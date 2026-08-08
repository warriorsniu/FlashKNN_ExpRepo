#!/usr/bin/env python3
"""Random-weight S3DIS latency benchmark for four Pointcept backbones."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path


CONFIGS = {
    "ptv3": "configs/s3dis/semseg-pt-v3m1-0-rpe.py",
    "octformer": "configs/s3dis/semseg-octformer-v1m1-0-base.py",
    "spunet": "configs/s3dis/semseg-spunet-v1m1-0-base.py",
    "minkunet34c": "configs/s3dis/semseg-minkunet34c-0-base.py",
}


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=CONFIGS, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--grid-size", type=float, default=.04)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--max-samples", type=int)
    return p.parse_args()


def summary(values):
    ordered = sorted(values)
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "p95_ms": ordered[round(.95 * (len(ordered) - 1))],
            "std_ms": statistics.pstdev(values)}


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from pointcept.datasets import build_dataset
    from pointcept.models import build_model
    from pointcept.utils.config import Config

    root = Path(__file__).resolve().parent
    cfg = Config.fromfile(root / CONFIGS[args.model])
    cfg.data.test.data_root = str(args.data_root.resolve())
    cfg.data.test.test_cfg.aug_transform = [[]]
    cfg.data.test.test_cfg.voxelize.grid_size = args.grid_size
    torch.manual_seed(47); torch.cuda.manual_seed_all(47)
    model = build_model(cfg.model).cuda().eval()
    dataset = build_dataset(cfg.data.test)
    gpu = {"name": torch.cuda.get_device_name(0)}
    try:
        line = subprocess.check_output([
            "nvidia-smi", "-i", str(args.gpu),
            "--query-gpu=uuid,driver_version,memory.total", "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()[0]
        uuid, driver, memory = (x.strip() for x in line.split(","))
        gpu.update(uuid=uuid, driver=driver, memory_mib=int(memory))
    except Exception:
        pass
    metadata = {
        "dataset": "S3DIS", "split": "Area_5", "model": args.model,
        "weights": "random initialization",
        "config": CONFIGS[args.model], "grid_size_m": args.grid_size,
        "timing_boundary": "CUDA-ready voxelized fragment; network forward only",
        "warmups": args.warmups, "repeats": args.repeats, "gpu": gpu,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    records = []
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        old = previous.get("metadata", {})
        fields = ("dataset", "split", "model", "config", "grid_size_m", "warmups",
                  "repeats", "torch", "torch_cuda")
        changed = {field: (old.get(field), metadata.get(field))
                   for field in fields if old.get(field) != metadata.get(field)}
        if old.get("gpu", {}).get("uuid") != metadata["gpu"].get("uuid"):
            changed["gpu.uuid"] = (old.get("gpu", {}).get("uuid"), metadata["gpu"].get("uuid"))
        if changed:
            raise SystemExit(f"Refusing to resume incompatible output {args.output}: {changed}")
        records = previous.get("records", [])
    completed = {record["room"] for record in records if "network" in record}
    payload = {"metadata": metadata, "records": records}
    with torch.inference_mode():
        sample_count = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
        for index in range(sample_count):
            sample = dataset[index]
            if sample["name"] in completed:
                print(f"skip completed {sample['name']}", flush=True)
                continue
            full_points = int(sample["segment"].shape[0])
            fragment = sample["fragment_list"][0]
            down_points = int(fragment["coord"].shape[0])
            for key, value in fragment.items():
                if isinstance(value, torch.Tensor):
                    fragment[key] = value.cuda(non_blocking=True)
            torch.cuda.synchronize()
            timings = []
            for iteration in range(args.warmups + args.repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(); model(fragment); end.record(); end.synchronize()
                if iteration >= args.warmups:
                    timings.append(float(start.elapsed_time(end)))
            records.append({"room": sample["name"], "num_full": full_points,
                            "num_down": down_points, "network": summary(timings)})
            completed.add(sample["name"])
            print(f"{sample['name']} N={down_points} {statistics.mean(timings):.3f} ms", flush=True)
            atomic_json(args.output, payload)
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
