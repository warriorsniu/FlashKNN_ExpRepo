#!/usr/bin/env python3
"""Construct every paper network and consume one room from the prepared data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CONFIGS = {
    "ptv3": "configs/s3dis/semseg-pt-v3m1-0-rpe.py",
    "octformer": "configs/s3dis/semseg-octformer-v1m1-0-base.py",
    "spunet": "configs/s3dis/semseg-spunet-v1m1-0-base.py",
    "minkunet34c": "configs/s3dis/semseg-minkunet34c-0-base.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3dis", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "Pointcept"))
    sys.path.insert(0, str(repo / "DeLA"))

    import torch
    from DeLA.S3DIS.benchmark_latency import load_room
    from Query.benchmark_s3dis import room_paths
    from pointcept.datasets import build_dataset
    from pointcept.models import build_model
    from pointcept.utils.config import Config

    data_root = args.s3dis.resolve()
    rooms = room_paths(data_root)
    xyz, color = load_room(torch, rooms[0])
    if xyz.shape != color.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise SystemExit(f"DeLA loader returned invalid shapes: {xyz.shape}, {color.shape}")
    print(f"query/DeLA data loader: {len(rooms)} rooms, first={rooms[0].name}")

    pointcept_root = repo / "Pointcept"
    for name, relative_config in CONFIGS.items():
        cfg = Config.fromfile(pointcept_root / relative_config)
        cfg.data.test.data_root = str(data_root)
        cfg.data.test.test_cfg.aug_transform = [[]]
        cfg.data.test.test_cfg.voxelize.grid_size = 0.04
        dataset = build_dataset(cfg.data.test)
        if len(dataset) != 68:
            raise SystemExit(f"{name} expected 68 Area 5 rooms, found {len(dataset)}")
        sample = dataset[0]
        fragment = sample["fragment_list"][0]
        model = build_model(cfg.model).eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"{name}: rooms={len(dataset)}, full={len(sample['segment'])}, "
            f"down={len(fragment['coord'])}, parameters={parameters}"
        )
        del model, dataset, sample, fragment


if __name__ == "__main__":
    main()
