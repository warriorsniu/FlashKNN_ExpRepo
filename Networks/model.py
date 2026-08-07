"""Load an unmodified DeLA or DeepLA segmentation model for 19 classes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


LIDAR_VOXEL_SIZES = (0.06, 0.12, 0.24, 0.48)
MODEL_COORDINATE_SCALE = 1.6 / 0.06
# Calibrated on 40 deterministic augmented/cropped training scans. These make
# the auxiliary relative-coordinate target's mean squared magnitude match the
# original S3DIS setup at each stage. See estimate_cor_std.py and README.md.
LIDAR_COR_STD = (3.0, 4.8, 8.4, 15.2)


def _load_source(path: Path, module_name: str):
    root = path.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_args(kind: str, variant: str = "default") -> SimpleNamespace:
    args = SimpleNamespace()
    args.ks = [24, 24, 24, 24]
    if kind == "dela":
        args.depths = [4, 4, 8, 4]
    elif variant == "24":
        args.depths = [4, 4, 12, 4]
    elif variant == "60":
        args.depths = [10, 10, 30, 10]
    else:
        args.depths = [20, 20, 60, 20]
    args.dims = [64, 128, 256, 512]
    args.nbr_dims = [32, 32]
    args.head_dim = 256
    args.num_classes = 19
    rates = torch.linspace(0.0, 0.1, sum(args.depths)).split(args.depths)
    args.drop_paths = [rate.tolist() for rate in rates]
    args.head_drops = torch.linspace(0.0, 0.15, len(args.depths)).tolist()
    args.bn_momentum = 0.02
    args.act = nn.GELU
    args.mlp_ratio = 2 if kind == "dela" else 1
    args.use_cp = False
    # Both models store 1/cor_std and multiply relative coordinates by it for
    # the auxiliary coordinate-regression target. LiDAR's actual neighbours
    # span multiple nominal voxels, so empirical target-scale matching is more
    # appropriate than voxel_size * coordinate_scale here.
    args.cor_std = list(LIDAR_COR_STD)
    return args


def create_model(kind: str, repo: Path, variant: str = "default") -> nn.Module:
    if kind == "dela":
        module = _load_source(repo / "S3DIS" / "delasemseg.py", "lidar_dela_semseg")
        return module.DelaSemSeg(model_args(kind, variant))
    if kind == "deepla":
        module = _load_source(repo / "S3DIS" / "deepla_semseg.py", "lidar_deepla_semseg")
        return module.DeepLA_semseg(model_args(kind, variant))
    raise ValueError(kind)
