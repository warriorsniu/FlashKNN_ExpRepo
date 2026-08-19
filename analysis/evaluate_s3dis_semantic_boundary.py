#!/usr/bin/env python3
"""Evaluate DeLA accuracy on semantic-boundary and non-boundary S3DIS points.

This script is intended to run from a DeLA/S3DIS checkout containing the
historical FlashKNN integration.  It deliberately uses the exact-kNN
dataloader to define semantic-boundary points, so the evaluation partition is
independent of the method being compared.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import dela_args, s3dis_args
from delasemseg import DelaSemSeg
from s3dis import S3DIS, default_collate_fn, s3dis_collate_fn

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.util as util


CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]
SUBSETS = ("all", "semantic_boundary", "non_boundary")
METHODS = ("FlashKNN", "ExactKNN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flash-checkpoint",
        default="output/model/DeLA_Seg_FlashKNN_StaticLoad_0509_6/best.pt",
    )
    parser.add_argument(
        "--exact-checkpoint", default="output/model/pretrained/best.pt"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partition", default="5")
    parser.add_argument("--same-class-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-rooms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_confusions(num_classes: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        method: {
            subset: np.zeros((num_classes, num_classes), dtype=np.int64)
            for subset in SUBSETS
        }
        for method in METHODS
    }


def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    encoded = target[mask].to(torch.int64) * num_classes + pred[mask].to(torch.int64)
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).cpu().numpy().astype(np.int64)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    confusion = np.asarray(confusion, dtype=np.int64)
    intersection = np.diag(confusion).astype(np.float64)
    target_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = target_count + pred_count - intersection
    total = int(confusion.sum())

    accuracy = float(intersection.sum() / total) if total else None
    valid_acc = target_count > 0
    class_accuracy = np.divide(
        intersection,
        target_count,
        out=np.full_like(intersection, np.nan),
        where=valid_acc,
    )
    valid_iou = union > 0
    class_iou = np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan),
        where=valid_iou,
    )

    def optional_float(value: float) -> float | None:
        return None if np.isnan(value) else float(value)

    return {
        "count": total,
        "accuracy": accuracy,
        "mean_class_accuracy": (
            float(np.nanmean(class_accuracy)) if valid_acc.any() else None
        ),
        "miou": float(np.nanmean(class_iou)) if valid_iou.any() else None,
        "class_iou": {
            name: optional_float(value)
            for name, value in zip(CLASS_NAMES, class_iou.tolist())
        },
        "class_target_count": {
            name: int(value) for name, value in zip(CLASS_NAMES, target_count.tolist())
        },
        "confusion": confusion.tolist(),
    }


def subset_masks(boundary: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "all": torch.ones_like(boundary, dtype=torch.bool),
        "semantic_boundary": boundary,
        "non_boundary": ~boundary,
    }


def scalar_or_blank(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def write_room_csv(path: Path, room_rows: list[dict[str, Any]]) -> None:
    fields = [
        "room",
        "points",
        "boundary_points",
        "boundary_fraction",
        "subset",
        "method",
        "count",
        "accuracy",
        "mean_class_accuracy",
        "miou",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in room_rows:
            writer.writerow({field: row[field] for field in fields})


def write_summary_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# S3DIS semantic-boundary evaluation",
        "",
        (
            "A point is classified as a semantic-boundary point when fewer than "
            f"{result['protocol']['same_class_threshold']:.0%} of its exact "
            f"{result['protocol']['k']}-NN support points have the center point's "
            "ground-truth class. The support set includes the query point itself."
        ),
        "",
        f"Evaluated rooms: {result['protocol']['rooms']}",
        f"Evaluated points: {result['point_partition']['total_points']}",
        (
            "Semantic-boundary points: "
            f"{result['point_partition']['semantic_boundary_points']} "
            f"({result['point_partition']['semantic_boundary_fraction']:.4%})"
        ),
        "",
        "| Subset | Method | Points | Accuracy | mAcc | mIoU |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for subset in SUBSETS:
        for method in METHODS:
            metric = result["aggregate"][method][subset]
            lines.append(
                f"| {subset} | {method} | {metric['count']} | "
                f"{metric['accuracy']:.4f} | "
                f"{metric['mean_class_accuracy']:.4f} | {metric['miou']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Differences below are FlashKNN minus ExactKNN in percentage points.",
            "",
            "| Subset | Accuracy difference | mIoU difference |",
            "| --- | ---: | ---: |",
        ]
    )
    for subset in SUBSETS:
        diff = result["difference_percentage_points"][subset]
        lines.append(
            f"| {subset} | {diff['accuracy']:+.3f} | {diff['miou']:+.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subsets = ("semantic_boundary", "non_boundary")
    labels = ("Semantic boundary", "Non-boundary")
    x = np.arange(len(subsets), dtype=np.float64)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    colors = ("#2474B5", "#E66B2E")
    for axis, metric_name, title in zip(
        axes, ("accuracy", "miou"), ("Point accuracy", "Subset mIoU")
    ):
        for offset, method, color in zip((-0.5, 0.5), METHODS, colors):
            values = [
                result["aggregate"][method][subset][metric_name]
                for subset in subsets
            ]
            bars = axis.bar(x + offset * width, values, width, label=method, color=color)
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        axis.set_xticks(x, labels)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(title)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.same_class_threshold <= 1.0:
        raise ValueError("--same-class-threshold must be in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flash_checkpoint = Path(args.flash_checkpoint).resolve()
    exact_checkpoint = Path(args.exact_checkpoint).resolve()

    model_flash = DelaSemSeg(dela_args).cuda()
    model_exact = DelaSemSeg(dela_args).cuda()
    util.load_state(str(flash_checkpoint), model=model_flash)
    util.load_state(str(exact_checkpoint), model=model_exact)
    model_flash.eval()
    model_exact.eval()

    exact_dataset = S3DIS(
        s3dis_args,
        partition=args.partition,
        loop=1,
        train=False,
        nbr_query="knn",
    )
    flash_dataset = S3DIS(
        s3dis_args,
        partition=args.partition,
        loop=1,
        train=False,
        nbr_query="grid_knn",
    )
    exact_paths = [Path(path).name for path in exact_dataset.paths]
    flash_paths = [Path(path).name for path in flash_dataset.paths]
    if exact_paths != flash_paths:
        raise RuntimeError("ExactKNN and FlashKNN dataset room ordering differs")

    room_limit = len(exact_dataset)
    if args.max_rooms > 0:
        room_limit = min(room_limit, args.max_rooms)
    loader_kwargs = {
        "batch_size": 1,
        "collate_fn": default_collate_fn,
        "pin_memory": True,
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }
    exact_loader = DataLoader(exact_dataset, **loader_kwargs)
    flash_loader = DataLoader(flash_dataset, **loader_kwargs)

    aggregate_confusions = new_confusions(len(CLASS_NAMES))
    room_rows: list[dict[str, Any]] = []
    total_points = 0
    total_boundary = 0
    same_class_hist = np.zeros(101, dtype=np.int64)
    processed_rooms: list[str] = []

    with torch.inference_mode():
        iterator = zip(exact_loader, flash_loader)
        for room_index, (batch_exact, batch_flash) in enumerate(
            tqdm(iterator, total=room_limit, desc="S3DIS rooms")
        ):
            if room_index >= room_limit:
                break
            room_name = Path(exact_dataset.paths[room_index]).stem

            xyz_exact, feature_exact, indices_exact, _, target_exact = (
                s3dis_collate_fn(batch_exact)
            )
            xyz_exact = xyz_exact.cuda(non_blocking=True)
            feature_exact = feature_exact.cuda(non_blocking=True)
            indices_exact = [
                item.cuda(non_blocking=True).long() for item in indices_exact[::-1]
            ]
            target_exact = target_exact.cuda(non_blocking=True).long()
            with autocast():
                logits_exact = model_exact(xyz_exact, feature_exact, indices_exact)

            xyz_flash, feature_flash, indices_flash, _, target_flash = (
                s3dis_collate_fn(batch_flash)
            )
            xyz_flash = xyz_flash.cuda(non_blocking=True)
            feature_flash = feature_flash.cuda(non_blocking=True)
            indices_flash = [
                item.cuda(non_blocking=True).long() for item in indices_flash[::-1]
            ]
            target_flash = target_flash.cuda(non_blocking=True).long()
            if xyz_exact.shape != xyz_flash.shape or not torch.allclose(
                xyz_exact, xyz_flash, atol=1e-5, rtol=0.0
            ):
                raise RuntimeError(f"Point alignment failed in {room_name}")
            if not torch.equal(target_exact, target_flash):
                raise RuntimeError(f"Label alignment failed in {room_name}")
            with autocast():
                logits_flash = model_flash(xyz_flash, feature_flash, indices_flash)

            exact_knn = indices_exact[-1]
            if exact_knn.ndim != 2 or exact_knn.shape[0] != target_exact.shape[0]:
                raise RuntimeError(
                    f"Unexpected finest exact-kNN shape in {room_name}: "
                    f"{tuple(exact_knn.shape)}"
                )
            if int(exact_knn.min()) < 0 or int(exact_knn.max()) >= len(target_exact):
                raise RuntimeError(f"Out-of-range exact-kNN index in {room_name}")
            query_ids = torch.arange(
                target_exact.shape[0], device=exact_knn.device, dtype=exact_knn.dtype
            )
            if not exact_knn.eq(query_ids[:, None]).any(dim=1).all():
                raise RuntimeError(
                    f"The exact-kNN support does not include every query in {room_name}"
                )

            same_class_fraction = (
                target_exact[exact_knn].eq(target_exact[:, None]).float().mean(dim=1)
            )
            boundary = same_class_fraction < args.same_class_threshold
            masks = subset_masks(boundary)
            predictions = {
                "FlashKNN": logits_flash.argmax(dim=-1),
                "ExactKNN": logits_exact.argmax(dim=-1),
            }
            room_confusions = new_confusions(len(CLASS_NAMES))
            for method, pred in predictions.items():
                for subset, mask in masks.items():
                    confusion = confusion_matrix(
                        pred, target_exact, mask, len(CLASS_NAMES)
                    )
                    room_confusions[method][subset] += confusion
                    aggregate_confusions[method][subset] += confusion

            point_count = int(target_exact.numel())
            boundary_count = int(boundary.sum().item())
            total_points += point_count
            total_boundary += boundary_count
            processed_rooms.append(room_name)
            hist_indices = torch.clamp(
                (same_class_fraction * 100).round().long(), min=0, max=100
            )
            same_class_hist += torch.bincount(
                hist_indices, minlength=101
            ).cpu().numpy().astype(np.int64)

            for subset in SUBSETS:
                for method in METHODS:
                    metric = metrics_from_confusion(room_confusions[method][subset])
                    room_rows.append(
                        {
                            "room": room_name,
                            "points": point_count,
                            "boundary_points": boundary_count,
                            "boundary_fraction": f"{boundary_count / point_count:.8f}",
                            "subset": subset,
                            "method": method,
                            "count": metric["count"],
                            "accuracy": scalar_or_blank(metric["accuracy"]),
                            "mean_class_accuracy": scalar_or_blank(
                                metric["mean_class_accuracy"]
                            ),
                            "miou": scalar_or_blank(metric["miou"]),
                        }
                    )
            del logits_exact, logits_flash
            torch.cuda.empty_cache()

    aggregate = {
        method: {
            subset: metrics_from_confusion(aggregate_confusions[method][subset])
            for subset in SUBSETS
        }
        for method in METHODS
    }
    difference_percentage_points: dict[str, dict[str, float]] = {}
    for subset in SUBSETS:
        difference_percentage_points[subset] = {
            metric: 100.0
            * (aggregate["FlashKNN"][subset][metric] - aggregate["ExactKNN"][subset][metric])
            for metric in ("accuracy", "mean_class_accuracy", "miou")
        }

    result = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_uuid": (
                torch.cuda.get_device_properties(0).uuid
                if hasattr(torch.cuda.get_device_properties(0), "uuid")
                else None
            ),
            "flash_checkpoint": str(flash_checkpoint),
            "flash_checkpoint_sha256": sha256(flash_checkpoint),
            "exact_checkpoint": str(exact_checkpoint),
            "exact_checkpoint_sha256": sha256(exact_checkpoint),
            "evaluation_script": str(Path(__file__).resolve()),
            "evaluation_script_sha256": sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "dataset": "S3DIS",
            "partition": args.partition,
            "rooms": len(processed_rooms),
            "room_names": processed_rooms,
            "voxel_size_m": float(s3dis_args.grid_size[0]),
            "k": int(s3dis_args.k[0]),
            "boundary_neighborhood": "finest-level exact kNN",
            "query_self_included": True,
            "same_class_threshold": args.same_class_threshold,
            "boundary_rule": "same_class_fraction < threshold",
            "evaluation": "single deterministic validation subsample (pick=0)",
            "seed": args.seed,
        },
        "point_partition": {
            "total_points": total_points,
            "semantic_boundary_points": total_boundary,
            "non_boundary_points": total_points - total_boundary,
            "semantic_boundary_fraction": total_boundary / total_points,
            "same_class_fraction_histogram_rounded_percent": same_class_hist.tolist(),
        },
        "aggregate": aggregate,
        "difference_percentage_points": difference_percentage_points,
    }

    (output_dir / "semantic_boundary_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_room_csv(output_dir / "semantic_boundary_per_room.csv", room_rows)
    write_summary_markdown(output_dir / "README.md", result)
    write_plot(output_dir / "semantic_boundary_accuracy_miou.pdf", result)
    print(json.dumps({
        "output_dir": str(output_dir),
        "rooms": len(processed_rooms),
        "points": total_points,
        "boundary_points": total_boundary,
        "boundary_fraction": total_boundary / total_points,
        "differences_pp": difference_percentage_points,
    }, indent=2))


if __name__ == "__main__":
    main()
