"""Shared training/evaluation helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


CLASS_COUNTS = torch.tensor([
    55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
    240942562, 17294618, 170599734, 6369672, 230413074, 101130274,
    476491114, 9833174, 129609852, 4506626, 1168181,
], dtype=torch.float64)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(device: torch.device) -> torch.Tensor:
    frequency = CLASS_COUNTS / CLASS_COUNTS.sum()
    weights = 1.0 / (frequency + 0.02)
    return (weights / weights.mean()).float().to(device)


class Confusion:
    def __init__(self, classes: int = 19) -> None:
        self.classes = classes
        self.matrix = torch.zeros((classes, classes), dtype=torch.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = logits.argmax(1).detach().cpu()
        target = target.detach().cpu()
        valid = (target >= 0) & (target < self.classes)
        bins = target[valid] * self.classes + prediction[valid]
        self.matrix += torch.bincount(bins, minlength=self.classes**2).reshape(self.classes, self.classes)

    def result(self) -> dict[str, object]:
        matrix = self.matrix.double()
        intersection = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = intersection / union.clamp_min(1)
        accuracy = intersection.sum() / matrix.sum().clamp_min(1)
        return {"miou": float(iou.mean()), "accuracy": float(accuracy), "iou": iou.tolist()}


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer=None) -> dict:
    # These are project-generated, trusted training checkpoints. PyTorch 2.6+
    # defaults weights_only to True, which rejects their pathlib metadata.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
