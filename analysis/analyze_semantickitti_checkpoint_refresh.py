#!/usr/bin/env python3
"""Summarize the current-kernel SemanticKITTI checkpoint compatibility run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    rows = []
    for model in ("dela", "deepla"):
        for seed in (47, 48, 49):
            path = args.input_dir / f"{model}_seed{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model") != model or int(payload.get("alpha", -1)) != 8:
                raise SystemExit(f"Unexpected model/alpha in {path}")
            if str(payload.get("split")) != "08":
                raise SystemExit(f"Unexpected split in {path}")
            rows.append({
                "model": model,
                "seed": seed,
                "miou": float(payload["miou"]) * 100.0,
                "accuracy": float(payload["accuracy"]) * 100.0,
                "checkpoint": payload["checkpoint"],
                "file": str(path.resolve()),
                "sha256": sha256(path),
            })
    models = {}
    for model in ("dela", "deepla"):
        subset = [row for row in rows if row["model"] == model]
        miou = [row["miou"] for row in subset]
        accuracy = [row["accuracy"] for row in subset]
        models[model] = {
            "runs": 3,
            "miou_mean": statistics.mean(miou),
            "miou_sample_sd": statistics.stdev(miou),
            "accuracy_mean": statistics.mean(accuracy),
            "accuracy_sample_sd": statistics.stdev(accuracy),
        }
    summary = {"protocol": "current production kernel, alpha=8, sequence 08", "rows": rows, "models": models}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# SemanticKITTI current-kernel checkpoint refresh",
        "",
        "| Model | Seed 47 | Seed 48 | Seed 49 | Mean mIoU | Sample SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("dela", "deepla"):
        values = [row["miou"] for row in rows if row["model"] == model]
        aggregate = models[model]
        lines.append(
            f"| {model} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | "
            f"{aggregate['miou_mean']:.3f} | {aggregate['miou_sample_sd']:.3f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
