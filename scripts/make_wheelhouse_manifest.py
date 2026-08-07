#!/usr/bin/env python3
"""Generate a checksummed compatibility manifest for native wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path


CUDA_DISTRIBUTIONS = {
    "faiss", "flashknn", "cukd", "pytorchcudaflann",
    "pointops", "dwconv", "minkowskiengine",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def distribution(wheel: Path) -> tuple[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
    return metadata["Name"].lower().replace("-", ""), metadata["Version"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--cuda-arch", required=True, help="Dotted capability, e.g. 8.9")
    parser.add_argument("--cuda-home", required=True, type=Path)
    args = parser.parse_args()
    import torch

    records = []
    for wheel in sorted(args.wheelhouse.glob("*.whl")):
        name, version = distribution(wheel)
        records.append({
            "file": wheel.name,
            "distribution": name,
            "version": version,
            "bytes": wheel.stat().st_size,
            "sha256": sha256(wheel),
            "cuda": name in CUDA_DISTRIBUTIONS,
        })
    payload = {
        "format_version": 1,
        "target_cuda_arch": args.cuda_arch,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "platform": platform.platform(),
        "nvcc": subprocess.check_output(
            [str(args.cuda_home / "bin/nvcc"), "--version"], text=True
        ).strip().splitlines()[-1],
        "wheels": records,
    }
    expected = {
        "faiss", "flashknn", "cukd", "pytorchcudaflann",
        "pytorchnanoflann", "pointops", "dwconv", "minkowskiengine",
    }
    found = {record["distribution"] for record in records}
    if found != expected:
        raise SystemExit(f"Wheelhouse distributions differ: missing={expected-found}, extra={found-expected}")
    path = args.wheelhouse / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path} with {len(records)} wheels")


if __name__ == "__main__":
    main()
