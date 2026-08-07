#!/usr/bin/env python3
"""Validate and install the repository's pinned native CUDA wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "faiss", "flashknn", "cukd", "pytorchcudaflann",
    "pytorchnanoflann", "pointops", "dwconv", "minkowskiengine",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--expected-arch", required=True)
    parser.add_argument("--cuobjdump", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    wheelhouse = args.wheelhouse.resolve()
    manifest_path = wheelhouse / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing wheelhouse manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    import torch

    runtime_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks = {
        "format_version": (manifest.get("format_version"), 1),
        "target_cuda_arch": (manifest.get("target_cuda_arch"), args.expected_arch),
        "python": (manifest.get("python"), runtime_python),
        "torch": (manifest.get("torch"), torch.__version__),
        "torch_cuda": (manifest.get("torch_cuda"), torch.version.cuda),
    }
    mismatches = [f"{key}: wheelhouse={actual!r}, runtime={expected!r}"
                  for key, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise SystemExit("Incompatible wheelhouse:\n  " + "\n  ".join(mismatches))

    records = manifest.get("wheels", [])
    found = {record.get("distribution") for record in records}
    if found != EXPECTED:
        raise SystemExit(
            f"Wheelhouse distributions differ: missing={EXPECTED-found}, extra={found-EXPECTED}"
        )
    wheels: list[Path] = []
    verifier = Path(__file__).with_name("verify_wheel_arch.py")
    for record in records:
        wheel = wheelhouse / record["file"]
        if not wheel.is_file():
            raise SystemExit(f"Missing wheel: {wheel}")
        if wheel.stat().st_size != record["bytes"] or digest(wheel) != record["sha256"]:
            raise SystemExit(f"Wheel checksum/size mismatch: {wheel}")
        if record.get("cuda"):
            subprocess.run([
                sys.executable, str(verifier), "--wheel", str(wheel),
                "--expected-arch", args.expected_arch.replace(".", ""),
                "--cuobjdump", str(args.cuobjdump),
            ], check=True)
        wheels.append(wheel)

    if args.validate_only:
        print(f"Validated {len(wheels)} native wheels from {wheelhouse}")
    else:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-deps",
            *[str(wheel) for wheel in wheels],
        ], check=True)
        print(f"Installed {len(wheels)} verified native wheels from {wheelhouse}")


if __name__ == "__main__":
    main()
