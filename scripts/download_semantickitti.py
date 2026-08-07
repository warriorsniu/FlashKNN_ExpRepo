#!/usr/bin/env python3
"""Legally download/import the efficiency-only SemanticKITTI benchmark data.

The preferred input is a small benchmark-pack archive supplied by the authors.
As an official fallback, a user-provided KITTI email download URL is used to
resume the 80 GB odometry archive, extract five frames from every sequence,
and build the pack.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--pack-url")
    source.add_argument("--kitti-velodyne-url")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-pack", type=Path, required=True)
    return p.parse_args()


def download(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "curl", "--fail", "--location", "--retry", "5",
        "--continue-at", "-", "--output", os.fspath(destination), url,
    ], check=True)


def safe_tar(archive: Path, destination: Path):
    destination = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            resolved = (destination / member.name).resolve()
            if resolved != destination and destination not in resolved.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination)


def find_manifest(root: Path) -> Path | None:
    candidates = [root / "manifest.json", *root.glob("*/manifest.json")]
    return next((p for p in candidates if p.is_file()), None)


def extract_pack(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            root = destination.resolve()
            for member in handle.infolist():
                resolved = (root / member.filename).resolve()
                if resolved != root and root not in resolved.parents:
                    raise RuntimeError(f"Unsafe archive member: {member.filename}")
            handle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        safe_tar(archive, destination)
    else:
        raise SystemExit(f"Pack URL did not produce a ZIP/TAR archive: {archive}")
    manifest = find_manifest(destination)
    if manifest is None:
        raise SystemExit("Downloaded pack archive does not contain manifest.json")
    return manifest.parent


def extract_uniform_frames(archive: Path, destination: Path, samples_per_sequence: int = 5):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        selected = []
        for sequence_number in range(22):
            sequence = f"{sequence_number:02d}"
            prefix = f"dataset/sequences/{sequence}/velodyne/"
            members = sorted(name for name in handle.namelist()
                             if name.startswith(prefix) and name.endswith(".bin"))
            if not members:
                raise SystemExit(f"No {prefix}*.bin files in {archive}")
            count = min(samples_per_sequence, len(members))
            # Match numpy.linspace(..., dtype=int) in the pack generator: the
            # intermediate positions are truncated toward zero, not rounded.
            positions = [i * (len(members) - 1) // (count - 1) for i in range(count)]
            selected.extend(members[position] for position in positions)
        for number, member in enumerate(selected, 1):
            handle.extract(member, destination)
            if number % 25 == 0:
                print(f"Extracted {number}/{len(selected)} scans", flush=True)
    print(f"Extracted {len(selected)} scans from 22 sequences (ZIP CRC checked)")


def main():
    args = arguments()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.pack_url:
        archive = args.work_dir / "semantickitti_benchmark_pack.download"
        download(args.pack_url, archive)
        pack = extract_pack(archive, args.work_dir / "pack_extracted")
    else:
        archive = args.work_dir / "data_odometry_velodyne.zip"
        download(args.kitti_velodyne_url, archive)
        raw = args.work_dir / "kitti_uniform_frames"
        ready = all(len(list((raw / f"dataset/sequences/{i:02d}/velodyne").glob("*.bin"))) >= 5
                    for i in range(22))
        if not ready:
            extract_uniform_frames(archive, raw)
        subprocess.run([
            sys.executable, os.fspath(Path(__file__).parent / "prepare_semantickitti_pack.py"),
            "--root", os.fspath(raw / "dataset"), "--voxel-size", "0.06",
            "--samples-per-sequence", "5",
            "--output-dir", os.fspath(args.output_pack),
        ], check=True)
        pack = args.output_pack
    print(f"SEMANTICKITTI_PACK={pack.resolve()}")
    (args.work_dir / "resolved_pack.txt").write_text(
        os.fspath(pack.resolve()) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
