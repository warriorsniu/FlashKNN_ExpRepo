#!/usr/bin/env python3
"""Prepare shared S3DIS and SemanticKITTI paths without redistributing data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--accept-s3dis-license", action="store_true")
    p.add_argument("--s3dis-existing", type=Path,
                   help="Existing Pointcept root (legacy PTH or current per-field NPY layout)")
    p.add_argument("--semantickitti-root", type=Path,
                   help="Licensed root containing sequences/*/velodyne")
    p.add_argument("--semantickitti-pack", type=Path,
                   help="Existing benchmark pack containing manifest.json")
    p.add_argument("--lidar-samples-per-sequence", type=int, default=5)
    return p.parse_args()


def replace_symlink(destination: Path, target: Path):
    target = target.resolve()
    if destination.is_symlink():
        if destination.resolve() == target:
            return
        destination.unlink()
    elif destination.exists():
        raise SystemExit(f"Refusing to replace existing non-symlink: {destination}")
    destination.symlink_to(target, target_is_directory=True)


def has_s3dis_rooms(root: Path) -> bool:
    for area_name in ("Area_1", "area_1"):
        area = root / area_name
        if not area.is_dir():
            continue
        if any(area.glob("*.pth")):
            return True
        if any(room.is_dir() and (room / "coord.npy").is_file() for room in area.iterdir()):
            return True
    return False


def find_s3dis(root: Path) -> Path | None:
    candidates = [root]
    candidates.extend(path.parent for name in ("Area_1", "area_1") for path in root.rglob(name))
    for candidate in candidates:
        if candidate.is_dir() and has_s3dis_rooms(candidate):
            return candidate
    return None


def safe_extract(archive: Path, destination: Path):
    destination = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            resolved = (destination / member.name).resolve()
            if destination not in resolved.parents and resolved != destination:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination)


def main():
    args = arguments()
    repo = Path(__file__).resolve().parents[1]
    data = repo / "data"
    downloads = data / "downloads"
    data.mkdir(exist_ok=True); downloads.mkdir(exist_ok=True)

    if args.s3dis_existing:
        s3dis = find_s3dis(args.s3dis_existing.resolve())
        if s3dis is None:
            raise SystemExit(
                "--s3dis-existing has neither legacy Area_*/*.pth nor current "
                "Area_*/room/coord.npy Pointcept data"
            )
    else:
        if not args.accept_s3dis_license:
            raise SystemExit(
                "S3DIS is license-gated. Read https://cvg-data.inf.ethz.ch/s3dis/ "
                "and rerun with --accept-s3dis-license."
            )
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download(
            repo_id="Pointcept/s3dis-compressed", repo_type="dataset",
            filename="s3dis.tar.gz", local_dir=downloads,
        ))
        extracted = data / "s3dis_extracted"
        if find_s3dis(extracted) is None:
            extracted.mkdir(exist_ok=True)
            safe_extract(archive, extracted)
        s3dis = find_s3dis(extracted)
        if s3dis is None:
            raise SystemExit(f"Could not find Pointcept S3DIS rooms after extracting {archive}")
    replace_symlink(data / "s3dis", s3dis)

    lidar = None
    if args.semantickitti_pack:
        lidar = args.semantickitti_pack.resolve()
        if not (lidar / "manifest.json").is_file():
            raise SystemExit("--semantickitti-pack does not contain manifest.json")
    elif args.semantickitti_root:
        source = args.semantickitti_root.resolve()
        if (source / "dataset" / "sequences").is_dir():
            source = source / "dataset"
        lidar = data / "semantickitti_pack"
        subprocess.run([
            sys.executable, os.fspath(Path(__file__).parent / "prepare_semantickitti_pack.py"),
            "--root", os.fspath(source), "--voxel-size", "0.06",
            "--samples-per-sequence", str(args.lidar_samples_per_sequence),
            "--output-dir", os.fspath(lidar),
        ], check=True)
    if lidar is not None:
        replace_symlink(data / "semantickitti", lidar)

    env = data / "paths.env"
    lines = [f"export EXPREPO_S3DIS={shlex_quote(os.fspath((data / 's3dis').resolve()))}"]
    if (data / "semantickitti").exists():
        lines.append(f"export EXPREPO_SEMANTICKITTI={shlex_quote(os.fspath((data / 'semantickitti').resolve()))}")
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"S3DIS: {s3dis}")
    print(f"Wrote {env}")


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


if __name__ == "__main__":
    main()
