#!/usr/bin/env python3
"""Create a clean cross-GPU handoff archive, optionally embedding data/wheels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


ROOT_NAME = "FlashKNN_ExpRepo"
EXCLUDED_PARTS = {
    ".git", ".runtime", ".vscode", "__pycache__", "build",
    "results", "s3dis_extracted", "downloads", "incoming", "wheelhouse",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a"}


def include_source(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if relative.parts[:2] == ("analysis", "output"):
        return False
    if relative.parts and relative.parts[0] == "data":
        return relative == Path("data/.gitkeep")
    if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def add_file(archive: tarfile.TarFile, source: Path, relative: Path) -> dict:
    archive.add(source, arcname=str(Path(ROOT_NAME) / relative), recursive=False)
    return {"path": relative.as_posix(), "bytes": source.stat().st_size,
            "sha256": digest(source)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--semantickitti-pack", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    sources = sorted(
        path for path in root.rglob("*")
        if path.resolve() != output and include_source(root, path)
    )
    if not sources:
        raise SystemExit("No repository sources selected")

    lidar_files: list[Path] = []
    if args.semantickitti_pack:
        lidar = args.semantickitti_pack.resolve()
        manifest = lidar / "manifest.json"
        if not manifest.is_file():
            raise SystemExit(f"SemanticKITTI pack has no manifest.json: {lidar}")
        lidar_files = sorted(path for path in lidar.iterdir() if path.is_file())

    wheel_files: list[Path] = []
    if args.wheelhouse:
        wheelhouse = args.wheelhouse.resolve()
        manifest = wheelhouse / "manifest.json"
        if not manifest.is_file():
            raise SystemExit(f"Wheelhouse has no manifest.json: {wheelhouse}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        wheel_files = [manifest]
        for record in payload.get("wheels", []):
            wheel = wheelhouse / record["file"]
            if not wheel.is_file() or digest(wheel) != record["sha256"]:
                raise SystemExit(f"Wheelhouse file missing or corrupt: {wheel}")
            wheel_files.append(wheel)

    records = []
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        for source in sources:
            records.append(add_file(archive, source, source.relative_to(root)))
        for source in lidar_files:
            relative = Path("data/incoming/semantickitti_pack") / source.name
            records.append(add_file(archive, source, relative))
        for source in wheel_files:
            relative = Path("wheelhouse") / source.name
            records.append(add_file(archive, source, relative))
        manifest_payload = {
            "format_version": 1,
            "root": ROOT_NAME,
            "source_files": len(sources),
            "embedded_semantickitti_files": len(lidar_files),
            "embedded_wheelhouse_files": len(wheel_files),
            "files": records,
        }
        encoded = (json.dumps(manifest_payload, indent=2) + "\n").encode()
        info = tarfile.TarInfo(str(Path(ROOT_NAME) / "PACKAGE_MANIFEST.json"))
        info.size = len(encoded)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(encoded))
    print(
        f"Created {output} ({output.stat().st_size} bytes): "
        f"{len(sources)} source files, {len(lidar_files)} LiDAR pack files, "
        f"{len(wheel_files)} wheelhouse files"
    )


if __name__ == "__main__":
    main()
