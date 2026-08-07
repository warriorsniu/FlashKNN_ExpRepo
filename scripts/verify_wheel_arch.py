#!/usr/bin/env python3
"""Inspect CUDA cubins inside a wheel without installing it."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--expected-arch", required=True)
    parser.add_argument("--cuobjdump", required=True, type=Path)
    args = parser.parse_args()
    expected = "sm_" + args.expected_arch.replace(".", "")
    architectures: set[str] = set()
    libraries = 0
    with tempfile.TemporaryDirectory(prefix="wheel-cubin-check-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(args.wheel) as archive:
            shared = [name for name in archive.namelist() if name.endswith(".so")]
            if not shared:
                raise SystemExit(f"No shared libraries in {args.wheel}")
            for name in shared:
                archive.extract(name, root)
                library = root / name
                output = subprocess.run(
                    [str(args.cuobjdump), "--list-elf", str(library)],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).stdout
                found = set(re.findall(r"sm_[0-9]+", output))
                if found:
                    libraries += 1
                    architectures.update(found)
    if expected not in architectures:
        raise SystemExit(
            f"{args.wheel.name} lacks {expected}; CUDA libraries contain "
            f"{sorted(architectures)}"
        )
    print(
        f"{args.wheel.name}: {expected} present in CUDA cubins "
        f"({libraries} CUDA shared libraries)"
    )


if __name__ == "__main__":
    main()
