#!/usr/bin/env python3
"""Require locally compiled Pointcept extensions to contain the target cubin."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-arch", required=True, help="8.9, 9.0, or compact 89/90")
    parser.add_argument("--cuobjdump", required=True, type=Path)
    args = parser.parse_args()
    expected = "sm_" + args.expected_arch.replace(".", "")

    import dwconv.core
    import dela_cutils_ext
    import MinkowskiEngineBackend._C
    import pointops._C

    modules = {
        "pointops": Path(pointops._C.__file__).resolve(),
        "MinkowskiEngine": Path(MinkowskiEngineBackend._C.__file__).resolve(),
        "dwconv": Path(dwconv.core.__file__).resolve(),
        "DeLA cutils": Path(dela_cutils_ext.__file__).resolve(),
    }
    for name, library in modules.items():
        listing = subprocess.check_output(
            [str(args.cuobjdump), "--list-elf", str(library)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        architectures = sorted(set(re.findall(r"sm_[0-9]+", listing)))
        if expected not in architectures:
            raise SystemExit(
                f"{name} lacks {expected}: {library} contains {architectures}"
            )
        print(f"{name}: {expected} cubin OK")


if __name__ == "__main__":
    main()
