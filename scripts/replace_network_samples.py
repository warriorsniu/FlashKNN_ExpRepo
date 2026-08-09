#!/usr/bin/env python3
"""Replace explicitly selected network records with compatible rerun samples."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def arguments() -> argparse.Namespace:
    """Parse the base, rerun, and provenance fields for one replacement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--reason", required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    """Rebuild aggregate statistics over per-sample latency means."""
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "std_ms": statistics.pstdev(values),
    }


def compatible(base: dict, rerun: dict) -> None:
    """Reject replacement records produced with a different benchmark setup."""
    base_meta, rerun_meta = base["metadata"], rerun["metadata"]
    fields = (
        "dataset", "model", "variant", "backends", "alpha", "voxel_sizes_m", "k",
        "checkpoint", "warmups", "repeats", "torch", "torch_cuda",
    )
    changed = {
        field: (base_meta.get(field), rerun_meta.get(field))
        for field in fields if base_meta.get(field) != rerun_meta.get(field)
    }
    base_gpu, rerun_gpu = base_meta.get("gpu", {}), rerun_meta.get("gpu", {})
    for field in ("name", "uuid", "driver", "memory_mib"):
        if base_gpu.get(field) != rerun_gpu.get(field):
            changed[f"gpu.{field}"] = (base_gpu.get(field), rerun_gpu.get(field))
    if changed:
        raise SystemExit(f"Incompatible rerun: {changed}")


def rebuild_aggregate(payload: dict) -> None:
    """Recompute the benchmark's backend aggregates after record replacement."""
    records = payload["samples"]
    aggregate = {}
    for backend in payload["metadata"]["backends"]:
        aggregate[backend] = {
            key: summarize([record["backends"][backend][key]["mean_ms"] for record in records])
            for key in ("hierarchy", "model", "end_to_end")
        }
        aggregate[backend]["single_sample_throughput_hz"] = (
            1000.0 / aggregate[backend]["end_to_end"]["mean_ms"]
        )
    payload["aggregate"] = aggregate


def main() -> None:
    """Replace records atomically and attach a machine-readable provenance entry."""
    args = arguments()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    rerun = json.loads(args.rerun.read_text(encoding="utf-8"))
    compatible(base, rerun)
    requested = set(args.sample)
    replacements = {
        record["sample"]: record for record in rerun["samples"]
        if record["sample"] in requested
    }
    if set(replacements) != requested:
        raise SystemExit(f"Rerun misses samples: {sorted(requested - set(replacements))}")
    found = set()
    for index, record in enumerate(base["samples"]):
        if record["sample"] in replacements:
            base["samples"][index] = replacements[record["sample"]]
            found.add(record["sample"])
    if found != requested:
        raise SystemExit(f"Base misses samples: {sorted(requested - found)}")
    base["metadata"].setdefault("timing_overrides", []).append({
        "samples": sorted(requested), "source": str(args.rerun), "reason": args.reason,
        "warmups": rerun["metadata"]["warmups"], "repeats": rerun["metadata"]["repeats"],
        "gpu": rerun["metadata"]["gpu"],
    })
    rebuild_aggregate(base)
    temporary = args.base.with_suffix(args.base.suffix + ".tmp")
    temporary.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.base)


if __name__ == "__main__":
    main()
