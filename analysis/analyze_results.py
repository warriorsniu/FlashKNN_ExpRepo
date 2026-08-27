#!/usr/bin/env python3
"""Aggregate benchmark JSON into an Excel workbook, markdown, and paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


def configure_paper_fonts() -> None:
    font_root = Path("/data/nyc/fonts")
    font_paths = [font_root / name for name in (
        "TIMES.TTF", "TIMESBD.TTF", "TIMESI.TTF", "TIMESBI.TTF"
    )]
    missing = [path for path in font_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Times New Roman fonts: {missing}")
    for font_path in font_paths:
        fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_paths[0]).get_name()
    plt.rcParams.update({
        "font.family": font_name,
        "font.serif": [font_name],
        "mathtext.fontset": "stix",
    })


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis/output"))
    return p.parse_args()


def avg(items, key):
    values = [float(item[key]) for item in items]
    return sum(values) / len(values) * 1000


def query_rows(path, payload):
    meta = payload.get("metadata", {})
    dataset = meta.get("dataset", "unknown")
    gpu = meta.get("gpu", {}).get("name", "unknown")
    rows = []
    if "records" in payload and payload["records"] and "methods" in payload["records"][0]:
        for record in payload["records"]:
            for method, value in record["methods"].items():
                timings = value.get("timings", [])
                if not timings:
                    continue
                recall = value.get("recall_vs_cukd", value.get("recall_vs_faiss_flat", 1.0))
                if isinstance(recall, dict): recall = recall.get("mean")
                training_ms = float(value.get("training_s", 0.0)) * 1000.0
                construction_ms = avg(timings, "construction_s") + training_ms
                query_ms = avg(timings, "query_s")
                rows.append({"file": str(path), "gpu": gpu, "dataset": dataset, "sample": record["room"],
                    "scope": record["scope"], "mode": record["mode"], "k": record["k"],
                    "method": method,
                    "alpha": (2 ** int(meta.get("num_down", 0))) if method == "flashknn" else None,
                    "num_support": record["num_support"], "num_query": record["num_query"],
                    "training_ms": training_ms, "construction_ms": construction_ms,
                    "query_ms": query_ms, "total_ms": construction_ms + query_ms,
                    "recall": recall})
    for record in payload.get("samples", []):
        if "flashknn" not in record:
            continue
        exact = record["exact"]
        timings = exact.get("timings", [])
        if timings:
            build = avg(timings, "construction_seconds")
            query = avg(timings, "query_seconds")
            rows.append({"file": str(path), "gpu": gpu, "dataset": dataset, "sample": record["sample"],
                "scope": "full", "mode": record["mode"], "k": record["k"],
                "method": exact["method"], "alpha": None, "num_support": record["num_support"],
                "num_query": record["num_query"], "construction_ms": build,
                "query_ms": query, "total_ms": build + query, "recall": 1.0})
        for flash in record["flashknn"]:
            timings = flash["timings"]
            build = avg(timings, "预处理耗时"); query = avg(timings, "查询耗时")
            rows.append({"file": str(path), "gpu": gpu, "dataset": dataset, "sample": record["sample"],
                "scope": "full", "mode": record["mode"], "k": record["k"],
                "method": "flashknn", "alpha": flash["alpha"],
                "num_support": record["num_support"], "num_query": record["num_query"],
                "construction_ms": build, "query_ms": query, "total_ms": build + query,
                "recall": flash["recall"]["mean"]})
        for method in ("flann_cuda", "nanoflann", "faiss_flat", "faiss_ivf"):
            if method not in record or not record[method].get("timings"):
                continue
            value = record[method]
            timings = value["timings"]
            training_ms = float(value.get("training_seconds", 0.0)) * 1000.0
            build = avg(timings, "construction_seconds") + training_ms
            query = avg(timings, "query_seconds")
            recall = value.get("recall_vs_exact", 1.0)
            if isinstance(recall, dict):
                recall = recall.get("mean", 1.0)
            rows.append({"file": str(path), "gpu": gpu, "dataset": dataset,
                "sample": record["sample"], "scope": "full", "mode": record["mode"],
                "k": record["k"], "method": method, "alpha": None,
                "num_support": record["num_support"], "num_query": record["num_query"],
                "training_ms": training_ms, "construction_ms": build,
                "query_ms": query, "total_ms": build + query,
                "recall": float(recall)})
    return rows


def network_rows(path, payload):
    meta = payload.get("metadata", {})
    if not meta or "model" not in meta:
        return []
    rows = []
    gpu = meta.get("gpu", {}).get("name", "unknown")
    for record in payload.get("records", []):
        if "backends" in record:
            for backend, value in record["backends"].items():
                rows.append({"file": str(path), "gpu": gpu, "dataset": meta["dataset"], "model": meta["model"],
                    "backend": backend, "sample": record["room"], "num_full": record["num_full"],
                    "num_down": value["num_down"], "preprocessing_ms": value["preprocessing"]["mean_ms"],
                    "network_ms": value["network"]["mean_ms"], "end_to_end_ms": value["end_to_end"]["mean_ms"]})
        else:
            rows.append({"file": str(path), "gpu": gpu, "dataset": meta["dataset"], "model": meta["model"],
                "backend": "native", "sample": record.get("room", record.get("sample")),
                "num_full": record.get("num_full", record.get("points")),
                "num_down": record.get("num_down", record.get("points")), "preprocessing_ms": 0.0,
                "network_ms": record["network"]["mean_ms"],
                "end_to_end_ms": record["network"]["mean_ms"]})
    for record in payload.get("samples", []):
        values = record.get("backends", {"flashknn": record})
        for backend, value in values.items():
            rows.append({"file": str(path), "gpu": gpu,
                "dataset": meta.get("dataset", meta.get("manifest", {}).get("dataset", "unknown")),
                "model": meta["model"], "backend": backend, "sample": record["sample"],
                "num_full": record["points"], "num_down": record["points"],
                "preprocessing_ms": value["hierarchy"]["mean_ms"],
                "network_ms": value["model"]["mean_ms"],
                "end_to_end_ms": value["end_to_end"]["mean_ms"]})
    return rows


def main():
    configure_paper_fonts()
    args = arguments(); args.output_dir.mkdir(parents=True, exist_ok=True)
    query, network = [], []
    for path in sorted(args.results.rglob("*.json")):
        try: payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError): continue
        query.extend(query_rows(path, payload)); network.extend(network_rows(path, payload))
    qdf, ndf = pd.DataFrame(query), pd.DataFrame(network)
    if qdf.empty and ndf.empty:
        raise SystemExit(f"No benchmark records found below {args.results}")
    main_table = pd.DataFrame()
    paper_query_table = pd.DataFrame()
    scaling_table = pd.DataFrame()
    semantickitti_representative = pd.DataFrame()
    if not qdf.empty:
        group = ["gpu", "dataset", "scope", "mode", "k", "method", "alpha"]
        main_table = qdf.groupby(group, dropna=False).agg(
            samples=("sample", "count"), support_points=("num_support", "mean"),
            query_points=("num_query", "mean"), construction_ms=("construction_ms", "mean"),
            query_ms=("query_ms", "mean"), total_ms=("total_ms", "mean"),
            recall=("recall", "mean")).reset_index()

        # Paper operating point for SemanticKITTI. Use alpha=8 to align the
        # operator and network tables with the matched LiDAR checkpoints while
        # retaining the full alpha sweep in main_table.
        semantic = main_table[
            (main_table.dataset == "SemanticKITTI") &
            (main_table.scope == "full") &
            (main_table.k == 24)
        ]
        representative_rows = []
        for (gpu_name, mode), subset in semantic.groupby(["gpu", "mode"]):
            flash = subset[(subset.method == "flashknn") & (subset.alpha == 8)]
            exact = subset[subset.method.isin(["cuda_kdtree", "cukd"])]
            if flash.empty or exact.empty:
                continue
            flash_row, exact_row = flash.iloc[0], exact.iloc[0]
            representative_rows.append({
                "gpu": gpu_name,
                "dataset": "SemanticKITTI",
                "mode": mode,
                "k": 24,
                "alpha": 8,
                "samples": int(flash_row["samples"]),
                "flashknn_total_ms": flash_row["total_ms"],
                "flashknn_recall": flash_row["recall"],
                "exact_method": exact_row["method"],
                "exact_total_ms": exact_row["total_ms"],
                "speedup_vs_exact": exact_row["total_ms"] / flash_row["total_ms"],
            })
        semantickitti_representative = pd.DataFrame(representative_rows)
        references = qdf[qdf.method.isin(["cuda_kdtree", "cukd"])]
        baseline = references.groupby(["gpu", "dataset", "sample", "mode", "k"])["total_ms"].mean()
        qdf["speedup_vs_exact"] = qdf.apply(lambda r: (
            baseline.get((r.gpu, r.dataset, r["sample"], r["mode"], r.k), float("nan")) / r.total_ms
        ), axis=1)
        flash = qdf[qdf.method == "flashknn"]
        if not flash.empty:
            for (gpu_name, mode), subset in flash.groupby(["gpu", "mode"]):
                plt.scatter(subset.num_support, subset.speedup_vs_exact, s=14, alpha=.65,
                            label=f"{gpu_name}: {mode}")
            plt.xscale("log"); plt.xlabel("Support points"); plt.ylabel("Speedup vs exact CUDA k-d tree")
            plt.legend(); plt.tight_layout(); plt.savefig(args.output_dir / "speedup_vs_points.png", dpi=220); plt.close()

        # Paper Table: Time Cost(ms) for Different k and Query Modes.
        paper_query_long = main_table[
            (main_table.dataset == "S3DIS") & (main_table.scope == "sample_part")
        ].copy()
        if not paper_query_long.empty:
            timing_long = paper_query_long.melt(
                id_vars=["gpu", "method", "mode", "k"],
                value_vars=["construction_ms", "query_ms", "total_ms"],
                var_name="metric", value_name="time_ms",
            )
            timing_long["mode_k"] = timing_long["mode"] + "_k" + timing_long["k"].astype(str)
            paper_query_table = timing_long.pivot_table(
                index=["gpu", "metric", "method"], columns="mode_k",
                values="time_ms",
            ).reset_index()
            ordered = [f"{mode}_k{k}" for mode in ("pre", "post")
                       for k in (8, 16, 24, 32, 48, 64)]
            paper_query_table = paper_query_table[
                ["gpu", "metric", "method"] +
                [column for column in ordered if column in paper_query_table]
            ]

        # Paper Fig.: query/construction speedups under different point counts,
        # with nanoflann as the baseline and k=32 pre-downsampling queries.
        scaling = qdf[(qdf.dataset == "S3DIS") & (qdf.scope == "full") &
                      (qdf["mode"] == "pre") & (qdf.k == 32)].copy()
        nano = scaling[scaling.method == "nanoflann"][[
            "gpu", "sample", "construction_ms", "query_ms"
        ]].rename(columns={"construction_ms": "nanoflann_construction_ms",
                           "query_ms": "nanoflann_query_ms"})
        scaling_table = scaling.merge(nano, on=["gpu", "sample"], how="left")
        scaling_table["construction_speedup_vs_nanoflann"] = (
            scaling_table.nanoflann_construction_ms / scaling_table.construction_ms
        )
        scaling_table["query_speedup_vs_nanoflann"] = (
            scaling_table.nanoflann_query_ms / scaling_table.query_ms
        )
        for metric, filename, ylabel in (
            ("query_ms", "speedup_of_query_under_different_number_of_point.png",
             "Query speedup vs nanoflann"),
            ("construction_ms", "speedup_of_construction_under_different_number_of_point.png",
             "Construction speedup vs nanoflann"),
        ):
            plt.figure(figsize=(7.2, 4.4))
            speedup_column = metric.replace("_ms", "_speedup_vs_nanoflann")
            display_names = {
                "flashknn": "FlashKNN",
                "cuda_kdtree": "cudaKDTree",
                "cukd": "cudaKDTree",
                "flann_cuda": "FLANN-CUDA",
                "nanoflann": "nanoflann",
                "faiss_ivf": "FAISS IVF",
                "faiss_flat": "FAISS Flat",
            }
            for (gpu_name, method), subset in scaling_table.groupby(["gpu", "method"]):
                subset = subset.sort_values("num_support")
                if not subset.empty:
                    plt.plot(subset.num_support, subset[speedup_column], marker="o",
                             markersize=3, linewidth=1,
                             label=display_names.get(method, method))
            plt.xlabel("Voxelized point count"); plt.ylabel(ylabel); plt.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(args.output_dir / filename, dpi=220); plt.close()
    network_table = pd.DataFrame()
    network_efficiency = pd.DataFrame()
    if not ndf.empty:
        network_table = ndf.groupby(["gpu", "dataset", "model", "backend"], dropna=False).agg(
            samples=("sample", "count"), points=("num_down", "mean"),
            preprocessing_ms=("preprocessing_ms", "mean"), network_ms=("network_ms", "mean"),
            end_to_end_ms=("end_to_end_ms", "mean")).reset_index()
        labels = network_table.apply(
            lambda r: f"{r['model']}\n{r['backend']}\n{r['gpu']}", axis=1)
        plt.figure(figsize=(max(7, len(labels) * .8), 4))
        plt.bar(labels, network_table.end_to_end_ms)
        plt.ylabel("Latency (ms / sample)"); plt.xticks(rotation=25, ha="right")
        plt.tight_layout(); plt.savefig(args.output_dir / "network_latency.png", dpi=220); plt.close()

        # Paper Fig.: preserve the point-count axis instead of collapsing every
        # model to one mean bar.
        def paper_network_label(row):
            if str(row["model"]).lower() == "dela":
                return "DeLA + FlashKNN" if row["backend"] == "flashknn" else "DeLA"
            return {"ptv3": "PTv3", "octformer": "OctFormer", "spunet": "SPUNet",
                    "minkunet34c": "MinkUNet"}.get(str(row["model"]).lower(), str(row["model"]))
        network_efficiency = ndf[ndf.dataset == "S3DIS"].copy()
        if not network_efficiency.empty:
            plt.figure(figsize=(7.2, 4.4))
            network_efficiency["paper_label"] = network_efficiency.apply(paper_network_label, axis=1)
            for (gpu_name, label), subset in network_efficiency.groupby(["gpu", "paper_label"]):
                subset = subset.sort_values("num_down")
                plt.plot(subset.num_down, subset.end_to_end_ms / 1000.0, marker="o",
                         markersize=3, linewidth=1, label=f"{label} ({gpu_name})")
            plt.xlabel("Voxelized point number"); plt.ylabel("Time cost (s)")
            plt.legend(fontsize=7); plt.tight_layout()
            plt.savefig(args.output_dir / "network_efficiency_comparison.png", dpi=220); plt.close()
    workbook = args.output_dir / "benchmark_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        if not main_table.empty: main_table.to_excel(writer, sheet_name="query_main_table", index=False)
        if not semantickitti_representative.empty:
            semantickitti_representative.to_excel(
                writer, sheet_name="semkitti_alpha8", index=False
            )
        if not paper_query_table.empty:
            paper_query_table.to_excel(writer, sheet_name="paper_query_table", index=False)
        if not scaling_table.empty:
            scaling_table.to_excel(writer, sheet_name="query_point_scaling", index=False)
        if not qdf.empty: qdf.to_excel(writer, sheet_name="query_per_sample", index=False)
        if not network_table.empty: network_table.to_excel(writer, sheet_name="network_summary", index=False)
        if not ndf.empty: ndf.to_excel(writer, sheet_name="network_per_sample", index=False)
        if not network_efficiency.empty:
            network_efficiency.to_excel(writer, sheet_name="network_efficiency_fig", index=False)
    report = ["# Benchmark summary", "", f"Workbook: `{workbook.name}`", ""]
    if not semantickitti_representative.empty:
        report += [
            "## SemanticKITTI representative operating point (alpha=8, k=24)",
            "",
            semantickitti_representative.to_markdown(index=False),
            "",
            "Speedup is total latency of the exact CUDA k-d tree divided by "
            "FlashKNN total latency on the same GPU and query mode.",
            "",
        ]
    if not main_table.empty: report += ["## Query main table", "", main_table.to_markdown(index=False), ""]
    if not network_table.empty: report += ["## Network latency", "", network_table.to_markdown(index=False), ""]
    (args.output_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(workbook)


if __name__ == "__main__":
    main()
