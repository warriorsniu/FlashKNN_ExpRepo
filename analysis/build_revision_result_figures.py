#!/usr/bin/env python3
"""Build the self-contained figures and CSV tables used by the revision booklet."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import network_rows, query_rows


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = PROJECT / "PaperRevise" / "revision_results_assets"

L20 = ROOT / "results/L20/l20_complete_20260807"
RTX = ROOT / "results/RTX3090/rtx3090_complete_20260808"
SEMANTIC = {
    "L20": L20 / "query/semantickitti.json",
    "RTX 3090": ROOT / (
        "results/RTX3090/rtx3090_semantickitti_unifiedk_alpha4ivf_20260818/"
        "query/semantickitti.json"
    ),
}
S3DIS = {
    "L20": L20 / "query/s3dis_sample_part.json",
    "RTX 3090": RTX / "query/s3dis_sample_part.json",
}

COLORS = {
    "flashknn": "#1f77b4",
    "cuda_kdtree": "#d62728",
    "cukd": "#d62728",
    "flann_cuda": "#2ca02c",
    "faiss_ivf": "#9467bd",
    "faiss_flat": "#8c564b",
    "nanoflann": "#7f7f7f",
}
LABELS = {
    "flashknn": "FlashKNN",
    "cuda_kdtree": "cudaKDTree",
    "cukd": "cudaKDTree",
    "flann_cuda": "FLANN-CUDA",
    "faiss_ivf": "FAISS-IVF",
    "faiss_flat": "FAISS-Flat",
    "nanoflann": "nanoflann",
}


def payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def grouped_query(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for short_gpu, path in paths.items():
        frame = pd.DataFrame(query_rows(path, payload(path)))
        frame["short_gpu"] = short_gpu
        rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    group = ["short_gpu", "dataset", "scope", "mode", "k", "method", "alpha"]
    return raw.groupby(group, dropna=False).agg(
        samples=("sample", "count"),
        support_points=("num_support", "mean"),
        query_points=("num_query", "mean"),
        construction_ms=("construction_ms", "mean"),
        query_ms=("query_ms", "mean"),
        total_ms=("total_ms", "mean"),
        recall=("recall", "mean"),
    ).reset_index()


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.28)
    ax.set_xticks([8, 16, 24, 32, 48, 64])


def plot_s3dis(frame: pd.DataFrame) -> None:
    methods = ["flashknn", "cuda_kdtree", "flann_cuda", "faiss_ivf"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    for row, gpu in enumerate(("RTX 3090", "L20")):
        for col, mode in enumerate(("pre", "post")):
            ax = axes[row, col]
            subset = frame[(frame.short_gpu == gpu) & (frame["mode"] == mode)]
            for method in methods:
                data = subset[subset.method == method].sort_values("k")
                if data.empty:
                    continue
                ax.plot(data.k, data.query_ms, marker="o", linewidth=1.7,
                        color=COLORS[method], label=LABELS[method])
            ax.set_yscale("log")
            style_axes(ax, f"{gpu}, {mode}-downsampling query", "Query latency (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("S3DIS: 250k-support query latency (81 rooms)", y=0.995, fontsize=12)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(OUT / "s3dis_query_latency_cross_gpu.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_semantic_latency(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    for row, gpu in enumerate(("RTX 3090", "L20")):
        for col, mode in enumerate(("pre", "post")):
            ax = axes[row, col]
            subset = frame[(frame.short_gpu == gpu) & (frame["mode"] == mode)]
            specs = [
                ("flashknn", 4, "FlashKNN, alpha=4"),
                ("cukd", None, "cudaKDTree"),
                ("faiss_ivf", None, "FAISS-IVF"),
            ]
            for method, alpha, label in specs:
                data = subset[subset.method == method]
                if alpha is not None:
                    data = data[data.alpha == alpha]
                data = data.sort_values("k")
                if data.empty:
                    continue
                ax.plot(data.k, data.total_ms, marker="o", linewidth=1.8,
                        color=COLORS[method], label=label)
            ax.set_yscale("log")
            style_axes(ax, f"{gpu}, {mode}", "Construction + query (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("SemanticKITTI: total neighbor-search latency (110 frames)", y=0.995, fontsize=12)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(OUT / "semantickitti_latency_cross_gpu.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_semantic_recall(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True, sharey=True)
    alpha_colors = {4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c", 32: "#d62728"}
    for row, gpu in enumerate(("RTX 3090", "L20")):
        for col, mode in enumerate(("pre", "post")):
            ax = axes[row, col]
            subset = frame[(frame.short_gpu == gpu) & (frame["mode"] == mode)]
            for alpha in (4, 8, 16, 32):
                data = subset[(subset.method == "flashknn") & (subset.alpha == alpha)].sort_values("k")
                ax.plot(data.k, data.recall * 100, marker="o", linewidth=1.6,
                        color=alpha_colors[alpha], label=f"FlashKNN alpha={alpha}")
            ivf = subset[subset.method == "faiss_ivf"].sort_values("k")
            ax.plot(ivf.k, ivf.recall * 100, marker="s", linewidth=1.5,
                    linestyle="--", color="#9467bd", label="FAISS-IVF (target alpha=4)")
            style_axes(ax, f"{gpu}, {mode}", "Recall vs exact (%)")
            ax.set_ylim(94, 100.1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("SemanticKITTI: accuracy across k and neighborhood scale", y=0.995, fontsize=12)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    fig.savefig(OUT / "semantickitti_recall_cross_gpu.pdf", bbox_inches="tight")
    plt.close(fig)


def semantic_memory_rows() -> pd.DataFrame:
    rows = []
    for gpu, path in SEMANTIC.items():
        data = payload(path)
        for sample in data["samples"]:
            for flash in sample["flashknn"]:
                value = flash.get("peak_incremental_allocated_bytes")
                if value is None:
                    continue
                rows.append({
                    "short_gpu": gpu,
                    "sample": sample["sample"],
                    "mode": sample["mode"],
                    "k": sample["k"],
                    "alpha": flash["alpha"],
                    "peak_mib": float(value) / (1024 ** 2),
                })
    raw = pd.DataFrame(rows)
    return raw.groupby(["short_gpu", "mode", "k", "alpha"]).agg(
        samples=("sample", "count"), peak_mib=("peak_mib", "mean")
    ).reset_index()


def plot_semantic_memory(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    alpha_colors = {4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c", 32: "#d62728"}
    for row, gpu in enumerate(("RTX 3090", "L20")):
        for col, mode in enumerate(("pre", "post")):
            ax = axes[row, col]
            subset = frame[(frame.short_gpu == gpu) & (frame["mode"] == mode)]
            for alpha in (4, 8, 16, 32):
                data = subset[subset.alpha == alpha].sort_values("k")
                ax.plot(data.k, data.peak_mib, marker="o", linewidth=1.7,
                        color=alpha_colors[alpha], label=f"alpha={alpha}")
            style_axes(ax, f"{gpu}, {mode}", "Peak incremental allocation (MiB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("SemanticKITTI: FlashKNN incremental GPU memory", y=0.995, fontsize=12)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(OUT / "semantickitti_memory_cross_gpu.pdf", bbox_inches="tight")
    plt.close(fig)


def grouped_network() -> pd.DataFrame:
    rows = []
    for gpu, root in (("L20", L20), ("RTX 3090", RTX)):
        for path in sorted((root / "network").glob("*.json")):
            values = network_rows(path, payload(path))
            for value in values:
                value["short_gpu"] = gpu
            rows.extend(values)
    raw = pd.DataFrame(rows)
    group = ["short_gpu", "dataset", "model", "backend"]
    return raw.groupby(group).agg(
        samples=("sample", "count"), points=("num_down", "mean"),
        preprocessing_ms=("preprocessing_ms", "mean"),
        network_ms=("network_ms", "mean"), end_to_end_ms=("end_to_end_ms", "mean")
    ).reset_index()


def plot_network(frame: pd.DataFrame) -> None:
    semantic = frame[frame.dataset == "SemanticKITTI"].copy()
    semantic["label"] = semantic.apply(
        lambda r: (
            f"{str(r.model).title()}\n{r.backend}"
            if r.backend != "native" else str(r.model).replace("minkunet34c", "MinkUNet34C")
        ), axis=1)
    labels = [
        "dela\ncpu_kdtree", "dela\nflashknn", "deepla\ncpu_kdtree", "deepla\nflashknn",
        "spunet", "minkunet34c", "ptv3", "octformer",
    ]
    display = ["DeLA\nCPU", "DeLA\nFlashKNN", "DeepLA\nCPU", "DeepLA\nFlashKNN",
               "SPUNet", "MinkUNet34C", "PTv3", "OctFormer"]
    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for offset, gpu, color in ((-width/2, "RTX 3090", "#4c78a8"), (width/2, "L20", "#f58518")):
        vals = []
        for key in labels:
            if "\n" in key:
                model, backend = key.split("\n")
                hit = semantic[(semantic.short_gpu == gpu) & (semantic.model.str.lower() == model) &
                               (semantic.backend == backend)]
            else:
                hit = semantic[(semantic.short_gpu == gpu) & (semantic.model.str.lower() == key) &
                               (semantic.backend == "native")]
            vals.append(float(hit.end_to_end_ms.iloc[0]) if not hit.empty else np.nan)
        ax.bar(x + offset, vals, width, label=gpu, color=color)
    ax.set_yscale("log")
    ax.set_ylabel("End-to-end latency (ms / frame, log scale)")
    ax.set_xticks(x, display)
    ax.grid(True, axis="y", which="both", linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("SemanticKITTI network latency (22 frames, alpha=4 for FlashKNN)")
    fig.tight_layout()
    fig.savefig(OUT / "semantickitti_network_latency_cross_gpu.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_final_figures() -> None:
    sources = {
        "core_memory_sorting.pdf": ROOT / "analysis/output/rtx3090_ablation_final_20260810/memory_sorting.pdf",
        "core_candidate_skip.pdf": ROOT / "analysis/output/rtx3090_ablation_final_20260810/candidate_skip.pdf",
        "thread_grouping.pdf": ROOT / (
            "analysis/output/rtx3090_thread_grouping_balanced_final_v2_20260811/"
            "thread_grouping/thread_grouping.pdf"
        ),
        "adaptive_neighborhood.pdf": ROOT / (
            "analysis/output/rtx3090_adaptive_neighborhood_final_v2_20260818/"
            "adaptive_neighborhood_latency_recall.pdf"
        ),
        "semantic_boundary.pdf": ROOT / (
            "results/RTX3090/rtx3090_s3dis_semantic_boundary_20260818/"
            "semantic_boundary_accuracy_miou.pdf"
        ),
        "l20_query_scaling.png": ROOT / "analysis/output/l20_complete_20260807/speedup_of_query_under_different_number_of_point.png",
        "l20_construction_scaling.png": ROOT / "analysis/output/l20_complete_20260807/speedup_of_construction_under_different_number_of_point.png",
        "l20_network_scaling.png": ROOT / "analysis/output/l20_complete_20260807/network_efficiency_comparison.png",
        "rtx3090_query_scaling.png": ROOT / "analysis/output/rtx3090_complete_20260808/speedup_of_query_under_different_number_of_point.png",
        "rtx3090_construction_scaling.png": ROOT / "analysis/output/rtx3090_complete_20260808/speedup_of_construction_under_different_number_of_point.png",
        "rtx3090_network_scaling.png": ROOT / "analysis/output/rtx3090_complete_20260808/network_efficiency_comparison.png",
    }
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.7, "pdf.fonttype": 42})

    s3dis = grouped_query(S3DIS)
    semantic = grouped_query(SEMANTIC)
    memory = semantic_memory_rows()
    network = grouped_network()

    s3dis.to_csv(OUT / "s3dis_summary.csv", index=False)
    semantic.to_csv(OUT / "semantickitti_summary.csv", index=False)
    memory.to_csv(OUT / "semantickitti_memory_summary.csv", index=False)
    network.to_csv(OUT / "network_summary.csv", index=False)

    plot_s3dis(s3dis)
    plot_semantic_latency(semantic)
    plot_semantic_recall(semantic)
    plot_semantic_memory(memory)
    plot_network(network)
    copy_final_figures()
    print(f"Wrote revision figures and CSV tables to {OUT}")


if __name__ == "__main__":
    main()
