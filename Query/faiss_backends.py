"""FAISS backends for the migrated FlashKNN benchmark entry points.

Inputs are CUDA tensors. Consequently, the timed sections contain no CPU-to-
GPU transfer. Flat/IVF index construction and GPU search are recorded in the
same Chinese-key schema as the historical EdgeAggr JSON files.
"""

from __future__ import annotations

import time
import math
from typing import Any

import torch


def import_faiss() -> Any:
    if not hasattr(torch.Tensor, "untyped_storage"):
        torch.Tensor.untyped_storage = torch.Tensor.storage  # type: ignore[attr-defined]
    import faiss
    import faiss.contrib.torch_utils  # noqa: F401
    return faiss


def _flat_index(faiss: Any, resources: Any) -> Any:
    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    config.useFloat16 = False
    return faiss.GpuIndexFlatL2(resources, 3, config)


def _ivf_index(faiss: Any, resources: Any, nlist: int, seed: int) -> Any:
    config = faiss.GpuIndexIVFFlatConfig()
    config.device = 0
    config.useFloat16 = False
    index = faiss.GpuIndexIVFFlat(resources, 3, nlist, faiss.METRIC_L2, config)
    if hasattr(index, "cp"):
        index.cp.seed = seed
    return index


def _search(index: Any, query: torch.Tensor, k: int,
            max_queries_per_launch: int = 131_072) -> tuple[float, torch.Tensor]:
    """Search every query while bounding FAISS temporary GPU memory."""
    distances = torch.empty((len(query), k), dtype=torch.float32, device="cuda")
    indices = torch.empty((len(query), k), dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for begin in range(0, len(query), max_queries_per_launch):
        end = min(len(query), begin + max_queries_per_launch)
        index.search(query[begin:end], k, distances[begin:end], indices[begin:end])
    torch.cuda.synchronize()
    return time.perf_counter() - start, indices


def _recall(approx: torch.Tensor, exact: torch.Tensor, k: int) -> tuple[float, float]:
    merged = torch.cat((approx, exact), dim=1).sort(dim=1).values
    per_query = (merged[:, 1:] == merged[:, :-1]).sum(dim=1).float() / k
    return float(per_query.mean()), float(per_query.min())


@torch.no_grad()
def benchmark_faiss_methods(
    support: torch.Tensor,
    query: torch.Tensor,
    k: int,
    warmups: int = 3,
    repeats: int = 10,
    nlist: int = 1024,
    nprobe: int | None = None,
    target_recall: float | None = None,
    seed: int = 47,
) -> dict[str, Any]:
    """Return exact Flat and approximate IVF-Flat fields for one room."""
    faiss = import_faiss()
    resources = faiss.StandardGpuResources()
    # FAISS requires at least nlist training vectors and recommends roughly 39
    # per centroid. Keep the paper setting (1024) for normal rooms, while making
    # the documented 1k-point smoke test a valid end-to-end preflight.
    max_trained_nlist = max(1, len(support) // 39)
    if nlist > max_trained_nlist:
        nlist = 1 << (max_trained_nlist.bit_length() - 1)
    query_type = "query" if support.data_ptr() == query.data_ptr() else "selected_query"
    heuristic_nprobe = 5 if k <= 32 else 4 if k == 48 else 3

    flat = _flat_index(faiss, resources)
    flat_times = []
    exact_indices = None
    for iteration in range(warmups + repeats):
        flat.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        flat.add(support)
        torch.cuda.synchronize()
        add_seconds = time.perf_counter() - start
        search_seconds, exact_indices = _search(flat, query, k)
        if iteration >= warmups:
            flat_times.append({
                "预处理耗时": add_seconds,
                "查询耗时": search_seconds,
                "查询类型": query_type,
                "查询方法": "faiss_gpu_flat_l2",
            })
    assert exact_indices is not None

    ivf = _ivf_index(faiss, resources, nlist, seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    ivf.train(support)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    ivf.reserveMemory(len(support))
    calibration = []
    calibration_queries = 0
    if nprobe is None and target_recall is not None:
        ivf.add(support)
        # Searching every inverted list can require tens of GiB of temporary
        # storage for full-room queries. IVF is an approximate baseline, so
        # calibrate over the practical range and select the closest recall.
        calibration_limit = min(len(query), 65_536)
        calibration_stride = max(1, math.ceil(len(query) / calibration_limit))
        calibration_query = query[::calibration_stride][:calibration_limit].contiguous()
        calibration_exact = exact_indices[::calibration_stride][:calibration_limit].contiguous()
        calibration_queries = len(calibration_query)
        candidates = tuple(range(1, 17)) + (24, 32, 48, 64)
        for candidate in candidates:
            if candidate > nlist:
                break
            ivf.nprobe = candidate
            _, candidate_indices = _search(ivf, calibration_query, k)
            candidate_mean, candidate_min = _recall(
                candidate_indices, calibration_exact, k
            )
            calibration.append({"nprobe": candidate, "mean_recall": candidate_mean,
                                "min_recall": candidate_min})
            if candidate_mean >= target_recall:
                nprobe = candidate
                break
        if nprobe is None:
            nprobe = min(calibration, key=lambda item: (
                abs(item["mean_recall"] - target_recall), item["nprobe"]
            ))["nprobe"]
        ivf.reset()
    elif nprobe is None:
        nprobe = min(heuristic_nprobe, nlist)
    ivf.nprobe = nprobe
    ivf_times = []
    approx_indices = None
    for iteration in range(warmups + repeats):
        ivf.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        ivf.add(support)
        torch.cuda.synchronize()
        add_seconds = time.perf_counter() - start
        search_seconds, approx_indices = _search(ivf, query, k)
        if iteration >= warmups:
            ivf_times.append({
                "预处理耗时": add_seconds,
                "查询耗时": search_seconds,
                "查询类型": query_type,
                "查询方法": "faiss_gpu_ivf_flat_l2",
            })
    assert approx_indices is not None
    mean_recall, min_recall = _recall(approx_indices, exact_indices, k)
    return {
        "faiss_flat_time_info": flat_times,
        "faiss_ivf_time_info": ivf_times,
        "faiss_ivf_training_time": training_seconds,
        "faiss_ivf_nlist": nlist,
        "faiss_ivf_nprobe": nprobe,
        "faiss_ivf_target_recall": target_recall,
        "faiss_ivf_calibration": calibration,
        "faiss_ivf_calibration_queries": calibration_queries,
        "faiss_ivf_mean_recall": mean_recall,
        "faiss_ivf_min_recall": min_recall,
    }
