"""Historical FLANN-CUDA and nanoflann baselines used by the paper tables."""

from __future__ import annotations

from typing import Any


def _timing(info: Any, method: str, query_type: str) -> dict[str, Any]:
    construction = float(info[0])
    query = float(info[1])
    return {
        "construction_s": construction,
        "query_s": query,
        "total_s": construction + query,
        "method": method,
        "query_type": query_type,
    }


def benchmark_legacy_methods(
    torch: Any,
    support: Any,
    query: Any,
    k: int,
    warmups: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    """Run the two wrappers with the same timing boundary as EdgeAggr.

    FLANN-CUDA receives CUDA tensors and reports its internal tree-build and
    query times. nanoflann copies CUDA inputs to CPU before starting its two
    timers, exactly matching the original ``exp_query*.py`` protocol; that
    device transfer is therefore deliberately excluded from the paper table.
    """
    from PyTorchCudaFlann import FlannCudaKnnQueryTorchWrapper
    from PyTorchNanoFlann import NanoFlannQuery

    query_type = "query" if support.data_ptr() == query.data_ptr() else "selected_query"
    results: dict[str, Any] = {}
    for name, function in (
        ("flann_cuda", FlannCudaKnnQueryTorchWrapper),
        ("nanoflann", NanoFlannQuery),
    ):
        timings = []
        indices = None
        for iteration in range(warmups + repeats):
            torch.cuda.synchronize()
            info = torch.zeros(2)
            if name == "flann_cuda":
                indices, _ = function(query, support, k, info, False)
            else:
                indices, _ = function(query, support, k, False, info)
            torch.cuda.synchronize()
            if iteration >= warmups:
                timings.append(_timing(info, name, query_type))
        results[name] = {"timings": timings, "indices": indices}
    return results
