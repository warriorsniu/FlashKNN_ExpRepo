#!/usr/bin/env python3
"""Runtime verification for source-built GPU FAISS, including torch tensors."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-arch", required=True)
    args = parser.parse_args()

    import faiss
    import faiss.contrib.torch_utils  # noqa: F401 - installs tensor adapters
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    actual = f"{major}{minor}"
    if actual != args.expected_arch.replace(".", ""):
        raise RuntimeError(
            f"visible GPU is sm_{actual}, but FAISS was built for sm_{args.expected_arch}"
        )

    device = torch.device("cuda:0")
    database = torch.rand((4096, 3), dtype=torch.float32, device=device)
    queries = database[:256].contiguous()
    resources = faiss.StandardGpuResources()

    flat = faiss.GpuIndexFlatL2(resources, 3)
    flat.add(database)
    distances, indices = flat.search(queries, 32)
    torch.cuda.synchronize()
    if indices.shape != (256, 32) or not torch.equal(indices[:, 0], torch.arange(256, device=device)):
        raise RuntimeError("FAISS GPU Flat torch-tensor smoke test returned invalid results")

    cpu_ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 64)
    ivf = faiss.index_cpu_to_gpu(resources, 0, cpu_ivf)
    ivf.train(database)
    ivf.add(database)
    ivf.nprobe = 8
    _, ivf_indices = ivf.search(queries, 32)
    torch.cuda.synchronize()
    if ivf_indices.shape != (256, 32):
        raise RuntimeError("FAISS GPU IVF-Flat smoke test returned invalid results")

    print(
        f"FAISS {faiss.__version__} GPU Flat/IVF smoke OK on "
        f"{torch.cuda.get_device_name(0)} (sm_{actual})"
    )


if __name__ == "__main__":
    main()
