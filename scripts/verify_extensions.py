#!/usr/bin/env python3
"""Run the bundled query extensions once after native compilation."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-arch", required=True)
    args = parser.parse_args()

    import torch
    import Cukd.CuFun as cukd
    try:
        from FlashKNN import FlashKNN
    except ImportError:  # in-place source-tree validation before wheel install
        from functions import FlashKNN
    from PyTorchCudaFlann import FlannCudaKnnQueryTorchWrapper
    from PyTorchNanoFlann import NanoFlannQuery

    capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
    if capability != args.expected_arch:
        raise SystemExit(
            f"Visible GPU changed during installation: expected {args.expected_arch}, got {capability}"
        )

    torch.manual_seed(47)
    xyz = torch.rand((1024, 3), device="cuda", dtype=torch.float32).contiguous()
    grid = torch.floor(xyz / 0.02).long().contiguous()
    batch = torch.zeros(len(xyz), device="cuda", dtype=torch.long)
    flash_indices = FlashKNN(num_nbr=8, num_down=2).query(grid, batch, xyz)

    exact_indices = torch.empty((len(xyz), 8), device="cuda", dtype=torch.int32)
    exact_distances = torch.empty((len(xyz), 8), device="cuda", dtype=torch.float32)
    cukd.CukdKnnQueryTorch(
        xyz, xyz, 8, exact_indices, exact_distances, torch.zeros(2), True
    )
    flann_indices, _ = FlannCudaKnnQueryTorchWrapper(
        xyz, xyz, 8, torch.zeros(2), False
    )
    nano_indices, _ = NanoFlannQuery(xyz, xyz, 8, False, torch.zeros(2))
    torch.cuda.synchronize()

    expected_shape = (len(xyz), 8)
    for name, indices in (("FlashKNN", flash_indices), ("cudaKDTree", exact_indices),
                          ("FLANN-CUDA", flann_indices), ("nanoflann", nano_indices)):
        if tuple(indices.shape) != expected_shape:
            raise SystemExit(f"{name} returned {tuple(indices.shape)}, expected {expected_shape}")

    # cuKD allocates power-of-two candidate storage but the historical API
    # uses an arbitrary runtime k.  Exercise both exact powers of two and the
    # paper's k=24/48 cases so an api.cu/knn.h mismatch cannot pass packaging.
    support_reference = torch.rand((1024, 3), device="cuda", dtype=torch.float32)
    query = torch.rand((97, 3), device="cuda", dtype=torch.float32).contiguous()
    reference_distances = torch.cdist(query, support_reference).square()
    for k in (8, 16, 24, 32, 48, 64):
        support = support_reference.clone().contiguous()
        indices = torch.empty((len(query), k), device="cuda", dtype=torch.int32)
        distances = torch.empty((len(query), k), device="cuda", dtype=torch.float32)
        cukd.CukdKnnQueryTorch(
            support, query, k, indices, distances, torch.zeros(2), False
        )
        expected = reference_distances.topk(k, dim=1, largest=False).values
        actual = distances.sort(dim=1).values
        if not torch.allclose(actual, expected, rtol=2e-5, atol=2e-6):
            error = (actual - expected).abs().max().item()
            raise SystemExit(f"cudaKDTree exactness failed for k={k}: max error {error:.3g}")
    print(f"Native extension runtime check passed on compute capability {capability}")


if __name__ == "__main__":
    main()
