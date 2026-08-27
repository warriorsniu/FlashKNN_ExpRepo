# S3DIS pre-250k fixed-grid execution diagnostic

This directed result compares the production FlashKNN execution path with its
historical GMSS path and the upstream `torch_knnquery` implementation. It is a
method-boundary diagnostic at one representative k, not a replacement for the
six-k main table.

## Protocol

- RTX 3090 physical GPU 0 (UUID `GPU-c568a174-078b-fcba-f91d-00dd48a900c7`),
  PyTorch 2.7.1+cu118 and CUDA 11.8; single GPU and no external compute process.
- The same 81 deterministic S3DIS pre-query 250,000-point crops used by the
  main table; 0.02 m input voxelization, alpha=4, fixed 3x3x3 neighborhood,
  k=16, 3 warmups and 10 recorded repeats.
- Upstream `torch_knnquery` commit `947957e`. Its native k limit is 20, so this
  diagnostic deliberately uses k=16 rather than modifying the algorithm to
  fabricate the paper's full k sweep.
- The only upstream patch changes five deprecated PyTorch dispatch calls from
  `Tensor.type()` to `Tensor.scalar_type()`; candidate traversal and selection
  are unchanged. No radius cutoff is applied.
- Query timing starts from CUDA-ready tensors. The upstream public-API row
  includes ray masking, compaction and output allocation; the core row measures
  only `query_along_ray`, i.e. its 3x3x3 scan and sequential local top-k update.

## Query latency

Values are the mean of the 81 per-room timing means. SD and 95% CI use rooms as
the statistical unit.

| Path | Mean (ms) | Room SD (ms) | 95% CI (ms) | Production speedup |
| --- | ---: | ---: | ---: | ---: |
| `torch_knnquery` public API | 6.4985 | 0.2441 | ±0.0540 | 6.115x |
| `torch_knnquery` core kernel | 2.5398 | 0.2233 | ±0.0494 | 2.390x |
| FlashKNN-GMSS | 1.3710 | 0.0450 | ±0.0099 | 1.290x |
| FlashKNN production SMPS | 1.0627 | 0.0542 | ±0.0120 | 1.000x |
| Exact cudaKDTree | 3.7379 | 0.6241 | ±0.1380 | 3.518x |

GMSS is the controlled `torch_knnquery`-style ablation: it preserves the
FlashKNN voxel graph, sorted layout and fixed candidate set, but reads support
points from global memory and assigns one CUDA thread to each query with a
serial max-heap top-k update. It must not be relabeled as the external
`torch_knnquery` result. The difference between GMSS and the upstream core also
contains grid representation, point-list construction and indexing effects.

## Correctness and truncation checks

| Path | Mean set recall vs cudaKDTree | Mean valid-neighbor fraction |
| --- | ---: | ---: |
| `torch_knnquery` | 0.999872 | 0.999985 |
| FlashKNN-GMSS | 0.999893 | 1.000000 |
| FlashKNN production SMPS | 0.999894 | 1.000000 |

The external public and core paths produced identical indices in all 81 rooms.
The largest observed parent-voxel occupancy was 58, below the configured
128-point capacity, so no voxel point list was truncated. The minimum per-room
valid-neighbor fraction was 0.9999005; the few missing entries occur where the
fixed stencil contains fewer than k supports and are reflected in recall.

Raw per-repeat timings, crop centers, occupancy checks, hashes and co-tenant
snapshots are stored in `torch_knnquery_s3dis_pre250k_k16.json`.
