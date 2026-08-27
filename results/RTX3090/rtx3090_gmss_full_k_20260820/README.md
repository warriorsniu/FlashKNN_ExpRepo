# FlashKNN-GMSS full-k design-ablation completion

This directed run completes the missing GMSS curve in the RTX 3090 final-kernel
design ablation. It is merged with the retained six-variant result only during
analysis; neither raw JSON is rewritten.

## Protocol and provenance

- Physical RTX 3090 GPU 5, UUID
  `GPU-97038723-1a8a-70df-e5b7-52a98de11890`, matching the retained formal
  design-ablation GPU.
- The same 81 deterministic S3DIS pre-query 250,000-point crops, 0.02 m
  voxelization, alpha=4 and k={8,16,24,32,40,48,56,64}.
- One idle GPU, 5 warmups and 20 recorded repetitions for every room-k pair;
  648 unique records. The starting compute-process snapshot is empty.
- GMSS uses global-memory support traversal, one CUDA thread per query, and
  serial max-heap top-k selection. Source and extension hashes are embedded in
  `ablation/s3dis_gmss_ablation.json`.

## GMSS results

SD and two-sided Student-t 95% CI use the 81 per-room timing means.

| k | Query mean (ms) | Room SD (ms) | 95% CI (ms) | Recall vs cudaKDTree |
|---:|---:|---:|---:|---:|
| 8  | 0.7206 | 0.0356 | ±0.0079 | 0.999806 |
| 16 | 1.3332 | 0.0432 | ±0.0096 | 0.999893 |
| 24 | 2.4819 | 0.0682 | ±0.0151 | 0.999930 |
| 32 | 3.7846 | 0.1269 | ±0.0281 | 0.999900 |
| 40 | 5.5095 | 0.1964 | ±0.0434 | 0.999757 |
| 48 | 7.1985 | 0.2728 | ±0.0603 | 0.999380 |
| 56 | 8.8598 | 0.3632 | ±0.0803 | 0.998174 |
| 64 | 10.2500 | 0.4676 | ±0.1034 | 0.994525 |

## Full memory--selection factorial

| k | SMPS | SMSS | GMPS | GMSS |
|---:|---:|---:|---:|---:|
| 8  | 0.794 | 0.995 | 0.970 | **0.721** |
| 16 | **1.098** | 1.644 | 1.332 | 1.333 |
| 24 | **1.583** | 2.700 | 1.916 | 2.482 |
| 32 | **1.712** | 3.815 | 2.100 | 3.785 |
| 40 | **2.374** | 5.713 | 2.807 | 5.509 |
| 48 | **2.533** | 7.596 | 2.983 | 7.198 |
| 56 | **2.701** | 9.443 | 3.152 | 8.860 |
| 64 | **2.860** | 10.929 | 3.315 | 10.250 |

GMSS is 9.3% faster than SMPS at k=8, where parallel coordination dominates a
very small selection problem. SMPS becomes fastest from k=16 onward, and its
speedup over GMSS grows from 1.21x at k=16 to 3.58x at k=64. Comparing SMPS
with GMPS isolates the shared-support path under parallel selection and gives a
1.16--1.23x speedup over the tested k values. The serial curves are close for
k>=32, with GMSS slightly faster than SMSS, showing that shared staging alone
does not compensate for serial selection overhead.

The merged Times New Roman figures and summary are in
`analysis/output/rtx3090_ablation_final_with_gmss_20260820/`. The manuscript
assets `core_memory_sorting.pdf` and `core_candidate_skip.pdf` were regenerated;
their fonts are embedded Times New Roman.
