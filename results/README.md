# Retained experiment results

This directory contains the final result sets used for the FlashKNN paper and
revision. Superseded runs, smoke tests, compiler experiments, and intermediate
result-selection notes are not included.

## NVIDIA L20

| Directory | Contents |
|---|---|
| `l20_final_20260824` | Final S3DIS query, full-room scaling, SemanticKITTI query, network-latency, ball-query, and S3DIS memory results |
| `l20_ncu_final_20260825` | Final Nsight Compute profiles for FlashKNN SMPS, the GMSS control, and cudaKDTree |

## NVIDIA RTX 3090

| Directory | Contents |
|---|---|
| `rtx3090_final_20260825` | Final S3DIS query, full-room scaling, ball-query, network-latency, and SemanticKITTI alpha=8 results |
| `rtx3090_ablation_final_20260810` | Sorting, candidate-storage, and skip ablations |
| `rtx3090_gmss_full_k_20260820` | GMSS results across the complete k sweep |
| `rtx3090_thread_grouping_balanced_final_v2_20260811` | Fixed and adaptive thread-grouping ablation |
| `rtx3090_adaptive_neighborhood_final_v2_20260818` | Fixed-grid and adaptive-neighborhood ablation with candidate statistics |
| `rtx3090_s3dis_memory_k32_20260819` | Four-method S3DIS memory comparison |
| `rtx3090_pytorch3d_ball_query_20260825` | PyTorch3D ball-query comparison |
| `rtx3090_s3dis_semantic_boundary_20260818` | Semantic-boundary accuracy and mIoU analysis |
| `rtx3090_semantickitti_checkpoint_alpha8_final_20260825` | SemanticKITTI checkpoint evaluations at alpha=8 |
| `rtx3090_semantickitti_training_wallclock_20260827` | Concurrent-training wall-clock measurements |
| `rtx3090_torch_knnquery_gmss_k16_20260820` | Direct fixed-grid execution diagnostic at k=16 |
| `rtx3090_ncu_final_kernel_20260824` | Final Nsight Compute profiles for FlashKNN SMPS, the GMSS control, and cudaKDTree |

## File conventions

Benchmark JSON files contain run metadata and per-room or per-frame records.
Timing arrays store the recorded repetitions in seconds unless the field name
states another unit. CSV files contain tabular summaries or profiler exports.
Each result-specific README documents any additional fields and the exact
profiling workload.

The metadata embedded in the result files records the GPU, software versions,
benchmark parameters, and relevant source or extension hashes. These fields
should be retained when generating derived tables or figures.

## Rebuilding analysis outputs

The primary platform result packs can be analyzed with:

```bash
python analysis/analyze_results.py \
  --results results/L20/l20_final_20260824 \
  --output-dir analysis/output/l20_final_20260824

python analysis/analyze_results.py \
  --results results/RTX3090/rtx3090_final_20260825 \
  --output-dir analysis/output/rtx3090_final_20260825
```

Specialized analyses are provided in `analysis/` for the ablation, memory,
SemanticKITTI checkpoint, and ball-query result sets. The detailed benchmark
protocols are listed in [`../EXPERIMENTS.md`](../EXPERIMENTS.md).
