# Final result selection

This registry prevents obsolete kernels, smoke runs, and superseded protocols
from being cited as paper results. A directory not listed under **Retained** is
not a supported paper result, even if it exists in an older Git revision.

## Retained L20 results

| Directory | Status | Intended use |
| --- | --- | --- |
| `L20/l20_stale_candidate_fix_complete_20260824/` | Final canonical query/network/memory | Current `2a697272` result pack: S3DIS 972/272, SemanticKITTI 1320, complete network matrix, memory 162 and unchanged ball-query baseline; full validator passed. |
| `L20/l20_stale_candidate_fix_ncu_20260825/` | Final current-source NCU | Current extension k=32 cudaKDTree/SMPS/GMSS reports with source/extension/report hashes; SMPS 31.33 threads/instruction, 96.996% uniform and 1.393 ms. |
| `L20/l20_complete_20260807/` | Superseded canonical reference | Pre-stale-candidate-fix canonical pack retained only for paired latency diffs and preserved third-party baseline provenance. |
| `L20/l20_s3dis_final_kernel_refresh_20260819/` | Final raw refresh | Direct source for final-kernel S3DIS fixed/full query and DeLA hierarchy results. |
| `L20/l20_ncu/` | Retained pre-branchless NCU reference | k=32 cudaKDTree/SMPS/GMSS reports used as the strict before-fix reference; not current source. |
| `L20/l20_branchless_sort_ncu_20260824/` | Superseded intermediate NCU | Commit `49ecd907` branchless evidence before upper-scratch stale-candidate repair; retained for A/B comparison only. |
| `L20/l20_s3dis_memory_k32_20260819/` | Final memory result | Same 81 fixed-250k crops, pre/post, k=32; FlashKNN/cudaKDTree/FAISS Flat/matched IVF peak incremental allocation with room SD/95% CI. |
| `L20/l20_semantickitti_six_k_alpha4_20260818/` | Final | Direct source for the canonical six-k SemanticKITTI query result. |
| `L20/l20_bitonic_generated_validation/` | Final algorithm, 5/20 protocol | Direct source for S3DIS k=24/32/48 optimized-kernel paired timings. |
| `L20/l20_bitonic_generated_k8_k16_k64_validation/` | Final algorithm, 5/20 protocol | Direct source for S3DIS k=8/16/64 optimized-kernel paired timings. |
| `L20/l20_lidar_network_alpha4_20260809/` | Final | Alpha=4 DeLA/DeepLA SemanticKITTI hierarchy results. |
| `L20/l20_lidar_network_alpha4_rerun_20260809/` | Final diagnostic override | Two documented outlier remeasurements used by the preceding result. |
| `L20/l20_ball_query_20260807/` | Final independent baseline | Ball-query comparison and one-room radius sweep. |
| `L20/l20_arkade_20260807/` | Final independent baseline | Arkade/TrueKNN RT-core comparison. |

The canonical L20 fixed-size, full-room and SemanticKITTI query files now use the verified `2a697272` production kernel and unified 3 warm-up/10-repeat protocol. Historical FLANN-CUDA/nanoflann fields and the kernel-independent ball query were preserved with stable deep hashes; all network files were refreshed with 10/30 timing. Direct current-build outputs, canonical merged files, build/GPU audit and validator output are in `L20/l20_stale_candidate_fix_complete_20260824/`.

The administrator enabled performance-counter access and all three current-source NCU profiles completed on physical GPU 0. `L20/l20_stale_candidate_fix_ncu_20260825/` is the final canonical NCU directory; it contains raw/source CSV, inspectable reports, complete requested metrics, production source hashes, extension hash and GPU provenance. `L20/l20_ncu/` and `L20/l20_branchless_sort_ncu_20260824/` remain only as pre-branchless and pre-correctness-fix A/B references.

## Retained RTX 3090 results

| Directory | Status | Intended use |
| --- | --- | --- |
| `RTX3090/rtx3090_semantickitti_alpha8_ivf_20260825/` | Final paper-default LiDAR query | Current production kernel, 110 frames, pre/post, six k values, alpha=8-only FlashKNN plus paired cudaKDTree/FAISS Flat and IVF recalibrated per record to alpha=8 recall; 1320 unique records and directed validator pass. |
| `RTX3090/rtx3090_lidar_network_alpha8_final_20260825/` | Final paper-default LiDAR network latency | DeLA/DeepLA, 22 stratified frames, paired CPU-KDTree/FlashKNN hierarchies, alpha=8, single GPU, 10 warmups/30 repeats. |
| `RTX3090/rtx3090_semantickitti_checkpoint_alpha8_final_20260825/` | Final current-kernel checkpoint compatibility | DeLA/DeepLA seeds 47--49 on sequence 08 with alpha=8 and the current production kernel; source, extension, checkpoint, environment and result hashes retained. |
| `RTX3090/rtx3090_final_corrected_20260824/` | Final corrected S3DIS canonical | Duplicate-safe production-kernel S3DIS fixed/full query pack and matched Pointcept ball-query reference. Its SemanticKITTI alpha=4 component is retained as a faster alternative, not the paper-default LiDAR table. |
| `RTX3090/rtx3090_pytorch3d_ball_query_20260825/` | Final matched operator extension | PyTorch3D 0.7.9 ball query on the same 81 S3DIS crops, pre/post modes, and Pointcept radii; 486 records and the cross-operator validator pass. |
| `RTX3090/rtx3090_complete_20260808/` | Superseded canonical reference | Historical main pack retained for provenance only; do not reuse its pre-duplicate-fix FlashKNN recall values. |
| `RTX3090/rtx3090_s3dis_final_kernel_refresh_20260819/` | Final direct source | Paired 3/10 S3DIS fixed/full query refresh and 10/30 DeLA S3DIS refresh on the same idle physical GPU. |
| `RTX3090/rtx3090_s3dis_memory_k32_20260819/` | Final memory result | Same 81 fixed-250k crops, pre/post, k=32; FlashKNN/cudaKDTree/FAISS Flat/matched IVF peak incremental allocation with room SD/95% CI. |
| `RTX3090/rtx3090_semantickitti_unifiedk_alpha4ivf_20260818/` | Superseded paper default; retained alternative | Six-k alpha=4/matched-IVF result retained for the faster LiDAR operating point and historical comparison. |
| `RTX3090/rtx3090_ablation_final_20260810/` | Final | SMPS/SMSS/GMPS and candidate-storage/skip ablation. |
| `RTX3090/rtx3090_thread_grouping_balanced_final_v2_20260811/` | Final | Balanced Fixed-8/16/32 versus Adaptive thread grouping. |
| `RTX3090/rtx3090_adaptive_neighborhood_final_v2_20260818/` | Final | Fixed 3x3x3 versus adaptive octree neighborhood versus cudaKDTree. |
| `RTX3090/rtx3090_s3dis_semantic_boundary_20260818/` | Final | Performance-matched checkpoint semantic-boundary analysis. |
| `RTX3090/rtx3090_gmss_full_k_20260820/` | Final directed ablation completion | Same physical GPU and 81-room 5/20 protocol as the retained design ablation; supplies the GMSS curve without rewriting the original six-variant result. |
| `RTX3090/rtx3090_torch_knnquery_gmss_k16_20260820/` | Final directed diagnostic | Same 81 pre-250k crops at k=16; production SMPS versus controlled GMSS and upstream `torch_knnquery` public/core paths. |
| `RTX3090/rtx3090_ncu_final_kernel_20260824/` | Final RTX 3090 NCU profile | Current production SMPS, GMSS, and cudaKDTree at S3DIS pre-250k, k=32, with raw/summary CSV and provenance. |
| `RTX3090/rtx3090_semantickitti_training_wallclock_20260827/` | Retained historical training evidence | Six paired DeLA/DeepLA-24 seed observations under the documented two-job/four-job concurrent regimes. |

The corrected RTX 3090 pack remains authoritative for S3DIS after the
stale/duplicate-candidate repair. SemanticKITTI paper-facing results now use
the separately validated alpha=8 query and network directories above; alpha=4
remains a documented faster alternative. PyTorch3D is stored separately because
it extends the common-operator comparison without changing the production FlashKNN result pack. Its radii,
crops, GPU, warm-ups, repeats, and record identities were checked against the
Pointcept file in the corrected canonical directory.

The 2026-08-19 RTX refresh replaced FlashKNN and the paired cudaKDTree control
in `query/s3dis_sample_part.json` and `query/s3dis_full_k32.json`, while
preserving all unaffected third-party baselines. It also replaced
`network/dela_s3dis.json` as one complete paired file. The raw refresh and
canonical timing overrides record the source commit, source/extension hashes,
production flags, physical GPU UUID, 3/10 or 10/30 protocol, and co-tenant
snapshots.

## Removed result classes

The 2026-08-19 cleanup removed the following classes of results:

- all smoke and co-tenant/idle probe directories;
- explicitly invalid concurrent SemanticKITTI output;
- obsolete three-k and pre-correctness-fix SemanticKITTI runs, regardless of alpha;
- fixed-order or incorrectly balanced thread-grouping runs superseded by balanced final v2;
- adaptive-neighborhood prototypes superseded by final v2;
- construction-fix and partial/top-p sorting prototypes that do not implement the final algorithm;
- the incomplete old L20 full-room batch and duplicate intermediate network matrices;
- NCU CSV files captured before the final sorting logic.

Tracked L20 deletions remain recoverable from Git history. Untracked local
smoke/prototype RTX directories were intentionally discarded and must be
regenerated if they are ever needed for debugging.
