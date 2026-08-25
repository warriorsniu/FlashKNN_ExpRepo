# L20 stale-candidate-fix complete refresh

## Background

This result pack refreshes every L20 experiment affected by commit `2a69727257a0ee46d214c4db2e49b7572a10adfe`. The fix clears selectively populated top-P scratch slots before compare-split, preventing stale winners from being reconsidered; recall is computed as duplicate-safe set recall. Repeated center indices remain possible only when a fixed candidate region contains fewer than k points, which was explicitly verified in the six-k smoke gate.

## Environment

All affected measurements used one NVIDIA L20, physical GPU 0, UUID `GPU-78990023-5606-bb80-49bf-8ddfe8683461`, PyTorch 2.7.1+cu128, CUDA toolkit 12.8 and an sm_89-only extension. The extension SHA256 is `95c5566be5d4a646e1300a6c77706f5c690bc6b6ba941a8a72ecfcdae32b6a44`; critical source hashes, GPU snapshots and build logs are in `audit/` and the raw JSON metadata.

## Coverage

| Result | Coverage | Protocol | Status |
| --- | ---: | --- | --- |
| S3DIS fixed 250k | 972 keys: 81 rooms × pre/post × six k | 3 warm-up / 10 repeats | Complete |
| S3DIS full-room | 272 rooms, pre, k=32 | 0.02 m voxelization, Morton input, 3/10 | Complete |
| SemanticKITTI query | 1320 keys: 110 frames × pre/post × six k | alpha=4/8/16/32, matched-IVF alpha=4, 3/10 | Complete |
| Network latency | S3DIS 5×68; SemanticKITTI 6×22 | alpha=4, 10/30 | Complete |
| S3DIS memory | 162 keys: 81 rooms × pre/post | k=32, four methods in one batch | Complete |
| Ball query | 486 historical keys | Kernel-independent, byte-identical carry-over | Preserved |

`audit/coverage_validator.txt` records the successful complete-pack validator. `audit/finalization.json` records the hashes of every preserved baseline and canonical output. Files ending in `_refresh.json` are direct current-build outputs; canonical filenames merge them with unchanged historical FLANN-CUDA/nanoflann fields and the independent ball-query result.

## Main query results

| Dataset/mode | k | FlashKNN query (ms) | cudaKDTree query (ms) | Query speedup | FlashKNN construction (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| S3DIS fixed pre | 8 | 0.638 | 0.731 | 1.15x | 1.163 |
| S3DIS fixed pre | 16 | 0.826 | 1.332 | 1.61x | 1.173 |
| S3DIS fixed pre | 24 | 1.148 | 2.052 | 1.79x | 1.177 |
| S3DIS fixed pre | 32 | 1.378 | 2.872 | 2.08x | 1.189 |
| S3DIS fixed pre | 48 | 2.561 | 4.524 | 1.77x | 1.204 |
| S3DIS fixed pre | 64 | 2.965 | 6.213 | 2.10x | 1.223 |
| S3DIS fixed post | 8 | 0.542 | 0.341 | 0.63x | 1.610 |
| S3DIS fixed post | 16 | 0.538 | 0.545 | 1.01x | 1.622 |
| S3DIS fixed post | 24 | 0.590 | 0.784 | 1.33x | 1.614 |
| S3DIS fixed post | 32 | 0.612 | 1.084 | 1.77x | 1.611 |
| S3DIS fixed post | 48 | 0.864 | 1.614 | 1.87x | 1.622 |
| S3DIS fixed post | 64 | 0.925 | 2.146 | 2.32x | 1.616 |
| S3DIS full-room pre | 32 | 1.444 | 2.159 | 1.50x | 1.156 |
| SemanticKITTI pre alpha=4 | 32 | 0.534 | 1.116 | 2.09x | — |
| SemanticKITTI post alpha=4 | 32 | 0.508 | 1.026 | 2.02x | — |

Relative to the preceding canonical S3DIS build, fixed pre k=32 and post k=32 improve by 9.84% and 11.57%, while full-room pre k=32 improves by 9.53%. Fixed pre k=48/64 regress slightly by 3.73%/2.06%; the complete room-level records are retained so these nonuniform changes are not hidden by the aggregate.

## Network and memory results

| Model/dataset | CPU hierarchy (ms) | Flash hierarchy (ms) | CPU end-to-end (ms) | Flash end-to-end (ms) | End-to-end speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeLA S3DIS | 757.943 | 20.227 | 775.464 | 37.564 | 20.64x |
| DeLA SemanticKITTI | 546.651 | 16.594 | 564.111 | 34.600 | 16.30x |
| DeepLA SemanticKITTI | 547.223 | 16.395 | 571.896 | 41.427 | 13.80x |

| Mode | FlashKNN | cudaKDTree | FAISS Flat | matched FAISS IVF |
| --- | ---: | ---: | ---: | ---: |
| Pre incremental peak | 290.584 MiB | 82.562 MiB | 1632.320 MiB | 1635.107 MiB |
| Post incremental peak | 121.123 MiB | 37.729 MiB | 1564.240 MiB | 1567.009 MiB |

The final current-build NCU evidence is in `../l20_stale_candidate_fix_ncu_20260825/`. The older `l20_branchless_sort_ncu_20260824/` profile remains useful only as the pre-correctness-fix intermediate comparison.

<details>
<summary>展开查看详细实验记录</summary>

`query/` contains both direct refresh and canonical merged JSON files. `network/` contains the complete S3DIS and SemanticKITTI network matrix. `memory/s3dis_memory_k32.json` contains per-room allocated/reserved accounting. `audit/` contains the build log, sm_89 cubin check, GPU/process snapshots, raw benchmark logs, correctness validation, finalization hashes and coverage validator output. No training was performed.

</details>
