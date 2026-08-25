# L20 stale-candidate-fix production NCU

## Background

This directory profiles the final production build at commit `2a69727257a0ee46d214c4db2e49b7572a10adfe`, which invalidates the upper scratch half before each selectively populated compare-split round and prevents stale winners from being reconsidered. The prior `l20_branchless_sort_ncu_20260824` directory corresponds to commit `49ecd907` and does not include this correctness fix.

## Environment and protocol

The run used physical NVIDIA L20 GPU 0, UUID `GPU-78990023-5606-bb80-49bf-8ddfe8683461`, PyTorch 2.7.1+cu128, CUDA toolkit 12.8, `TORCH_CUDA_ARCH_LIST=8.9`, S3DIS `Area_1/WC_1`, a deterministic 250,000-point pre-query crop, k=32 and seed 47. The FlashKNN extension SHA256 is `95c5566be5d4a646e1300a6c77706f5c690bc6b6ba941a8a72ecfcdae32b6a44`; `provenance.json` records all critical source, extension and report hashes.

## Main results

| Backend | HBM read sectors | HBM write sectors | Total sectors | Uniform branch targets (%) | Threads / warp instruction | Duration (ms) | Active warps (%) | Registers/thread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cudaKDTree | 414,112 | 4,599,500 | 5,013,612 | 69.58 | 8.87 | 4.478 | 55.23 | 34 |
| Final FlashKNN-SMPS | 344,984 | 638,424 | 983,408 | 97.00 | 31.33 | 1.393 | 63.66 | 58 |
| FlashKNN-GMSS control | 334,340 | 1,505,704 | 1,840,044 | 85.52 | 15.95 | 2.498 | 40.67 | 38 |

The final SMPS kernel is 3.21x faster than the same-run cudaKDTree kernel and reduces total DRAM sectors by 80.39%. Relative to commit `49ecd907`, synchronized threads remain effectively unchanged (31.33 versus 31.39), uniform branch targets remain 97.00%, registers fall from 62 to 58, and NCU duration falls from 1.483 ms to 1.393 ms. Thus the correctness repair does not reintroduce the earlier divergence problem.

<details>
<summary>展开查看详细实验记录</summary>

`microarch/*.ncu-rep` contains inspectable Nsight Compute reports, `microarch/*_raw.csv` contains every requested paper metric, and `microarch/flash-smps_k32_source.csv` contains the CUDA/SASS source view. `provenance.json` binds these artifacts to the production source and extension hashes. The complete latency, memory and network refresh is stored in `../l20_stale_candidate_fix_complete_20260824/`.

</details>
