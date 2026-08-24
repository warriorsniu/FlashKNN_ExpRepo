# L20 branchless compare-exchange NCU validation

## Background

The retained pre-fix L20 profile reported only 23.13 threads per executed warp instruction for FlashKNN-SMPS, although its generated top-P sorting network assigns one 32-thread group to each query. Source-correlated SASS inspection showed that the compiler lowered data-dependent `if (take_peer)` compare-and-swap blocks into divergent control flow. Commit `49ecd90712e3ea8525d186d8c76e5d49d658c42a` replaces those conditional blocks with unconditional paired distance/index selections while preserving the comparison network and candidate set.

## Environment and protocol

The run used NVIDIA L20 physical GPU 0 (UUID `GPU-78990023-5606-bb80-49bf-8ddfe8683461`, CC 8.9), PyTorch 2.7.1+cu128, CUDA 12.8, S3DIS `Area_1/WC_1`, a deterministic 250,000-point pre-query crop, k=32 and seed 47. cudaKDTree, FlashKNN-SMPS and FlashKNN-GMSS were profiled with the same runner and requested metric set as `../l20_ncu/`. The branchless FlashKNN extension SHA256 is `04f6b82e4d800755baaf77737212191001661b2683a688abb1f680b26fbae080`; `provenance.json` records the committed source hashes and immutable artifact hashes.

## Main results

| Backend | HBM read sectors | HBM write sectors | Total sectors | Uniform branch targets (%) | Threads / warp instruction | Duration (ms) | Active warps (%) | Registers/thread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cudaKDTree | 414,592 | 4,599,532 | 5,014,124 | 69.58 | 8.87 | 4.505 | 55.29 | 34 |
| Branchless FlashKNN-SMPS | 346,840 | 638,360 | 985,200 | 97.00 | 31.39 | 1.483 | 63.55 | 62 |
| FlashKNN-GMSS control | 334,280 | 1,519,988 | 1,854,268 | 85.52 | 15.95 | 2.528 | 40.58 | 38 |

| SMPS revision | Uniform branch targets (%) | Threads / warp instruction | Duration (ms) | Active warps (%) | Registers/thread |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pre-fix retained profile | 67.29 | 23.13 | 1.655 | 70.93 | 55 |
| Branchless compare-exchange | 97.00 | 31.39 | 1.483 | 63.55 | 62 |
| Change | +29.70 pp | +8.26 | -10.37% | -7.38 pp | +7 |

The branchless SMPS kernel is 3.04x faster than the same-run cudaKDTree kernel and reduces total DRAM sectors by 80.35%. Relative to the retained pre-fix SMPS profile, its read/write traffic is effectively unchanged while duration falls by 10.37%. Source-correlated SASS changes from 163 to 31 unique branch instructions over the whole kernel and from 60 to zero branches over the sorting source range; sorting `SEL/FSEL` instructions increase from 70 to 224. This confirms that the previous 23.13 value came from compiler-generated compare-and-swap control flow rather than an L20 warp-width anomaly.

## Validation status and next step

A local one-room smoke test covered pre/post and k=8/16/24/32/48/64; every FlashKNN recall value matched the retained canonical value for the same room and configuration. This directory commits the NCU evidence only. Existing canonical S3DIS, SemanticKITTI and network latency JSON files still describe the pre-fix production extension and have not been overwritten; the required latency rerun scope will be selected separately before those results are refreshed.

<details>
<summary>展开查看详细实验记录</summary>

`microarch/*_raw.csv` contains the NCU raw page, `microarch/*.ncu-rep` is the inspectable Nsight Compute report, and `microarch/flash-smps_k32_source.csv` is the CUDA/SASS source view used for static branch accounting. `analysis/sass_branch_summary.json` records the deduplicated source-view counts and comparison inputs. The pre-fix reference remains in `../l20_ncu/` and is retained only as the A/B baseline for this fix.

</details>
