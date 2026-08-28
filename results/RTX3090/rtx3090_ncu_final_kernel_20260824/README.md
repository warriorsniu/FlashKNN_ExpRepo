# RTX 3090 final-kernel NCU profile

This directory contains a clean, single-GPU Nsight Compute profile of the
production FlashKNN kernel. The run used physical GPU 0 (RTX 3090, sm86),
PyTorch 2.7.1+cu118, CUDA 11.8, and Nsight Compute 2022.3. The deterministic
input is the first valid S3DIS 250k pre-query crop (`Area_1/WC_1`), with
`k=32` and seed 47. No compute process was present before or after profiling.

The FlashKNN extension SHA256 is
`ef8fc17b25f9daa6c7d9e8b24654a63b014d78b359b2d894ab71efbcd5741130`.
The production dynamic-load source SHA256 is
`d09b091ee345a509b55742f058a0bab0bf0b0ae01344b0a14258d6e5fe7a51ac`;
the generated bitonic top-P header SHA256 is
`323bf535be8078635cf5e8f0d83a24a189c9b14dee088cfa94dcf7afbd3b28aa`.
These hashes match the final-kernel RTX 3090 result pack from 2026-08-19.

## Results

| Backend | DRAM read sectors | DRAM write sectors | Uniform branch targets (%) | Active threads/instruction | NCU duration (ms) | Active warps (%) | Registers/thread | Shared memory/block (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cudaKDTree | 113,537,408 | 58,628,148 | 69.58 | 8.87 | 10.510 | 59.10 | 35 | 1.024 |
| FlashKNN-SMPS | 229,188 | 1,979,028 | 98.86 | 31.38 | 1.529 | 71.56 | 51 | 5.120 |
| FlashKNN-GMSS | 7,400,328 | 24,655,948 | 85.52 | 15.95 | 3.495 | 29.62 | 38 | 1.024 |

`Active threads/instruction` is the kernel-wide NCU metric
`smsp__thread_inst_executed_per_inst_executed.ratio`; it is not a direct count
of threads assigned to the sorting network. NCU-instrumented duration is used
only for the paired backends in this run and does not replace the formal query
benchmark latency.

## Cross-platform note

The final profiles report 31.38 active threads/instruction on RTX 3090 with
CUDA 11.8 and 31.33 on L20 with CUDA 12.8. Both launches use `(32,4,1)` blocks,
and the k=32 specialization uses a 32-lane sorting network. Profiling metrics
remain platform- and compiler-specific and should not be treated as
architecture-independent invariants.

Machine-readable source, extension, GPU, launch, metric, and artifact hashes
are stored in `provenance.json`; summary and raw profiler CSV files are retained
under `ncu/microarch/`.
