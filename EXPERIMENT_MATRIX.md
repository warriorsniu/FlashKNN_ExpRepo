# Paper-aligned experiment coverage

This file is the acceptance checklist for the three efficiency artifacts in
the paper. `run_all.sh` is not successful unless
`scripts/validate_result_coverage.py` verifies every required result.

## Time Cost(ms) for Different k and Query Modes

- Dataset: S3DIS, 0.02 m voxelization.
- `sample_part`: exactly 250,000 support points when available.
- Modes: pre-downsampling and post-downsampling.
- k: 8, 16, 24, 32, 48, 64.
- Paper methods: FlashKNN, cudaKDTree, FLANN-CUDA, CPU nanoflann.
- Added strong baselines: exact FAISS GPU Flat and matched-recall FAISS
  GPU IVF-Flat.
- Timings: construction, query, and total, with 3 warmups and 10 recorded
  repetitions. CUDA inputs are ready before timing. The historical nanoflann
  protocol copies inputs to CPU before starting its construction/query timers.

## Speedups under different number of point

- Dataset: every Pointcept-format S3DIS room, `full`, pre-downsampling, k=32.
- Baseline denominator: CPU nanoflann for both query and construction speedup.
- Curves: FlashKNN, cudaKDTree, FLANN-CUDA, nanoflann, FAISS Flat, and
  matched-recall FAISS IVF-Flat.
- Output figures preserve per-room voxelized point counts on the x-axis.

## Network efficiency comparison

- Dataset/split: all 68 S3DIS Area 5 validation rooms, 0.04 m voxelization.
- Models: SPUNet, MinkUNet34C, PTv3, OctFormer, original DeLA with CPU
  nanoflann preprocessing, and DeLA with FlashKNN preprocessing.
- Weights on the cross-GPU efficiency host: deterministic random
  initialization; no checkpoint is needed because only execution time is
  compared.
- Output: per-room voxelized point count versus latency, plus the aggregate
  summary table.

## Additional SemanticKITTI coverage

- Pack: sequences 00--21, five evenly spaced frames per sequence (110 total).
- Modes: pre/post; k: 16, 24, 32; FlashKNN alpha: 4, 8, 16, 32.
- Baselines: cudaKDTree, FLANN-CUDA, CPU nanoflann, FAISS GPU Flat, and
  matched-recall FAISS GPU IVF-Flat.
- Network latency: DeLA and DeepLA on 22 stratified frames, one per sequence.
