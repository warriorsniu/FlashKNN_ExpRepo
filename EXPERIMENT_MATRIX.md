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
- Recall is row-wise set recall: each exact neighbor is counted at most once,
  and repeated predictions are reported separately instead of inflating recall.

## Speedups under different number of point

- Dataset: every Pointcept-format S3DIS room, `full`, pre-downsampling, k=32.
- Each room is voxelized at 0.02 m before querying; `full` means no subsequent 250,000-point crop, not the raw unvoxelized cloud.
- Baseline denominator: CPU nanoflann for both query and construction speedup.
- Curves: FlashKNN, cudaKDTree, FLANN-CUDA, and nanoflann, matching the paper.
- FAISS Flat/IVF-Flat are evaluated in the fixed-size 250,000-point table; exact Flat is intentionally not extended to the million-point scaling sweep.
- Output figures preserve per-room voxelized point counts on the x-axis.

## S3DIS fixed-250k memory footprint

- Dataset/crops: the same 81 S3DIS `sample_part` rooms and deterministic
  250,000-point crops as the main table.
- Representative configuration: pre/post query, k=32, alpha=4.
- RTX 3090 methods: FlashKNN, cudaKDTree, exact FAISS GPU Flat, and the
  canonical per-room matched-recall FAISS GPU IVF-Flat configuration.
- L20 methods: FlashKNN, cudaKDTree, exact FAISS GPU Flat, and the same
  canonical per-room matched-recall FAISS GPU IVF-Flat configuration. The L20
  latency table remains compact (FlashKNN/cudaKDTree only), while the memory
  comparison uses the same four-method set on both GPUs.
- Boundary: peak incremental method-owned GPU allocation above CUDA-ready
  support/query/grid/batch inputs; includes construction/index, workspace, and
  outputs; excludes file I/O, voxelization, crop, H2D, and input tensors.
- Accounting: FlashKNN uses the PyTorch active-allocation high-water mark;
  cudaKDTree uses an instrumented native memory resource because its spatial
  tree is invisible to the PyTorch allocator; FAISS combines its
  `StandardGpuResources` allocation ledger with PyTorch output tensors.
- Statistics: per-room memory, room mean, sample SD, and Student-t 95% CI.
  Formal runs require one idle GPU and identical default FAISS scratch policy.

## Final-kernel design ablations

- Dataset: the same 81 S3DIS `sample_part`, pre-query, 250,000-point crops.
- k: 8, 16, 24, 32, 40, 48, 56, 64, matching the historical ablation figures.
- Memory/sorting variants: SMPS, SMSS, and GMPS.
- Candidate/skip factorial: register/shared candidate storage with skip enabled/disabled.
- Every PS variant uses the current generated bitonic top-P network; variants are selected in one build.
- Formal protocol: 5 warmups and 20 recorded repetitions, single GPU, no co-tenant training.
- Provenance: git/dirty state, source hashes, extension hash, GPU UUID, and co-tenant snapshots.

## Thread-grouping ablation

- Dataset: the same 81 S3DIS `sample_part`, pre-query, 250,000-point crops.
- k: 8, 16, 24, 32, 48, 64.
- Strategies: Fixed-8, Fixed-16, Fixed-32, and the production Adaptive rule.
- Formal protocol: 5 warmups and 20 recorded repetitions, one GPU, with no co-tenant process.
- Ordering: balanced cyclic order over the 81 eligible rooms; for every k, each strategy appears first 20 or 21 times.
- Statistics: query mean/p95 over recorded timings; SD and 95% CI over per-room timing means.
- Correctness: report index-set recall against CUKD and require per-query sorted neighbor-distance equivalence across all four strategies, allowing only equal-distance index tie-breaking.
- Result: the balanced final v2 run completed 81 rooms and 486 room-k records; the formal runner remains locked for deliberate reproduction via `ALLOW_FORMAL_THREAD_GROUPING=1`.

## Adaptive octree-neighborhood ablation

- Dataset: the same 81 S3DIS `sample_part`, pre-query, 250,000-point crops.
- k: 8, 16, 24, 32, 48, 64.
- Strategies: fixed 3x3x3, adaptive multilevel 3x3x3 neighborhoods, and exact cudaKDTree measured in the same balanced run.
- Selection: descend above 8k candidates; accept a child only at or above 2k; no geometric guard or post-query retry.
- Kernel ABI: unchanged production `FlashKNN_Query_Dynamic_Load` inputs. Variable-level groups are flattened to compatible query/support descriptors and processed in one query-kernel call.
- Timings: octree construction, level selection, compatible-input construction, query/output remapping, and their total are reported separately.
- Diagnostics: CUKD set recall, distance/index consistency, per-level choice and candidate-band counts, compatible-coordinate expansion, incremental peak memory, and kernel-call count.
- Formal protocol/result: 5 warmups and 20 recorded repetitions, one GPU, no co-tenant process; the final v2 run completed 81 rooms and 486 room-k records. Reproduction remains explicitly locked.

## Radius/ball-query operator

- Dataset: the same 81 S3DIS `sample_part` crops used by the kNN table.
- Modes: pre/post; `nsample=k=24,32,48`.
- Radius: one global 90th-percentile exact kth-neighbor distance for each mode and k.
- Output: query latency, valid-neighbor ratio, insufficient-query ratio, truncation ratio, and set recall against cudaKDTree.

## Semantic-boundary accuracy analysis

- Dataset: all 68 S3DIS Area 5 validation rooms; 5,327,301 evaluated points.
- Definition: an exact k=24 neighborhood includes the query itself; a point is a semantic boundary when fewer than 50% of neighbors share its ground-truth class.
- Coverage: 49,144 boundary points (0.922%) and 5,278,157 non-boundary points.
- Comparison: the preselected performance-matched FlashKNN and exact-KNN checkpoints, with checkpoint and evaluation-script SHA256 recorded.
- Result fields: sample count, accuracy, mean class accuracy, mIoU, per-class IoU and confusion matrix for all/boundary/non-boundary partitions.
- Interpretation: this class-mixing partition complements, but does not replace, the sparse/low-recall candidate-count bins.

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
- Modes: pre/post; k: 8, 16, 24, 32, 48, 64, matching the fixed-size S3DIS
  query sweep; FlashKNN alpha: 4, 8, 16, 32.
- Baselines: cudaKDTree, FLANN-CUDA, CPU nanoflann, FAISS GPU Flat, and
  FAISS GPU IVF-Flat matched to the paper-default FlashKNN alpha=4 recall.
- Formal results: L20 and RTX 3090 each contain 1,320 unique sample-mode-k records (110 x 2 x 6), 3 warmups and 10 recorded repetitions; the L20 canonical file passed full coverage validation and the RTX 3090 directed result passed the same SemanticKITTI protocol checks.
- Network latency: 22 stratified frames, one per sequence. DeLA and DeepLA each compare the paper-compatible CPU KDTree hierarchy against FlashKNN with the paper-default alpha=4; PTv3, OctFormer, SPUNet, and MinkUNet34C measure CUDA-ready network forward latency on the same frames.
