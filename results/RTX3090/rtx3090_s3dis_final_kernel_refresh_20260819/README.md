# RTX 3090 S3DIS production-kernel refresh

This run refreshes every S3DIS result affected by the final generated-bitonic
production kernel. All measurements used physical GPU 1
(`GPU-8998eefa-dc46-f1dc-5f7e-547fb11dd3c0`) with no co-tenant training or
inference process.

- `query/s3dis_sample_part.json`: 81 rooms, pre/post, 250,000 points,
  k=8/16/24/32/48/64, FlashKNN and paired cudaKDTree, 3 warm-ups/10 repeats,
  972 unique records.
- `query/s3dis_full_k32.json`: all 272 voxelized full rooms, pre, k=32,
  FlashKNN and paired cudaKDTree, 3 warm-ups/10 repeats, 272 unique records.
- `network/dela_s3dis.json`: all 68 Area 5 rooms, CPU KDTree and alpha=4
  FlashKNN hierarchy, 10 warm-ups/30 repeats.

Every raw JSON records the Git commit, source and extension SHA256, effective
production flags, actual GPU UUID, timing boundary, and co-tenant snapshots.
The query files were merged into the canonical pack by replacing only
FlashKNN and cudaKDTree. Unaffected FLANN, nanoflann, and FAISS fields were
verified by a deterministic JSON hash. The DeLA paired file was replaced as a
unit.

Representative canonical results:

| Configuration | FlashKNN total | cudaKDTree total | Total speedup | Recall |
| --- | ---: | ---: | ---: | ---: |
| 250k pre, k=32 | 2.409 ms | 13.334 ms | 5.54x | 0.999901 |
| 250k post, k=32 | 1.541 ms | 6.212 ms | 4.03x | 0.999853 |
| Full-room pre, k=32 | 2.536 ms | 10.151 ms | 4.00x | 0.999914 |

For DeLA, CPU KDTree end-to-end latency is 473.800 ms and FlashKNN
end-to-end latency is 22.785 ms, a 20.79x speedup. The FlashKNN hierarchy
component is 12.428 ms.
