# L20 S3DIS final-kernel refresh

本目录保存提交 `2dd4049f3d5ffe34f32b4f18154d4dd4c51d64e8` 上最终 production `SM+PS` generated-bitonic top-P kernel 的独立原始补跑结果。所有正式实验固定使用物理 GPU 0（UUID `GPU-78990023-5606-bb80-49bf-8ddfe8683461`），开始和结束时均无其他 compute process；环境为 Python 3.10.12、PyTorch 2.7.1+cu128、CUDA 12.8、driver 580.159.03，FlashKNN 扩展以 `TORCH_CUDA_ARCH_LIST=8.9` 重新编译。

## 实验协议与 coverage

| 文件 | 协议 | Coverage | 状态 |
|---|---|---:|---|
| `query/s3dis_sample_part.json` | 81 rooms，250k support，pre/post，k=8/16/24/32/48/64，3 warm-up/10 repeats，alpha=4，仅 FlashKNN/cudaKDTree | 972 个唯一 key | 通过 key、点数、timing、有限值和 recall 校验 |
| `query/s3dis_full_k32.json` | 272 rooms，完整 0.02 m 体素化点云，pre，k=32，3/10，alpha=4，仅 FlashKNN/cudaKDTree | 272 个唯一 key，21,579–2,424,985 点 | 通过 key、点数、timing、有限值和 recall 校验 |
| `network/dela_s3dis.json` | Area 5，CPU KDTree/FlashKNN paired backend，10 warm-up/30 repeats，FlashKNN alpha=4 | 68 个唯一 room | 双 backend 和全部 latency summary 校验通过 |

## 代表性结果

250k fixed-size 的 k=32 平均 query latency 如下；speedup 为 cudaKDTree/FlashKNN。

| Mode | FlashKNN query (ms) | cudaKDTree query (ms) | Speedup |
|---|---:|---:|---:|
| pre | 1.528 | 2.878 | 1.883× |
| post | 0.692 | 1.083 | 1.565× |

full-room 的分箱结果表明 FlashKNN 吞吐在 21k–2.425M 点范围内保持约 153–163 Mquery/s，而 cudaKDTree 吞吐随规模增大从约 107 Mquery/s 上升到约 147 Mquery/s，因此 query speedup 从小点云的 1.549× 收敛到 2.425M 点的 1.040×。完整分箱和 250k/500k/1M/2.425M 代表房间的新旧对比位于 `analysis/output/l20_complete_20260807/full_room_throughput.{json,md}`。

DeLA 68-room 平均 latency 为：CPU KDTree preprocessing/network/end-to-end 761.524/16.850/778.373 ms，FlashKNN preprocessing/network/end-to-end 20.020/17.258/37.278 ms，端到端加速 20.880×。

## Provenance 与 canonical 合并

三个 raw JSON 的 `metadata.provenance` 记录 source commit、七个关键源码 SHA256、已安装扩展 SHA256、物理 GPU、effective FlashKNN flags、数据根、room manifest 和共租快照路径。`provenance/gpu_{start,end}.csv` 与 `provenance/compute_processes_{start,end}.csv` 证明正式批次首尾目标 GPU 空闲。canonical `l20_complete_20260807` 中 fixed/full 仅替换 `flashknn` 和 `cuda_kdtree`，历史 FLANN/nanoflann/FAISS 字段经写入前后深比较确认未改变；DeLA canonical 替换为本目录的完整 paired 文件。

## NCU 状态

最终源码的 NCU profiling 已尝试，但当前驱动设置为 `RmProfilingAdminOnly: 1`，本用户没有 performance-counter 权限，cudaKDTree 首个 profile 返回 `ERR_NVGPUCTRPERM`，没有生成 `.ncu-rep`，因此未继续 SMPS/GMSS，也没有将旧 kernel 的 NCU 数据恢复为论文结果。原始错误保存在 `provenance/ncu_permission_error.log`。管理员将 `NVreg_RestrictProfilingToAdminUsers=0` 应用到 NVIDIA 模块并重载驱动或重启后，可运行 `NCU_GPU=0 NCU_OUTPUT_DIR=results/L20/l20_s3dis_final_kernel_refresh_20260819/ncu/microarch bash scripts/run_knn_ncu_microarch.sh`；脚本会生成三种 backend 的 `.ncu-rep`、summary CSV 和 raw CSV，并可用 `scripts/summarize_ncu_profiles.py` 校验和记录 provenance。

## 验证与重建

`python scripts/validate_result_coverage.py --run-dir results/L20/l20_complete_20260807` 已通过。`analysis/analyze_results.py` 已重新生成 `analysis/output/l20_complete_20260807` 下的 workbook、summary 和论文图，`analysis/analyze_l20_full_room_refresh.py` 另生成 full-room 分箱及历史对比；历史结果只用于对比，不再进入最终图表。
