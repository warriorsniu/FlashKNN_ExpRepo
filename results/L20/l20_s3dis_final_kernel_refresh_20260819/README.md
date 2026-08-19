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

## NCU 微架构结果

管理员开放 performance counter 后，已在同一物理 GPU 0 上完成最终 kernel 的 cudaKDTree、FlashKNN-SMPS 和 FlashKNN-GMSS profiling。协议为确定性的 S3DIS 250k pre-query、k=32、seed=47；每种方法均保留 `.ncu-rep`、NCU console CSV 和 raw wide CSV，机器可读汇总及源码 SHA256 位于 `provenance/ncu_provenance.json`。最初的 `ERR_NVGPUCTRPERM` 记录 `provenance/ncu_permission_error.log` 仅保留为权限修复历史，不再表示当前状态。

| Backend | HBM read sectors | HBM write sectors | Total sectors | Uniform branch targets (%) | Synchronized threads | Duration (ms) | Active warps (%) | Registers/thread | Shared memory/block (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cudaKDTree | 413,936 | 4,608,776 | 5,022,712 | 69.58 | 8.87 | 4.471 | 55.34 | 34 | 1.024 |
| FlashKNN-SMPS | 345,412 | 640,840 | 986,252 | 67.29 | 23.13 | 1.655 | 70.93 | 55 | 5.120 |
| FlashKNN-GMSS | 335,448 | 1,513,248 | 1,848,696 | 85.52 | 15.95 | 2.545 | 40.58 | 38 | 1.024 |

SMPS 相对 cudaKDTree 的总 DRAM sector 数减少 80.37%，NCU-instrumented kernel duration 加速 2.70×，同步线程指标提高 2.61×，active-warps 占用率提高 15.59 个百分点。GMSS 的 uniform-branch 指标最高，但总 I/O、同步线程数和 duration 均不及 SMPS。NCU duration 受 replay/instrumentation 影响，只用于同轮微架构对照，不替代正式 3/10 benchmark latency。

本轮绝对 sector 数不复现论文原表的 3090 数值：论文记录的 FlashKNN/cudaKDTree 总 sector 分别为 2,212,036/178,565,712，而 L20 最终 kernel 为 986,252/5,022,712；I/O 降低方向一致，但降低比例由 98.76% 变为 80.37%。最终 generated-bitonic SMPS 的 Uniform Warp Ratio 也由旧 kernel 的约 99.0% 变为 67.29%，而 synchronized-threads ratio 仍明显高于 cudaKDTree。因此论文若加入 L20 表，应按架构和 kernel revision 分栏，不能将两组绝对计数当作复现实验或直接合并。

| SMPS revision | HBM read sectors | HBM write sectors | Uniform branch targets (%) | Synchronized threads | Duration (ms) |
|---|---:|---:|---:|---:|---:|
| 旧完整双调排序 kernel | 347,320 | 1,479,008 | 99.03 | 31.40 | 2.706 |
| 最终 generated top-P kernel | 345,412 | 640,840 | 67.29 | 23.13 | 1.655 |

旧报告的 kernel 实例为 `FlashKNN_Query_dynamic_load_kernel<float,2,5>`，最终报告为带编译期 ablation 参数的 `<float,2,5,false,true>`，二者 launch 形状相同。最终 top-P 删除了完整排序中大量全线程共同执行的 stage/loop 指令，同时保留按 lane 位置和比较结果触发的 compare-exchange，因此以动态指令为分母的 synchronized-threads 平均值下降并不等同于总体线程效率退化：新 kernel 的 duration 下降38.9%，DRAM writes 下降56.7%。由于同一版本还引入了编译期 `CandidateInShared/EnableSkip` 参数，现有证据支持“变化主要与 top-P 重构一致”，但若要在论文中声称唯一因果，仍需在同一源码中只切换完整排序/top-P 做 NCU A/B。

## 验证与重建

`python scripts/validate_result_coverage.py --run-dir results/L20/l20_complete_20260807` 已通过。`scripts/summarize_ncu_profiles.py` 已验证三份 NCU report、summary CSV、raw CSV 和全部必需指标。`analysis/analyze_results.py` 已重新生成 `analysis/output/l20_complete_20260807` 下的 workbook、summary 和论文图，`analysis/analyze_l20_full_room_refresh.py` 另生成 full-room 分箱及历史对比；历史结果只用于对比，不再进入最终图表。
