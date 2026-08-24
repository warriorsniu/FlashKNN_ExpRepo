# L20 pre-branchless NCU profiles

本目录保留 L20 k=32 的 pre-branchless 微架构结果，内容来自 `../l20_s3dis_final_kernel_refresh_20260819/ncu/microarch/`。测试使用 NVIDIA L20（物理 GPU 0，UUID `GPU-78990023-5606-bb80-49bf-8ddfe8683461`，CC 8.9）、S3DIS 250k pre-query、k=32、seed=47，并 profile cudaKDTree、FlashKNN-SMPS/generated-bitonic 和 FlashKNN-GMSS。`microarch/provenance.json` 记录提交、关键源码哈希、GPU 身份、协议和抽取指标；`*_raw.csv` 是机器可读 NCU raw page，`*.ncu-rep` 可在 Nsight Compute 中复查，短 `*.csv` 只记录 profile 进程和 report 路径。当前 branchless compare-exchange 结果位于 `../l20_branchless_sort_ncu_20260824/`；本目录不再代表当前源码，只作为严格 A/B 的修复前参照。

| Backend | HBM read sectors | HBM write sectors | Total sectors | Uniform branch targets (%) | Synchronized threads | Duration (ms) | Active warps (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cudaKDTree | 413,936 | 4,608,776 | 5,022,712 | 69.58 | 8.87 | 4.471 | 55.34 |
| FlashKNN-SMPS | 345,412 | 640,840 | 986,252 | 67.29 | 23.13 | 1.655 | 70.93 |
| FlashKNN-GMSS | 335,448 | 1,513,248 | 1,848,696 | 85.52 | 15.95 | 2.545 | 40.58 |

SMPS 相比 cudaKDTree 减少 80.37% 的总 DRAM sectors，并在 NCU profile 中获得 2.70× duration speedup。绝对 sector 数和最终 SMPS 的 uniform-branch 指标与论文旧 3090 profile 不一致，因此这组结果只应作为独立 L20/final-kernel 表引用；正式 query latency 仍以 `../l20_complete_20260807/` 的 3 warm-up/10-repeat JSON 为准。

此前将31.40降至23.13解释为删除完整排序stage并不成立：重新核查确认两批源码都已经使用三阶段top-P网络。当前源码/SASS A/B定位到`if (take_peer)`被编译为数据相关控制流；提交`49ecd907`将compare-and-swap改成无分支select后，threads-per-instruction恢复到31.39、uniform branch targets恢复到97.00%，duration降至1.483 ms。完整证据和新report见`../l20_branchless_sort_ncu_20260824/`。
