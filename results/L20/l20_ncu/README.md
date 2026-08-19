# L20 final-kernel NCU profiles

本目录是 L20 k=32 微架构结果的 canonical 副本，内容来自 `../l20_s3dis_final_kernel_refresh_20260819/ncu/microarch/`。测试使用 NVIDIA L20（物理 GPU 0，UUID `GPU-78990023-5606-bb80-49bf-8ddfe8683461`，CC 8.9）、S3DIS 250k pre-query、k=32、seed=47，并 profile cudaKDTree、最终 production FlashKNN-SMPS/generated-bitonic 和 FlashKNN-GMSS。`microarch/provenance.json` 记录提交、关键源码哈希、GPU 身份、协议和抽取指标；`*_raw.csv` 是机器可读 NCU raw page，`*.ncu-rep` 可在 Nsight Compute 中复查，短 `*.csv` 只记录 profile 进程和 report 路径。

| Backend | HBM read sectors | HBM write sectors | Total sectors | Uniform branch targets (%) | Synchronized threads | Duration (ms) | Active warps (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cudaKDTree | 413,936 | 4,608,776 | 5,022,712 | 69.58 | 8.87 | 4.471 | 55.34 |
| FlashKNN-SMPS | 345,412 | 640,840 | 986,252 | 67.29 | 23.13 | 1.655 | 70.93 |
| FlashKNN-GMSS | 335,448 | 1,513,248 | 1,848,696 | 85.52 | 15.95 | 2.545 | 40.58 |

SMPS 相比 cudaKDTree 减少 80.37% 的总 DRAM sectors，并在 NCU profile 中获得 2.70× duration speedup。绝对 sector 数和最终 SMPS 的 uniform-branch 指标与论文旧 3090 profile 不一致，因此这组结果只应作为独立 L20/final-kernel 表引用；正式 query latency 仍以 `../l20_complete_20260807/` 的 3 warm-up/10-repeat JSON 为准。

旧完整双调排序 SMPS report 的 synchronized-threads ratio、uniform-branch 指标、duration 和 DRAM writes 分别为31.40、99.03%、2.706 ms和1,479,008 sectors；最终 generated top-P 对应23.13、67.29%、1.655 ms和640,840 sectors。下降的线程平均指标与 top-P 删除大量全线程共同执行的完整排序指令一致，而更短的 duration 和更低的写流量表明不能用该平均值单独判断总体效率。由于最终 revision 同时模板化了 candidate-storage/skip 开关，唯一因果结论仍需仅切换排序实现的同源码 A/B。
