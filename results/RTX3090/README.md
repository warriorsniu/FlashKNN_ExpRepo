# NVIDIA RTX 3090 单卡正式结果

`rtx3090_complete_20260808/` 是论文修改期间在本机 RTX 3090 上生成的 canonical 增量合并目录。实验固定使用物理 GPU 1（UUID `GPU-8998eefa-dc46-f1dc-5f7e-547fb11dd3c0`），未使用 DataParallel、DistributedDataParallel、NCCL 或任何多卡并行。15 个实验 JSON 的 `metadata.gpu.uuid` 都已核对为该 UUID；环境清单 `system.json` 另行记录 `selected_gpu=1`。

软件环境为 Python 3.10.20、PyTorch 2.7.1+cu118、CUDA runtime/toolkit 11.8；所有本地 CUDA 扩展为 RTX 3090 生成 `sm_86` cubin。与 L20 结果保持同一数据 manifest、warm-up/repeat 和计时边界，但按硬件平台使用 cu118，不与 L20/cu128 结果混合平均。2026-08-19 已在同一空闲物理 GPU 1 上用最终 generated top-P production kernel 补跑 S3DIS fixed/full 与 DeLA S3DIS，并增量写入 canonical；未受影响的第三方/native 基线保持不变。

## 覆盖与方法集

- S3DIS 250k fixed-size：81 个房间、pre/post、k=8/16/24/32/48/64，共 972 条。包含 FlashKNN、cudaKDTree、FLANN-CUDA、nanoflann、FAISS Flat 和 matched-recall IVF-Flat。
- S3DIS full-point：272 个 0.02 m 体素化完整房间、pre、k=32，共 272 条。只包含 FlashKNN、cudaKDTree、FLANN-CUDA 和 nanoflann；由于完整点云上 brute-force 复杂度和内存成本过高，该组明确跳过 FAISS Flat/IVF。
- S3DIS ball query：81 个房间、pre/post、k=24/32/48，共 486 条。
- S3DIS memory：与主表相同的81个250k crop、pre/post、k=32，共162条；比较FlashKNN、cudaKDTree、FAISS GPU Flat和逐房间matched-recall IVF。
- SemanticKITTI query：canonical 文件已由2026-08-18正式结果替换，覆盖110帧、pre/post、k=8/16/24/32/48/64，共1320条，FlashKNN alpha=4/8/16/32，FAISS IVF逐条匹配alpha=4。
- 网络：S3DIS Area 5 的 DeLA、PTv3、OctFormer、SPUNet、MinkUNet34C 各 68 个房间；SemanticKITTI 的 DeLA/DeepLA 分别比较 CPU KDTree 与 FlashKNN hierarchy，并包含 PTv3、OctFormer、SPUNet、MinkUNet34C，各 22 帧、10 次 warm-up、30 次记录。

`python scripts/validate_result_coverage.py --run-dir results/RTX3090/rtx3090_complete_20260808` 的结构覆盖检查已通过；该检查只证明样本/方法字段齐全，不会把上述旧 FlashKNN 时间认证为最终算法结果。六k SemanticKITTI 文件独立检查为1320条唯一记录、全部方法/alpha完整且IVF target无误，并已逐字节写入 canonical；两处SHA256均为 `ddd31c113d4e6e6326aafd83713b9f5ec0972e1f0de05fdf6084b6955aaa4594`。

## 代表性均值与可用状态

| 场景 | FlashKNN total | cudaKDTree total | 加速比 | FlashKNN recall |
| --- | ---: | ---: | ---: | ---: |
| S3DIS 250k, pre, k=32 | 2.409 ms | 13.334 ms | 5.54x | 0.999901 |
| S3DIS 250k, post, k=32 | 1.541 ms | 6.212 ms | 4.03x | 0.999853 |
| S3DIS full, pre, k=32 | 2.536 ms | 10.151 ms | 4.00x | 0.999914 |
| SemanticKITTI, pre, k=24, alpha=4 | 1.392 ms | 4.325 ms | 3.11x | 0.981446 |
| SemanticKITTI, post, k=24, alpha=4 | 1.402 ms | 3.661 ms | 2.61x | 0.978889 |

表中前三个 S3DIS 行来自2026-08-19最终 kernel 3/10补跑；后两个 SemanticKITTI 行来自最终六k文件，均可引用。

## S3DIS fixed-250k 显存占用

`rtx3090_s3dis_memory_k32_20260819/` 使用与主表相同的81个deterministic
250k crop，在空闲物理GPU 1上测量pre/post、k=32。指标是CUDA-ready输入之上的
method-owned peak incremental GPU allocation：包含construction/index、workspace和
output，排除文件I/O、voxelization、crop、H2D及输入tensor。FAISS沿用latency实验的
默认`StandardGpuResources` scratch policy，IVF逐房间复用canonical matched-recall
`nlist/nprobe`。

| Mode | FlashKNN | cudaKDTree | FAISS Flat | FAISS IVF |
| --- | ---: | ---: | ---: | ---: |
| Pre | 290.61 MiB | 82.56 MiB | 1632.32 MiB | 1635.11 MiB |
| Post | 121.12 MiB | 37.73 MiB | 1564.22 MiB | 1566.99 MiB |

表中是81个房间的均值；逐房间sample SD和Student-t 95% CI位于
`rtx3090_s3dis_memory_k32_20260819/analysis/summary.md`。FlashKNN的显存footprint
高于cudaKDTree，但仍低于300 MiB；其I/O优势指HBM traffic和访问模式，并不等价于
比树方法使用更少的allocated memory。FAISS的约1.5--1.6 GiB主要包括默认GPU resource
scratch。cudaKDTree的树由native `cudaMallocAsync`分配，因此通过instrumented memory
resource统计，不能用PyTorch allocator单独估算。

## S3DIS DeLA 最终内核延迟

2026-08-19补跑覆盖Area 5全部68房间，CPU KDTree与FlashKNN采用相同随机初始化模型，
每个backend为10次warm-up、30次正式记录。CPU端到端平均473.800 ms；FlashKNN hierarchy
平均12.428 ms、网络forward平均10.356 ms、端到端平均22.785 ms，对CPU端到端加速20.79×。
CPU批次相对旧结果只变化0.3%，说明机器负载稳定；最终Flash批次相对旧实现慢约4.7%，因此
canonical采用本次最终实现数字，不回退选择旧批次的更快结果。

直接来源目录为 `rtx3090_s3dis_final_kernel_refresh_20260819/`。fixed/full canonical只替换
FlashKNN和同轮cudaKDTree字段，FLANN、nanoflann、FAISS保持原值；metadata中的
`timing_overrides`记录未变基线哈希、源码/扩展哈希、GPU UUID及production flags。

SemanticKITTI 使用 alpha=4 作为代表性效率设置，是因为多种子网络实验已在 matched training/evaluation 协议下验证 alpha=4 相对 alpha=8 未显示系统性精度下降。该结论不表示可以把使用 alpha=8 训练的 checkpoint 在推理时直接切换为 alpha=4 而完全没有影响。

## SemanticKITTI 网络延迟

下表均使用相同的 22 个分层扫描，体素化输入点数平均为 84,287。DeLA/DeepLA 行报告 hierarchy 与 network forward 的端到端时延；其余四个网络只报告 CUDA-ready forward，因此跨网络比较必须同时披露计时边界。

| 模型 | 后端 | End-to-end / forward（ms） | 同模型端到端加速 |
| --- | --- | ---: | ---: |
| DeLA | CPU KDTree | 374.350 | 1.00x |
| DeLA | FlashKNN (alpha=4) | 27.275 | 13.72x |
| DeepLA | CPU KDTree | 378.660 | 1.00x |
| DeepLA | FlashKNN (alpha=4) | 33.184 | 11.41x |
| SPUNet | Native forward | 55.169 | — |
| MinkUNet34C | Native forward | 79.310 | — |
| PTv3 | Native forward | 106.882 | — |
| OctFormer | Native forward | 114.828 | — |

早期仅 FlashKNN hierarchy、alpha=8 CPU/Flash 配对、smoke/probe 和其他被最终 v2 取代的目录已在2026-08-19筛选中删除。canonical run 只保留 alpha=4 配对结果，避免重复计数或 operating point 混淆。完整保留/删除规则见 `results/RESULT_SELECTION.md`。

## 最终排序逻辑设计消融

`rtx3090_ablation_final_20260810/` 是机器空闲后在物理 GPU 5（UUID
`GPU-97038723-1a8a-70df-e5b7-52a98de11890`）完成的正式消融。覆盖 81 个 S3DIS
250k pre-query 房间、k=8/16/24/32/40/48/56/64、六个变体，每项 5 次 warm-up 和
20 次正式记录。运行开始时 metadata 中的 training/compute process 快照均为空；共
648 个 room-k 记录，`scripts/validate_ablation.py` 已通过。

原始结果：`rtx3090_ablation_final_20260810/ablation/s3dis_design_ablation.json`。
分析与两张最终 PDF/SVG 位于
`analysis/output/rtx3090_ablation_final_20260810/`。

| k | SMPS | SMSS | GMPS | CandidateSM | NoSkip | CandidateSM+NoSkip |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.794 | 0.995 | 0.970 | 1.015 | 0.974 | 1.866 |
| 32 | 1.712 | 3.816 | 2.100 | 2.706 | 1.755 | 3.246 |
| 64 | 2.860 | 10.929 | 3.315 | 5.344 | 2.699 | 5.920 |

表中为 query mean（ms）。SMPS 在全部 k 上快于 GMPS/SMSS；register candidate 在全部
k 上快于 shared candidate。Skip 对小 k 有明显收益，但约从 k=40 开始，register NoSkip
略快，说明此时 skip 的线程通信成本超过被跳过排序带来的收益。六变体在每个 k 下的 recall
一致到汇总精度。

## 线程分组策略消融

推荐正式结果是 `rtx3090_thread_grouping_balanced_final_v2_20260811/`：物理 GPU 5
（UUID `GPU-97038723-1a8a-70df-e5b7-52a98de11890`）、81 个 S3DIS 250k pre-query
房间、k=8/16/24/32/48/64、Adaptive 与 Fixed-8/16/32、5 次 warm-up/20 次正式记录。
共 486 个 room-k、1,944 个策略记录，coverage、源码/扩展 hash、目标 GPU 无共租快照、
严格 row-wise set recall 和逐查询邻居距离等价性均通过。

| k | Adaptive | Fixed-8 | Fixed-16 | Fixed-32 | Adaptive 相对最优 fixed |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.7571 | 0.7567 | 0.8302 | 1.0213 | +0.06% |
| 16 | 1.0575 | 1.1445 | 1.0579 | 1.3001 | -0.04% |
| 24 | 1.5672 | 2.0632 | 1.6391 | 1.5673 | -0.01% |
| 32 | 1.7605 | 2.2648 | 1.8211 | 1.7583 | +0.12% |
| 48 | 2.5684 | 3.7112 | 2.9424 | 2.5567 | +0.46% |
| 64 | 2.8656 | 4.2559 | 3.2683 | 2.8653 | +0.01% |

表中为 query mean（ms）。Adaptive 在全部 k 上与对应最优 fixed 相差不超过 0.46%，
同时相对不合适的固定分组最多快 32.7%。四条曲线、per-room SD/Student-t 95% CI 和 p95
位于 `analysis/output/rtx3090_thread_grouping_balanced_final_v2_20260811/thread_grouping/`。

测量顺序按实际纳入的 81 个房间和 k 平衡循环；每个 k 上四策略分别位于首位 20 或 21 次。
固定顺序的早期正式批次出现约3.5%顺序偏差，第一版 balanced 又因按全部房间而非实际81个
eligible房间轮换而被validator拒绝；二者及所有smoke目录已删除，不得引用其latency。

## 自适应八叉树邻域消融

推荐正式结果为 `rtx3090_adaptive_neighborhood_final_v2_20260818/`：物理 GPU 5、
PyTorch 2.7.1+cu118、81 个 S3DIS 250k pre-query 房间、k=8/16/24/32/48/64、
Fixed-3³、`[2k,8k]` Adaptive 和 exact cudaKDTree，5 次 warm-up/20 次正式记录。
三种策略按 room–k 平衡循环；486 个记录通过正式 coverage validation。

| k | Fixed total | Adaptive total | cudaKDTree total | Adaptive/Fixed | Adaptive/cudaKDTree |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.467 | 15.862 | 4.892 | 10.81× | 3.24× |
| 16 | 1.750 | 15.525 | 6.899 | 8.87× | 2.25× |
| 24 | 2.240 | 15.287 | 9.978 | 6.82× | 1.53× |
| 32 | 2.456 | 15.579 | 13.325 | 6.34× | 1.17× |
| 48 | 3.337 | 16.542 | 21.250 | 4.96× | 0.78× |
| 64 | 3.687 | 17.083 | 29.655 | 4.63× | 0.58× |

Adaptive 的 octree construction 约 9.75 ms，compatible-input construction 约
2.71--4.27 ms，兼容坐标为原输入的 2.09--2.47 倍，平均峰值增量显存随 k 从约
383 MiB 增至 958 MiB。正式汇总及 per-room SD/Student-t 95% CI 位于
`analysis/output/rtx3090_adaptive_neighborhood_final_v2_20260818/`。结果目录中的
`adaptive_neighborhood/fixed_alpha4_candidate_counts.json` 记录二维流形候选规模核验：81 个房间的逐查询
候选中位数之逐房间平均为 146.44，接近 `(3×4)^2=144`。
`sparse_candidate_recall_bins.json` 进一步按固定邻域候选数分箱：少于 `2k` 的稀疏
query 上 Adaptive recall 提高约 0.029--0.061，但网络常用 k=8--32 时这类点只占
0.0045%--0.3851%，解释了为何局部收益不足以抵偿全局自适应开销。

## S3DIS 语义边界分析

`rtx3090_s3dis_semantic_boundary_20260818/` 使用两个性能接近的既定 checkpoint，在
Area 5 全部68房间上评估。语义边界定义为 exact k=24 邻域（包含query自身）内与center
同类的点少于50%。总计5,327,301个点，其中49,144个边界点，占0.922%。

| 子集 | FlashKNN mIoU | ExactKNN mIoU | 差值（百分点） |
| --- | ---: | ---: | ---: |
| 全部点 | 72.953% | 72.862% | +0.090 |
| 语义边界点 | 34.765% | 34.648% | +0.118 |
| 非边界点 | 73.395% | 73.294% | +0.100 |

边界点 accuracy 在两种方法下均为65.371%。该结果与 recall/稀疏分箱互补：前者描述局部
类别混合，后者描述候选不足或低 recall，不能将二者视为同一分组。
