# NVIDIA L20 实验结果说明

本目录集中保存论文修改期间在两张同型号 NVIDIA L20（48 GB，driver 580.159.03）上生成的结果。正式 JSON 使用 Python 3.10.12、PyTorch 2.7.1+cu128 和 CUDA 12.8；每个 JSON 的 `metadata.gpu.uuid` 记录实际执行该文件的物理 GPU。除明确标记为 smoke、定向验证或无效并发记录的目录外，正式计时均使用 CUDA-ready 输入并排除文件 I/O、体素化、裁块与 H2D。

## 使用建议

论文主实验和复现检查应首先使用 `l20_complete_20260807/`，该目录已经通过 `scripts/validate_result_coverage.py` 的完整覆盖校验。S3DIS fixed-size 的全部 k=8/16/24/32/48/64 均已将 FlashKNN 和同轮 cudaKDTree 对照更新为最终 generated-bitonic build 的20次正式记录；JSON 的 `metadata.timing_overrides` 分别记录两批补跑的来源、GPU UUID 和计时参数。若需要微架构指标、半径敏感性、Arkade RT-core 对比或优化过程消融，再分别读取 `l20_ncu/`、`l20_ball_query_20260807/`、`l20_arkade_20260807/` 和 optimization validation 目录。Smoke 结果只用于证明安装与调用链可运行，不能用于加速比或论文性能结论。

## 目录来源、目的与覆盖关系

| 目录 | 结果来源与实验目的 | 覆盖关系与使用状态 |
| --- | --- | --- |
| `l20_complete_20260807/` | 最终正式汇总。S3DIS fixed-size query 包含81个房间、pre/post、k=8/16/24/32/48/64，共972条；所有 k 的 FlashKNN 与 cudaKDTree 均来自最终 generated-bitonic build 的配对验证，其余方法保留第一轮正式计时。S3DIS full 使用0.02 m体素化后的272个完整房间、pre、k=32，共272条；ball query 共486条；SemanticKITTI 使用22个 sequence 各5帧、pre/post、k=16/24/32，共660条；网络结果包含5个 S3DIS 模型各68个 Area 5 房间，以及 SemanticKITTI 上 DeLA/DeepLA 的 CPU KDTree/`alpha=4` FlashKNN 成对层次构建和4个 Pointcept 网络，各22帧。`system.json` 保存统一环境。 | **论文主结果与推荐下载目录。** 已覆盖正式 query、ball query、LiDAR 和网络延迟矩阵，并通过 coverage validation。S3DIS fixed-size 的两批配对计时来源见 JSON metadata；它不替代 NCU、Arkade、单房间 radius sweep 和特定 kernel revision 的定向消融。 |
| `l20_full_20260806/` | 第一轮正式 S3DIS 运行。`s3dis_sample_part.json` 已完成972条；当时的 `s3dis_full_k32.json` 仅有92条 pre/post 记录，是旧版 full 定义下的部分运行。 | **已被后续全量覆盖。** sample_part 的 FLANN-CUDA、nanoflann 和 FAISS 记录仍构成最终汇总中这些未变更基线的数据；全部 FlashKNN/cudaKDTree 记录均已被最终 build 的配对结果替换。92条旧 full 记录被最终272房间、仅 pre、正确0.02 m体素化的 full 实验取代。仅用于追溯历史。 |
| `l20_supplement_gpu1_20260807/` | 第二张 L20 上补齐 SemanticKITTI query、S3DIS 网络结果和早期仅 FlashKNN hierarchy 的 DeLA/DeepLA LiDAR 网络结果，随后由 merge 脚本汇入最终目录。`semantickitti.concurrent-invalid.json` 是早期与 CPU nanoflann 并发运行时保存的424条中间记录。 | SemanticKITTI query 和5个 S3DIS 网络 JSON 已汇入最终目录；早期2个 LiDAR 网络 JSON 已被 `l20_lidar_network_matrix_20260808/` 的成对 CPU/Flash 结果取代。`concurrent-invalid` 明确无效，绝不可用于论文数字。保留该目录只为记录来源。 |
| `l20_lidar_network_matrix_20260808/` | 22个分层 SemanticKITTI 帧的完整网络矩阵：PTv3、OctFormer、SPUNet、MinkUNet34C 网络 forward 保留原始正式结果；DeLA/DeepLA 文件已用2026-08-09的 `alpha=4` 正式结果更新。 | **已汇入最终全量目录。** 四个 Pointcept 文件是其最终直接来源，DeLA/DeepLA 的直接来源和异常点复测 provenance 分别见后两行。 |
| `l20_lidar_network_alpha4_20260809/` | 在同型号、同驱动、同显存的另一张 L20 上以论文默认 `alpha=4` 重跑 DeLA/DeepLA 的22帧正式 CPU KDTree/FlashKNN 配对实验，参数为10次 warmup、30次记录；`co_tenant_audit.txt` 保存物理卡、驻留但无 SM 活动的服务及进程级采样。 | **已替换原 `alpha=8` 结果并汇入最终目录。** 两个 JSON 的 `metadata.timing_overrides` 指向定向复测来源。 |
| `l20_lidar_network_alpha4_rerun_20260809/` | 对正式批次逐帧审查后，使用相同 GPU 和10/30参数定向复测 `03_000000` 与 `09_000000`。 | DeLA 仅采用 `09_000000`、DeepLA 仅采用 `03_000000` 覆盖明确的 model-latency 异常；其他复测记录仅作诊断，不进入最终均值。 |
| `l20_network_matrix_smoke_20260808/` | 新增 LiDAR CPU KNN 与四个 Pointcept 网络入口后的单样本、1次 warmup、1次记录连通性检查。 | **已被正式矩阵完全覆盖。** 只能用于环境与调用链回归，不能用于性能结论。 |
| `l20_ball_query_20260807/` | 审稿补充的 Pointcept `pointops.ball_query` 对比。正式文件使用81个250k S3DIS crop、pre/post、k=24/32/48和全局 exact 第k邻居距离的90%分位半径；另有一个房间上的多 percentile radius sweep。 | 正式486条 ball-query 文件已被最终汇总逐字节覆盖；`s3dis_radius_sweep_one_room.json` 没有被全量目录替代，仍用于说明半径、coverage、truncation、recall 与延迟之间的关系。 |
| `l20_arkade_20260807/` | 审稿补充的 Arkade/TrueKNN OptiX 8.1 RT-core 基线，覆盖81个250k S3DIS crop、pre/post、k=24/32/48，并分别记录 BVH construction、TrueKNN radius-refit query、轮数和相对 cudaKDTree recall。 | **未被最终全量目录覆盖。** 这是独立硬件路径的补充对比；公开实现较慢且正式运行观察到两次可恢复的子进程故障，因此应与稳定性说明和 recall 一起引用，不作为主基线。 |
| `l20_ncu/` | 按论文微架构口径运行 Nsight Compute。顶层 `cukd_k32_io_raw.csv` 是 cudaKDTree k=32 的初始 I/O/DRAM sector 核验；`microarch/` 统一采集 cudaKDTree、FlashKNN GMSS 和 FlashKNN SMPS 的 DRAM read/write、kernel duration、Uniform Warp Ratio 和 Synchronized Threads 等指标。 | **未被计时全量实验覆盖。** CSV 是论文微架构表的可移植数据源；本地 `.ncu-rep` 可用于 Nsight Compute 交互复查，但因仓库全局忽略二进制 profile report，远端默认只保存 CSV。 |
| `l20_construct_fix_validation/` | 修复 construction 阶段排序与数组去重实现后的定向验证，使用81个250k S3DIS crop、pre/post、k=32、5次 warm-up 和20次记录，只比较 FlashKNN 与 cudaKDTree。 | 后续正式和 bitonic 测试覆盖相同基本场景，但该目录拥有更高重复次数并隔离 construction 修复，仍适合作为回归证据；论文主表优先使用最终或最新定向结果。 |
| `l20_bitonic_top_p_validation/` | 对“只执行得到前k候选所需的 bitonic top-p 阶段”原型进行定向验证，覆盖81个250k S3DIS crop、pre/post、k=24/32/48、5次 warm-up 和20次记录。 | **被后续 generated comparator 方案取代为最终实现方向。** 保留用于比较重构过程，不应单独作为最终算法数字。 |
| `l20_bitonic_generated_validation/` | 对 k≤32 静态生成比较网络以及非2次幂 k 的重构实现做最终定向验证，设置与 top-p validation 相同，并同时包含 k=48 fallback 路径。 | 该目录的 FlashKNN 与配对 cudaKDTree 记录已汇入最终 fixed-size 文件的 k=24/32/48 配置；原目录继续作为20次重复、物理GPU和特定 kernel revision 的完整来源记录。 |
| `l20_bitonic_generated_k8_k16_k64_validation/` | 为消除最终汇总中残留的旧 FlashKNN 计时而补跑，覆盖81个250k S3DIS crop、pre/post、k=8/16/64、5次 warm-up 和20次记录，只运行 FlashKNN 与配对 cudaKDTree。 | 该目录的全部486条配对记录已汇入最终 fixed-size 文件；原目录是 k=8/16/64 最终 build 计时、召回和物理GPU信息的独立来源。 |
| `l20_smoke/` | 最早的 SemanticKITTI 环境连通性检查，仅包含1个样本、1次计时以及 DeLA/DeepLA 各1条网络记录。 | **已被后续 smoke 和正式全量完全覆盖。** 只能追溯早期 LiDAR 调用链，不能用于性能结论。 |
| `l20_smoke_full_20260806/` | 第一版统一 `run_all` smoke，S3DIS crop 到1000点，query 和网络各取1个样本、1次计时；当时尚未纳入 ball query。 | **已被后续 smoke 与正式全量覆盖。** 仅用于证明当时 Conda/uv、CUDA 扩展和各网络入口可运行。 |
| `l20_smoke_ball_env_20260807/` | 加入 uv/Conda 自动环境选择、ball query、断点恢复和严格 coverage validator 后的最终端到端 smoke。S3DIS crop 到1000点，所有 query/ball/network 路径各取1个样本、1次计时。 | **数值已被正式全量覆盖。** 仍可作为当前安装和运行脚本的快速功能参考，但其裁块与单次计时不具备性能解释力。 |

## `l20_complete_20260807` 文件索引

| 文件 | 内容 |
| --- | --- |
| `system.json` | L20、驱动、CUDA、PyTorch、Python、CPU 和依赖环境快照。 |
| `query/s3dis_sample_part.json` | S3DIS 250k fixed-size pre/post kNN 主表数据，包含 FlashKNN、cudaKDTree、FLANN-CUDA、nanoflann、FAISS Flat 和 matched-recall IVF-Flat；全部 k 的 FlashKNN/cudaKDTree 均使用最终 generated-bitonic build 的配对记录，详细 provenance 位于 `metadata.timing_overrides`。 |
| `query/s3dis_full_k32.json` | 272个房间经0.02 m体素化后的完整 pre-query 点数缩放实验，k=32；不运行不具实际可行性的百万点 exact FAISS Flat。 |
| `query/ball_query_s3dis_sample_part.json` | Pointcept ball-query 的 latency、valid-neighbor ratio、insufficient/truncation ratio 和相对 exact kNN set recall。 |
| `query/semantickitti.json` | SemanticKITTI 110帧的 pre/post kNN，包含 FlashKNN alpha=4/8/16/32及全部基线。 |
| `network/dela_s3dis.json` | DeLA 在 S3DIS Area 5 上使用 CPU KD-tree 与 FlashKNN preprocessing 的配对端到端延迟。 |
| `network/{ptv3,octformer,spunet,minkunet34c}_s3dis.json` | 四个 Pointcept 网络在 S3DIS Area 5 上的网络延迟。 |
| `network/{dela,deepla}_semantickitti_backends.json` | DeLA 与 DeepLA 在22个分层 LiDAR 帧上使用 CPU KDTree 和 FlashKNN hierarchy 的成对 preprocessing、network 与 end-to-end 延迟。 |
| `network/{ptv3,octformer,spunet,minkunet34c}_semantickitti.json` | 四个 Pointcept 网络在相同22个分层 LiDAR 帧上的 CUDA-ready network forward latency。 |

## SemanticKITTI 网络延迟结果

下表来自22个 sequence 各取1帧、10次 warmup 和30次记录的正式均值。DeLA/DeepLA 的 CPU 与 Flash 行是相同模型内的有效成对比较；四个 Pointcept 行只报告 CUDA-ready forward。所有模型均为随机初始化，输入通道和执行形状保持一致，但未比较精度，因此不同网络之间的延迟只能说明本实验形状下的执行成本。

| 模型 | 邻域/后端 | Hierarchy / preprocessing (ms) | Network (ms) | End-to-end (ms) | 同模型端到端加速 |
| --- | --- | ---: | ---: | ---: | ---: |
| DeLA | CPU KDTree | 550.583 | 18.015 | 568.598 | 1.00× |
| DeLA | FlashKNN (`alpha=4`) | 16.464 | 17.555 | 34.019 | 16.71× |
| DeepLA | CPU KDTree | 548.357 | 25.095 | 573.453 | 1.00× |
| DeepLA | FlashKNN (`alpha=4`) | 16.423 | 24.704 | 41.128 | 13.94× |
| SPUNet | Native forward | — | 44.727 | 44.727 | — |
| MinkUNet34C | Native forward | — | 102.656 | 102.656 | — |
| OctFormer | Native forward | — | 113.929 | 113.929 | — |
| PTv3 | Native forward | — | 137.299 | 137.299 | — |

将相同22帧的旧 `alpha=8` 结果与更新后的 `alpha=4` 结果对照，DeLA/DeepLA 的 Flash hierarchy 分别下降8.6%和9.7%，端到端分别下降3.6%和3.9%。`alpha` 只改变 FlashKNN 候选规模；network forward 的小幅批次差异属于计时波动，因此论文应优先引用 hierarchy 和 end-to-end 变化。

| 模型 | Alpha | Flash hierarchy (ms) | Flash end-to-end (ms) | 对 CPU 端到端加速 |
| --- | ---: | ---: | ---: | ---: |
| DeLA | 8（旧） | 18.019 | 35.276 | 16.10× |
| DeLA | 4（最终） | 16.464 | 34.019 | 16.71× |
| DeepLA | 8（旧） | 18.192 | 42.800 | 13.44× |
| DeepLA | 4（最终） | 16.423 | 41.128 | 13.94× |

## 重建汇总

在仓库根目录运行 `python analysis/analyze_results.py --results results/L20/l20_complete_20260807 --output-dir analysis/output/l20_complete_20260807` 可重新生成 Excel、Markdown 摘要和论文图；运行 `python scripts/validate_result_coverage.py --run-dir results/L20/l20_complete_20260807` 可重新执行完整性校验。
