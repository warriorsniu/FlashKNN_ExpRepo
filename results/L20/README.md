# NVIDIA L20 实验结果说明

本目录集中保存论文修改期间在两张同型号 NVIDIA L20（48 GB，driver 580.159.03）上生成的结果。正式 JSON 使用 Python 3.10.12、PyTorch 2.7.1+cu128 和 CUDA 12.8；每个 JSON 的 `metadata.gpu.uuid` 记录实际执行该文件的物理 GPU。除明确标记为 smoke、定向验证或无效并发记录的目录外，正式计时均使用 CUDA-ready 输入并排除文件 I/O、体素化、裁块与 H2D。

## 使用建议

论文主实验和复现检查应首先使用 `l20_complete_20260807/`。远端提交 `7f6cf67` 将其 SemanticKITTI query 更新为六个k、1320条、IVF匹配alpha=4的正式结果；canonical文件与直接来源文件的SHA256均为 `5bc4294251bdbff62f4b2adb628650f2e1a6a5eab626e120b8158fdc00f65ab8`。S3DIS fixed-size 的全部 k=8/16/24/32/48/64 已将 FlashKNN 和同轮 cudaKDTree 对照更新为最终 generated-bitonic build 的20次正式记录；JSON 的 `metadata.timing_overrides` 分别记录两批补跑的来源、GPU UUID 和计时参数。需要注意，这些覆盖项采用5/20，而文件中未覆盖的方法采用主实验3/10协议；此外 full-room 与 DeLA S3DIS 的 FlashKNN 路径仍待最终 production kernel 补跑。它们在替换前只作为增量合并来源，不作为最终统一协议数字。当前保留/删除清单见 `results/RESULT_SELECTION.md`。

## 目录来源、目的与覆盖关系

| 目录 | 结果来源与实验目的 | 覆盖关系与使用状态 |
| --- | --- | --- |
| `l20_complete_20260807/` | 正式汇总。S3DIS fixed-size query 包含81个房间、pre/post、k=8/16/24/32/48/64，共972条；所有 k 的 FlashKNN 与 cudaKDTree 均来自最终 generated-bitonic build 的配对验证，其余方法保留第一轮正式计时。S3DIS full 使用0.02 m体素化后的272个完整房间、pre、k=32，共272条；ball query 共486条；SemanticKITTI 使用22个 sequence 各5帧、pre/post、k=8/16/24/32/48/64，共1320条，IVF 显式匹配 FlashKNN `alpha=4` recall；网络结果包含5个 S3DIS 模型各68个 Area 5 房间，以及 SemanticKITTI 上 DeLA/DeepLA 的 CPU KDTree/`alpha=4` FlashKNN 成对层次构建和4个 Pointcept 网络，各22帧。`system.json` 保存统一环境。 | **推荐增量合并目录。** SemanticKITTI、ball query 和 LiDAR 网络可直接使用；S3DIS fixed-size 的 FlashKNN/cudaKDTree 需统一回3/10，full-room 与 DeLA S3DIS 需由最终 production kernel 替换后，才是完全统一的最终主结果。 |
| `l20_semantickitti_six_k_alpha4_20260818/` | 使用最终 build 从空文件运行110帧、pre/post、k=8/16/24/32/48/64，共1320条；每条包含四个 FlashKNN alpha 和全部基线，FAISS IVF 匹配 `alpha=4` recall，参数为3次 warmup、10次记录。 | **已替换旧660条文件并汇入最终目录。** 这是最终 SemanticKITTI query 的直接来源，且已通过严格 coverage validation。 |
| `l20_lidar_network_alpha4_20260809/` | 在同型号、同驱动、同显存的另一张 L20 上以论文默认 `alpha=4` 重跑 DeLA/DeepLA 的22帧正式 CPU KDTree/FlashKNN 配对实验，参数为10次 warmup、30次记录；`co_tenant_audit.txt` 保存物理卡、驻留但无 SM 活动的服务及进程级采样。 | **已替换原 `alpha=8` 结果并汇入最终目录。** 两个 JSON 的 `metadata.timing_overrides` 指向定向复测来源。 |
| `l20_lidar_network_alpha4_rerun_20260809/` | 对正式批次逐帧审查后，使用相同 GPU 和10/30参数定向复测 `03_000000` 与 `09_000000`。 | DeLA 仅采用 `09_000000`、DeepLA 仅采用 `03_000000` 覆盖明确的 model-latency 异常；其他复测记录仅作诊断，不进入最终均值。 |
| `l20_ball_query_20260807/` | 审稿补充的 Pointcept `pointops.ball_query` 对比。正式文件使用81个250k S3DIS crop、pre/post、k=24/32/48和全局 exact 第k邻居距离的90%分位半径；另有一个房间上的多 percentile radius sweep。 | 正式486条 ball-query 文件已被最终汇总逐字节覆盖；`s3dis_radius_sweep_one_room.json` 没有被全量目录替代，仍用于说明半径、coverage、truncation、recall 与延迟之间的关系。 |
| `l20_arkade_20260807/` | 审稿补充的 Arkade/TrueKNN OptiX 8.1 RT-core 基线，覆盖81个250k S3DIS crop、pre/post、k=24/32/48，并分别记录 BVH construction、TrueKNN radius-refit query、轮数和相对 cudaKDTree recall。 | **未被最终全量目录覆盖。** 这是独立硬件路径的补充对比；公开实现较慢且正式运行观察到两次可恢复的子进程故障，因此应与稳定性说明和 recall 一起引用，不作为主基线。 |
| `l20_bitonic_generated_validation/` | 对 k≤32 静态生成比较网络以及非2次幂 k 的重构实现做最终定向验证，设置与 top-p validation 相同，并同时包含 k=48 fallback 路径。 | 该目录的 FlashKNN 与配对 cudaKDTree 记录已汇入最终 fixed-size 文件的 k=24/32/48 配置；原目录继续作为20次重复、物理GPU和特定 kernel revision 的完整来源记录。 |
| `l20_bitonic_generated_k8_k16_k64_validation/` | 为消除最终汇总中残留的旧 FlashKNN 计时而补跑，覆盖81个250k S3DIS crop、pre/post、k=8/16/64、5次 warm-up 和20次记录，只运行 FlashKNN 与配对 cudaKDTree。 | 该目录的全部486条配对记录已汇入最终 fixed-size 文件；原目录是 k=8/16/64 最终 build 计时、召回和物理GPU信息的独立来源。 |

## `l20_complete_20260807` 文件索引

| 文件 | 内容 |
| --- | --- |
| `system.json` | L20、驱动、CUDA、PyTorch、Python、CPU 和依赖环境快照。 |
| `query/s3dis_sample_part.json` | S3DIS 250k fixed-size pre/post kNN 主表数据，包含 FlashKNN、cudaKDTree、FLANN-CUDA、nanoflann、FAISS Flat 和 matched-recall IVF-Flat；全部 k 的 FlashKNN/cudaKDTree 均使用最终 generated-bitonic build 的配对记录，详细 provenance 位于 `metadata.timing_overrides`。 |
| `query/s3dis_full_k32.json` | 272个房间经0.02 m体素化后的完整 pre-query 点数缩放实验，k=32；不运行不具实际可行性的百万点 exact FAISS Flat。 |
| `query/ball_query_s3dis_sample_part.json` | Pointcept ball-query 的 latency、valid-neighbor ratio、insufficient/truncation ratio 和相对 exact kNN set recall。 |
| `query/semantickitti.json` | SemanticKITTI 110帧的 pre/post、k=8/16/24/32/48/64 kNN，共1320条，包含 FlashKNN alpha=4/8/16/32及全部基线；FAISS IVF 逐条匹配 alpha=4 recall。 |
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
