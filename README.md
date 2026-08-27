# FlashKNN cross-GPU reproducibility package

本仓库用于复现论文修订中的两类**效率**实验：S3DIS / SemanticKITTI 上的
KNN query，以及随机初始化网络的单样本 latency。随机权重实验不报告 mIoU；它只保持
网络结构、张量形状和算子路径一致，用来比较不同 GPU 架构上的执行时间。

## 1. 数据关系与选择

本地曾出现三种 S3DIS：

| 路径/来源 | 表示 | 与其他版本的关系 | 用途 |
|---|---|---|---|
| ETH 官网 `Stanford3dDataset_v1.2_Aligned_Version` | 每个房间的 annotation TXT，XYZRGB | 原始授权数据；aligned 版只旋转对齐，不归一化或平移 | DeLA 原始预处理输入 |
| `s3disfull/raw/*.npy` | 每房间 `N×7`，XYZRGB+label | 由官方 annotation 合并而来，房间局部坐标 | 历史 query 输入 |
| Pointcept legacy `Area_*/*.pth` 或 current `Area_*/room/*.npy` | coord/color/label/instance/normal | 同一批 272 个房间；新版将每个字段拆成 NPY | 本仓库统一输入 |

两种处理表示的房间命名和语义一致。对 272 个房间作了逐项核对：将 Pointcept
坐标减去各轴最小值并重新舍入到官方毫米精度后，269 个房间的 2 cm 体素 support/post
点数与历史 `s3disfull/raw` 完全一致。Pointcept 压缩数据中
`Area_4/hallway_2`、`Area_4/office_21`、`Area_5/lobby_1` 的原始点数与官方 aligned
TXT 不同；其中前者和后者会影响 81-room sample-part 中的两个裁块。所以该下载方案
适合使所有 GPU 共享同一可复现 workload，但不应声称它逐点等同于历史 NPY。当需要
与原论文输入逐房间完全一致时，使用本地已验证的 `s3disfull/raw`；跨 GPU 不得混用两个版本。
推荐下载 Pointcept 的约 2.04 GB 压缩包；当前归档解压约需 8 GB。下载产物是每个房间目录下
`coord.npy`、`color.npy`、`normal.npy`、`segment.npy`、`instance.npy` 的多文件布局；本仓库
直接读取该布局，不再错误地假定下载产物为旧版 PTH，也不生成第二份离线副本。
S3DIS 仍受原许可约束，下载前须阅读并接受官网条款，数据不随 Git 仓库分发。

SemanticKITTI 的 efficiency pack 覆盖 00--21 全部 22 个序列，每个序列在完整帧范围内
等距抽取 5 帧，共 110 帧。已有完整、合法下载的数据时由脚本生成；仓库不重新分发原始数据。

## 2. 环境安装

### L20（Ada SM89 / CUDA 12.8）

要求 x86_64 Linux（glibc 2.35 或兼容版本）、NVIDIA 驱动可支持 CUDA 12.8、CUDA 12.8
toolkit（含 `nvcc`）、
git，以及 Python 3.10。预编译 wheel 与 CPython/PyTorch C++ ABI 绑定，因此不要改用
Python 3.11；安装脚本会拒绝不匹配环境。

主脚本只创建一套软件栈：Python 3.10、PyTorch 2.7.1+cu128。L20 必须由脚本检测为
compute capability `8.9`，所有本地 CUDA 扩展必须包含 `sm_89` cubin。交付压缩包若包含
与当前 GPU 匹配的 `wheelhouse/`，其中包括 FAISS、FlashKNN、cudaKDTree、
FLANN-CUDA、nanoflann、Pointcept pointops、OctFormer dwconv 和带 CUDA 12/PyTorch 2.7
兼容补丁的 MinkowskiEngine。安装脚本会先校验 manifest 中的版本、架构、文件大小及
SHA256，再安装这些 wheel。若 wheelhouse 属于 H20/`sm_90` 或其他架构，脚本会明确跳过，
并自动为可见的 L20 从源码编译 `sm_89`，不需要编辑任何 `setup.py`。DeLA 与
DeepLA 共用同一份 cutils 源码，安装脚本会在安装阶段构建并安装一个共享扩展
`dela_cutils_ext`；之后每次运行 smoke/正式实验都直接加载该扩展，不再触发 JIT 或 Ninja。

PyPI 的 `faiss-gpu-cu12` wheel 只包含有限的旧架构 cubin 且没有可供 SM89 使用的 PTX，
不能作为 L20 安装方案。本仓库从官方 FAISS v1.12.0 源码构建，并在 L20 上设置
`CMAKE_CUDA_ARCHITECTURES=89`。安装末尾会真实运行 FAISS
Flat/IVF 和四个 query 扩展的 GPU smoke test，不只检查 import；正式结果不得排除
MinkUNet 或任何其他论文图中方法，也不得切换到另一套旧版 PyTorch 环境。

安装开始前建议只暴露将用于实验的一张 L20；脚本会读取该卡并完成架构选择：

```bash
export CUDA_VISIBLE_DEVICES=0
conda create -n flashknn-exp python=3.10 -y
conda activate flashknn-exp
bash scripts/install.sh
```

日志中应出现 `Detected CUDA capability: 8.9; native target: 8.9`。如果不是 `8.9`，
应先检查 `CUDA_VISIBLE_DEVICES` 和实际 GPU 型号，不要手工伪造 `TORCH_CUDA_ARCH_LIST`。
脚本优先自动选择 `/usr/local/cuda-12.8`；只有 toolkit 安装在非标准位置时才需要在执行前
设置一次 `CUDA_HOME=/实际/cuda-12.8/路径`，其余代码和实验参数均无需修改。

### RTX 3090 本机环境（Ampere SM86 / CUDA 11.8）

本轮 RTX 3090 复跑保持与 L20 相同的 Python 3.10、PyTorch 2.7.1、源码 commit、
数据和计时协议，但使用本机 CUDA 11.8 toolkit 与官方 `cu118` wheels。该设置用于新的
跨平台实验；原论文历史软件栈应作为历史结果单独记录，不能与新 run 合并。

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-11.8
conda create -n flashknn-exp-cu118 python=3.10 -y
conda activate flashknn-exp-cu118
bash scripts/install.sh
```

安装器会检测 RTX 3090 的 compute capability `8.6`，并为所有本地扩展生成 `sm_86`
cubin。正式运行默认写入 `results/RTX3090/<RUN_ID>`；L20 仍写入
`results/L20/<RUN_ID>`，避免两个平台共用结果目录。每个平台的 JSON 会分别记录
PyTorch CUDA runtime、nvcc、driver、GPU UUID 和完整依赖列表。

## 3. 准备数据

### L20 推荐的零配置方式

将单独提供的、已授权 SemanticKITTI benchmark pack 放到固定位置：

```text
data/incoming/semantickitti_pack/manifest.json
data/incoming/semantickitti_pack/*.npz
```

阅读 S3DIS 许可后只需执行：

```bash
ACCEPT_S3DIS_LICENSE=1 bash prepare_data.sh
```

脚本会下载并解压 Pointcept S3DIS、自动发现 LiDAR pack、创建路径配置，并检查 272 个
S3DIS 房间及 pack 中的全部文件。无需编辑任何 Python、Shell 或配置文件。若 pack 位于其他
目录，可使用 `SEMANTICKITTI_PACK=/path/to/pack`；若已有 S3DIS，则可同时设置
`S3DIS_ROOT=/path/to/s3dis`，仍然不需要修改代码。

也可以完全由脚本下载。推荐给同学提供私有/受控的约 130 MB pack URL：

```bash
SEMANTICKITTI_PACK_URL='https://server/path/pack.tar.gz' \
ACCEPT_S3DIS_LICENSE=1 bash prepare_data.sh
```

若只能使用官方数据，先在 KITTI 下载页提交邮箱、接受许可，并复制邮件中的 Velodyne
授权链接。之后脚本负责断点下载约 80 GB 整包、从 00--21 每个序列只解压 5 个均匀分布
的帧、体素化并生成 110 帧 pack：

```bash
KITTI_VELODYNE_URL='邮件中的授权下载链接' \
ACCEPT_S3DIS_LICENSE=1 bash prepare_data.sh
```

效率实验不需要 SemanticKITTI label ZIP；它只使用点云几何和 remission。
官方 Velodyne 包是单一整包，因此即使只提取 110 帧，也仍须先下载约 80 GB。许可确认
和邮箱授权不能由仓库替用户完成。

只下载一次 S3DIS（需要明确确认许可）：

```bash
python scripts/prepare_data.py --accept-s3dis-license
```

复用已有 Pointcept 格式数据，并从完整 SemanticKITTI 生成 110 帧 pack：

```bash
python scripts/prepare_data.py \
  --s3dis-existing /path/to/Stanford3dDataset_v1.2_Aligned_Ptv3Style \
  --semantickitti-root /path/to/SemanticKITTI/dataset \
  --lidar-samples-per-sequence 5
```

若同学直接收到 benchmark pack：

```bash
python scripts/prepare_data.py --s3dis-existing /path/to/s3dis \
  --semantickitti-pack /path/to/semantickitti_pack
```

脚本只在 `data/` 下建立符号链接，并生成 `data/paths.env`；不会覆盖现有实体目录。

## 4. 一键运行

安装和数据准备完成后，正式实验只需：

```bash
bash run_all.sh
```

该入口自动选择当前显存占用最低的 GPU，执行依赖/数据 preflight，统一生成一个 run ID，并在当前已激活的 PyTorch 2.7.1+cu118/cu128 环境中依次完成 query、ball query、network latency 和 Excel/绘图分析。Python 选择顺序为显式 `PYTHON_BIN`、已激活的 uv/venv、已激活的 Conda、仓库 `.venv`、最后才是 `PATH`，因此原始 Conda 安装与本机 `uv pip` 环境使用同一个入口；所有子脚本继承同一解释器，不会混用系统 Python。最终路径会打印在终端，不需要设置 GPU、环境名或输出目录。快速端到端验收使用 `SMOKE=1 bash run_all.sh`。

已有同一 GPU、PyTorch/CUDA、算法 build、数据 manifest 和 warmup/repeat 配置的结果时，可用冒号分隔的 `REUSE_RUN_DIRS` 合并后断点续跑，例如 `REUSE_RUN_DIRS=results/L20/l20_complete_20260807 RUN_ID=l20_refresh GPU=0 bash run_all.sh`。只能复用 `results/RESULT_SELECTION.md` 标为保留、且算法与协议相容的字段；不得从旧 Git revision 恢复 smoke、旧 kernel 或被取代的结果。`scripts/merge_run_results.py` 保留原有 JSON 顶层结构和记录字段；元数据不同的文件会被拒绝或跳过，smoke 结果不会混入正式结果。结果根目录按 GPU 自动选择：L20 写入 `results/L20/<RUN_ID>`，RTX 3090 写入 `results/RTX3090/<RUN_ID>`；也可用 `RESULTS_ROOT` 显式覆盖。

先确认机器空闲并指定从 **0 开始编号**的物理 GPU：

```bash
GPU=0 RUN_ID=l20_01 bash run_query.sh
GPU=0 RUN_ID=l20_01 bash run_network_latency.sh
```

若机器上已有历史的 `s3disfull/raw/*.npy`，并需要与原论文 query 输入完全一致，
不用改脚本，只需在命令前指定：

```bash
EXPREPO_S3DIS_QUERY=/path/to/s3disfull/raw GPU=0 RUN_ID=l20_01 bash run_query.sh
```

网络 latency 仍使用 `EXPREPO_S3DIS` 指向的 Pointcept 多文件 NPY 数据，两者不会混淆。

`run_query.sh` 分两份运行：

- `sample_part`: 250,000 个 support 点，k=8/16/24/32/48/64，pre/post；
- `full`: 先按 0.02 m 体素化、随后不作 250,000 点裁块的完整房间，固定 k=32，仅运行论文点数缩放图所需的 pre-query；
- 若已准备 LiDAR pack，追加 SemanticKITTI pre/post query，k=8/16/24/32/48/64。
- Pointcept ball query：S3DIS `sample_part`、pre/post、`nsample=k=24/32/48`，分别使用全局 exact 第 k 邻居距离的 90% 分位半径，并保存 coverage、truncation 和 recall。

S3DIS `sample_part` 和 SemanticKITTI query 均对比 FlashKNN、精确 cudaKDTree、FLANN-CUDA、CPU nanoflann、FAISS GPU Flat 和 matched-recall IVF-Flat；S3DIS `full` 点数缩放实验按论文方法只运行 FlashKNN、cudaKDTree、FLANN-CUDA 和 nanoflann，避免在百万点完整查询上运行复杂度近似二次的 exact FAISS Flat。S3DIS 以论文默认 FlashKNN `alpha=4` recall 校准 IVF（`num_down=2`）；SemanticKITTI 则以 LiDAR 默认 `alpha=8` 校准，并通过 `faiss_ivf_match_alpha=8` 显式记录。需要注意，canonical S3DIS 后续只用最终 build 覆盖了 FlashKNN/cudaKDTree timing 与 recall，未重跑未变更的 IVF，因此其 `target_recall` 保留首次正式 alpha=4 校准值，与覆盖后的 FlashKNN recall 可能存在极小差异；新的 SemanticKITTI 文件逐条严格匹配当前 alpha=8 recall。IVF 在实用范围 `nprobe=1...64` 中选择第一个达到目标的设置；若因等距候选的 ID tie-breaking 始终不能精确达到，则选择 recall 最接近的设置，并保存完整校准轨迹。限制实用范围可避免因等距候选的 ID tie-breaking 无法精确达到目标时，退化为扫描全部 IVF lists 并产生过大显存临时开销。所有 GPU 方法的计时输入已在 GPU，排除文件 I/O、体素化、裁块与 CPU→GPU 传输；FAISS Flat 的 `add` 与 search 分别计时，IVF 原始 JSON 分别保留 training、add 和 search，论文主表的 construction/total 按 `training + add` 计算。nanoflann 严格复用原论文脚本口径：先把输入复制到 CPU，再开始计时，故其 GPU→CPU 传输不在表中。

SemanticKITTI 的论文代表性 FlashKNN operating point 使用 `alpha=8`，与主要的 matched
training/evaluation checkpoint 保持一致，并提高稀疏 LiDAR 场景的大 (k) recall。2026-08-18
的 `alpha=4` matched-IVF 目录保留为历史/快速 operating point 证据，不得改写标签后作为
`alpha=8` 结果使用；论文表格使用 2026-08-25 以后按 `alpha=8` 重新校准的结果。

`run_network_latency.sh` 测试：

- DeLA S3DIS：CPU KDTree 预处理 + 网络，以及 FlashKNN GPU 预处理 + 网络；
- PTv3、OctFormer、SpUNet、MinkUNet34C：4 cm 体素后的网络 forward；
- SemanticKITTI：DeLA / DeepLA 使用 `alpha=8`，分别采用 CPU KDTree 与 FlashKNN 构建四层 hierarchy，并记录 preprocessing、network 和 end-to-end；PTv3、OctFormer、SPUNet、MinkUNet34C 在相同 LiDAR 帧上记录 CUDA-ready network forward。

正式参数为 10 次 warmup、30 次记录；S3DIS 使用与论文一致的
Area 5 全部 68 个验证房间，
SemanticKITTI 使用 22 帧（每个序列 1 帧）。快速检查用：

```bash
SMOKE=1 GPU=0 RUN_ID=smoke bash run_query.sh
SMOKE=1 GPU=0 RUN_ID=smoke bash run_network_latency.sh
```

`SMOKE=1` 只用于检查安装、数据接口和结果覆盖：它把 S3DIS `sample_part`
裁到 1,000 点，并且只记录 1 次。该尺度下 kernel 启动、输出分配和索引重排等固定开销
可能使 FlashKNN 的 query 暂时慢于 cudaKDTree，因此 smoke 数值不能用于论文中的
加速比或跨 GPU 结论。正式 query 实验必须去掉 `SMOKE=1`，保留默认 250,000 点裁块、
完整房间和 3/10 次 warmup/repeat。

同一台 GPU 上的所有方法应单卡串行运行；不要同时启动训练。正式测试建议先记录空载
`nvidia-smi`，并在结果中检查 GPU UUID，防止设备编号变化。

### 最终排序逻辑的设计消融

`run_ablation.sh` 复现原论文的两组设计消融，但所有 PS 变体统一使用当前
generated bitonic top-P 逻辑，不再通过复制 CUDA backup 并反复重编译来切换实现：

- support 数据位置与排序方式：SMPS、SMSS、GMPS；
- 候选列表位置与 skip：Register+Skip、CandidateSM+Skip、Register+NoSkip、
  CandidateSM+NoSkip。

正式实验使用同一批 81 个 S3DIS 250k pre-query crop，覆盖
`k=8,16,24,32,40,48,56,64`，默认 5 次 warm-up 和 20 次记录。JSON 保存源码 SHA256、
扩展二进制 SHA256、GPU UUID、共租进程快照以及每次 construction/query/total timing。

```bash
GPU=5 RUN_ID=rtx3090_ablation_final_20260810 bash run_ablation.sh
python scripts/validate_ablation.py \
  results/RTX3090/rtx3090_ablation_final_20260810/ablation/s3dis_design_ablation.json
python analysis/analyze_ablation.py \
  results/RTX3090/rtx3090_ablation_final_20260810/ablation/s3dis_design_ablation.json \
  --output-dir analysis/output/rtx3090_ablation_final_20260810
```

功能检查使用 `SMOKE=1`，只运行一个房间和 k=8/32/64。正式运行前必须用 SMPS
重复探针确认同机训练任务没有造成可观察的性能漂移；若与安静基线相比出现稳定的数个百分点
变化，应保留 smoke、等待机器空闲，不得将共租结果用于论文。

### 线程分组策略消融

`run_thread_grouping_ablation.sh` 在同一个最终 SMPS/generated bitonic top-P build 中比较
Fixed-8、Fixed-16、Fixed-32 和生产 Adaptive 策略。正式协议为同一批 81 个 S3DIS
250k pre-query crop、`k=8,16,24,32,48,64`、单卡无共租、5 次 warm-up 和 20 次正式记录。
分析脚本按所有 query timing 报告均值和 p95，并以每个房间的 timing mean 为统计单位报告
room SD 与 Student-t 95% CI。四种策略按实际纳入的 room–k 做平衡循环排序，避免固定
测量顺序与 GPU 升频混入 thread-group 差异；每个 k 上每种策略位于首位 20 或 21 次。

```bash
# 只做功能检查：一个房间，但覆盖全部 6 个 k 和 4 种策略
SMOKE=1 GPU=0 RUN_ID=thread_grouping_smoke \
  bash run_thread_grouping_ablation.sh

# 复现正式实验仍需在机器空闲且获得明确授权后解锁
ALLOW_FORMAL_THREAD_GROUPING=1 GPU=0 RUN_ID=rtx3090_thread_grouping_final \
  bash run_thread_grouping_ablation.sh
```

RTX 3090 推荐正式结果为
`results/RTX3090/rtx3090_thread_grouping_balanced_final_v2_20260811/`。固定顺序的早期
正式目录不用于论文，因为同一 CUDA 模板也显示了可观察的顺序偏差。

JSON 同时保存相对 CUKD 的 index-set recall，以及每种固定策略相对 Adaptive 的逐查询排序
邻居距离等价性。后者用于正确处理等距点：不同线程遍历顺序可能选择不同但距离完全相同的
合法邻居索引，不能把这种 tie-breaking 差异误判为 recall 退化。正式命令默认被锁定，未设置
`ALLOW_FORMAL_THREAD_GROUPING=1` 时会在 GPU 探测和结果目录创建前退出。

### 自适应八叉树邻域消融

`run_adaptive_neighborhood.sh` 在同一运行内比较固定 3×3×3 邻域、多层 Morton 线性八叉树的
自适应 3×3×3 邻域和 exact cudaKDTree。自适应策略从粗到细遍历：当前候选多于 `8k` 时尝试子层，只有子层
候选不少于 `2k` 才采用子层；不使用几何边界 guard，也不在查询后重试。若子层少于 `2k`，
会保留候选可能多于 `8k` 的父层，这是无额外回退近似范式的预期行为。

生产 CUDA kernel 的函数签名和五类输入语义保持不变。层级选择后，脚本把不同层的 query
节点和实际引用的 support 节点副本降平为兼容的 `GridCoord`、`Parent2Child`、
`ParentNeigh` 与 `CumCntInNeigh`，所有层只调用一次
`FlashKNN_Query_Dynamic_Load`。结果分别记录 octree construction、level selection、
compatible-input construction 和 query（含输出映射）时间，同时保存点副本倍率、峰值增量显存、
逐层选择比例、落在/超出 `[2k,8k]` 的 query 数以及相对 CUKD recall。

```bash
# 一个 250k 房间、全部 6 个 k 的功能检查
PYTHON_BIN=/path/to/pytorch-2.7.1-cu118/bin/python \
SMOKE=1 GPU=0 RUN_ID=adaptive_neighborhood_smoke \
  bash run_adaptive_neighborhood.sh

# 复现81房间正式实验仍需明确授权，并要求目标 GPU 无共租进程
PYTHON_BIN=/path/to/pytorch-2.7.1-cu118/bin/python \
ALLOW_FORMAL_ADAPTIVE_NEIGHBORHOOD=1 GPU=0 RUN_ID=adaptive_neighborhood_final \
  bash run_adaptive_neighborhood.sh
```

RTX 3090 推荐正式结果为
`results/RTX3090/rtx3090_adaptive_neighborhood_final_v2_20260818/`：81房间、六个k、
486个room-k记录，5次warm-up/20次记录，Fixed/Adaptive/exact三种策略通过完整coverage。

### 2026-08-18 SemanticKITTI 六 k 正式结果

L20 直接来源为 `results/L20/l20_semantickitti_six_k_alpha4_20260818/`，并已逐字节汇入
`results/L20/l20_complete_20260807/query/semantickitti.json`。RTX 3090 对应结果为
`results/RTX3090/rtx3090_semantickitti_unifiedk_alpha4ivf_20260818/`。两者均覆盖110帧、
pre/post、`k=8,16,24,32,48,64`、FlashKNN `alpha=4,8,16,32` 和全部基线，各1320条；
参数为3次warm-up/10次记录，FAISS IVF target逐条等于同配置FlashKNN `alpha=4` recall。
RTX 3090 任务正常以退出码0完成，结果检查为1320条唯一记录且无方法、alpha或target缺失。

### 2026-08-25 SemanticKITTI 论文默认 alpha=8 刷新

论文默认 LiDAR operating point 已改为 `alpha=8`。RTX 3090 的定向正式结果位于
`results/RTX3090/rtx3090_semantickitti_alpha8_ivf_20260825/`：110 帧、pre/post、
`k=8,16,24,32,48,64`、3/10 协议，FlashKNN 仅运行 `alpha=8`，并在同一轮重跑
cudaKDTree、FAISS Flat 和逐配置重新校准的 IVF。结果包含1320条唯一记录，IVF target
逐条与当前 FlashKNN recall 完全一致。对应的 DeLA/DeepLA 网络延迟位于
`results/RTX3090/rtx3090_lidar_network_alpha8_final_20260825/`，覆盖22个分层帧、
10/30 协议和配对的 CPU-KDTree/FlashKNN hierarchy。旧 `alpha=4` 目录仅作为更快
operating point 和历史证据保留，不再向论文默认表格供数。当前 production kernel 对
DeLA/DeepLA 三个论文 seed 的 sequence-08 兼容性复核位于
`results/RTX3090/rtx3090_semantickitti_checkpoint_alpha8_final_20260825/`；聚合 mIoU
分别为 65.455±0.394 和 65.992±0.473，与历史数值仅有约千分之一百分点的舍入变化。

LiDAR 网络实验统一使用 pack 中已经按 0.06 m 体素化的 XYZ 与 remission。DeLA/DeepLA 的 CPU 层次计时沿用论文口径，不计 CPU→GPU 传输；FlashKNN 层次默认使用论文设置 `alpha=8` 并在 GPU 上计时，两者的 model forward 均使用 CUDA event。可用 `LIDAR_ALPHA` 显式覆盖探索性实验，但正式 coverage validator 只接受 `alpha=8`。四个 Pointcept 模型采用随机初始化，仅比较执行形状与 latency：SPUNet/MinkUNet 使用 SemanticKITTI 配置，PTv3 使用 NuScenes backbone 并适配为4输入通道和19类，同时沿用 S3DIS 基线的 non-FlashAttention 128-point patch fallback；OctFormer 适配为4输入通道、19类和覆盖 LiDAR 空间范围的12层 octree，并因 efficiency pack 不含 normal 而输入零 normal。网络权重未经训练，因此这些结果不能解释精度。

同一个 JSON 文件内的全部方法和样本必须来自同一物理 GPU。若主机有多张型号、显存和 driver 完全相同的 L20，不同且互不直接求加速比的 JSON 文件可以分配到不同物理卡并行补跑；覆盖检查比较 GPU platform signature，而每个文件继续保存实际 UUID。不能在同一张论文对比表内部混用不同 UUID 的记录，也不能混合不同型号或 driver。

### Ball-query 与 Arkade RT-core 补充基线

审稿补充实验提供 Pointcept `pointops.ball_query` 和 Arkade/TrueKNN 两个代表性基线。Ball query 与 kNN 的语义不同：前者返回固定半径内的邻居并由 `nsample` 截断或补齐，后者固定返回最近的 \(k\) 个点。因此脚本从同一批 cudaKDTree exact kNN 距离中校准全局第 \(k\) 邻居距离的 90% 分位半径，并同时保存 latency、valid-neighbor ratio、insufficient/truncated query ratio 和 set recall，不能只引用 latency。Pointcept kernel 是实际点云 pipeline 中使用的全 support 扫描实现，其结果只代表该公开 CUDA operator，不应泛化为所有 radius-search 算法。

Arkade 依赖 NVIDIA OptiX 8 SDK。SDK 不随仓库再分发；从 NVIDIA Developer 下载并解压后设置 `OptiX_INSTALL_DIR`。L20 属于带 RT core 的 Ada 数据中心 GPU，可以运行 OptiX。仓库中的 Arkade benchmark frontend 使用二进制 CUDA-ready 点集、分离 BVH build 与同步 TrueKNN query/refit 计时，并将官方未排序输出与 cudaKDTree ground truth 做 set-recall 校验。Arkade 官方实现使用 pinned host output 和 host-side TrueKNN 轮次检查，计时会保留这些实现成本。

```bash
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64
GPU=0 RUN_ID=l20_related bash run_related_baselines.sh
```

正式设置使用 250,000 点、\(k=24/32/48\)、pre/post、3 次 warm-up 和 10 次记录。`Query/benchmark_ball_query.py` 与 `Query/benchmark_arkade.py` 也可以分别运行，并用 `--max-samples 1 --crop-points 10000 --warmups 1 --repeats 1` 进行功能检查。Arkade 驱动默认允许有限次子进程重试，并可用 `--resume` 从原子保存的记录继续，以应对公开 OptiX 示例偶发的进程级故障；失败尝试不会进入计时样本。OptiX context/pipeline 创建、文件 I/O、体素化、裁块和 H2D 不计入算法 latency；Arkade 的 BVH build 和查询过程中发生的 radius refit 分别保留在 construction/query 边界中。

正式 L20 结果保存在 `results/L20/l20_ball_query_20260807/`、`results/L20/l20_arkade_20260807/`，论文侧汇总表为 `analysis/l20_related_baselines.md`。该表可用 `analysis/analyze_related_baselines.py --knn <matched-knn.json> --ball <ball.json> --ball-sweep <optional-sweep.json> --arkade <arkade.json> --output <summary.md>` 重新生成。所有 L20 目录的来源、目的和覆盖关系见 `results/L20/README.md`。

## 5. 汇总论文结果

```bash
bash analyze.sh
```

输出位于 `analysis/output/`：

- `benchmark_results.xlsx`：pre/post 主表、逐样本 query、网络汇总及逐样本时延；
- `paper_query_table` sheet：直接对应论文 *Time Cost(ms) for Different k and Query Modes*；
- `speedup_of_query_under_different_number_of_point.png` 与
  `speedup_of_construction_under_different_number_of_point.png`：以 nanoflann 为基线的点数曲线；
- `network_efficiency_comparison.png`：SPUNet、MinkUNet、PTv3、OctFormer、DeLA 和
  DeLA+FlashKNN 的逐房间点数—时延曲线；
- `network_latency.png`：每种网络的均值辅助图；
- `summary.md`：便于人工复核的表格。

在论文中横向比较 GPU 前，应使用同一 commit、同一数据 manifest、相同 warmup/repeat，且
按 GPU 型号分别报告，不把不同 GPU 的数值混入同一平均值。

## 6. 上游来源

固定版本见 [SOURCES.md](SOURCES.md)。官方数据入口：

- S3DIS: <https://cvg-data.inf.ethz.ch/s3dis/>
- Pointcept 压缩数据: <https://huggingface.co/datasets/Pointcept/s3dis-compressed/tree/main>
- SemanticKITTI: <https://semantic-kitti.org/dataset.html>

维护者需要重新生成 L20 交付包时执行：

```bash
TARGET_CUDA_ARCH=8.9 \
  bash scripts/build_native_wheelhouse.sh ../wheelhouse_sm89_cp310_torch271_cu128
python scripts/package_release.py \
  --wheelhouse ../wheelhouse_sm89_cp310_torch271_cu128 \
  --semantickitti-pack /path/to/semantickitti_pack \
  --output ../FlashKNN_L20_bundle_with_LiDAR.tar.gz
```

第一条命令可以在非 SM89 GPU 的 x86_64 构建机上执行；它通过 `nvcc` 交叉编译 SM89，
不运行 L20 kernel。第二条命令排除环境、临时编译产物、结果和 S3DIS 数据，只将代码、
校验过的 wheelhouse 与明确指定的 110 帧 LiDAR pack 写入压缩包，并在包内生成逐文件
SHA-256 清单。L20 上的安装脚本仍会执行真实 GPU smoke test，作为交叉编译之后不可省略的
最终验证。

如果没有提前构建 SM89 wheelhouse，也可以不传 `--wheelhouse` 打包。L20 用户仍只需运行
`bash scripts/install.sh`；安装器会联网下载固定版本源码并原生编译，区别只是首次安装耗时更长。
