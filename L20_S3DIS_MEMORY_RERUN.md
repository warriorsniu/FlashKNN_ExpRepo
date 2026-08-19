# L20 S3DIS memory补测任务

## 目的

Reviewer #2 要求将 cross-GPU latency and memory results 写入正文。L20 的
S3DIS fixed-250k latency 和 network latency 已完成，但 canonical/final-refresh
JSON 目前没有四种主要 GPU 方法的显存 footprint。`system.json` 中的 GPU 总显存
以及 NCU 的 HBM read/write traffic 都不能替代该指标。

本任务只补 L20 + S3DIS 上 FlashKNN、cudaKDTree、FAISS GPU Flat 和 matched-recall
FAISS GPU IVF-Flat 的 memory，不重跑 latency。L20 latency 表仍只展示 FlashKNN 与
cudaKDTree；memory 则在两平台使用完全一致的四方法集合，以便进行严格的 cross-GPU
对照。

## 冻结协议

- 单张空闲 L20，不使用多卡，不允许训练或其他 compute co-tenant；
- 与主表相同的 81 个 S3DIS fixed-250k deterministic crops；
- pre/post query，代表性 `k=32`，FlashKNN `alpha=4`；
- methods：`flashknn cuda_kdtree faiss_flat faiss_ivf`；
- FAISS Flat 使用精确 GPU Flat；matched IVF 逐房间复用 L20 canonical S3DIS
  结果中已冻结的 `nlist`/`nprobe`，不得重新进行 recall matching；
- CUDA-ready input boundary：排除文件 I/O、voxelization、crop、H2D 和输入
  tensor；包含 construction/tree、workspace 和 output；
- 输出逐房间结果以及 room mean、sample SD、Student-t 95% CI。

cudaKDTree 的 tree/build workspace 由原生 `cudaMallocAsync` 分配，不能使用
`torch.cuda.max_memory_allocated()` 单独测量。本提交新增
`CukdKnnQueryTorchMemory`，通过跟踪其 native memory resource 统计 builder/tree
峰值，并加上输出 tensor；不要改回仅使用 PyTorch allocator 的实现。

## L20执行步骤

从远端 `main` 拉取本提交后，先在 L20 的 PyTorch 2.7.1+cu128/CUDA 12.8 环境中
重编 cudaKDTree（`sm_89`）：

```bash
cd /path/to/FlashKNN_ExpRepo
git pull --ff-only origin main
source .venv/bin/activate
source scripts/runtime_env.sh
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
python -m pip install -v --no-build-isolation --force-reinstall --no-deps \
  Query/ThirdParty/cudaKDTree
python - <<'PY'
import Cukd.CuFun as cukd
assert hasattr(cukd, "CukdKnnQueryTorchMemory")
print(cukd.__file__)
PY
```

确认目标 GPU 没有 compute process 后运行正式实验：

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
GPU=0 \
METHODS="flashknn cuda_kdtree faiss_flat faiss_ivf" \
RUN_ID=l20_s3dis_memory_k32_20260819 \
bash run_s3dis_memory.sh
```

runner 会在目标 GPU 存在 compute co-tenant 时拒绝正式运行。预期输出：

```text
results/L20/l20_s3dis_memory_k32_20260819/
  system.json
  s3dis_memory_k32.json
  analysis/summary.json
  analysis/README.md
  analysis/summary.md
```

正式结果必须包含 162 条唯一记录（81 rooms × pre/post），且 analysis validator
通过。完成后：

1. 将上述完整结果目录提交到远端 `main`；其中`analysis/README.md`是应被Git
   保留的人类可读汇总，`summary.md`若受ignore规则影响可不单独强制加入；
2. 在 `results/L20/README.md` 增加均值表和测量边界；
3. 在 `results/RESULT_SELECTION.md` 将该目录登记为 retained/final；
4. 在提交说明中写明：仅补 memory，未覆盖或更改任何既有 latency；
5. 回报 commit SHA、物理 GPU UUID、环境版本、记录数和 pre/post 四方法汇总。

## 本机参照

RTX 3090 使用同一 runner 和同一 measurement boundary，正式结果目录为
`results/RTX3090/rtx3090_s3dis_memory_k32_20260819/`。L20 结果不得与 RTX 3090
混合平均；只在同平台比较方法，并在正文分平台报告。

RTX 3090 的81房间均值如下，仅用于L20运行后的量级sanity check：

| Mode | FlashKNN | cudaKDTree | FAISS Flat | matched IVF |
| --- | ---: | ---: | ---: | ---: |
| Pre | 290.61 MiB | 82.56 MiB | 1632.32 MiB | 1635.11 MiB |
| Post | 121.12 MiB | 37.73 MiB | 1564.22 MiB | 1566.99 MiB |

L20 要求完整四列。由于两平台的 CUDA allocator/runtime 不同，不要求逐字节相同；若差异
明显，应先检查 measurement boundary、输入点数、`k`、extension hash 和
FAISS/allocator 配置，不得直接挑选更接近 RTX 3090 的重复批次。
