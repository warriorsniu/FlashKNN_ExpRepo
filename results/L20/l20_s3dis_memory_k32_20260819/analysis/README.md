# S3DIS fixed-250k GPU memory summary

GPU: `NVIDIA L20`; $k=32$; 81 rooms; pre/post query.

Metric: peak incremental method-owned GPU allocation above CUDA-ready inputs. It includes construction/index, workspace and outputs, and excludes file I/O, voxelization, crop, H2D and input tensors.

| Mode | Method | Mean (MiB) | Room SD | 95% CI | Min--max (MiB) |
|---|---|---:|---:|---:|---:|
| post | cuda_kdtree | 37.73 | 0.30 | [37.66, 37.80] | 36.92--38.83 |
| post | faiss_flat | 1564.26 | 0.54 | [1564.14, 1564.38] | 1563.51--1566.47 |
| post | faiss_ivf | 1567.03 | 0.59 | [1566.90, 1567.16] | 1566.34--1569.50 |
| post | flashknn | 121.15 | 1.74 | [120.76, 121.53] | 117.85--128.79 |
| pre | cuda_kdtree | 82.56 | 0.00 | [82.56, 82.56] | 82.56--82.57 |
| pre | faiss_flat | 1632.32 | 0.18 | [1632.28, 1632.36] | 1631.37--1633.24 |
| pre | faiss_ivf | 1635.11 | 0.20 | [1635.06, 1635.15] | 1634.14--1636.03 |
| pre | flashknn | 290.66 | 1.12 | [290.41, 290.90] | 288.01--294.21 |

FAISS uses the same default `StandardGpuResources` scratch policy as the latency benchmark. cudaKDTree memory includes PyTorch output tensors and all allocations made by its native spatial-tree builder.
