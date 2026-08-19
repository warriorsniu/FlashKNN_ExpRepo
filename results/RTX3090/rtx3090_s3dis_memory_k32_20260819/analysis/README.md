# S3DIS fixed-250k GPU memory summary

GPU: `NVIDIA GeForce RTX 3090`; $k=32$; 81 rooms; pre/post query.

Metric: peak incremental method-owned GPU allocation above CUDA-ready inputs. It includes construction/index, workspace and outputs, and excludes file I/O, voxelization, crop, H2D and input tensors.

| Mode | Method | Mean (MiB) | Room SD | 95% CI | Min--max (MiB) |
|---|---|---:|---:|---:|---:|
| post | cuda_kdtree | 37.73 | 0.30 | [37.66, 37.80] | 36.92--38.83 |
| post | faiss_flat | 1564.22 | 0.49 | [1564.11, 1564.33] | 1563.52--1566.32 |
| post | faiss_ivf | 1566.99 | 0.53 | [1566.87, 1567.11] | 1566.34--1569.19 |
| post | flashknn | 121.12 | 1.78 | [120.73, 121.52] | 117.85--128.48 |
| pre | cuda_kdtree | 82.56 | 0.00 | [82.56, 82.56] | 82.56--82.57 |
| pre | faiss_flat | 1632.32 | 0.18 | [1632.28, 1632.36] | 1631.37--1633.24 |
| pre | faiss_ivf | 1635.11 | 0.20 | [1635.06, 1635.15] | 1634.13--1636.04 |
| pre | flashknn | 290.61 | 0.95 | [290.40, 290.82] | 287.50--292.56 |

FAISS uses the same default `StandardGpuResources` scratch policy as the latency benchmark. cudaKDTree memory includes PyTorch output tensors and all allocations made by its native spatial-tree builder.
