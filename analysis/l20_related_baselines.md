# L20 ball-query and Arkade comparison

All values use the same 250,000-point S3DIS crops on NVIDIA L20 (81 rooms). The tables report the median of per-room median times. Pointcept ball query uses a global 90th-percentile radius and `nsample=k`; Arkade uses the official TrueKNN radius-doubling path starting from 0.02 m. These operators have different semantics, so latency must be read together with recall and coverage.

| Mode | k | FlashKNN ms | cudaKDTree ms | Ball query ms | Ball recall | Ball valid | Arkade ms | Arkade recall | Arkade rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 24 | 1.2735 | 1.9805 | 349.1011 | 0.871796 | 0.987769 | 110.4410 | 0.992115 | 5 |
| pre | 32 | 1.5274 | 2.7729 | 349.2008 | 0.877319 | 0.987572 | 141.7555 | 0.998212 | 5 |
| pre | 48 | 2.4701 | 4.4100 | 349.4555 | 0.881446 | 0.986589 | 198.7305 | 0.998406 | 5 |
| post | 24 | 0.6365 | 0.6831 | 116.7201 | 0.862756 | 0.985016 | 38.3126 | 0.991917 | 5 |
| post | 32 | 0.6858 | 0.9235 | 116.7395 | 0.868399 | 0.984394 | 49.9041 | 0.998230 | 5 |
| post | 48 | 0.9018 | 1.4035 | 116.8179 | 0.872077 | 0.983046 | 70.6205 | 0.998107 | 5 |

## Ball-query coverage

`Insufficient` is the fraction of queries with fewer than k points in the radius; `truncated` is the fraction with more than k and therefore subject to `nsample` truncation.

| Mode | k | Radius m | Valid slots | Insufficient | Truncated |
|---|---:|---:|---:|---:|---:|
| pre | 24 | 0.057983 | 0.987769 | 0.101139 | 0.813581 |
| pre | 32 | 0.066528 | 0.987572 | 0.100306 | 0.830417 |
| pre | 48 | 0.081031 | 0.986589 | 0.100506 | 0.848895 |
| post | 24 | 0.058215 | 0.985016 | 0.099644 | 0.828785 |
| post | 32 | 0.066888 | 0.984394 | 0.099649 | 0.843384 |
| post | 48 | 0.081400 | 0.983046 | 0.099886 | 0.860914 |

### One-room radius sensitivity

This diagnostic uses Area_1/WC_1 and locally calibrated distance quantiles. A larger radius does not improve kNN recall in this operator once more than `nsample=k` candidates are present, because the fixed-radius operator truncates candidates rather than selecting the nearest k.

| Mode | k | Quantile | Radius m | Query ms | Recall | Insufficient | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre | 24 | 0.50 | 0.054378 | 349.1140 | 0.906983 | 0.503396 | 0.347932 |
| pre | 24 | 0.90 | 0.058138 | 349.1073 | 0.875198 | 0.099672 | 0.821368 |
| pre | 24 | 0.99 | 0.069829 | 349.0994 | 0.637221 | 0.011048 | 0.985556 |
| pre | 32 | 0.50 | 0.063063 | 349.1688 | 0.911995 | 0.498240 | 0.375192 |
| pre | 32 | 0.90 | 0.066940 | 349.2598 | 0.878155 | 0.095620 | 0.846140 |
| pre | 32 | 0.99 | 0.082240 | 349.2675 | 0.609242 | 0.009784 | 0.987748 |
| pre | 48 | 0.50 | 0.077427 | 349.3930 | 0.914465 | 0.499268 | 0.401836 |
| pre | 48 | 0.90 | 0.081468 | 349.4893 | 0.881208 | 0.098484 | 0.863756 |
| pre | 48 | 0.99 | 0.102879 | 349.7645 | 0.579756 | 0.009064 | 0.989296 |
| post | 24 | 0.50 | 0.054452 | 116.6674 | 0.902614 | 0.499698 | 0.353527 |
| post | 24 | 0.90 | 0.058464 | 116.7016 | 0.865038 | 0.101529 | 0.833685 |
| post | 24 | 0.99 | 0.072884 | 116.7399 | 0.592439 | 0.009423 | 0.987154 |
| post | 32 | 0.50 | 0.063071 | 116.7441 | 0.907392 | 0.499714 | 0.375449 |
| post | 32 | 0.90 | 0.067261 | 116.7534 | 0.867682 | 0.098348 | 0.856331 |
| post | 32 | 0.99 | 0.084225 | 116.8520 | 0.586142 | 0.010644 | 0.986416 |
| post | 48 | 0.50 | 0.077473 | 116.7047 | 0.909504 | 0.496653 | 0.404396 |
| post | 48 | 0.90 | 0.082012 | 116.7670 | 0.869380 | 0.099659 | 0.872946 |
| post | 48 | 0.99 | 0.105045 | 116.9165 | 0.561225 | 0.009785 | 0.988541 |

## Index construction

Ball query has no persistent index construction in this implementation. Arkade's value is its OptiX BVH build; FlashKNN and cudaKDTree use their respective index-build timing boundaries.

| Mode | k | FlashKNN ms | cudaKDTree ms | Arkade BVH ms |
|---|---:|---:|---:|---:|
| pre | 24 | 1.1016 | 3.1021 | 4.1002 |
| pre | 32 | 1.1065 | 3.1024 | 4.1276 |
| pre | 48 | 1.1261 | 3.1019 | 4.1352 |
| post | 24 | 1.5043 | 3.1204 | 4.1113 |
| post | 32 | 1.5035 | 3.1229 | 4.1177 |
| post | 48 | 1.5074 | 3.1235 | 4.1208 |

## Interpretation

The Pointcept operator is a representative pipeline implementation, but it assigns one thread to each query and scans the full support set. Its latency therefore characterizes this public operator rather than all possible grid- or tree-accelerated radius searches. Its lower kNN recall is expected because fixed-radius sampling is not required to return the nearest k points.

Arkade exercises a genuinely different hardware path through OptiX RT cores. The benchmark retains its pinned-host output, host-side completion scan, iterative radius expansion, and acceleration-structure refits. Its measured recall, rather than an assumed exact label, is reported because the public AABB implementation and voxel-distance ties do not always return the same index set as cudaKDTree.
