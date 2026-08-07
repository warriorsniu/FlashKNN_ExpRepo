# FlashKNN: Fast GPU k-Nearest Neighbor Search for Point Cloud Parsing

**FlashKNN** is an I/O-aware fast approximate k-nearest neighbor (kNN) search algorithm for GPU, designed to accelerate neighbor search-based point cloud parsing neural networks. It achieves up to **126.9x** speedup over CPU-based kNN (nanoflann) and **2.8x** over GPU-based kNN (cudaKDTree) while maintaining **over 99% recall**.

## Performance

FlashKNN enables neighbor search-based point cloud networks to surpass voxel-based and serialization-based methods in inference efficiency without compromising segmentation accuracy.

![time cost comparison](fig/time_cost_comparison.png)

### kNN Search Efficiency (Pre-downsampling Query, 250K points)

| k | Construct (ms) | Query (ms) | Total (ms) | Speedup vs. nanoflann |
|---|---------------|------------|------------|----------------------|
| 16 | 1.44 | 1.83 | 3.27 | 107.5x |
| 32 | 1.43 | 3.10 | 4.53 | 126.9x |
| 48 | 1.41 | 5.20 | 6.61 | 122.8x |
| 64 | 1.42 | 5.70 | 7.12 | 145.1x |

### Scalability

Query and construction speedups across different point cloud scales (baseline: nanoflann):

![speedup query](fig/speedup_query.png)
![speedup construction](fig/speedup_construction.png)

## Requirements

- NVIDIA GPU with CUDA 11.6+
- Python 3.10

## Installation

```bash
conda create -n flashknn python=3.10
conda activate flashknn

# PyTorch with CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 numpy==1.26.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install tqdm
python -m pip install "setuptools<82.0.0"
pip install . --no-build-isolation
```

## Usage

```bash
python test.py
```

```python
from functions.FlashKnnWrapper import FlashKNN

knn = FlashKNN(num_nbr=32, num_down=2, debug=True, print_time=True)
nbr_indices = knn.query(grid_coord, batch, coord, memory_mode="SM", sorting_mode="PS")
```

`test.py` provides a complete benchmark including recall evaluation against exact kNN.

## Project Structure

```
├── csrc/            # CUDA kernel implementations
├── functions/       # Python wrapper and utility functions
├── test.py          # Benchmark script
├── test_data/       # Sample point cloud data
├── setup.py         # Build configuration
└── fig/             # Figures
```

## License

See [LICENSE](LICENSE) for details.
