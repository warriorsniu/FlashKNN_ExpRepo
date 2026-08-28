# FlashKNN experiment repository

This repository contains the CUDA implementation, benchmark adapters, and
scripts used to reproduce the FlashKNN experiments on S3DIS and
SemanticKITTI. All benchmark launchers use one GPU per process.

Detailed experiment protocols are listed in [EXPERIMENTS.md](EXPERIMENTS.md),
the retained result sets are documented in [results/README.md](results/README.md),
and upstream revisions are listed in [SOURCES.md](SOURCES.md).

## Supported environments

| GPU | Python | PyTorch | CUDA toolkit | CUDA architecture |
|---|---:|---:|---:|---:|
| NVIDIA RTX 3090 | 3.10 | 2.7.1+cu118 | 11.8 | sm_86 |
| NVIDIA L20 | 3.10 | 2.7.1+cu128 | 12.8 | sm_89 |

The installation script builds native extensions for the visible GPU. Use the
matching CUDA toolkit and expose only the GPU on which the experiments will
run.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/warriorsniu/FlashKNN_ExpRepo.git
cd FlashKNN_ExpRepo
conda create -n flashknn-exp python=3.10 -y
conda activate flashknn-exp
export CUDA_VISIBLE_DEVICES=0
```

If CUDA is not detected automatically, set `CUDA_HOME` before installation:

```bash
# RTX 3090
export CUDA_HOME=/usr/local/cuda-11.8

# L20
# export CUDA_HOME=/usr/local/cuda-12.8
```

Install Python dependencies and compile the native extensions:

```bash
bash scripts/install.sh
```

The installer verifies the PyTorch/CUDA versions, compiled GPU architecture,
and required extension imports before it exits.

## Data preparation

Datasets are not redistributed by this repository.

For S3DIS, read and accept the dataset license before allowing the preparation
script to obtain or process the data:

```bash
ACCEPT_S3DIS_LICENSE=1 bash prepare_data.sh
```

An existing S3DIS directory can be supplied with `S3DIS_ROOT`:

```bash
S3DIS_ROOT=/path/to/s3dis \
SEMANTICKITTI_ROOT=/path/to/semantic-kitti \
bash prepare_data.sh
```

SemanticKITTI can alternatively be supplied as a prepared pack through
`SEMANTICKITTI_PACK`, placed at
`data/incoming/semantickitti_pack`, or downloaded from a user-authorized source
through `SEMANTICKITTI_PACK_URL` or `KITTI_VELODYNE_URL`.

Successful preparation writes resolved paths to `data/paths.env`. The
preflight check can then be run independently:

```bash
bash scripts/preflight.sh --data-only
```

## Running the benchmarks

Run a one-sample functional check before collecting formal timings:

```bash
GPU=0 SMOKE=1 RUN_ID=smoke bash run_all.sh
```

Smoke timings are only for functional validation. For a formal single-GPU run,
use an idle GPU and omit `SMOKE`:

```bash
GPU=0 RUN_ID=my_formal_run bash run_all.sh
```

`run_all.sh` executes the query and network-latency suites, validates their
coverage, and writes the analysis output. Individual experiment groups can be
run with the following launchers:

| Launcher | Experiment |
|---|---|
| `run_query.sh` | S3DIS and SemanticKITTI query benchmarks |
| `run_network_latency.sh` | S3DIS and SemanticKITTI network latency |
| `run_s3dis_memory.sh` | S3DIS GPU-memory comparison |
| `run_ablation.sh` | sorting, storage, and skip ablations |
| `run_thread_grouping_ablation.sh` | thread-grouping strategies |
| `run_adaptive_neighborhood.sh` | fixed and adaptive neighborhoods |
| `run_gmss_ablation.sh` | GMSS k sweep |
| `run_torch_knnquery.sh` | direct fixed-grid execution diagnostic |
| `run_pytorch3d_ball_query.sh` | PyTorch3D ball-query comparison |

All launchers accept `GPU` and `RUN_ID`. Some formal ablation launchers include
an explicit opt-in guard; the required variable is printed when such a runner
is invoked.

## Results and analysis

Results are written beneath a hardware-specific directory:

```text
results/
  L20/<RUN_ID>/
  RTX3090/<RUN_ID>/
```

The main result packs use the following layout when the corresponding suite is
enabled:

```text
<RUN_ID>/
  system.json
  query/*.json
  network/*.json
  memory/*.json
```

Analyze one run explicitly:

```bash
python analysis/analyze_results.py \
  --results results/RTX3090/<RUN_ID> \
  --output-dir analysis/output/<RUN_ID>
```

Validate a complete run produced by the current `run_all.sh` protocol:

```bash
python scripts/validate_result_coverage.py \
  --run-dir results/RTX3090/<RUN_ID>
```

Use `--smoke` with the validator only for a run produced with `SMOKE=1`.

## Source and license information

Bundled upstream components retain their original licenses. Exact upstream
revisions and local adapter boundaries are recorded in [SOURCES.md](SOURCES.md).
