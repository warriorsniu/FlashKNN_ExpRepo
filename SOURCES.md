# Source manifest

| Component | Source | Revision |
|---|---|---|
| DeLA | https://github.com/Matrix-ASC/DeLA | `0f427c36c5a991a28397ffdaaf8c205b80d6b30d` |
| DeepLA-Net | https://github.com/zeng-ziyin/DeepLA-Net | `7f572899de7db26d2c5eac538395d9932faafb89` |
| Pointcept | https://github.com/Pointcept/Pointcept | tag `v1.5.1`, `72a799353c568c29a3365c02a15129ebe7fe637a` |
| cudaKDTree | https://github.com/ingowald/cudaKDTree | `37ce4c377170d13670632143b976192d5b69ed03` |
| FLANN / FLANN-CUDA wrapper | https://github.com/flann-lib/flann | `f9caaf609d8b8cb2b7104a85cf59eb92c275a25d`; local PyTorch wrapper bundled |
| nanoflann / PyTorch wrapper | https://github.com/jlblancoc/nanoflann | header version 1.5.0 with the historical DeLA heap modification; local wrapper bundled |
| FAISS GPU | https://github.com/facebookresearch/faiss | tag `v1.12.0`; built from source for the executing GPU architecture |
| Arkade | https://github.com/MDurgaKeerthi/Arkade | `45b9425e14ed120f0e3fdfb3626131b1ac47d1fa`; Apache-2.0 source vendored with a local reproducible benchmark frontend |
| FlashKNN | this paper's implementation | bundled source; portable CUDA architecture flags added |

The added benchmark/configuration files are experiment glue. Upstream licenses remain in their respective subdirectories. Dataset files are intentionally not tracked or redistributed.
