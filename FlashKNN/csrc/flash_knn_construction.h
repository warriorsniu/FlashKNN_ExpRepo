#pragma once

// Fused Morton-key primitives used by FlashKNN's voxel-graph construction.

#include <torch/extension.h>

torch::Tensor FlashKNN_Morton_Encode(
    const torch::Tensor& grid_coord,
    const torch::Tensor& batch_idx);

torch::Tensor FlashKNN_Morton_Neighbor_Keys(
    const torch::Tensor& grid_coord_down,
    const torch::Tensor& parent_batch);
