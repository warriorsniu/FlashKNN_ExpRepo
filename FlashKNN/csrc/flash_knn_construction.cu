// Fused CUDA primitives for Morton encoding and 3x3x3 neighbor-key creation.

#include "flash_knn_construction.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace {

__device__ __forceinline__ int64_t morton_key(
    int64_t x, int64_t y, int64_t z, int64_t batch) {
  const uint32_t ux = static_cast<uint32_t>(x) & 0xffffU;
  const uint32_t uy = static_cast<uint32_t>(y) & 0xffffU;
  const uint32_t uz = static_cast<uint32_t>(z) & 0xffffU;
  uint64_t key = 0;
#pragma unroll
  for (int bit = 0; bit < 16; ++bit) {
    key |= static_cast<uint64_t>((ux >> bit) & 1U) << (3 * bit + 2);
    key |= static_cast<uint64_t>((uy >> bit) & 1U) << (3 * bit + 1);
    key |= static_cast<uint64_t>((uz >> bit) & 1U) << (3 * bit);
  }
  key |= static_cast<uint64_t>(batch) << 48;
  return static_cast<int64_t>(key);
}

__global__ void morton_encode_kernel(
    const int64_t* grid_coord,
    const int64_t* batch_idx,
    int64_t* keys,
    int64_t count) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int64_t* point = grid_coord + 3 * index;
  keys[index] = morton_key(point[0], point[1], point[2], batch_idx[index]);
}

__device__ __forceinline__ int neighbor_offset(int position) {
  return position == 0 ? 0 : (position == 1 ? -1 : 1);
}

__global__ void morton_neighbor_keys_kernel(
    const int64_t* grid_coord_down,
    const int64_t* parent_batch,
    int64_t* neighbor_keys,
    int64_t parent_count) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = parent_count * 27;
  if (index >= total) {
    return;
  }
  const int64_t parent = index / 27;
  const int neighbor = static_cast<int>(index - parent * 27);
  const int64_t* point = grid_coord_down + 3 * parent;
  const int64_t x = point[0] + neighbor_offset(neighbor / 9);
  const int64_t y = point[1] + neighbor_offset((neighbor / 3) % 3);
  const int64_t z = point[2] + neighbor_offset(neighbor % 3);
  neighbor_keys[index] = morton_key(x, y, z, parent_batch[parent]);
}

void validate_inputs(
    const torch::Tensor& coordinates,
    const torch::Tensor& batch,
    const char* operation) {
  TORCH_CHECK(coordinates.is_cuda() && batch.is_cuda(), operation,
              ": inputs must be CUDA tensors");
  TORCH_CHECK(coordinates.scalar_type() == torch::kInt64 &&
                  batch.scalar_type() == torch::kInt64,
              operation, ": inputs must use int64");
  TORCH_CHECK(coordinates.is_contiguous() && batch.is_contiguous(), operation,
              ": inputs must be contiguous");
  TORCH_CHECK(coordinates.dim() == 2 && coordinates.size(1) == 3, operation,
              ": coordinates must have shape [N, 3]");
  TORCH_CHECK(batch.dim() == 1 && batch.size(0) == coordinates.size(0), operation,
              ": batch must have shape [N]");
  TORCH_CHECK(coordinates.device() == batch.device(), operation,
              ": inputs must be on the same device");
}

}  // namespace

torch::Tensor FlashKNN_Morton_Encode(
    const torch::Tensor& grid_coord,
    const torch::Tensor& batch_idx) {
  validate_inputs(grid_coord, batch_idx, "FlashKNN_Morton_Encode");
  const c10::cuda::CUDAGuard device_guard(grid_coord.device());
  torch::Tensor keys = torch::empty(
      {grid_coord.size(0)}, grid_coord.options().dtype(torch::kInt64));
  if (grid_coord.size(0) == 0) {
    return keys;
  }
  constexpr int threads = 256;
  const int64_t blocks = (grid_coord.size(0) + threads - 1) / threads;
  morton_encode_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      grid_coord.data_ptr<int64_t>(), batch_idx.data_ptr<int64_t>(),
      keys.data_ptr<int64_t>(), grid_coord.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return keys;
}

torch::Tensor FlashKNN_Morton_Neighbor_Keys(
    const torch::Tensor& grid_coord_down,
    const torch::Tensor& parent_batch) {
  validate_inputs(
      grid_coord_down, parent_batch, "FlashKNN_Morton_Neighbor_Keys");
  const c10::cuda::CUDAGuard device_guard(grid_coord_down.device());
  torch::Tensor keys = torch::empty(
      {grid_coord_down.size(0) * 27},
      grid_coord_down.options().dtype(torch::kInt64));
  if (grid_coord_down.size(0) == 0) {
    return keys;
  }
  constexpr int threads = 256;
  const int64_t total = grid_coord_down.size(0) * 27;
  const int64_t blocks = (total + threads - 1) / threads;
  morton_neighbor_keys_kernel<<<blocks, threads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      grid_coord_down.data_ptr<int64_t>(), parent_batch.data_ptr<int64_t>(),
      keys.data_ptr<int64_t>(), grid_coord_down.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return keys;
}
