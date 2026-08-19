#include "cukd/builder.h"
#include "cukd/knn.h"
#include "cukd/fcp.h"
#include "cukd/common.h"
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <unordered_map>

using data_t = float3;
using data_traits = cukd::default_data_traits<float3>;

// cudaKDTree builds its spatial tree with cudaMallocAsync rather than the
// PyTorch caching allocator.  torch.cuda.max_memory_allocated() therefore
// cannot observe the tree/build workspace.  This resource is used only by the
// dedicated memory benchmark and accounts for every allocation made by the
// upstream builder without changing the production timing entry point.
struct TrackingGpuMemoryResource final : cukd::GpuMemoryResource {
  std::unordered_map<void *, size_t> allocation_sizes;
  size_t current_bytes = 0;
  size_t peak_bytes = 0;

  cudaError_t malloc(void **ptr, size_t size, cudaStream_t stream) override {
#if CUDART_VERSION >= 11020
    cudaError_t error = cudaMallocAsync(ptr, size, stream);
#else
    cudaStreamSynchronize(stream);
    cudaError_t error = cudaMalloc(ptr, size);
#endif
    if (error == cudaSuccess) {
      allocation_sizes[*ptr] = size;
      current_bytes += size;
      peak_bytes = std::max(peak_bytes, current_bytes);
    }
    return error;
  }

  cudaError_t free(void *ptr, cudaStream_t stream) override {
    auto iterator = allocation_sizes.find(ptr);
    if (iterator != allocation_sizes.end()) {
      current_bytes -= iterator->second;
      allocation_sizes.erase(iterator);
    }
#if CUDART_VERSION >= 11020
    return cudaFreeAsync(ptr, stream);
#else
    cudaStreamSynchronize(stream);
    return cudaFree(ptr);
#endif
  }
};

template<int max_candidate>
__global__ void KnnKernel(const float3* queries, int num_queries,
                          const cukd::SpatialKDTree<float3, data_traits> tree,
                          float* distances, int* indices, int k, float radius) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_queries) return;
  // Match the historical FlashKNN/cuKD pair: max_candidate is the
  // power-of-two storage capacity, while the heap itself contains exactly k
  // entries. This is required for the paper's k=24 and k=48 workloads.
  cukd::HeapCandidateList<max_candidate> result(radius, k);
  cukd::stackBased::knn<decltype(result), float3, data_traits>(result, tree, queries[tid]);
  for (int i = 0; i < k; ++i) {
    int id = result.get_pointID(i);
    distances[tid * k + i] = id < 0 ? INFINITY : result.get_dist2(i);
    indices[tid * k + i] = id < 0 ? tid : id;
  }
}

void CukdKnnQueryTorch(const torch::Tensor& support, const torch::Tensor& query,
                       int k, torch::Tensor& indices, torch::Tensor& distances,
                       torch::Tensor& time_info, bool debug = false) {
  TORCH_CHECK(support.is_cuda() && query.is_cuda(), "support/query must be CUDA tensors");
  TORCH_CHECK(support.scalar_type() == torch::kFloat32 && query.scalar_type() == torch::kFloat32,
              "support/query must be float32");
  TORCH_CHECK(support.is_contiguous() && query.is_contiguous(), "support/query must be contiguous");
  TORCH_CHECK(k > 0 && k <= 64, "k must be in [1, 64]");
  float3* data = reinterpret_cast<float3*>(support.data_ptr<float>());
  const float3* query_data = reinterpret_cast<const float3*>(query.data_ptr<float>());
  float* output_distances = distances.data_ptr<float>();
  int* output_indices = indices.data_ptr<int>();
  cukd::SpatialKDTree<float3, data_traits> tree;
  int num_data = support.size(0);
  int num_queries = query.size(0);

  double start = cukd::common::getCurrentTime();
  cukd::buildTree(tree, data, num_data);
  cudaDeviceSynchronize();
  time_info[0] = cukd::common::getCurrentTime() - start;

  int threads = 1024;
  int blocks = (num_queries + threads - 1) / threads;
  start = cukd::common::getCurrentTime();
  int bit_length = 0;
  for (int value = k - 1; value > 0; value >>= 1) ++bit_length;
  switch (bit_length) {
    case 0: KnnKernel<1><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 1: KnnKernel<2><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 2: KnnKernel<4><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 3: KnnKernel<8><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 4: KnnKernel<16><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 5: KnnKernel<32><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 6: KnnKernel<64><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
  }
  cudaDeviceSynchronize();
  time_info[1] = cukd::common::getCurrentTime() - start;
  // SpatialKDTree owns node and primitive-index buffers.  Release them after
  // timing so repeated benchmark calls do not accumulate GPU allocations.
  cukd::free(tree);
}

void CukdKnnQueryTorchMemory(const torch::Tensor& support,
                             const torch::Tensor& query,
                             int k,
                             torch::Tensor& indices,
                             torch::Tensor& distances,
                             torch::Tensor& memory_info) {
  TORCH_CHECK(support.is_cuda() && query.is_cuda(),
              "support/query must be CUDA tensors");
  TORCH_CHECK(support.scalar_type() == torch::kFloat32 &&
                  query.scalar_type() == torch::kFloat32,
              "support/query must be float32");
  TORCH_CHECK(support.is_contiguous() && query.is_contiguous(),
              "support/query must be contiguous");
  TORCH_CHECK(k > 0 && k <= 64, "k must be in [1, 64]");
  TORCH_CHECK(!memory_info.is_cuda() &&
                  memory_info.scalar_type() == torch::kInt64 &&
                  memory_info.is_contiguous() && memory_info.numel() >= 4,
              "memory_info must be a contiguous CPU int64 tensor with >=4 entries");

  float3 *data = reinterpret_cast<float3 *>(support.data_ptr<float>());
  const float3 *query_data =
      reinterpret_cast<const float3 *>(query.data_ptr<float>());
  float *output_distances = distances.data_ptr<float>();
  int *output_indices = indices.data_ptr<int>();
  const int num_data = support.size(0);
  const int num_queries = query.size(0);

  TrackingGpuMemoryResource memory_resource;
  cukd::SpatialKDTree<float3, data_traits> tree;
  cukd::buildTree(tree, data, num_data, {}, 0, memory_resource);
  cudaDeviceSynchronize();
  const size_t persistent_tree_bytes = memory_resource.current_bytes;

  const int threads = 1024;
  const int blocks = (num_queries + threads - 1) / threads;
  int bit_length = 0;
  for (int value = k - 1; value > 0; value >>= 1) ++bit_length;
  switch (bit_length) {
    case 0: KnnKernel<1><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 1: KnnKernel<2><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 2: KnnKernel<4><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 3: KnnKernel<8><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 4: KnnKernel<16><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 5: KnnKernel<32><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
    case 6: KnnKernel<64><<<blocks, threads>>>(query_data, num_queries, tree, output_distances, output_indices, k, INFINITY); break;
  }
  cudaDeviceSynchronize();

  const int64_t output_bytes =
      static_cast<int64_t>(indices.numel() * indices.element_size() +
                           distances.numel() * distances.element_size());
  int64_t *values = memory_info.data_ptr<int64_t>();
  values[0] = output_bytes;
  values[1] = static_cast<int64_t>(memory_resource.peak_bytes);
  values[2] = static_cast<int64_t>(persistent_tree_bytes);
  values[3] = output_bytes + static_cast<int64_t>(memory_resource.peak_bytes);

  cukd::free(tree, 0, memory_resource);
  cudaDeviceSynchronize();
  TORCH_CHECK(memory_resource.current_bytes == 0,
              "tracked cudaKDTree allocations were not fully released");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("CukdKnnQueryTorch", &CukdKnnQueryTorch, "Exact cudaKDTree kNN");
  module.def("CukdKnnQueryTorchMemory", &CukdKnnQueryTorchMemory,
             "Exact cudaKDTree kNN with allocation accounting");
}
