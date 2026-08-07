#include "cukd/builder.h"
#include "cukd/knn.h"
#include "cukd/fcp.h"
#include "cukd/common.h"
#include <pybind11/stl.h>
#include <torch/extension.h>

using data_t = float3;
using data_traits = cukd::default_data_traits<float3>;

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("CukdKnnQueryTorch", &CukdKnnQueryTorch, "Exact cudaKDTree kNN");
}
