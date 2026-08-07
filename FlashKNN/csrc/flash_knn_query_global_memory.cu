#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <assert.h>


#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <thrust/extrema.h>

#include "flash_knn_query_global_memory.h"

constexpr uint64_t WARP_SIZE = 32;
constexpr uint64_t WARP_SIZE_1 = WARP_SIZE - 1;
constexpr uint64_t grid_x = 512;
constexpr uint64_t ArraySizePerThread = 2;
constexpr uint64_t MaxArrayIdx = ArraySizePerThread-1;

using namespace std;

template<int MaxCandidates>
__global__ void FlashKNN_Query_GM_kernel(
    int* Indices_out,                   // Nq, K
    float* Dis_out,                     // Nq, K
    const float* xyz_Support,           // Ns, 3
    const float* xyz_Query,             // Nq, 3
    const int* Parent2Child_Support,    // Nd+1
    const int* Parent2Child_Query,      // Nd+1
    const int* QueryIndex,              // Nq
    const int* Query2Parent,            // Nq
    const int* ParentNeigh,             // Nd, 27
    const int* CumCntInNeigh,           // Nd, 28
    const int num_nbr,
    const int N_PRT,
    const int N_query,
    const float cut_radiu2 = INFINITY
){
    int block_idx = blockIdx.x + blockIdx.y*gridDim.x;
    int thread_num_in_block = blockDim.x*blockDim.y;
    int thread_idx_in_grid = block_idx*thread_num_in_block + threadIdx.y*blockDim.x + threadIdx.x;
    
    for(int query_idx = thread_idx_in_grid;query_idx < N_query;query_idx+=thread_num_in_block*gridDim.x*gridDim.y){
        // if(blockIdx.x == 1 && blockIdx.y == 0){
        //     printf("query_idx: %d, thread_idx_in_grid: %d, thread_num_in_block: %d, block_idx: %d, threadIdx.x: %d, threadIdx.y: %d", query_idx, thread_idx_in_grid, thread_num_in_block, block_idx, threadIdx.x, threadIdx.y);
        // }
        float x_query = xyz_Query[query_idx*3 + 0];
        float y_query = xyz_Query[query_idx*3 + 1];
        float z_query = xyz_Query[query_idx*3 + 2];
        int prt_idx_query = Query2Parent[query_idx];
        // 初始化候选列表
        float dis_candidate_list[MaxCandidates];
        int idx_candidate_list[MaxCandidates];
        #pragma unroll
        for(int i = 0;i < MaxCandidates;i++){
            dis_candidate_list[i] = cut_radiu2;
            idx_candidate_list[i] = QueryIndex[query_idx];
        }
        //在降采样图上寻找候选邻域
        for(int nbr_idx = 0;nbr_idx<27;nbr_idx++){
            int parent_nbr_idx = ParentNeigh[27*prt_idx_query+nbr_idx];
            if(parent_nbr_idx != -1){
                int child_idx_support_start = Parent2Child_Support[parent_nbr_idx];
                int child_idx_support_end = Parent2Child_Support[parent_nbr_idx+1];
                for(int child_idx_support = child_idx_support_start;child_idx_support < child_idx_support_end;child_idx_support+=1){
                    float dis_x = xyz_Support[child_idx_support*3 + 0] - x_query;
                    float dis_y = xyz_Support[child_idx_support*3 + 1] - y_query;
                    float dis_z = xyz_Support[child_idx_support*3 + 2] - z_query;
                    float dis = dis_x*dis_x + dis_y*dis_y + dis_z*dis_z;
                    if(dis < dis_candidate_list[0]){
                        Re_Heap(child_idx_support, dis, idx_candidate_list, dis_candidate_list, num_nbr);
                    }
                }
            }
        }
        #pragma unroll
        for(int i = 0;i < MaxCandidates;i++){
            if(i < num_nbr){
                Indices_out[query_idx*num_nbr + i] = idx_candidate_list[i];
                Dis_out[query_idx*num_nbr + i] = dis_candidate_list[i];
            }
        }
    }
}

void FlashKNN_Query_GM(
    const torch::Tensor& xyz_Support,             // Ns, 3
    const torch::Tensor& xyz_Query,               // Nq, 3
    const torch::Tensor& Parent2Child_Support,    // Nd+1
    const torch::Tensor& Parent2Child_Query,      // Nd+1
    const torch::Tensor& QueryIndex,              // Nq
    const torch::Tensor& Query2Parent,            // Nq
    const torch::Tensor& ParentNeigh,             // Nd, 27
    const torch::Tensor& CumCntInNeigh,           // Nd, 28
    torch::Tensor& Indices_out,                   // Nq, K
    torch::Tensor& Dis_out,                     // Nq, K
    const int num_nbr,
    const float cut_radiu2
){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(WARP_SIZE,1,1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const uint64_t TotalShareMemory = at::cuda::getCurrentDeviceProperties()->sharedMemPerMultiprocessor;
    const uint64_t TotalRegs = at::cuda::getCurrentDeviceProperties()->regsPerMultiprocessor;
    // const uint64_t max_point_num_loaded = 2000;
    uint64_t dtypesize = 4;
    const uint64_t max_point_num_loaded = 256; //留出余量，防止共享内存占用过多影响性能
    // const uint64_t maxGridX = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
    // std::cout<<"TotalShareMemory: "<<TotalShareMemory<<endl;
    // std::cout<<"max_point_num_loaded: "<<max_point_num_loaded<<endl;
    // std::cout<<"sizeof(GridCoord.dtype()): "<<sizeof(GridCoord.dtype())<<endl;
    const dim3 blocks(grid_x, std::min((uint64_t)4, maxGridY), 1);
    int MemoryCost = 0; // 27K  B
    int bitlen = 0;
    for(int i = num_nbr-1;i > 0;i>>=1){bitlen ++;}
    if(xyz_Support.dtype() == torch::kFloat32){
        switch (bitlen){
            case 0:
            FlashKNN_Query_GM_kernel<1><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices_out.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_Support.data_ptr(),
                (const float*) xyz_Query.data_ptr(),
                (const int*) Parent2Child_Support.data_ptr(),
                (const int*) Parent2Child_Query.data_ptr(),
                (const int*) QueryIndex.data_ptr(),
                (const int*) Query2Parent.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                QueryIndex.size(0),
                cut_radiu2
            );
            break;
            case 1:
            FlashKNN_Query_GM_kernel<2><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices_out.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_Support.data_ptr(),
                (const float*) xyz_Query.data_ptr(),
                (const int*) Parent2Child_Support.data_ptr(),
                (const int*) Parent2Child_Query.data_ptr(),
                (const int*) QueryIndex.data_ptr(),
                (const int*) Query2Parent.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                QueryIndex.size(0),
                cut_radiu2
            );
            break;
            case 2:
            FlashKNN_Query_GM_kernel<4><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices_out.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_Support.data_ptr(),
                (const float*) xyz_Query.data_ptr(),
                (const int*) Parent2Child_Support.data_ptr(),
                (const int*) Parent2Child_Query.data_ptr(),
                (const int*) QueryIndex.data_ptr(),
                (const int*) Query2Parent.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                QueryIndex.size(0),
                cut_radiu2
            );
            break;
            case 3:
            FlashKNN_Query_GM_kernel<8><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices_out.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_Support.data_ptr(),
                (const float*) xyz_Query.data_ptr(),
                (const int*) Parent2Child_Support.data_ptr(),
                (const int*) Parent2Child_Query.data_ptr(),
                (const int*) QueryIndex.data_ptr(),
                (const int*) Query2Parent.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                QueryIndex.size(0),
                cut_radiu2
            );
            break;
            case 4:
            FlashKNN_Query_GM_kernel<16><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices_out.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_Support.data_ptr(),
                (const float*) xyz_Query.data_ptr(),
                (const int*) Parent2Child_Support.data_ptr(),
                (const int*) Parent2Child_Query.data_ptr(),
                (const int*) QueryIndex.data_ptr(),
                (const int*) Query2Parent.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                QueryIndex.size(0),
                cut_radiu2
            );
            break;
            case 5:
            FlashKNN_Query_GM_kernel<32><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices_out.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_Support.data_ptr(),
                (const float*) xyz_Query.data_ptr(),
                (const int*) Parent2Child_Support.data_ptr(),
                (const int*) Parent2Child_Query.data_ptr(),
                (const int*) QueryIndex.data_ptr(),
                (const int*) Query2Parent.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                QueryIndex.size(0),
                cut_radiu2
            );
            break;
            case 6:
            FlashKNN_Query_GM_kernel<64><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices_out.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_Support.data_ptr(),
                (const float*) xyz_Query.data_ptr(),
                (const int*) Parent2Child_Support.data_ptr(),
                (const int*) Parent2Child_Query.data_ptr(),
                (const int*) QueryIndex.data_ptr(),
                (const int*) Query2Parent.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                QueryIndex.size(0),
                cut_radiu2
            );
            break;
            default:
            break;
        }
    }
    else{
        throw cudaErrorNotYetImplemented;
    }
    cudaDeviceSynchronize();
}