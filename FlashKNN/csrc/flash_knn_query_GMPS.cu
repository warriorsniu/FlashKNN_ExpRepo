// 从global memory中读取数据，但是使用并行排序
// 先读取进shared memory中，但是使用串行排序

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

#include "flash_knn_query_GMPS.h"

constexpr uint64_t WARP_SIZE = 32;
constexpr uint64_t WARP_SIZE_1 = WARP_SIZE - 1;
constexpr uint64_t grid_x = 512;

using namespace std;

template <typename T>
struct SharedMemory;
template <>
struct SharedMemory <int>
{
    __device__ int *getPointer()
    {
        extern __shared__ int s_int[];
        return s_int;
    }
};

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};


template <typename CoordDType>
__device__ void Dynamic_Load_Support_Points(
    int parent_idx,
    int total_support_points_num,
    int max_support_point_num_loaded_per_batch,
    int &loaded_support_points_num,
    int &loaded_support_points_num_cur_batch,
    int thread_idx_in_block,
    const int* Parent2Child,
    const int* CumCntInNeigh,
    const int* ParentNeigh,
    const CoordDType* GridCoord,
    CoordDType* buf,
    int* buf_index,
    int & neigh_idx = 0
){
    // int neigh_idx = 0;  // 0 - 26
    //读取一个批次的数据
    __syncthreads();
    for(;loaded_support_points_num_cur_batch<max_support_point_num_loaded_per_batch;){
        int support_point_index2load = (
            thread_idx_in_block 
            + loaded_support_points_num_cur_batch 
            + loaded_support_points_num);                            //当前邻域内的子节点index
        while(
            neigh_idx < 27
            && *(CumCntInNeigh + 28*parent_idx + neigh_idx + 1) <= support_point_index2load){
            neigh_idx += 1;
        }// 潜在瓶颈
        int idx_offset = support_point_index2load - *(CumCntInNeigh + 28*parent_idx + neigh_idx);

        if( support_point_index2load < total_support_points_num      //不能超过当前邻域内总数量
            && loaded_support_points_num_cur_batch + thread_idx_in_block 
                < max_support_point_num_loaded_per_batch              //不能超过每批次读取的数量
            ){
            // bool flag = (blockIdx.x == 159 && blockIdx.y == 12 && threadIdx.x < 32 && threadIdx.x > 23 && threadIdx.y <= 1);
            // if(flag){
            //     printf("读取数据, threadIdx.x: %d, threadIdx.y: %d, parent_idx: %d, neigh_idx: %d \n", threadIdx.x, threadIdx.y, parent_idx, neigh_idx);
            // }
            int cur_prt_nbr_idx = *(ParentNeigh + 27*parent_idx + neigh_idx);
            // if(flag){
            //     printf("读取数据, threadIdx.x: %d, threadIdx.y: %d, parent_idx: %d, neigh_idx: %d, cur_prt_nbr_idx: %d, Parent2Child addr: %p \n", threadIdx.x, threadIdx.y, parent_idx, neigh_idx, cur_prt_nbr_idx, Parent2Child+cur_prt_nbr_idx);
            // }
            int cur_child_idx = Parent2Child[cur_prt_nbr_idx] + idx_offset;   
            // if(flag){
            //     printf("读取数据, threadIdx.x: %d, threadIdx.y: %d, parent_idx: %d, neigh_idx: %d, cur_prt_nbr_idx: %d, cur_child_idx: %d \n", threadIdx.x, threadIdx.y, parent_idx, neigh_idx, cur_prt_nbr_idx, cur_child_idx);
            // }
            const CoordDType *CurCoord = GridCoord + 3*cur_child_idx;
            CoordDType x = CurCoord[0];
            CoordDType y = CurCoord[1];
            CoordDType z = CurCoord[2];
            int save_base = (
                thread_idx_in_block 
                + loaded_support_points_num_cur_batch 
                );
            buf[save_base*3] = x;
            buf[save_base*3+1] = y;
            buf[save_base*3+2] = z;
            buf_index[save_base] = cur_child_idx;         
        }
        // 更新loaded_support_points_num_cur_batch 
        loaded_support_points_num_cur_batch += blockDim.x*blockDim.y;
    }
    loaded_support_points_num_cur_batch = max_support_point_num_loaded_per_batch;
    if(loaded_support_points_num + loaded_support_points_num_cur_batch >= total_support_points_num){
        loaded_support_points_num_cur_batch = total_support_points_num - loaded_support_points_num;
    } //溢出修正
    loaded_support_points_num += loaded_support_points_num_cur_batch;
}


// 还可以尝试优化为使用共享内存维护候选列表，这样比较器数量更多，目前比较器数量为WARPSIZE/2，使用共享内存，比较器数量可以增加到WARPSIZE
template <typename CoordDType, int ArrayLengthPerThread, int depth_K>
__device__ void Bitonic_Sort(
    CoordDType* best_dis,
    int* best_idx
){
    int depth = 1;
    for(int step = 1;step < ArrayLengthPerThread*blockDim.x;step <<= 1){
        int sub_depth = depth-1;
        for(int cmp_step = step;cmp_step > 0;cmp_step >>= 1){
            for(int i = 0;i<ArrayLengthPerThread;i++){
                if(cmp_step < blockDim.x){
                    int array_idx = threadIdx.x + (blockDim.x*i);
                    int tgt_thread_ = threadIdx.x^cmp_step;
                    int cmp_drct_ = ((array_idx >> depth)&1) ^ ((array_idx >> sub_depth)&1); // 0表示低位，1表示高位
                    cmp_drct_ = 1 - 2*cmp_drct_;
                    CoordDType cur_dis = best_dis[i];
                    int cur_idx = best_idx[i];
                    CoordDType cmp_dis = WARP_SHFL(cur_dis, tgt_thread_, blockDim.x);
                    int cmp_idx = WARP_SHFL(cur_idx, tgt_thread_, blockDim.x);
                    if((cur_dis - cmp_dis)*cmp_drct_ > 0){
                        best_dis[i] = cmp_dis;
                        best_idx[i] = cmp_idx;
                    }
                }
                else if(ArrayLengthPerThread == 2){
                    int array_idx = threadIdx.x + (blockDim.x*i);
                    int idx_step = cmp_step >> depth_K;   //32对应5
                    int cmp_drct_ = ((array_idx >> depth)&1) ^ ((array_idx >> sub_depth)&1);
                    cmp_drct_ = 1 - 2*cmp_drct_;
                    CoordDType cur_dis = best_dis[i];
                    int cur_idx = best_idx[i];
                    CoordDType cmp_dis = best_dis[i+idx_step];
                    int cmp_idx = best_idx[i+idx_step];
                    if((cur_dis - cmp_dis)*cmp_drct_ > 0){
                        CoordDType temp = cur_dis;
                        best_dis[i] = cmp_dis;
                        best_dis[i+idx_step] = temp;

                        int temp_idx = cur_idx;
                        best_idx[i] = cmp_idx;
                        best_idx[i+idx_step] = temp_idx;
                    }
                    break;
                }
                else if(i < (ArrayLengthPerThread >> 1)){
                    int tid = threadIdx.x + (blockDim.x*i);
                    int array_idx = ((tid>>sub_depth)<<(sub_depth+1)) | ((tid)&(cmp_step-1));
                    int array_idx_in_thread = array_idx >> depth_K;
                    int idx_step = cmp_step >> depth_K;   //32对应5
                    int cmp_drct_ = ((array_idx >> depth)&1) ^ ((array_idx >> sub_depth)&1);
                    cmp_drct_ = 1 - 2*cmp_drct_;
                    CoordDType cur_dis = best_dis[array_idx_in_thread];
                    int cur_idx = best_idx[array_idx_in_thread];
                    CoordDType cmp_dis = best_dis[array_idx_in_thread+idx_step];
                    int cmp_idx = best_idx[array_idx_in_thread+idx_step];
                    if((cur_dis - cmp_dis)*cmp_drct_ > 0){
                        CoordDType temp = cur_dis;
                        best_dis[array_idx_in_thread] = cmp_dis;
                        best_dis[array_idx_in_thread+idx_step] = temp;

                        int temp_idx = cur_idx;
                        best_idx[array_idx_in_thread] = cmp_idx;
                        best_idx[array_idx_in_thread+idx_step] = temp_idx;
                    }
                }
            }
            sub_depth -= 1;
        }
        depth += 1;
    }
}


template <typename CoordDType, int ArrayLengthPerThread, int depth_K>
__global__ void FlashKNN_Query_GMPS_kernel(
    int* Indices_out,
    CoordDType* Dis_out,
    const CoordDType* GridCoord,          // N
    const int* Parent2Child,       // N_+1
    const int* ParentNeigh,        // N_
    const int* CumCntInNeigh,      // N_, 28   
    const int K,                   //  
    const int N_prt,
    const int SM_index_offset,
    const int max_point_num_loaded,
    const int batch_for_prune = 8,
    const CoordDType cut_radiu2 = INFINITY
){  
    // 声明共享内存
    // SharedMemory<CoordDType> shared;
    // int* buf_thread_tree_depth_worker = (int*)(buf) + SM_thread_tree_depth_offset;
    // int* buf_warp_thread_nbr_idx = (int*)(buf) + SM_warp_thread_nbr_idx_offset;
    // int max_query_point_num_loaded_per_batch = max_point_num_loaded / 16;
    // int loaded_query_points_num = 0;
    // int total_query_points_num = 0;

    // int max_support_point_num_loaded_per_batch = max_point_num_loaded;
    // int loaded_support_points_num = 0;
    int total_support_points_num = 0;
    
    //一个block处理一个parent及其邻域
    int parent_idx = blockIdx.x + blockIdx.y*gridDim.x;
    
    for(;parent_idx < N_prt;parent_idx += gridDim.x*gridDim.y){
        int thread_idx_in_block = threadIdx.y*blockDim.x + threadIdx.x;
        total_support_points_num = *(CumCntInNeigh + 28*parent_idx + 27);

        int child_idx_start = Parent2Child[parent_idx];
        int child_idx_end = Parent2Child[parent_idx + 1];
        int child_idx_finished = child_idx_start;
        int batch = (ArrayLengthPerThread >> 1);
        for(;child_idx_finished < child_idx_end;child_idx_finished += blockDim.y){
            int nbr_i = 0;
            // loaded_support_points_num = 0;
            int child_idx_origin = min(child_idx_finished+threadIdx.y, child_idx_end-1);
            // 读取中心子节点
            CoordDType center_x = GridCoord[child_idx_origin*3 + 0];
            CoordDType center_y = GridCoord[child_idx_origin*3 + 1];
            CoordDType center_z = GridCoord[child_idx_origin*3 + 2];
            CoordDType best_dis[ArrayLengthPerThread];
            int best_idx[ArrayLengthPerThread];
            int nbr_idx_start = 0;
            for(int i = 0;i < ArrayLengthPerThread;i++){best_idx[i]=child_idx_origin;best_dis[i] = INFINITY;} //初始化结果
            
            nbr_idx_start = max(child_idx_origin - child_idx_start - K, 0);
            // 遍历所有邻域子节点
            bool skip = true;
            int offset = threadIdx.x;
            int S_points_num_traversed = 0;
            while (S_points_num_traversed < total_support_points_num){    
                CoordDType max_dis = best_dis[(K-1)>>depth_K];
                max_dis = WARP_SHFL(max_dis, (K-1)&(blockDim.x - 1), blockDim.x);
                for(int batch_idx = 0;batch_idx < batch && offset < total_support_points_num;batch_idx++){
                    int child_nbr_idx_with_offset = ((nbr_idx_start + offset)%total_support_points_num);
                    while (
                        CumCntInNeigh[parent_idx*28+nbr_i] > child_nbr_idx_with_offset 
                        || CumCntInNeigh[parent_idx*28+nbr_i+1] <= child_nbr_idx_with_offset){
                        nbr_i += 1;
                        if(nbr_i > 26){
                            nbr_i -= 27;
                        }
                    }
                    int parent_nbr_idx = ParentNeigh[parent_idx*27+nbr_i];
                    int candidate_idx = child_nbr_idx_with_offset - CumCntInNeigh[parent_idx*28+nbr_i] + Parent2Child[parent_nbr_idx];
                    
                    CoordDType dis_x = GridCoord[candidate_idx*3 + 0] - center_x;
                    CoordDType dis_y = GridCoord[candidate_idx*3 + 1] - center_y;
                    CoordDType dis_z = GridCoord[candidate_idx*3 + 2] - center_z;
                    CoordDType dis = dis_x*dis_x + dis_y*dis_y + dis_z*dis_z;
                    // if(child_idx_origin == 106042){
                    //     printf("thread.x: %d, child_idx_origin: %d, candidate_idx: %d, dis: %f, max_dis: %f, offset: %d, total_support_points_num: %d 读取信息 \n", threadIdx.x, child_idx_origin, candidate_idx, dis, max_dis, offset, total_support_points_num);
                    // }
                    if(dis < max_dis && dis < cut_radiu2){
                        best_dis[batch_idx + batch] = dis;
                        best_idx[batch_idx + batch] = candidate_idx;
                        skip = false;
                    }
                    offset += blockDim.x;
                }

                __syncwarp();
                for (int l = 0;  l < 5;  ++l) {
                    int srcLaneB = (thread_idx_in_block^(1<<l))&31;
                    bool skip_other = WARP_SHFL(skip, srcLaneB);
                    skip = skip_other&skip;
                }
                skip = WARP_SHFL(skip, 0);
                if(skip){
                    continue;
                }
                //调用双调排序
                Bitonic_Sort<CoordDType, ArrayLengthPerThread, depth_K>(best_dis, best_idx);
                S_points_num_traversed += batch*blockDim.x;
            }
            if(child_idx_finished+threadIdx.y < child_idx_end){
                for(int nbr_idx = threadIdx.x;nbr_idx<K;nbr_idx += blockDim.x){
                    // if(child_idx_origin == 106042){
                    //     printf("thread.x: %d, child_idx_origin: %d, nbr_idx: %d, dis: %f, idx: %d, total_support_points_num: %d 写入信息 \n", threadIdx.x, child_idx_origin, nbr_idx, best_dis[0], best_idx[0], total_support_points_num);
                    // }
                    Indices_out[K*child_idx_origin+nbr_idx] = best_idx[nbr_idx >> depth_K];
                    Dis_out[K*child_idx_origin+nbr_idx] = best_dis[nbr_idx >> depth_K];
                }
            }
        }//双调排序support节点遍历循环
    }//block循环
}



template <typename CoordDType, int ArrayLengthPerThread, int depth_K>
__global__ void FlashKNN_Query_SMSS_kernel(
    int* Indices_out,
    CoordDType* Dis_out,
    const CoordDType* GridCoord,          // N
    const int* Parent2Child,       // N_+1
    const int* ParentNeigh,        // N_
    const int* CumCntInNeigh,      // N_, 28   
    const int K,                   //  
    const int N_prt,
    const int SM_index_offset,
    const int max_point_num_loaded,
    const int batch_for_prune = 8,
    const CoordDType cut_radiu2 = INFINITY
){
    // 声明共享内存
    SharedMemory<CoordDType> shared;
    CoordDType* buf = shared.getPointer();
    int* buf_index = (int*)(buf) + SM_index_offset;
    // int* buf_thread_tree_depth_worker = (int*)(buf) + SM_thread_tree_depth_offset;
    // int* buf_warp_thread_nbr_idx = (int*)(buf) + SM_warp_thread_nbr_idx_offset;
    // int max_query_point_num_loaded_per_batch = max_point_num_loaded / 16;
    // int loaded_query_points_num = 0;
    // int total_query_points_num = 0;

    int max_support_point_num_loaded_per_batch = max_point_num_loaded;
    int loaded_support_points_num = 0;
    int total_support_points_num = 0;
    
    //一个block处理一个parent及其邻域
    int parent_idx = blockIdx.x + blockIdx.y*gridDim.x;
    
    for(;parent_idx < N_prt;parent_idx += gridDim.x*gridDim.y){
        int thread_idx_in_block = threadIdx.y*blockDim.x + threadIdx.x;
        total_support_points_num = *(CumCntInNeigh + 28*parent_idx + 27);
        // if(blockIdx.x == 13 && blockIdx.y == 0){
        //     printf("block.x:%d, block.y: %d, thread.x: %d, thread.y: %d, parent_idx: %d, 当前批次query点读取完毕, loaded_query_points_num: %d, loaded_query_points_num_cur_batch:%d, max_point_num_loaded:%d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, parent_idx, loaded_query_points_num, loaded_query_points_num_cur_batch, max_point_num_loaded);
        // }

        //循环读取一个批次的support点，并进行排序
        loaded_support_points_num = 0;
        int neigh_idx = 0;
        for(;loaded_support_points_num<total_support_points_num;){
            
            int loaded_support_points_num_cur_batch = 0;
            Dynamic_Load_Support_Points<CoordDType>(
                parent_idx,
                total_support_points_num,
                max_support_point_num_loaded_per_batch,
                loaded_support_points_num,
                loaded_support_points_num_cur_batch,
                thread_idx_in_block,
                Parent2Child,
                CumCntInNeigh,
                ParentNeigh,
                GridCoord,
                buf,
                buf_index,
                neigh_idx
            );
            __syncthreads();//当前批次的support点读取完毕
            
            // if(blockIdx.x == 0 && blockIdx.y == 0){
            //     printf("thread.x: %d, thread.y, %d, loaded_support_points_num_cur_batch: %d, blockdim.x: %d, blockdim.y: %d, 数据读取进入共享内存完毕 \n", threadIdx.x, threadIdx.y, loaded_support_points_num_cur_batch, blockDim.x, blockDim.y);
            // }

            int child_idx_start = Parent2Child[parent_idx];
            // int child_idx_end = Parent2Child[parent_idx + 1];
            // int child_idx_finished = child_idx_start;

            for(int center_child_idx = child_idx_start+thread_idx_in_block;center_child_idx < Parent2Child[parent_idx + 1];center_child_idx += blockDim.x*blockDim.y){
                // 读取中心子节点
                CoordDType center_x = GridCoord[center_child_idx*3 + 0];
                CoordDType center_y = GridCoord[center_child_idx*3 + 1];
                CoordDType center_z = GridCoord[center_child_idx*3 + 2];
                CoordDType best_dis[1<<depth_K];
                int best_idx[1<<depth_K];
                int nbr_idx_start = 0;
                for(int i = 0;i < (1<<depth_K);i++){best_idx[i]=center_child_idx;best_dis[i] = INFINITY;} //初始化结果
                if(loaded_support_points_num != loaded_support_points_num_cur_batch){ //不是第一个循环，从已有结果中读取数据
                    for(int i = 0;i < K;i++){
                        // int k = i*blockDim.x+threadIdx.x;
                        best_idx[i]=Indices_out[center_child_idx*K+i];
                        best_dis[i]=Dis_out[center_child_idx*K+i];
                    
                    }
                }
                else{  //第一个循环，从附近开始读取
                    // best_dis[MaxArrayIdx] = 0;
                    nbr_idx_start = max(center_child_idx - child_idx_start - K, 0);
                }
                // 遍历所有邻域子节点
                
                // int offset = threadIdx.x;
                int S_points_num_traversed = 0;
                while(S_points_num_traversed < loaded_support_points_num_cur_batch){
                    int child_nbr_idx_with_offset = ((nbr_idx_start + S_points_num_traversed)%loaded_support_points_num_cur_batch);
                    CoordDType dis_x = buf[child_nbr_idx_with_offset*3 + 0] - center_x;
                    CoordDType dis_y = buf[child_nbr_idx_with_offset*3 + 1] - center_y;
                    CoordDType dis_z = buf[child_nbr_idx_with_offset*3 + 2] - center_z;
                    CoordDType dis = dis_x*dis_x + dis_y*dis_y + dis_z*dis_z;
                    // if(center_child_idx == 106042){
                    //     printf("候选点： thread_idx_in_block: %d, center_child_idx: %d, idx: %d, dis: %f, total_support_points_num: %d \n", thread_idx_in_block, center_child_idx, buf_index[child_nbr_idx_with_offset], dis, total_support_points_num);
                    // }
                    if(dis < best_dis[0] && dis < cut_radiu2){
                        Re_Heap(buf_index[child_nbr_idx_with_offset], dis, best_idx, best_dis, K);
                    }
                    S_points_num_traversed += 1;
                }//

                for(int nbr_idx = 0;nbr_idx<K;nbr_idx += 1){
                    // if(center_child_idx == 106042){
                    //     printf("写入此轮次结果： thread_idx_in_block %d, center_child_idx: %d, nbr_idx: %d, dis: %f, idx: %d\n", threadIdx.x, center_child_idx, nbr_idx, best_dis[nbr_idx], best_idx[nbr_idx]);
                    // }
                    Indices_out[K*center_child_idx+nbr_idx] = best_idx[nbr_idx];
                    Dis_out[K*center_child_idx+nbr_idx] = best_dis[nbr_idx];
                }
            }//双调排序query节点遍历循环
        }//support读取循环
    }//block循环
}




void FlashKNN_Query_GMPS(
    const torch::Tensor &GridCoord,
    const torch::Tensor &Parent2Child,
    const torch::Tensor &ParentNeigh,
    const torch::Tensor &CumCntInNeigh,
    const int K,
    torch::Tensor &Indices,
    torch::Tensor &Dis,
    int batch_for_prune,
    float cut_radiu2
){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // const dim3 threads(WARP_SIZE,4,1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const uint64_t TotalShareMemory = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
    at::cuda::getCurrentDeviceProperties()->maxBlocksPerMultiProcessor;
    // const uint64_t max_point_num_loaded = 2000;
    uint64_t dtypesize = 4;
    const uint64_t max_point_num_loaded = 64; //留出余量，防止共享内存占用过多影响性能
    int bit_len_K = (32 - __builtin_clz(K-1));
    int blockdimx = WARP_SIZE;
    int blockdimy = 4;
    int ArrayLengthPerThread = 2;
    if(bit_len_K > 5){ // 33~64
        ArrayLengthPerThread >>= bit_len_K - 5;
    }
    if(bit_len_K < 5){  // 5~31
        blockdimx >>= min((5 - bit_len_K), 2);
        blockdimy <<= min((5 - bit_len_K), 2);
    }
    const dim3 threads(blockdimx,blockdimy,1);
    
    const dim3 blocks(grid_x, std::min((uint64_t)16, maxGridY), 1);
    // int MemoryCost = max_point_num_loaded*(3*dtypesize+4); // 27K  B
    // int MemoryOffset = (max_point_num_loaded*3*dtypesize) / sizeof(int);
    int MemoryCost = 0;
    int MemoryOffset = 0;
    // std::cout<<"子节点数量"<<GridCoord.size(0)<<std::endl;
    // std::cout<<"父节点数量"<<ParentNeigh.size(0)<<std::endl;
    if(GridCoord.dtype() == torch::kFloat32){
        if(bit_len_K == 6){
            FlashKNN_Query_GMPS_kernel<float, 4, 5><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) ParentNeigh.data_ptr(),
            (const int*) CumCntInNeigh.data_ptr(),
            K,
            ParentNeigh.size(0),
            MemoryOffset,
            max_point_num_loaded,
            batch_for_prune,
            cut_radiu2
            );
        }
        else if(bit_len_K == 5){
            FlashKNN_Query_GMPS_kernel<float, 2, 5><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) ParentNeigh.data_ptr(),
            (const int*) CumCntInNeigh.data_ptr(),
            K,
            ParentNeigh.size(0),
            MemoryOffset,
            max_point_num_loaded,
            batch_for_prune,
            cut_radiu2
            );
        }
        else if(bit_len_K == 4){
            FlashKNN_Query_GMPS_kernel<float, 2, 4><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) ParentNeigh.data_ptr(),
            (const int*) CumCntInNeigh.data_ptr(),
            K,
            ParentNeigh.size(0),
            MemoryOffset,
            max_point_num_loaded,
            batch_for_prune,
            cut_radiu2
            );
        }
        else if (bit_len_K < 4){
            FlashKNN_Query_GMPS_kernel<float, 2, 3><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) ParentNeigh.data_ptr(),
            (const int*) CumCntInNeigh.data_ptr(),
            K,
            ParentNeigh.size(0),
            MemoryOffset,
            max_point_num_loaded,
            batch_for_prune,
            cut_radiu2
            );
        }
        else{
            throw cudaErrorNotYetImplemented;
        }
    }
    else{
        throw cudaErrorNotYetImplemented;
    }
    cudaDeviceSynchronize();
}


void FlashKNN_Query_SMSS(
    const torch::Tensor &GridCoord,
    const torch::Tensor &Parent2Child,
    const torch::Tensor &ParentNeigh,
    const torch::Tensor &CumCntInNeigh,
    const int K,
    torch::Tensor &Indices,
    torch::Tensor &Dis,
    int batch_for_prune,
    float cut_radiu2
){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // const dim3 threads(WARP_SIZE,4,1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const uint64_t TotalShareMemory = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
    at::cuda::getCurrentDeviceProperties()->maxBlocksPerMultiProcessor;
    // const uint64_t max_point_num_loaded = 2000;
    uint64_t dtypesize = 4;
    const uint64_t max_point_num_loaded = 256; //留出余量，防止共享内存占用过多影响性能
    int bit_len_K = (32 - __builtin_clz(K-1));
    int blockdimx = WARP_SIZE;
    int blockdimy = 4;
    int ArrayLengthPerThread = 2;
    // if(bit_len_K > 5){ // 33~64
    //     ArrayLengthPerThread >>= bit_len_K - 5;
    // }
    // if(bit_len_K < 5){  // 5~31
    //     blockdimx >>= min((5 - bit_len_K), 2);
    //     blockdimy <<= min((5 - bit_len_K), 2);
    // }
    const dim3 threads(blockdimx,blockdimy,1);
    
    const dim3 blocks(grid_x, std::min((uint64_t)16, maxGridY), 1);
    int MemoryCost = max_point_num_loaded*(3*dtypesize+4); // 27K  B
    int MemoryOffset = (max_point_num_loaded*3*dtypesize) / sizeof(int);
    // int MemoryCost = 0;
    // int MemoryOffset = 0;
    // std::cout<<"子节点数量"<<GridCoord.size(0)<<std::endl;
    // std::cout<<"父节点数量"<<ParentNeigh.size(0)<<std::endl;
    if(GridCoord.dtype() == torch::kFloat32){
        if(bit_len_K == 6){
            FlashKNN_Query_SMSS_kernel<float, 2, 6><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) ParentNeigh.data_ptr(),
            (const int*) CumCntInNeigh.data_ptr(),
            K,
            ParentNeigh.size(0),
            MemoryOffset,
            max_point_num_loaded,
            batch_for_prune,
            cut_radiu2
            );
        }
        else if(bit_len_K == 5){
            FlashKNN_Query_SMSS_kernel<float, 2, 5><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) ParentNeigh.data_ptr(),
            (const int*) CumCntInNeigh.data_ptr(),
            K,
            ParentNeigh.size(0),
            MemoryOffset,
            max_point_num_loaded,
            batch_for_prune,
            cut_radiu2
            );
        }
        else if(bit_len_K == 4){
            FlashKNN_Query_SMSS_kernel<float, 2, 4><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) ParentNeigh.data_ptr(),
            (const int*) CumCntInNeigh.data_ptr(),
            K,
            ParentNeigh.size(0),
            MemoryOffset,
            max_point_num_loaded,
            batch_for_prune,
            cut_radiu2
            );
        }
        else if (bit_len_K < 4){
            FlashKNN_Query_SMSS_kernel<float, 2, 3><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) ParentNeigh.data_ptr(),
            (const int*) CumCntInNeigh.data_ptr(),
            K,
            ParentNeigh.size(0),
            MemoryOffset,
            max_point_num_loaded,
            batch_for_prune,
            cut_radiu2
            );
        }
        else{
            throw cudaErrorNotYetImplemented;
        }
    }
    else{
        throw cudaErrorNotYetImplemented;
    }
    cudaDeviceSynchronize();
}