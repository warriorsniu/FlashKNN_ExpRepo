//使用寄存器作为候选列表


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

#include "flash_knn_query_dynamic_load.h"

constexpr uint64_t WARP_SIZE = 32;
constexpr uint64_t WARP_SIZE_1 = WARP_SIZE - 1;
constexpr uint64_t grid_x = 512;
constexpr uint64_t ArraySizePerThread = 2;
constexpr uint64_t MaxArrayIdx = ArraySizePerThread-1;


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
    int& neigh_idx = 0 // 0 - 26
){
    
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


// Compile-time bitonic top-P network.  The lower register half contains the
// sorted values retained from the preceding pruning round; the upper half
// contains new candidates.  Template recursion makes every stage, stride and
// register pair static for both two- and four-register layouts.
template <typename CoordDType, int Register, int RegisterEnd,
          int RegisterBegin, int LaneWidth, int Size, int Stride>
__device__ __forceinline__ void BitonicSortDescendingShuffle(
    CoordDType* distance, int* index) {
    if constexpr (Register < RegisterEnd) {
        constexpr int LocalRegister = Register - RegisterBegin;
        const int logical_index = LocalRegister * LaneWidth + threadIdx.x;
        CoordDType current_distance = distance[Register];
        int current_index = index[Register];
        CoordDType peer_distance = WARP_SHFL(
            current_distance, threadIdx.x ^ Stride, LaneWidth);
        int peer_index = WARP_SHFL(
            current_index, threadIdx.x ^ Stride, LaneWidth);
        const bool ascending_group = (logical_index & Size) != 0;
        const bool lower_lane = (logical_index & Stride) == 0;
        const bool take_peer = ascending_group
            ? (lower_lane ? current_distance > peer_distance
                          : current_distance < peer_distance)
            : (lower_lane ? current_distance < peer_distance
                          : current_distance > peer_distance);
        distance[Register] = take_peer ? peer_distance : current_distance;
        index[Register] = take_peer ? peer_index : current_index;
        BitonicSortDescendingShuffle<
            CoordDType, Register + 1, RegisterEnd, RegisterBegin,
            LaneWidth, Size, Stride>(distance, index);
    }
}

template <typename CoordDType, int Register, int RegisterEnd,
          int RegisterBegin, int LaneWidth, int Size, int Stride>
__device__ __forceinline__ void BitonicSortDescendingRegisters(
    CoordDType* distance, int* index) {
    if constexpr (Register < RegisterEnd) {
        constexpr int LocalRegister = Register - RegisterBegin;
        constexpr int RegisterStride = Stride / LaneWidth;
        if constexpr ((LocalRegister & RegisterStride) == 0) {
            constexpr int PeerRegister = Register + RegisterStride;
            constexpr bool AscendingGroup =
                ((LocalRegister * LaneWidth) & Size) != 0;
            const bool swap = AscendingGroup
                ? distance[Register] > distance[PeerRegister]
                : distance[Register] < distance[PeerRegister];
            const CoordDType current_distance = distance[Register];
            const CoordDType peer_distance = distance[PeerRegister];
            const int current_index = index[Register];
            const int peer_index = index[PeerRegister];
            distance[Register] = swap ? peer_distance : current_distance;
            distance[PeerRegister] = swap ? current_distance : peer_distance;
            index[Register] = swap ? peer_index : current_index;
            index[PeerRegister] = swap ? current_index : peer_index;
        }
        BitonicSortDescendingRegisters<
            CoordDType, Register + 1, RegisterEnd, RegisterBegin,
            LaneWidth, Size, Stride>(distance, index);
    }
}

template <typename CoordDType, int RegisterBegin, int RegisterEnd,
          int LaneWidth, int Size, int Stride>
__device__ __forceinline__ void BitonicSortDescendingStrides(
    CoordDType* distance, int* index) {
    if constexpr (Stride < LaneWidth) {
        BitonicSortDescendingShuffle<
            CoordDType, RegisterBegin, RegisterEnd, RegisterBegin,
            LaneWidth, Size, Stride>(distance, index);
    } else {
        BitonicSortDescendingRegisters<
            CoordDType, RegisterBegin, RegisterEnd, RegisterBegin,
            LaneWidth, Size, Stride>(distance, index);
    }
    if constexpr (Stride > 1) {
        BitonicSortDescendingStrides<
            CoordDType, RegisterBegin, RegisterEnd, LaneWidth,
            Size, Stride / 2>(distance, index);
    }
}

template <typename CoordDType, int RegisterBegin, int RegisterEnd,
          int LaneWidth, int Size = 2>
__device__ __forceinline__ void BitonicSortDescending(
    CoordDType* distance, int* index) {
    constexpr int Length = (RegisterEnd - RegisterBegin) * LaneWidth;
    BitonicSortDescendingStrides<
        CoordDType, RegisterBegin, RegisterEnd, LaneWidth,
        Size, Size / 2>(distance, index);
    if constexpr (Size < Length) {
        BitonicSortDescending<
            CoordDType, RegisterBegin, RegisterEnd, LaneWidth,
            Size * 2>(distance, index);
    }
}

template <typename CoordDType, int Register, int HalfRegisters>
__device__ __forceinline__ void BitonicCompareSplit(
    CoordDType* distance, int* index) {
    if constexpr (Register < HalfRegisters) {
        constexpr int PeerRegister = Register + HalfRegisters;
        const CoordDType current_distance = distance[Register];
        const CoordDType peer_distance = distance[PeerRegister];
        const int current_index = index[Register];
        const int peer_index = index[PeerRegister];
        const bool swap = current_distance > peer_distance;
        distance[Register] = swap ? peer_distance : current_distance;
        distance[PeerRegister] = swap ? current_distance : peer_distance;
        index[Register] = swap ? peer_index : current_index;
        index[PeerRegister] = swap ? current_index : peer_index;
        BitonicCompareSplit<
            CoordDType, Register + 1, HalfRegisters>(distance, index);
    }
}

template <typename CoordDType, int Register, int RegisterEnd,
          int LaneWidth, int Stride>
__device__ __forceinline__ void BitonicMergeAscendingShuffle(
    CoordDType* distance, int* index) {
    if constexpr (Register < RegisterEnd) {
        const int logical_index = Register * LaneWidth + threadIdx.x;
        CoordDType current_distance = distance[Register];
        int current_index = index[Register];
        CoordDType peer_distance = WARP_SHFL(
            current_distance, threadIdx.x ^ Stride, LaneWidth);
        int peer_index = WARP_SHFL(
            current_index, threadIdx.x ^ Stride, LaneWidth);
        const bool lower_lane = (logical_index & Stride) == 0;
        const bool take_peer = lower_lane
            ? current_distance > peer_distance
            : current_distance < peer_distance;
        distance[Register] = take_peer ? peer_distance : current_distance;
        index[Register] = take_peer ? peer_index : current_index;
        BitonicMergeAscendingShuffle<
            CoordDType, Register + 1, RegisterEnd,
            LaneWidth, Stride>(distance, index);
    }
}

template <typename CoordDType, int Register, int RegisterEnd,
          int LaneWidth, int Stride>
__device__ __forceinline__ void BitonicMergeAscendingRegisters(
    CoordDType* distance, int* index) {
    if constexpr (Register < RegisterEnd) {
        constexpr int RegisterStride = Stride / LaneWidth;
        if constexpr ((Register & RegisterStride) == 0) {
            constexpr int PeerRegister = Register + RegisterStride;
            const CoordDType current_distance = distance[Register];
            const CoordDType peer_distance = distance[PeerRegister];
            const int current_index = index[Register];
            const int peer_index = index[PeerRegister];
            const bool swap = current_distance > peer_distance;
            distance[Register] = swap ? peer_distance : current_distance;
            distance[PeerRegister] = swap ? current_distance : peer_distance;
            index[Register] = swap ? peer_index : current_index;
            index[PeerRegister] = swap ? current_index : peer_index;
        }
        BitonicMergeAscendingRegisters<
            CoordDType, Register + 1, RegisterEnd,
            LaneWidth, Stride>(distance, index);
    }
}

template <typename CoordDType, int RegisterEnd, int LaneWidth, int Stride>
__device__ __forceinline__ void BitonicMergeAscending(
    CoordDType* distance, int* index) {
    if constexpr (Stride < LaneWidth) {
        BitonicMergeAscendingShuffle<
            CoordDType, 0, RegisterEnd, LaneWidth,
            Stride>(distance, index);
    } else {
        BitonicMergeAscendingRegisters<
            CoordDType, 0, RegisterEnd, LaneWidth,
            Stride>(distance, index);
    }
    if constexpr (Stride > 1) {
        BitonicMergeAscending<
            CoordDType, RegisterEnd, LaneWidth,
            Stride / 2>(distance, index);
    }
}

template <typename CoordDType, int ArrayLengthPerThread, int depth_K>
__device__ __forceinline__ void Bitonic_Sort(
    CoordDType* best_dis,
    int* best_idx) {
    static_assert(ArrayLengthPerThread >= 2);
    static_assert((ArrayLengthPerThread & (ArrayLengthPerThread - 1)) == 0);
    constexpr int LaneWidth = 1 << depth_K;
    constexpr int HalfRegisters = ArrayLengthPerThread / 2;
    constexpr int RetainedLength = HalfRegisters * LaneWidth;
    BitonicSortDescending<
        CoordDType, HalfRegisters, ArrayLengthPerThread,
        LaneWidth>(best_dis, best_idx);
    BitonicCompareSplit<
        CoordDType, 0, HalfRegisters>(best_dis, best_idx);
    BitonicMergeAscending<
        CoordDType, HalfRegisters, LaneWidth,
        RetainedLength / 2>(best_dis, best_idx);
}


template <typename CoordDType, int ArrayLengthPerThread, int depth_K,
          bool CandidateInShared = false, bool EnableSkip = true>
__global__ void FlashKNN_Query_dynamic_load_kernel(
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
    int* candidate_idx_buffer = buf_index + max_point_num_loaded;
    CoordDType* candidate_dis_buffer = reinterpret_cast<CoordDType*>(
        candidate_idx_buffer
        + blockDim.x * blockDim.y * ArrayLengthPerThread);
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
            int child_idx_end = Parent2Child[parent_idx + 1];
            int child_idx_finished = child_idx_start;

            for(;child_idx_finished < child_idx_end;child_idx_finished += blockDim.y){
                int batch = (ArrayLengthPerThread >> 1);
                int child_idx_origin = min(child_idx_finished+threadIdx.y, child_idx_end-1);
                // 读取中心子节点
                CoordDType center_x = GridCoord[child_idx_origin*3 + 0];
                CoordDType center_y = GridCoord[child_idx_origin*3 + 1];
                CoordDType center_z = GridCoord[child_idx_origin*3 + 2];
                CoordDType register_best_dis[ArrayLengthPerThread];
                int register_best_idx[ArrayLengthPerThread];
                CoordDType* best_dis = register_best_dis;
                int* best_idx = register_best_idx;
                if constexpr (CandidateInShared) {
                    const int candidate_offset =
                        thread_idx_in_block * ArrayLengthPerThread;
                    best_dis = candidate_dis_buffer + candidate_offset;
                    best_idx = candidate_idx_buffer + candidate_offset;
                }
                int nbr_idx_start = 0;
                for(int i = 0;i < ArrayLengthPerThread;i++){best_idx[i]=child_idx_origin;best_dis[i] = INFINITY;} //初始化结果
                if(loaded_support_points_num != loaded_support_points_num_cur_batch){ //不是第一个循环，从已有结果中读取数据
                    for(int i = 0;i < ArrayLengthPerThread;i++){
                        int k = i*blockDim.x+threadIdx.x;
                        if(k < K){
                            best_idx[i]=Indices_out[child_idx_origin*K+k];
                            best_dis[i]=Dis_out[child_idx_origin*K+k];
                        }
                        else{
                            break;
                        }
                    }
                }
                else{  //第一个循环，从附近开始读取
                    // best_dis[MaxArrayIdx] = 0;
                    nbr_idx_start = max(child_idx_origin - child_idx_start - K, 0);
                }
                // 遍历所有邻域子节点
                
                int offset = threadIdx.x;
                int S_points_num_traversed = 0;
                while(S_points_num_traversed < loaded_support_points_num_cur_batch){
                    bool skip = true;
                    CoordDType max_dis = INFINITY;
                    if constexpr (EnableSkip) {
                        max_dis = best_dis[(K-1)>>depth_K];
                        max_dis = WARP_SHFL(
                            max_dis, (K-1)&(blockDim.x - 1), blockDim.x);
                    }
                    for(int batch_idx = 0;batch_idx < batch && offset < loaded_support_points_num_cur_batch;batch_idx++){
                        int child_nbr_idx_with_offset = ((nbr_idx_start + offset)%loaded_support_points_num_cur_batch);
                        CoordDType dis_x = buf[child_nbr_idx_with_offset*3 + 0] - center_x;
                        CoordDType dis_y = buf[child_nbr_idx_with_offset*3 + 1] - center_y;
                        CoordDType dis_z = buf[child_nbr_idx_with_offset*3 + 2] - center_z;
                        CoordDType dis = dis_x*dis_x + dis_y*dis_y + dis_z*dis_z;
                        if constexpr (EnableSkip) {
                            if(dis < max_dis && dis < cut_radiu2){
                                best_dis[batch_idx + batch] = dis;
                                best_idx[batch_idx + batch] =
                                    buf_index[child_nbr_idx_with_offset];
                                skip = false;
                            }
                        } else if(dis < cut_radiu2) {
                            best_dis[batch_idx + batch] = dis;
                            best_idx[batch_idx + batch] =
                                buf_index[child_nbr_idx_with_offset];
                        }
                        offset += blockDim.x;
                    }
                    
                    __syncwarp();
                    S_points_num_traversed += batch*blockDim.x;
                    // if(S_points_num_traversed >= (K<<1)){
                    //     batch = batch_for_prune;
                    // }
                    if constexpr (EnableSkip) {
                        for (int l = 0;  l < 5;  ++l) {
                            int srcLaneB = (thread_idx_in_block^(1<<l))&31;
                            bool skip_other = WARP_SHFL(skip, srcLaneB);
                            skip = skip_other&skip;
                        }
                        skip = WARP_SHFL(skip, 0);
                        if(skip){
                            continue;
                        }
                    }
                    //调用双调排序
                    Bitonic_Sort<CoordDType, ArrayLengthPerThread, depth_K>(best_dis, best_idx);
                }//双调排序support节点遍历循环

                if(child_idx_finished+threadIdx.y < child_idx_end){
                    for(int nbr_idx = threadIdx.x;nbr_idx<K;nbr_idx += blockDim.x){
                        // if(child_idx_origin == 106042){
                        //     printf("thread.x: %d, child_idx_origin: %d, nbr_idx: %d, dis: %f, idx: %d, 写入信息 \n", threadIdx.x, child_idx_origin, nbr_idx, best_dis[0], best_idx[0]);
                        // }
                        Indices_out[K*child_idx_origin+nbr_idx] = best_idx[nbr_idx >> depth_K];
                        Dis_out[K*child_idx_origin+nbr_idx] = best_dis[nbr_idx >> depth_K];
                    }
                }
            }//双调排序query节点遍历循环
        }//support读取循环
    }//block循环
}

template <bool CandidateInShared, bool EnableSkip>
void FlashKNN_Query_Dynamic_Load_Impl(
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
    if(bit_len_K < 5){  // 5~31
        blockdimx >>= min((5 - bit_len_K), 2);
        blockdimy <<= min((5 - bit_len_K), 2);
    }
    const dim3 threads(blockdimx,blockdimy,1);
    
    const dim3 blocks(grid_x, std::min((uint64_t)16, maxGridY), 1);
    const int candidate_array_length = bit_len_K == 6 ? 4 : 2;
    const int CandidateMemoryCost = CandidateInShared
        ? blockdimx * blockdimy * candidate_array_length
            * (dtypesize + sizeof(int))
        : 0;
    int MemoryCost = max_point_num_loaded*(3*dtypesize+4)
        + CandidateMemoryCost;
    int MemoryOffset = (max_point_num_loaded*3*dtypesize) / sizeof(int);
    // std::cout<<"子节点数量"<<GridCoord.size(0)<<std::endl;
    // std::cout<<"父节点数量"<<ParentNeigh.size(0)<<std::endl;
    if(GridCoord.dtype() == torch::kFloat32){
        if(bit_len_K == 6){
            // const dim3 threads(32,4,1);
            FlashKNN_Query_dynamic_load_kernel<
                float, 4, 5, CandidateInShared, EnableSkip>
                <<<blocks, threads, MemoryCost, stream>>>(
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
            // const dim3 threads(32,4,1);
            FlashKNN_Query_dynamic_load_kernel<
                float, 2, 5, CandidateInShared, EnableSkip>
                <<<blocks, threads, MemoryCost, stream>>>(
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
            // const dim3 threads(16,8,1);
            FlashKNN_Query_dynamic_load_kernel<
                float, 2, 4, CandidateInShared, EnableSkip>
                <<<blocks, threads, MemoryCost, stream>>>(
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
            // const dim3 threads(8,16,1);
            FlashKNN_Query_dynamic_load_kernel<
                float, 2, 3, CandidateInShared, EnableSkip>
                <<<blocks, threads, MemoryCost, stream>>>(
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

void FlashKNN_Query_Dynamic_Load(
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
    FlashKNN_Query_Dynamic_Load_Impl<false, true>(
        GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
        Indices, Dis, batch_for_prune, cut_radiu2);
}

void FlashKNN_Query_Dynamic_Load_Ablation(
    const torch::Tensor &GridCoord,
    const torch::Tensor &Parent2Child,
    const torch::Tensor &ParentNeigh,
    const torch::Tensor &CumCntInNeigh,
    const int K,
    torch::Tensor &Indices,
    torch::Tensor &Dis,
    int batch_for_prune,
    float cut_radiu2,
    bool candidate_in_shared,
    bool enable_skip
){
    if (candidate_in_shared && enable_skip) {
        FlashKNN_Query_Dynamic_Load_Impl<true, true>(
            GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
            Indices, Dis, batch_for_prune, cut_radiu2);
    } else if (candidate_in_shared) {
        FlashKNN_Query_Dynamic_Load_Impl<true, false>(
            GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
            Indices, Dis, batch_for_prune, cut_radiu2);
    } else if (enable_skip) {
        FlashKNN_Query_Dynamic_Load_Impl<false, true>(
            GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
            Indices, Dis, batch_for_prune, cut_radiu2);
    } else {
        FlashKNN_Query_Dynamic_Load_Impl<false, false>(
            GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
            Indices, Dis, batch_for_prune, cut_radiu2);
    }
}

template <int ArrayLengthPerThread, int DepthK>
void FlashKNN_Query_Thread_Group_Launch(
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
    static_assert(ArrayLengthPerThread >= 2);
    static_assert(
        (ArrayLengthPerThread & (ArrayLengthPerThread - 1)) == 0);
    constexpr int BlockDimX = 1 << DepthK;
    constexpr int BlockDimY = 128 / BlockDimX;
    constexpr int RetainedCapacity =
        (ArrayLengthPerThread / 2) * BlockDimX;
    TORCH_CHECK(
        K <= RetainedCapacity,
        "thread-group candidate capacity ", RetainedCapacity,
        " is smaller than k=", K);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t maxGridY =
        at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    constexpr int MaxPointNumLoaded = 256;
    constexpr int MemoryCost = MaxPointNumLoaded * (3 * sizeof(float) + sizeof(int));
    constexpr int MemoryOffset =
        (MaxPointNumLoaded * 3 * sizeof(float)) / sizeof(int);
    const dim3 threads(BlockDimX, BlockDimY, 1);
    const dim3 blocks(grid_x, std::min((uint64_t)16, maxGridY), 1);

    FlashKNN_Query_dynamic_load_kernel<
        float, ArrayLengthPerThread, DepthK, false, true>
        <<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) ParentNeigh.data_ptr(),
            (const int*) CumCntInNeigh.data_ptr(),
            K,
            ParentNeigh.size(0),
            MemoryOffset,
            MaxPointNumLoaded,
            batch_for_prune,
            cut_radiu2);
    cudaDeviceSynchronize();
}

void FlashKNN_Query_Dynamic_Load_Thread_Group(
    const torch::Tensor &GridCoord,
    const torch::Tensor &Parent2Child,
    const torch::Tensor &ParentNeigh,
    const torch::Tensor &CumCntInNeigh,
    const int K,
    torch::Tensor &Indices,
    torch::Tensor &Dis,
    int batch_for_prune,
    float cut_radiu2,
    int thread_group_size
){
    TORCH_CHECK(
        GridCoord.dtype() == torch::kFloat32,
        "thread-group ablation currently supports float32 coordinates only");
    TORCH_CHECK(K >= 2 && K <= 64, "thread-group ablation requires 2 <= k <= 64");
    TORCH_CHECK(
        thread_group_size == 8 || thread_group_size == 16
            || thread_group_size == 32,
        "thread_group_size must be 8, 16, or 32");
    const int bit_len_K = 32 - __builtin_clz(K - 1);

    if (thread_group_size == 8) {
        if (bit_len_K <= 3) {
            FlashKNN_Query_Thread_Group_Launch<2, 3>(
                GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
                Indices, Dis, batch_for_prune, cut_radiu2);
        } else if (bit_len_K == 4) {
            FlashKNN_Query_Thread_Group_Launch<4, 3>(
                GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
                Indices, Dis, batch_for_prune, cut_radiu2);
        } else if (bit_len_K == 5) {
            FlashKNN_Query_Thread_Group_Launch<8, 3>(
                GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
                Indices, Dis, batch_for_prune, cut_radiu2);
        } else {
            FlashKNN_Query_Thread_Group_Launch<16, 3>(
                GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
                Indices, Dis, batch_for_prune, cut_radiu2);
        }
    } else if (thread_group_size == 16) {
        if (bit_len_K <= 4) {
            FlashKNN_Query_Thread_Group_Launch<2, 4>(
                GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
                Indices, Dis, batch_for_prune, cut_radiu2);
        } else if (bit_len_K == 5) {
            FlashKNN_Query_Thread_Group_Launch<4, 4>(
                GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
                Indices, Dis, batch_for_prune, cut_radiu2);
        } else {
            FlashKNN_Query_Thread_Group_Launch<8, 4>(
                GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
                Indices, Dis, batch_for_prune, cut_radiu2);
        }
    } else {
        if (bit_len_K <= 5) {
            FlashKNN_Query_Thread_Group_Launch<2, 5>(
                GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
                Indices, Dis, batch_for_prune, cut_radiu2);
        } else {
            FlashKNN_Query_Thread_Group_Launch<4, 5>(
                GridCoord, Parent2Child, ParentNeigh, CumCntInNeigh, K,
                Indices, Dis, batch_for_prune, cut_radiu2);
        }
    }
}


void __global__ FlashKNN_Nearest_Back_Query_DL_kernel(
    int* Indices_out,
    float* Dis_out,
    const float* xyz_H,          // N
    const float* xyz_L,          // N
    const int* Parent2Child_H,       // N_+1
    const int* Parent2Child_L,       // N_+1
    const int* ParentNeigh,        // N_
    const int* CumCntInNeigh,      // N_, 28
    const int N_PRT,
    const int SM_index_offset,
    const int max_point_num_loaded
){
    //候选点读取进共享内存
    // 声明共享内存
    SharedMemory<float> shared;
    float* buf = shared.getPointer();
    int* buf_index = (int*)(buf + SM_index_offset);

    int max_support_point_num_loaded_per_batch = max_point_num_loaded;
    int loaded_support_points_num = 0;
    int total_support_points_num = 0;
    
    //一个block处理一个parent及其邻域
    int parent_idx = blockIdx.x + blockIdx.y*gridDim.x;
    for(;parent_idx < N_PRT;parent_idx += gridDim.x*gridDim.y){
        int thread_idx_in_block = threadIdx.y*blockDim.x + threadIdx.x;
        total_support_points_num = *(CumCntInNeigh + 28*parent_idx + 27);
        //循环读取一个批次的support点，并进行排序
        loaded_support_points_num = 0;
        int neigh_idx = 0;
        for(;loaded_support_points_num<total_support_points_num;){
            //读取一个parent3*3邻域格中的所有Low-level-point
            __syncthreads();//等待前一轮所有线程信息写入完毕
            int loaded_support_points_num_cur_batch = 0;
            Dynamic_Load_Support_Points<float>(
                    parent_idx,
                    total_support_points_num,
                    max_support_point_num_loaded_per_batch,
                    loaded_support_points_num,
                    loaded_support_points_num_cur_batch,
                    thread_idx_in_block,
                    Parent2Child_L,
                    CumCntInNeigh,
                    ParentNeigh,
                    xyz_L,
                    buf,
                    buf_index,
                    neigh_idx
            );
            __syncthreads();//等待写入完毕
            //对候选点进行查询
            // TODO 查询点多的时候采用逐点计算，查询点少的时候采用并行排序
            int H_idx = Parent2Child_H[parent_idx] + threadIdx.y*blockDim.x + threadIdx.x;
            for(;H_idx < Parent2Child_H[parent_idx+1];H_idx += blockDim.x*blockDim.y){
                const float *CenterCoord = xyz_H + 3*H_idx;
                float x = CenterCoord[0];
                float y = CenterCoord[1];
                float z = CenterCoord[2];
                float best_dis = -1;
                int best_idx = 0;
                if(loaded_support_points_num != loaded_support_points_num_cur_batch){
                    best_dis = Dis_out[H_idx];
                    best_idx = Indices_out[H_idx];
                }
                for(int cmp_idx = 0;cmp_idx < loaded_support_points_num_cur_batch;cmp_idx++){
                    float dis_x = buf[3*cmp_idx+0] - x;
                    float dis_y = buf[3*cmp_idx+1] - y;
                    float dis_z = buf[3*cmp_idx+2] - z;
                    float dis = dis_x*dis_x + dis_y*dis_y + dis_z*dis_z;
                    if(dis < best_dis || best_dis < 0){
                        best_dis = dis;
                        best_idx = buf_index[cmp_idx];
                    }
                }
                Indices_out[H_idx] = best_idx;
                Dis_out[H_idx] = best_dis;
            }
        }
    }
}


template<typename CoordDType, int ArrayLengthPerThread, int depth_K>
void __global__ FlashKNN_Back_Query_DL_kernel(
    int* Indices_out,            // N, k
    float * Dis_out,             // N, k
    const float* xyz_H,          // N   高分辨率点云
    const float* xyz_L,          // N   低分辨率点云
    const int* Parent2Child_H,       // N_+1
    const int* Parent2Child_L,       // N_+1
    const int* ParentNeigh,        // N_
    const int* CumCntInNeigh,      // N_, 28
    const int K,
    const int N_PRT,
    const int SM_index_offset,
    int max_point_num_loaded,
    const CoordDType cut_radiu2 = INFINITY
){
    //候选点读取进共享内存
    // 声明共享内存
    SharedMemory<CoordDType> shared;
    float* buf = shared.getPointer();
    int* buf_index = (int*)(buf + SM_index_offset);

    int max_support_point_num_loaded_per_batch = max_point_num_loaded;
    int loaded_support_points_num = 0;
    int total_support_points_num = 0;
    
    //一个block处理一个parent及其邻域
    int parent_idx = blockIdx.x + blockIdx.y*gridDim.x;
    for(;parent_idx < N_PRT;parent_idx += gridDim.x*gridDim.y){
        int thread_idx_in_block = threadIdx.y*blockDim.x + threadIdx.x;
        total_support_points_num = *(CumCntInNeigh + 28*parent_idx + 27);
        //循环读取一个批次的support点，并进行排序
        loaded_support_points_num = 0;
        int neigh_idx = 0;
        for(;loaded_support_points_num<total_support_points_num;){
            //读取一个parent3*3邻域格中的所有Low-level-point
            __syncthreads();//等待前一轮所有线程信息写入完毕
            int loaded_support_points_num_cur_batch = 0;
            Dynamic_Load_Support_Points<CoordDType>(
                    parent_idx,
                    total_support_points_num,
                    max_support_point_num_loaded_per_batch,
                    loaded_support_points_num,
                    loaded_support_points_num_cur_batch,
                    thread_idx_in_block,
                    Parent2Child_L,
                    CumCntInNeigh,
                    ParentNeigh,
                    xyz_L,
                    buf,
                    buf_index,
                    neigh_idx
            );
            __syncthreads();//等待写入完毕
            //对候选点进行查询，逐点计算
            int H_idx = Parent2Child_H[parent_idx] + threadIdx.y*blockDim.x + threadIdx.x;
            for(;H_idx < Parent2Child_H[parent_idx+1];H_idx += blockDim.x*blockDim.y){
                float dis_candidate_list[1<<depth_K];
                int idx_candidate_list[1<<depth_K];
                // 初始化
                if(loaded_support_points_num != loaded_support_points_num_cur_batch){
                    for(int i = 0;i<K;i++){
                        dis_candidate_list[i] = Dis_out[H_idx*K + i];
                        idx_candidate_list[i] = Indices_out[H_idx*K + i];
                    }
                }
                else{
                    for(int i = 0;i<K;i++){
                    dis_candidate_list[i] = INFINITY;
                    idx_candidate_list[i] = -1;
                    }
                }
                const float* CenterCoord = xyz_H + 3*H_idx;
                float x = CenterCoord[0];
                float y = CenterCoord[1];
                float z = CenterCoord[2];
                for(int cmp_idx = 0;cmp_idx < loaded_support_points_num_cur_batch;cmp_idx++){
                    float dis_x = buf[3*cmp_idx+0] - x;
                    float dis_y = buf[3*cmp_idx+1] - y;
                    float dis_z = buf[3*cmp_idx+2] - z;
                    float dis = dis_x*dis_x + dis_y*dis_y + dis_z*dis_z;
                    if(dis < dis_candidate_list[0]){
                        Re_Heap(buf_index[cmp_idx], dis, idx_candidate_list, dis_candidate_list, K);
                    }
                    // Re_Order(cmp_idx, dis, idx_candidate_list, dis_candidate_list, num_nbr);
                }
                for(int i = 0;i<K;i++){
                    Indices_out[H_idx*K + i] = idx_candidate_list[i];
                    Dis_out[H_idx*K + i] = dis_candidate_list[i];
                }
            }

            // 并行排序
            // int child_idx_start = Parent2Child_H[parent_idx]; // 读取query点
            // int child_idx_end = Parent2Child_H[parent_idx + 1];
            // int child_idx_finished = child_idx_start;

            // for(;child_idx_finished < child_idx_end;child_idx_finished += blockDim.y){
            //     int batch = (ArrayLengthPerThread >> 1);
            //     int child_idx_origin = min(child_idx_finished+threadIdx.y, child_idx_end-1);
            //     // 读取中心子节点
            //     CoordDType center_x = xyz_H[child_idx_origin*3 + 0];
            //     CoordDType center_y = xyz_H[child_idx_origin*3 + 1];
            //     CoordDType center_z = xyz_H[child_idx_origin*3 + 2];
            //     CoordDType best_dis[ArrayLengthPerThread];
            //     int best_idx[ArrayLengthPerThread];
            //     int nbr_idx_start = 0;
            //     for(int i = 0;i < ArrayLengthPerThread;i++){best_idx[i]=child_idx_origin;best_dis[i] = INFINITY;} //初始化结果
            //     if(loaded_support_points_num != loaded_support_points_num_cur_batch){ //不是第一个循环，从已有结果中读取数据
            //         for(int i = 0;i < ArrayLengthPerThread;i++){
            //             int k = i*blockDim.x+threadIdx.x;
            //             if(k < K){
            //                 best_idx[i]=Indices_out[child_idx_origin*K+k];
            //                 best_dis[i]=Dis_out[child_idx_origin*K+k];
            //             }
            //             else{
            //                 break;
            //             }
            //         }
            //     }
            //     else{  //第一个循环，从附近开始读取
            //         // best_dis[MaxArrayIdx] = 0;
            //         nbr_idx_start = max(child_idx_origin - child_idx_start - K, 0);
            //     }
            //     // 遍历所有邻域子节点
                
            //     int offset = threadIdx.x;
            //     int S_points_num_traversed = 0;
            //     while(S_points_num_traversed < loaded_support_points_num_cur_batch){
            //         bool skip = true;
            //         CoordDType max_dis = best_dis[(K-1)>>depth_K];
            //         max_dis = WARP_SHFL(max_dis, (K-1)&(blockDim.x - 1), blockDim.x);
            //         for(int batch_idx = 0;batch_idx < batch && offset < loaded_support_points_num_cur_batch;batch_idx++){
            //             int child_nbr_idx_with_offset = ((nbr_idx_start + offset)%loaded_support_points_num_cur_batch);
            //             CoordDType dis_x = buf[child_nbr_idx_with_offset*3 + 0] - center_x;
            //             CoordDType dis_y = buf[child_nbr_idx_with_offset*3 + 1] - center_y;
            //             CoordDType dis_z = buf[child_nbr_idx_with_offset*3 + 2] - center_z;
            //             CoordDType dis = dis_x*dis_x + dis_y*dis_y + dis_z*dis_z;
            //             if(dis < max_dis && dis < cut_radiu2){
            //                 best_dis[batch_idx + batch] = dis;
            //                 best_idx[batch_idx + batch] = buf_index[child_nbr_idx_with_offset];
            //                 skip = false;
            //             }
            //             offset += blockDim.x;
            //         }
                    
            //         __syncwarp();
            //         for (int l = 0;  l < 5;  ++l) {
            //             int srcLaneB = (thread_idx_in_block^(1<<l))&31;
            //             bool skip_other = WARP_SHFL(skip, srcLaneB);
            //             skip = skip_other&skip;
            //         }
            //         skip = WARP_SHFL(skip, 0);
            //         S_points_num_traversed += batch*blockDim.x;
            //         // if(S_points_num_traversed >= (K<<1)){
            //         //     batch = batch_for_prune;
            //         // }
            //         if(skip){
            //             continue;
            //         }
            //         //调用双调排序
            //         Bitonic_Sort<CoordDType, ArrayLengthPerThread, depth_K>(best_dis, best_idx);
            //     }//双调排序support节点遍历循环

            //     if(child_idx_finished+threadIdx.y < child_idx_end){
            //         for(int nbr_idx = threadIdx.x;nbr_idx<K;nbr_idx += blockDim.x){
            //             // if(child_idx_origin == 106042){
            //             //     printf("thread.x: %d, child_idx_origin: %d, nbr_idx: %d, dis: %f, idx: %d, 写入信息 \n", threadIdx.x, child_idx_origin, nbr_idx, best_dis[0], best_idx[0]);
            //             // }
            //             Indices_out[K*child_idx_origin+nbr_idx] = best_idx[nbr_idx >> depth_K];
            //             Dis_out[K*child_idx_origin+nbr_idx] = best_dis[nbr_idx >> depth_K];
            //         }
            //     }
            // }//双调排序query节点遍历循环
        }
    }
}


template <typename CoordDType, int ArrayLengthPerThread, int depth_K>
__global__ void FlashKNN_Selected_Query_DL_kernel(
    int* Indices_out,
    CoordDType* Dis_out,
    const CoordDType* GridCoord,        // N, 3
    const int* Queryindex,              // N
    const int* Parent2Child,            // N_+1
    const int* Parent2ChildQuery,       // N_+1
    const int* ParentNeigh,             // N_
    const int* CumCntInNeigh,           // N_, 28
    const int K,                     
    const int N_prt,
    const int SM_index_offset,
    const int max_point_num_loaded,
    const int batch_for_prune = 8,
    const float cut_radiu2 = INFINITY
){  
    // 声明共享内存
    SharedMemory<float> shared;
    CoordDType* buf = shared.getPointer();
    int* buf_index = (int*)(buf + SM_index_offset);

    int max_support_point_num_loaded_per_batch = max_point_num_loaded;
    int loaded_support_points_num = 0;
    int total_support_points_num = 0;
    // bool flag = (blockIdx.x == 159 && blockIdx.y == 12 && threadIdx.x < 32 && threadIdx.x > 23 && threadIdx.y <= 1);
    
    //一个block处理一个parent及其邻域
    int parent_idx = blockIdx.x + blockIdx.y*gridDim.x;
    for(;parent_idx < N_prt;parent_idx += gridDim.x*gridDim.y){
        int thread_idx_in_block = threadIdx.y*blockDim.x + threadIdx.x;
        // if(flag){
        //     printf("准备读取support数据, threadIdx.x: %d, threadIdx.y: %d, parent_idx: %d, CumCntInNeigh addr: %p \n", threadIdx.x, threadIdx.y, parent_idx, CumCntInNeigh + 28*parent_idx + 27);
        // }
        total_support_points_num = *(CumCntInNeigh + 28*parent_idx + 27);
        //循环读取一个批次的support点，并进行排序
        loaded_support_points_num = 0;
        int neigh_idx = 0;
        for(;loaded_support_points_num<total_support_points_num;){
            //读取一个parent3*3邻域格中的所有Low-level-point
            __syncthreads();//等待前一轮所有线程信息写入完毕
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
            // if(flag){
            //     printf("当前批次support点读取完毕, 准备读取query数据, threadIdx.x: %d, threadIdx.y: %d, parent_idx: %d, Parent2ChildQuery addr: %p \n", threadIdx.x, threadIdx.y, parent_idx, Parent2ChildQuery + parent_idx + 1);
            // }
            __syncthreads();//等待写入完毕
            //一个warp处理一个子节点
            // int child_idx_num = CumCntInNeigh[28*parent_idx + 1];   //父节点对应子节点数量
            int child_idx_start = Parent2ChildQuery[parent_idx];
            int child_idx_end = Parent2ChildQuery[parent_idx + 1];
            int child_idx_finished = child_idx_start;

            for(;child_idx_finished < child_idx_end;child_idx_finished += blockDim.y){
                int child_idx_origin = min(child_idx_finished+threadIdx.y, child_idx_end-1);
                int batch = ArrayLengthPerThread >> 1;
                // 读取中心子节点
                int query_index = Queryindex[child_idx_origin];
                CoordDType center_x = GridCoord[query_index*3 + 0];
                CoordDType center_y = GridCoord[query_index*3 + 1];
                CoordDType center_z = GridCoord[query_index*3 + 2];
                CoordDType best_dis[ArrayLengthPerThread];
                int best_idx[ArrayLengthPerThread];
                int nbr_idx_start = 0;
                for(int i = 0;i < ArrayLengthPerThread;i++){best_idx[i]=query_index;best_dis[i] = INFINITY;} //初始化结果
                if(loaded_support_points_num != loaded_support_points_num_cur_batch){ //不是第一个循环，从已有结果中读取数据
                    for(int i = 0;i < ArrayLengthPerThread;i++){
                        int k = i*blockDim.x+threadIdx.x;
                        if(k < K){
                            best_idx[i]=Indices_out[child_idx_origin*K+k];
                            best_dis[i]=Dis_out[child_idx_origin*K+k];
                        }
                        else{
                            best_idx[i]=query_index;
                        }
                    }
                }
                else{  //第一个循环，从附近开始读取
                    // best_dis[MaxArrayIdx] = 0;
                    nbr_idx_start = max(query_index - Parent2Child[parent_idx] - K, 0);
                }
                // 遍历所有邻域子节点
                
                int offset = threadIdx.x;
                int S_points_num_traversed = 0;
                while(S_points_num_traversed < loaded_support_points_num_cur_batch){
                    bool skip = true;
                    CoordDType max_dis = best_dis[(K-1)>>depth_K];
                    max_dis = WARP_SHFL(max_dis, (K-1)&(blockDim.x - 1), blockDim.x);
                    for(int batch_idx = 0;batch_idx < batch && offset < loaded_support_points_num_cur_batch;batch_idx++){
                        int child_nbr_idx_with_offset = ((nbr_idx_start + offset)%loaded_support_points_num_cur_batch);
                        CoordDType dis_x = buf[child_nbr_idx_with_offset*3 + 0] - center_x;
                        CoordDType dis_y = buf[child_nbr_idx_with_offset*3 + 1] - center_y;
                        CoordDType dis_z = buf[child_nbr_idx_with_offset*3 + 2] - center_z;
                        CoordDType dis = dis_x*dis_x + dis_y*dis_y + dis_z*dis_z;
                        if(dis < max_dis && dis < cut_radiu2){
                            // max_dis = dis;
                            best_dis[batch_idx + batch] = dis;
                            best_idx[batch_idx + batch] = buf_index[child_nbr_idx_with_offset];
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
                    S_points_num_traversed += batch*blockDim.x;
                    // if(child_idx_origin == 2000){
                    //     printf("准备排序, thread.x: %d, blockDim.x: %d, max_dis: %f, best_dis: %f, %f, best_idx: %d, %d, child_idx_origin: %d, query_index: %d, skip: %d \n", threadIdx.x, blockDim.x, max_dis, best_dis[0], best_dis[1], best_idx[0], best_idx[1], child_idx_origin, query_index, skip);
                    // }
                    // if(S_points_num_traversed >= (K<<1)){
                    //     batch = batch_for_prune;
                    // }
                    if(skip){
                        continue;
                    }
                    //调用双调排序
                    Bitonic_Sort<CoordDType, ArrayLengthPerThread, depth_K>(best_dis, best_idx);
                }
                // printf("排序完毕，准备写入信息 \n");
                // 邻域候选节点遍历完毕, 写入索引信息
                if(child_idx_finished+threadIdx.y < child_idx_end){
                    for(int nbr_idx = threadIdx.x;nbr_idx<K;nbr_idx += blockDim.x){
                    Indices_out[K*child_idx_origin+nbr_idx] = best_idx[nbr_idx >> depth_K];
                    Dis_out[K*child_idx_origin+nbr_idx] = best_dis[nbr_idx >> depth_K];
                    // if(blockIdx.x == 159 && blockIdx.y == 12 && threadIdx.x < 32 && threadIdx.x >= 24 && threadIdx.y <= 1){
                    //     printf("threadIdx.x: %d, threadIdx.y: %d, child_idx_origin: %d, query_index: %d, Indices_out: %d, Dis_out: %f \n", threadIdx.x, threadIdx.y, child_idx_origin, query_index, Indices_out[K*child_idx_origin+nbr_idx], Dis_out[K*child_idx_origin+nbr_idx]);
                    // }
                }
                }
                // __syncthreads();
                // printf("写入信息完毕 \n");
            }
        }
    }
}


void FlashKNN_Nearest_Back_Query_DL(
    const torch::Tensor &xyz_H,
    const torch::Tensor &xyz_L,
    const torch::Tensor &Parent2Child_H,
    const torch::Tensor &Parent2Child_L,
    const torch::Tensor &ParentNeigh,
    const torch::Tensor &CumCntInNeigh_L,
    torch::Tensor &Indices,
    torch::Tensor &Dis_out
){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(WARP_SIZE,2,1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const uint64_t TotalShareMemory = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
    // const uint64_t max_point_num_loaded = 2000;
    uint64_t dtypesize = 4;
    const uint64_t max_point_num_loaded = 256; //留出余量，防止共享内存占用过多影响性能
    // const uint64_t maxGridX = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
    // std::cout<<"TotalShareMemory: "<<TotalShareMemory<<endl;
    // std::cout<<"max_point_num_loaded: "<<max_point_num_loaded<<endl;
    // std::cout<<"sizeof(GridCoord.dtype()): "<<sizeof(GridCoord.dtype())<<endl;
    const dim3 blocks(grid_x, std::min((uint64_t)16, maxGridY), 1);
    int MemoryCost = max_point_num_loaded*(3*dtypesize+4); // 27K  B
    int MemoryOffset = (max_point_num_loaded*3*dtypesize) / sizeof(int);

    FlashKNN_Nearest_Back_Query_DL_kernel<<<blocks, threads, MemoryCost, stream>>>(
        (int*) Indices.data_ptr(),
        (float*) Dis_out.data_ptr(),
        (const float*) xyz_H.data_ptr(),
        (const float*) xyz_L.data_ptr(),
        (const int*) Parent2Child_H.data_ptr(),
        (const int*) Parent2Child_L.data_ptr(),
        (const int*) ParentNeigh.data_ptr(),
        (const int*) CumCntInNeigh_L.data_ptr(),
        ParentNeigh.size(0),
        MemoryOffset,
        max_point_num_loaded
    );
    cudaDeviceSynchronize();
}

void FlashKNN_Back_Query_DL(
    const torch::Tensor &xyz_H,
    const torch::Tensor &xyz_L,
    const torch::Tensor &Parent2Child_H,
    const torch::Tensor &Parent2Child_L,
    const torch::Tensor &ParentNeigh,
    const torch::Tensor &CumCntInNeigh_L,
    const int num_nbr,
    torch::Tensor &Indices,
    torch::Tensor &Dis_out
){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // const dim3 threads(WARP_SIZE,2,1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const uint64_t TotalShareMemory = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
    // const uint64_t max_point_num_loaded = 2000;
    uint64_t dtypesize = 4;
    const uint64_t max_point_num_loaded = 128; //留出余量，防止共享内存占用过多影响性能
    // const uint64_t maxGridX = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
    // std::cout<<"TotalShareMemory: "<<TotalShareMemory<<endl;
    // std::cout<<"max_point_num_loaded: "<<max_point_num_loaded<<endl;
    // std::cout<<"sizeof(GridCoord.dtype()): "<<sizeof(GridCoord.dtype())<<endl;
    const dim3 blocks(grid_x, std::min((uint64_t)16, maxGridY), 1);
    int MemoryCost = max_point_num_loaded*(3*dtypesize+4); // 27K  B
    int MemoryOffset = (max_point_num_loaded*3*dtypesize) / sizeof(int);

    int bit_len_K = (32 - __builtin_clz(num_nbr-1));
    int blockdimx = WARP_SIZE;
    int blockdimy = 4;
    if(bit_len_K < 5){  // 5~31
        blockdimx >>= min((5 - bit_len_K), 2); //min: 8
        blockdimy <<= min((5 - bit_len_K), 2); //max: 16
    }
    const dim3 threads(blockdimx,blockdimy,1);
    if(xyz_H.dtype() == torch::kFloat32){
        if(bit_len_K == 6){
            FlashKNN_Back_Query_DL_kernel<float, 4, 5><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_H.data_ptr(),
                (const float*) xyz_L.data_ptr(),
                (const int*) Parent2Child_H.data_ptr(),
                (const int*) Parent2Child_L.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh_L.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                MemoryOffset,
                max_point_num_loaded
            );
        }
        else if(bit_len_K == 5){
            FlashKNN_Back_Query_DL_kernel<float, 2, 5><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_H.data_ptr(),
                (const float*) xyz_L.data_ptr(),
                (const int*) Parent2Child_H.data_ptr(),
                (const int*) Parent2Child_L.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh_L.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                MemoryOffset,
                max_point_num_loaded
            );
        }
        else if(bit_len_K == 4){
            FlashKNN_Back_Query_DL_kernel<float, 2, 4><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_H.data_ptr(),
                (const float*) xyz_L.data_ptr(),
                (const int*) Parent2Child_H.data_ptr(),
                (const int*) Parent2Child_L.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh_L.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                MemoryOffset,
                max_point_num_loaded
            );
        }
        else if(bit_len_K <= 3){
            FlashKNN_Back_Query_DL_kernel<float, 2, 3><<<blocks, threads, MemoryCost, stream>>>(
                (int*) Indices.data_ptr(),
                (float*) Dis_out.data_ptr(),
                (const float*) xyz_H.data_ptr(),
                (const float*) xyz_L.data_ptr(),
                (const int*) Parent2Child_H.data_ptr(),
                (const int*) Parent2Child_L.data_ptr(),
                (const int*) ParentNeigh.data_ptr(),
                (const int*) CumCntInNeigh_L.data_ptr(),
                num_nbr,
                ParentNeigh.size(0),
                MemoryOffset,
                max_point_num_loaded
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

void FlashKNN_Selected_Query_DL(
    const torch::Tensor &GridCoord,
    const torch::Tensor &Queryindex,
    const torch::Tensor &Parent2Child,
    const torch::Tensor &Parent2ChildQuery,
    const torch::Tensor &ParentNeigh,
    const torch::Tensor &CumCntInNeigh,
    const int K,
    torch::Tensor &Indices,
    torch::Tensor &Dis,
    int batch_for_prune,
    float cut_radiu2
){
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // const dim3 threads(WARP_SIZE,2,1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const uint64_t TotalShareMemory = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
    // const uint64_t max_point_num_loaded = 2000;
    uint64_t dtypesize = 4;
    const uint64_t max_point_num_loaded = 256; //留出余量，防止共享内存占用过多影响性能

    int bit_len_K = (32 - __builtin_clz(K-1));
    int blockdimx = WARP_SIZE;
    int blockdimy = 2;
    if(bit_len_K < 5){  // 5~31
        blockdimx >>= min((5 - bit_len_K), 2);
        blockdimy <<= min((5 - bit_len_K), 2);
    }
    // blockdimy = 2;
    const dim3 threads(blockdimx,blockdimy,1);
    // const dim3 threads(32,2,1);

    const dim3 blocks(grid_x, std::min((uint64_t)16, maxGridY), 1);
    int MemoryCost = max_point_num_loaded*(3*dtypesize+4); // 27K  B
    int MemoryOffset = (max_point_num_loaded*3*dtypesize) / sizeof(int);
    // std::cout<<"子节点数量"<<GridCoord.size(0)<<std::endl;
    // std::cout<<"父节点数量"<<ParentNeigh.size(0)<<std::endl;
    if(GridCoord.dtype() == torch::kFloat32){
        if(bit_len_K == 6){
            FlashKNN_Selected_Query_DL_kernel<float, 4, 5><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Queryindex.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) Parent2ChildQuery.data_ptr(),
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
            FlashKNN_Selected_Query_DL_kernel<float, 2, 5><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Queryindex.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) Parent2ChildQuery.data_ptr(),
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
            FlashKNN_Selected_Query_DL_kernel<float, 2, 4><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Queryindex.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) Parent2ChildQuery.data_ptr(),
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
        else if(bit_len_K <= 3){
            FlashKNN_Selected_Query_DL_kernel<float, 2, 3><<<blocks, threads, MemoryCost, stream>>>(
            (int*) Indices.data_ptr(),
            (float*) Dis.data_ptr(),
            (const float*) GridCoord.data_ptr(),
            (const int*) Queryindex.data_ptr(),
            (const int*) Parent2Child.data_ptr(),
            (const int*) Parent2ChildQuery.data_ptr(),
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
