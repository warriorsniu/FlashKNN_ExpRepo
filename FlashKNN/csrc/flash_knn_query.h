#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

using namespace std;


void inline __device__ Re_Heap(
    int new_idx,
    float new_dis,
    int* idx_list,
    float* dis_list,
    int k
){  
    if (new_dis < dis_list[0])
    {
        idx_list[0] = new_idx;
        dis_list[0] = new_dis;

        // 下移调整堆
        int i = 0;
        while (true)
        {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int largest = i;
            //寻找当前节点和左右子节点中的最大者
            if (left < k && dis_list[left] > dis_list[largest])
                largest = left;

            if (right < k && dis_list[right] > dis_list[largest])
                largest = right;

            if (largest != i)
            {
                // 交换索引
                int temp_idx = idx_list[i];
                idx_list[i] = idx_list[largest];
                idx_list[largest] = temp_idx;

                // 交换距离
                float temp_dis = dis_list[i];
                dis_list[i] = dis_list[largest];
                dis_list[largest] = temp_dis;

                i = largest;
            }
            else
            {
                break;
            }
        }
    }
}

//降序排列
void inline __device__ Re_Order(
    int new_idx,
    float new_dis,
    int* idx_list,
    float* dis_list,
    int k
){
    if(new_dis < dis_list[0]){
        dis_list[0] = new_dis;
        idx_list[0] = new_idx;
    }
    else{
        return;
    }
    int pos = 0;
    while (pos+1 < k && dis_list[pos] < dis_list[pos+1])
    {
        float dis_temp = dis_list[pos+1];
        dis_list[pos+1] = dis_list[pos];
        dis_list[pos] = dis_temp;

        int idx_temp = idx_list[pos+1];
        idx_list[pos+1] = idx_list[pos];
        idx_list[pos] = idx_temp;
        pos++;
    }
}



int get_cuda_shared_mem();

// void FlashKNN_Query(
//     const torch::Tensor &GridCoord,
//     const torch::Tensor &Parent2Child,
//     const torch::Tensor &Child2Parent,
//     const torch::Tensor &ParentNeigh,
//     const torch::Tensor &CumCntInNeigh,
//     const torch::Tensor &TotalCntInNeigh,
//     const int K,
//     const int down_times,
//     torch::Tensor &Indices
// );

// void FlashKNN_Nearest_Back_Query(
//     const torch::Tensor &xyz_H,
//     const torch::Tensor &xyz_L,
//     const torch::Tensor &Parent2Child_H,
//     const torch::Tensor &Parent2Child_L,
//     const torch::Tensor &ParentNeigh,
//     const torch::Tensor &CumCntInNeigh_L,
//     const torch::Tensor &TotalCntInNeigh_L,
//     const int down_ratio,
//     torch::Tensor &Indices
// );

// void FlashKNN_Back_Query(
//     const torch::Tensor &xyz_H,
//     const torch::Tensor &xyz_L,
//     const torch::Tensor &Parent2Child_H,
//     const torch::Tensor &Parent2Child_L,
//     const torch::Tensor &ParentNeigh,
//     const torch::Tensor &CumCntInNeigh_L,
//     const torch::Tensor &TotalCntInNeigh_L,
//     const int down_ratio,
//     const int num_nbr,
//     torch::Tensor &Indices,
//     torch::Tensor &Dis_out
// );
// void FlashKNN_Selected_Query(
//     const torch::Tensor &GridCoord,
//     const torch::Tensor &Queryindex,
//     const torch::Tensor &Parent2Child,
//     const torch::Tensor &Parent2ChildQuery,
//     const torch::Tensor &Child2Parent,
//     const torch::Tensor &ParentNeigh,
//     const torch::Tensor &CumCntInNeigh,
//     const torch::Tensor &TotalCntInNeigh,
//     const int K,
//     const int down_ratio,
//     torch::Tensor &Indices
// );

void FlashKNN_Query_Dynamic_Load(
    const torch::Tensor &GridCoord,
    const torch::Tensor &Parent2Child,
    const torch::Tensor &ParentNeigh,
    const torch::Tensor &CumCntInNeigh,
    const int K,
    torch::Tensor &Indices,
    torch::Tensor &Dis,
    int batch_for_prune = 8,
    float cut_radiu2 = INFINITY
);

void FlashKNN_Nearest_Back_Query_DL(
    const torch::Tensor &xyz_H,
    const torch::Tensor &xyz_L,
    const torch::Tensor &Parent2Child_H,
    const torch::Tensor &Parent2Child_L,
    const torch::Tensor &ParentNeigh,
    const torch::Tensor &CumCntInNeigh_L,
    torch::Tensor &Indices,
    torch::Tensor &Dis_out
);

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
);

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
);

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
    torch::Tensor& Dis_out,                       // Nq, K
    const int num_nbr,
    const float cut_radiu2 = INFINITY
);

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
);

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
);
