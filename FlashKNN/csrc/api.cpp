#include "flash_knn_query.h"
#include "flash_knn_construction.h"
#include <pybind11/stl.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_cuda_shared_mem", &get_cuda_shared_mem, "获取cuda的共享内存信息");
    m.def("FlashKNN_Query_Dynamic_Load", &FlashKNN_Query_Dynamic_Load, "网格KNN查询, 使用动态数据读取, 避免共享内存溢出");
    m.def("FlashKNN_Nearest_Back_Query_DL", &FlashKNN_Nearest_Back_Query_DL, "网格最近邻逆向查询");
    m.def("FlashKNN_Back_Query_DL", &FlashKNN_Back_Query_DL, "网格K近邻逆向查询");
    m.def("FlashKNN_Selected_Query_DL", &FlashKNN_Selected_Query_DL, "子集做query, 全集做support");
    m.def("FlashKNN_Query_GM", &FlashKNN_Query_GM, "使用global内存进行查询");
    m.def("FlashKNN_Query_GMPS", &FlashKNN_Query_GMPS, "数据从全局内存中读取，但是使用并行排序");
    m.def("FlashKNN_Query_SMSS", &FlashKNN_Query_SMSS, "数据从共享内存内存中缓存，但是使用串行排序");
    m.def("FlashKNN_Morton_Encode", &FlashKNN_Morton_Encode, "Fused Morton key encoding");
    m.def("FlashKNN_Morton_Neighbor_Keys", &FlashKNN_Morton_Neighbor_Keys, "Fused 3x3x3 Morton neighbor keys");
    
} 
