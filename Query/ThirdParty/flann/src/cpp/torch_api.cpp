#define FLANN_USE_CUDA
#include <flann/flann.hpp>
// #include <flann/algorithms/kdtree_cuda_builder.h>
// #include <flann/algorithms/kdtree_cuda_3d_index.h>
#include <pybind11/stl.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>

// flann::Index<flann::L2<float>> FlannBuildIndexTorch(
//     const torch::Tensor &Coord_n_4_support
// ){
//     int n_points = Coord_n_4_support.size(0);
//     float* gpu_pointer = (float*)Coord_n_4_support.data_ptr();
//     flann::Matrix<float> matrix_gpu(gpu_pointer,n_points,3, 4);
//     flann::KDTreeCuda3dIndexParams params;
//     params["input_is_gpu_float4"]=true;
//     flann::Index<flann::L2<float> > flannindex( matrix_gpu, params );
//     flannindex.buildIndex();
//     return flannindex;
// };

// void FlannKnnQueryTorch(
//     const torch::Tensor &Coord_n_4_query,
//     const flann::Index<flann::L2<float>> &flannindex, 
//     const int k,
//     torch::Tensor &nn_idx,
//     torch::Tensor &nn_dis
// ){
//     flann::SearchParams params;
//     params.matrices_in_gpu_ram = true;
//     int n_points = Coord_n_4_query.size(0);
//     float* gpu_pointer_query = (float*)Coord_n_4_query.data_ptr();
//     flann::Matrix<float> matrix_gpu_query(gpu_pointer_query,n_points,3, 4);

//     int * gpu_pointer_indices = (int*) nn_idx.data_ptr();
//     float * gpu_pointer_dists = (float*) nn_dis.data_ptr();
//     int stride = 1;
//     flann::Matrix<int> indices_gpu(gpu_pointer_indices,n_points, k, stride);
//     flann::Matrix<float> dists_gpu(gpu_pointer_dists,n_points, k, stride);

//     flannindex.knnSearch(
//         matrix_gpu_query,
//         indices_gpu,
//         dists_gpu,
//         k,
//         params
//     );
// };

class Timer{
public:
    clock_t start_time_;

    void start_timer(const std::string& message = "")
    {
        if (!message.empty()) {
            printf("%s", message.c_str());
            fflush(stdout);
        }
        start_time_ = clock();
    }

    double stop_timer()
    {
        return double(clock()-start_time_)/CLOCKS_PER_SEC;
    }

};

#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif

inline std::string prettyDouble(const double val) {
      const double absVal = abs(val);
      char result[1000];

      if      (absVal >= 1e+18f) osp_snprintf(result,1000,"%.1f%c",float(val/1e18f),'E');
      else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",float(val/1e15f),'P');
      else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",float(val/1e12f),'T');
      else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",float(val/1e09f),'G');
      else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",float(val/1e06f),'M');
      else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",float(val/1e03f),'k');
      else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",float(val*1e15f),'f');
      else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",float(val*1e12f),'p');
      else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",float(val*1e09f),'n');
      else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",float(val*1e06f),'u');
      else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",float(val*1e03f),'m');
      else osp_snprintf(result,1000,"%f",(float)val);

      return result;
    }

void FlannKnnQueryTorch(
    const torch::Tensor &Coord_n_4_support,
    const torch::Tensor &Coord_n_4_query,
    // const flann::Index<flann::L2<float>> &flannindex, 
    const int k,
    torch::Tensor &nn_idx,
    torch::Tensor &nn_dis,
    torch::Tensor &time_info,
    bool debug = false
){
    int n_points = Coord_n_4_support.size(0);
    float* gpu_pointer_support = (float*)Coord_n_4_support.data_ptr();
    flann::Matrix<float> matrix_gpu_support(gpu_pointer_support,n_points,3,4*4);
    flann::KDTreeCuda3dIndexParams params_index;
    params_index["input_is_gpu_float4"]=true;
    flann::Index<flann::L2<float> > flannindex( matrix_gpu_support, params_index );

    Timer timer;
    if(debug){
        std::cout << "calling builder..." << std::endl;
    }
    
    timer.start_timer();
    flannindex.buildIndex();
    torch::cuda::synchronize();
    double time_build_tree = timer.stop_timer();
    if(debug){
        std::cout << "done building tree, took "<< prettyDouble(time_build_tree) << "s" << std::endl;
    }
    flann::SearchParams params_search;
    params_search.matrices_in_gpu_ram = true;
    int n_points_query = Coord_n_4_query.size(0);
    float* gpu_pointer_query = (float*)Coord_n_4_query.data_ptr();
    flann::Matrix<float> matrix_gpu_query(gpu_pointer_query,n_points_query,3,4*4);

    int * gpu_pointer_indices = (int*) nn_idx.data_ptr();
    float * gpu_pointer_dists = (float*) nn_dis.data_ptr();
    int stride = k;
    flann::Matrix<int> indices_gpu(gpu_pointer_indices,n_points_query, k, 0);
    flann::Matrix<float> dists_gpu(gpu_pointer_dists,n_points_query, k, 0);

    if(debug){
        std::cout << "calling query..." << std::endl;
    }
    timer.start_timer();
    flannindex.knnSearch(
        matrix_gpu_query,
        indices_gpu,
        dists_gpu,
        k,
        params_search
    );
    torch::cuda::synchronize();
    double time_query_tree = timer.stop_timer();
    if(debug){
        std::cout << "done knn query, took "<< prettyDouble(time_query_tree) << "s" << std::endl;
    }
    time_info[0] = float(time_build_tree);
    time_info[1] = float(time_query_tree);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("FlannBuildIndexTorch", &FlannBuildIndexTorch, "Flann cuda kdtree build");
    m.def("FlannKnnQueryTorch", &FlannKnnQueryTorch, "Flann cuda kdtree query");
}