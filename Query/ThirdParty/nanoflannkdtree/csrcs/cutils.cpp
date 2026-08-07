#include <torch/extension.h>

std::vector<size_t> kdtree_build(const torch::Tensor &pc, const size_t max_leaf_size);
void kdtree_free(size_t kdtree, size_t pca);
void kdtree_knn(size_t kdtree, const torch::Tensor &qpc, torch::Tensor &indices, torch::Tensor &dists, const bool sorted);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("kdtree_build", &kdtree_build);
    m.def("kdtree_free", &kdtree_free);
    m.def("kdtree_knn", &kdtree_knn);
}