import torch
from .Kdtree import kdtree_build, kdtree_free, kdtree_knn


class KDTree():
    r"""
    kdt = KDTree(xyz) 
    indices, squared_dists = kdt.knn(query_xyz, k=16, ordered=True)
    indices: int32
    dists: float

    Setting ordered = False (default) can be 1.1-1.2x faster. 
    If there are not enough neighbors, the nearest point is used for padding. 
    Resources (reference to xyz, built tree) are freed when kdt goes out of life scope.
    """
    def __init__(self, xyz: torch.Tensor, max_leaf_size=20):
        assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
        if xyz.stride(0) != 3:
            xyz = xyz.contiguous()
        # reserve xyz for knn search
        self.xyz = xyz
        self.n = xyz.shape[0]
        self.tree, self.pca = kdtree_build(xyz, max_leaf_size)
    
    def __del__(self):
        kdtree_free(self.tree, self.pca)
    
    def knn(self, query: torch.Tensor, k=1, ordered=False):
        assert query.ndim == 2 and query.shape[1] == 3 and query.dtype == torch.float
        if query.stride(0) != 3:
            query = query.contiguous()
        queries = query.shape[0]
        nbrs = min(self.n, k)
        if self.n < k : ordered = True
        indices = torch.empty((queries, nbrs), dtype=torch.int32)
        dists = torch.empty((queries, nbrs), dtype=torch.float)
        kdtree_knn(self.tree, query, indices, dists, ordered)
        if self.n < k:
            indices = torch.cat([indices, indices[:, :1].expand(-1, k - self.n)], dim=1)
            dists = torch.cat([dists, dists[:, :1].expand(-1, k - self.n)], dim=1)
        return indices, dists

@torch.no_grad()
def NanoFlannQuery(
        xyz_query:torch.Tensor, 
        xyz_support:torch.Tensor, 
        K, 
        debug=False, 
        time_info:torch.Tensor=None):

    if xyz_query.is_cuda:
        xyz_query = xyz_query.cpu()
        xyz_support = xyz_support.cpu()
    import time
    ts = time.time()
    kdt = KDTree(xyz_support)
    time_cost_construct = time.time() - ts;ts=time.time()
    if debug:
        print("构建kdtree耗时： ", time_cost_construct)

    indices, dists = kdt.knn(xyz_query, K)
    time_cost_query = time.time() - ts

    if debug:
        print("查询kdtree耗时： ", time_cost_query)
    
    if time_info is not None:
        time_info[0] = time_cost_construct
        time_info[1] = time_cost_query

    return indices, dists