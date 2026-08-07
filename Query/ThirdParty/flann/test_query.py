import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from glob import glob
import torch
import torch.nn as nn
import EdgeAggr
from EdgeAggr.GridKnnWrapper import GridKnn
from PyTorchCudaFlann import FlannCudaKnnQueryTorchWrapper
from z_order import xyz2key
import time

device = "cuda"
K = 32
data_folder = "/data2/PUB_DS/scannet/train/"
data_pathes = sorted(glob(f"{data_folder}*.pth"))
# data = torch.load(data_pathes[10])
data = torch.load("/data2/PUB_DS/scannet/train/scene0054_00.pth")

# print(data.keys()) #(['coord', 'color', 'scene_id', 'normal', 'semantic_gt20', 'semantic_gt200', 'instance_gt'])

coord:torch.Tensor = torch.from_numpy(data["coord"]).to(device)
voxel_size = 0.01
grid_coord = ((coord - coord.min()) / voxel_size).long()
key = xyz2key(grid_coord[:,0], grid_coord[:,1], grid_coord[:,2])
keys_unique, inverse, counts = torch.unique(key, return_inverse = True, return_counts = True)
index = torch.argsort(key)
grid_coord = grid_coord[index][nn.functional.pad(torch.cumsum(counts, dim=0), (1,0))[:len(keys_unique)]]
coord = coord[index][nn.functional.pad(torch.cumsum(counts, dim=0), (1,0))[:len(keys_unique)]]

totallen = 250000
if(len(grid_coord) >= totallen):
    center_idx = torch.randint(high=len(grid_coord), size = (1,))
    dis = ((grid_coord[center_idx] - grid_coord)**2).sum(dim=-1)
    sel_idx = dis.argsort()[:totallen]
    loc = grid_coord[sel_idx]
    loc_raw = coord[sel_idx]
else:
    totallen = len(grid_coord)
    loc = grid_coord
    loc_raw = coord

batch = torch.zeros(totallen, device=device, dtype=torch.long)

nbr_indices = torch.zeros((totallen, K), device=device, dtype=torch.int32)
nbr_dis = torch.zeros((totallen, K), device=device, dtype=torch.float)

# KNN = GridKnn(num_nbr=K, num_down=2, debug=True)

loc_raw_n_4 = torch.zeros((totallen, 4), device = loc_raw.device)
loc_raw_n_4[:,:3] = loc_raw

print("点云数量: ", totallen)
for _ in range(10):
    ts = time.time()
    # nbr_indices = KNN.query(loc, batch)
    time_cost = torch.zeros(2)
    debug = True

    nbr_indices, nbr_dis = FlannCudaKnnQueryTorchWrapper(loc_raw, loc_raw, K, time_cost, debug)

    print(nbr_indices)
    torch.cuda.synchronize()
    print("总耗时: ", time.time() - ts)