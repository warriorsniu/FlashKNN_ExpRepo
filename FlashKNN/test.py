import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from glob import glob
import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import argparse
import json
# import faiss

import functions
from functions.FlashKnnWrapper import FlashKNN
from functions.z_order import xyz2key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-K', type=int, default=32, help='邻域数量')
    args = parser.parse_args()

    return args

args = parse_args()

# seed = torch.randint(10000000, size=(1,))
seed = 47
print("seed:", seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = "cuda"
K = args.K

data = np.load("test_data/Area_1_conferenceRoom_2.npy")
coord:torch.Tensor = torch.from_numpy(data[:,:3]).to(device).float()

voxel_size = 0.02
grid_coord = torch.floor((coord)/ voxel_size).long()
# grid_coord = ((coord - coord.min()) / voxel_size).long()
# print(f"grid_coord.max(): {grid_coord.max()}, grid_coord.min(): {grid_coord.min()}")
key = xyz2key(grid_coord[:,0], grid_coord[:,1], grid_coord[:,2])
keys_unique, inverse, counts = torch.unique(key, return_inverse = True, return_counts = True)
index = torch.argsort(key)
grid_coord = grid_coord[index][nn.functional.pad(torch.cumsum(counts, dim=0), (1,0))[:len(keys_unique)]]
coord = coord[index][nn.functional.pad(torch.cumsum(counts, dim=0), (1,0))[:len(keys_unique)]]
print(len(keys_unique))
totallen = 250000
if(len(grid_coord) >= totallen):
    center_idx = torch.randint(high=len(grid_coord), size = (1,))
    dis = ((grid_coord[center_idx] - grid_coord)**2).sum(dim=-1)
    sel_idx = dis.argsort()[:totallen]
    loc = grid_coord[sel_idx]
    coord = coord[sel_idx]
else:
    totallen = len(grid_coord)
    loc = grid_coord
    coord = coord

batch = torch.zeros(totallen, device=device, dtype=torch.long)

# test pre-downsampling query mode

exp_repeats = 10
warmup_num = 3
KNN = FlashKNN(num_nbr=K, num_down=2, debug=True, print_time=True)
for _ in range(exp_repeats+warmup_num):
    # ts = time.time()
    print("SMSS:")
    nbr_indices_SMSS = KNN.query(loc, batch, coord, memory_mode="SM", sorting_mode="SS")
    print("GMPS:")
    nbr_indices_GMPS = KNN.query(loc, batch, coord, memory_mode="GM", sorting_mode="PS")
    print("SMPS:")
    nbr_indices_SMPS = KNN.query(loc, batch, coord, memory_mode="SM", sorting_mode="PS")


_, indices_gt = functions.vanilla_Knn(coord, coord, K, 10000, 10000)

counter_recall = functions.cal_recall(nbr_indices_SMPS, indices_gt)
print("mean recall", counter_recall.mean())
print("min recall", counter_recall.min())
print("finished")

# test post-downsampling query mode




# test backquery mode