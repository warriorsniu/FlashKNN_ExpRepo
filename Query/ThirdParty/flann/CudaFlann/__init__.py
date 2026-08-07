import torch
from .CuFun import FlannKnnQueryTorch

def FlannCudaKnnQueryTorchWrapper(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    num_nbr,
    time_info: torch.Tensor,
    debug = False
):
    assert len(query_points.shape) == len(support_points.shape) == 2, ""
    if query_points.shape[1] == 3:
        query_points_input = torch.zeros((len(query_points), 4), device=query_points.device, dtype=torch.float)
        query_points_input[:,:3] = query_points
    
    if support_points.shape[1] == 3:
        support_points_input = torch.zeros((len(support_points), 4), device=support_points.device, dtype=torch.float)
        support_points_input[:,:3] = support_points


    assert support_points_input.shape[1] == query_points_input.shape[1] == 4, ""
    nbr_indices = torch.zeros((len(query_points), num_nbr), device=query_points.device, dtype=torch.int32)
    nbr_dis = torch.zeros((len(query_points), num_nbr), device=query_points.device, dtype=torch.float)

    FlannKnnQueryTorch(support_points_input, query_points_input, num_nbr, nbr_indices, nbr_dis, time_info, debug)

    return nbr_indices, nbr_dis
