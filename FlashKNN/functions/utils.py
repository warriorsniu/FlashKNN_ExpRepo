import torch
import torch.nn as nn
import numpy as np

def vanilla_Knn(query: torch.Tensor, support: torch.Tensor, K: int, 
                query_batchsize: int = 1000, support_batchsize: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized K-nearest neighbors with L2 distance, merging topk operations in inner loop.
    
    Args:
        query: Tensor of query points (M, D)
        support: Tensor of support points (N, D)
        K: Number of nearest neighbors
        query_batchsize: Batch size for query points
        support_batchsize: Batch size for support points
    
    Returns:
        distances: L2 distances of K nearest neighbors (M, K)
        indices: Indices of K nearest neighbors in support (M, K)
    """
    assert query.device == support.device, "query and support must be on the same device"
    assert query.size(1) == support.size(1), "Feature dimensions must match"
    device = query.device
    M, D = query.shape
    N, _ = support.shape

    global_distances = torch.full((M, K), float('inf'), device=device)
    global_indices = torch.full((M, K), -1, device=device, dtype=torch.long)
    support_sq = torch.sum(support** 2, dim=1, keepdim=True)  # (N, 1)

    # Outer loop: batch over query
    for q_start in range(0, M, query_batchsize):
        q_end = min(q_start + query_batchsize, M)
        batch_query = query[q_start:q_end]  # (Bq, D)
        Bq = batch_query.shape[0]
        batch_query_sq = torch.sum(batch_query **2, dim=1, keepdim=True)  # (Bq, 1)

        # Initialize candidate buffer with size (Bq, K + Bs) to avoid repeated concatenation
        # We'll dynamically use the first (K + current_Bs) columns
        max_candidate_size = K + support_batchsize
        candidate_dists_sq = torch.full((Bq, max_candidate_size), float('inf'), device=device)
        candidate_indices = torch.full((Bq, max_candidate_size), -1, device=device, dtype=torch.long)
        # Fill initial K positions with infinity (placeholder for global top-K)
        candidate_dists_sq[:, :K] = float('inf')
        candidate_indices[:, :K] = -1

        # Inner loop: batch over support
        for s_start in range(0, N, support_batchsize):
            s_end = min(s_start + support_batchsize, N)
            Bs = s_end - s_start
            batch_support = support[s_start:s_end]  # (Bs, D)
            batch_support_sq = support_sq[s_start:s_end]  # (Bs, 1)

            # Compute L2 squared distance (Bq, Bs)
            cross_dot = torch.matmul(batch_query, batch_support.t())
            distances_sq = batch_query_sq + batch_support_sq.t() - 2 * cross_dot

            # Copy current top-K candidates to the buffer's first K positions
            # and new distances to the next Bs positions
            candidate_dists_sq[:, :K] = candidate_dists_sq[:, :K]  # Keep previous top-K
            candidate_dists_sq[:, K:K+Bs] = distances_sq  # Add new candidates
            candidate_indices[:, :K] = candidate_indices[:, :K]  # Keep previous indices
            candidate_indices[:, K:K+Bs] = torch.arange(s_start, s_end, device=device).view(1, -1).repeat(Bq, 1)  # New indices

            # Single topk to get the best K from (previous K + current Bs) candidates
            topk_dists_sq, topk_indices = torch.topk(
                candidate_dists_sq[:, :K+Bs],  # Only use valid columns
                k=K,
                dim=1,
                largest=False
            )

            # Update candidates with new top-K
            candidate_dists_sq[:, :K] = topk_dists_sq
            candidate_indices[:, :K] = candidate_indices.gather(1, topk_indices)

        # Convert to L2 distance and update global results
        local_distances = torch.sqrt(candidate_dists_sq[:, :K])
        global_distances[q_start:q_end] = local_distances
        global_indices[q_start:q_end] = candidate_indices[:, :K]

    return global_distances, global_indices

def cal_recall(nbr_indices_pred: torch.Tensor, nbr_indices_gt: torch.Tensor):
    device = nbr_indices_pred.device
    totallen = len(nbr_indices_pred)
    K = nbr_indices_pred.shape[1]

    result = torch.hstack((nbr_indices_gt, nbr_indices_pred))
    result_ordered, order = result.sort(dim=1)
    order_index = nn.functional.pad(torch.cumsum(result_ordered[:,1:] != result_ordered[:,:-1], dim=1), (1,0), value=0)
    inverse = torch.zeros_like(order)
    pt_idx_expand= torch.arange(totallen, device=device)[:,None].repeat(1, 2*K)
    inverse[pt_idx_expand, order] = torch.arange(2*K, device=device)[None, :].repeat(totallen, 1)
    order_index = order_index[pt_idx_expand, inverse]
    counter_gt = torch.zeros_like(result_ordered)
    counter_pre = torch.zeros_like(result_ordered)

    counter_gt = counter_gt.scatter_add(dim=1, index=order_index[:, :K], src=torch.ones_like(nbr_indices_pred))
    counter_pre = counter_pre.scatter_add(dim=1, index=order_index[:, K:], src=torch.ones_like(nbr_indices_pred))

    counter_recall = torch.minimum(counter_pre, counter_gt).sum(dim=1) / K

    return counter_recall