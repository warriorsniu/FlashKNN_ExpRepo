"""Adaptive 3x3x3-neighborhood search over a linear Morton octree.

The hierarchy is sparse: only occupied nodes are stored.  Points are Morton
sorted once and every coarser level is formed from a key prefix.  The coarsest
3x3x3 graph is built by key lookup; finer graphs are propagated from their
parent graph with the same lookup-table construction used by OCNN.

This module intentionally keeps the existing FlashKNN CUDA query ABI.  After
level selection, variable-level query groups and their referenced support nodes
are flattened into compatible Parent2Child/ParentNeigh inputs so that all
levels are processed by one production query-kernel launch.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.nn.functional as F

from .CuFun import (
    FlashKNN_Morton_Encode,
    FlashKNN_Morton_Neighbor_Keys,
    FlashKNN_Query_Dynamic_Load,
)


_SPATIAL_KEY_MASK = (1 << 48) - 1
_MAX_COORD = 1 << 16


@dataclass
class AdaptiveOctreeLevel:
    """One occupied-node level; ``shift`` is its voxel edge log2 scale."""

    shift: int
    keys: torch.Tensor
    counts: torch.Tensor
    steps: torch.Tensor
    point_to_node: torch.Tensor
    neighbors: torch.Tensor
    candidate_counts: torch.Tensor

    @property
    def node_count(self) -> int:
        return int(self.counts.numel())


@dataclass
class AdaptiveOctreeHierarchy:
    """Linear octree and the permutation shared by all its levels."""

    order: torch.Tensor
    inverse_order: torch.Tensor
    sorted_grid: torch.Tensor
    sorted_batch: torch.Tensor
    coordinate_offsets: torch.Tensor
    levels: list[AdaptiveOctreeLevel]
    construction_ms: float


@dataclass
class CompatibleQueryInputs:
    """Variable-level groups flattened to the production FlashKNN ABI."""

    coordinates: torch.Tensor
    parent_steps: torch.Tensor
    neighbors: torch.Tensor
    cumulative_counts: torch.Tensor
    query_order: torch.Tensor
    compat_to_sorted: torch.Tensor
    level_stats: list[dict[str, Any]]
    group_count: int
    support_descriptor_count: int
    support_copy_count: int


def _grid(values: torch.Tensor) -> torch.Tensor:
    mesh = torch.meshgrid(values, values, values, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 3)


def _child_neighbor_lut(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return parent-neighbor positions and child slots for 8 x 27 cases."""

    # Keep FlashKNN's center-first ordering.  Position zero must remain the
    # center because the query kernel uses the center cell as its fast start.
    offsets = _grid(torch.tensor([0, -1, 1], device=device, dtype=torch.long))
    octants = _grid(torch.tensor([0, 1], device=device, dtype=torch.long))
    target = octants[:, None, :] + offsets[None, :, :]
    parent_delta = torch.div(target, 2, rounding_mode="floor")
    child_coord = torch.remainder(target, 2)

    def offset_position(delta: torch.Tensor) -> torch.Tensor:
        # Axis order [0, -1, 1] maps to positions [0, 1, 2].
        return torch.where(delta == 0, 0, torch.where(delta < 0, 1, 2))

    parent_axis = offset_position(parent_delta)
    parent_position = (
        parent_axis[..., 0] * 9
        + parent_axis[..., 1] * 3
        + parent_axis[..., 2]
    )
    child_slot = (
        child_coord[..., 0] * 4
        + child_coord[..., 1] * 2
        + child_coord[..., 2]
    )
    return parent_position.long(), child_slot.long()


def _validate_inputs(grid_coord: torch.Tensor, batch_idx: torch.Tensor) -> None:
    if not grid_coord.is_cuda or not batch_idx.is_cuda:
        raise ValueError("adaptive octree inputs must be CUDA tensors")
    if grid_coord.dtype != torch.int64 or batch_idx.dtype != torch.int64:
        raise ValueError("grid_coord and batch_idx must use torch.int64")
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise ValueError("grid_coord must have shape [N, 3]")
    if batch_idx.ndim != 1 or batch_idx.shape[0] != grid_coord.shape[0]:
        raise ValueError("batch_idx must have shape [N]")
    if grid_coord.shape[0] == 0:
        raise ValueError("cannot construct an octree for an empty point cloud")
    if bool((batch_idx < 0).any().item()) or bool((batch_idx >= 32768).any().item()):
        raise ValueError("batch indices must lie in [0, 32768)")


def _normalize_coordinates(
    grid_coord: torch.Tensor, batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate each batch to nonnegative Morton coordinates."""

    batch_slots = int(batch_idx.max().item()) + 1
    offsets = torch.full(
        (batch_slots, 3), torch.iinfo(torch.int64).max,
        dtype=torch.int64, device=grid_coord.device,
    )
    offsets.scatter_reduce_(
        0, batch_idx[:, None].expand(-1, 3), grid_coord,
        reduce="amin", include_self=True,
    )
    normalized = (grid_coord - offsets[batch_idx]).contiguous()
    if bool((normalized >= _MAX_COORD).any().item()):
        raise ValueError("each batch's coordinate span must be smaller than 65536")
    return normalized, offsets


def _direct_neighbors(
    keys: torch.Tensor,
    coords: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    neighbor_keys = FlashKNN_Morton_Neighbor_Keys(
        coords.contiguous(), batch.contiguous()
    )
    positions = torch.searchsorted(keys, neighbor_keys)
    safe = positions.clamp_max(keys.numel() - 1)
    present = (positions < keys.numel()) & (keys[safe] == neighbor_keys)
    return torch.where(present, safe, -1).reshape(-1, 27).to(torch.int32)


def _propagate_neighbors(
    child_keys: torch.Tensor,
    child_coords: torch.Tensor,
    parent: AdaptiveOctreeLevel,
    parent_position_lut: torch.Tensor,
    child_slot_lut: torch.Tensor,
) -> torch.Tensor:
    child_batch = child_keys >> 48
    child_spatial = child_keys & _SPATIAL_KEY_MASK
    parent_keys = (child_batch << 48) | (child_spatial >> 3)
    parent_index = torch.searchsorted(parent.keys, parent_keys)
    if bool((parent_index >= parent.keys.numel()).any().item()) or not bool(
        (parent.keys[parent_index] == parent_keys).all().item()
    ):
        raise RuntimeError("invalid child-to-parent Morton mapping")

    octant = (
        ((child_coords[:, 0] & 1) << 2)
        | ((child_coords[:, 1] & 1) << 1)
        | (child_coords[:, 2] & 1)
    )
    child_lookup = torch.full(
        (parent.node_count * 8,), -1, dtype=torch.int32,
        device=child_keys.device,
    )
    child_lookup[parent_index * 8 + octant] = torch.arange(
        child_keys.numel(), dtype=torch.int32, device=child_keys.device,
    )

    parent_positions = parent_position_lut[octant]
    child_slots = child_slot_lut[octant]
    parent_neighbors = parent.neighbors.long()[
        parent_index[:, None], parent_positions
    ]
    invalid = parent_neighbors < 0
    lookup_position = parent_neighbors.clamp_min(0) * 8 + child_slots
    neighbors = child_lookup[lookup_position]
    neighbors[invalid] = -1
    return neighbors


def _candidate_counts(
    counts: torch.Tensor, neighbors: torch.Tensor,
) -> torch.Tensor:
    padded = F.pad(counts.long(), (0, 1))
    return padded[neighbors.long()].sum(dim=1).to(torch.int32)


@torch.no_grad()
def build_adaptive_octree(
    grid_coord: torch.Tensor,
    batch_idx: torch.Tensor,
    max_shift: int | None = None,
) -> AdaptiveOctreeHierarchy:
    """Build all occupied Morton levels and their 3x3x3 neighbor graphs.

    ``shift=0`` is the input voxel resolution.  If ``max_shift`` is omitted,
    enough levels are generated to obtain one root node per batch.
    """

    _validate_inputs(grid_coord, batch_idx)
    torch.cuda.synchronize(grid_coord.device)
    start = time.perf_counter()

    normalized_grid, coordinate_offsets = _normalize_coordinates(
        grid_coord, batch_idx
    )
    full_key = FlashKNN_Morton_Encode(
        normalized_grid, batch_idx.contiguous()
    )
    order = torch.argsort(full_key)
    sorted_key = full_key[order]
    sorted_grid = normalized_grid[order].contiguous()
    sorted_batch = batch_idx[order].contiguous()
    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(order.numel(), device=order.device)

    if max_shift is None:
        coordinate_max = int(sorted_grid.max().item())
        max_shift = coordinate_max.bit_length()
    if max_shift < 0 or max_shift > 16:
        raise ValueError("max_shift must lie in [0, 16]")

    spatial_key = sorted_key & _SPATIAL_KEY_MASK
    level_parts: list[dict[str, Any]] = []
    for shift in range(max_shift + 1):
        point_key = (sorted_batch << 48) | (spatial_key >> (3 * shift))
        keys, point_to_node, counts = torch.unique_consecutive(
            point_key, return_inverse=True, return_counts=True
        )
        steps = F.pad(torch.cumsum(counts, dim=0), (1, 0))
        starts = steps[:-1]
        coords = (sorted_grid[starts] >> shift).contiguous()
        node_batch = sorted_batch[starts].contiguous()
        level_parts.append({
            "shift": shift,
            "keys": keys,
            "counts": counts.to(torch.int32),
            "steps": steps.to(torch.int32),
            "point_to_node": point_to_node.to(torch.int32),
            "coords": coords,
            "batch": node_batch,
        })

    parent_position_lut, child_slot_lut = _child_neighbor_lut(grid_coord.device)
    levels: list[AdaptiveOctreeLevel | None] = [None] * len(level_parts)
    for shift in range(max_shift, -1, -1):
        part = level_parts[shift]
        if shift == max_shift:
            neighbors = _direct_neighbors(
                part["keys"], part["coords"], part["batch"]
            )
        else:
            parent = levels[shift + 1]
            assert parent is not None
            neighbors = _propagate_neighbors(
                part["keys"], part["coords"], parent,
                parent_position_lut, child_slot_lut,
            )
        center = neighbors[:, 0]
        expected = torch.arange(
            center.numel(), dtype=torch.int32, device=center.device
        )
        if not bool((center == expected).all().item()):
            raise RuntimeError(f"level {shift} has an invalid center-neighbor map")
        candidate_counts = _candidate_counts(part["counts"], neighbors)
        levels[shift] = AdaptiveOctreeLevel(
            shift=shift,
            keys=part["keys"],
            counts=part["counts"],
            steps=part["steps"],
            point_to_node=part["point_to_node"],
            neighbors=neighbors,
            candidate_counts=candidate_counts,
        )

    torch.cuda.synchronize(grid_coord.device)
    construction_ms = (time.perf_counter() - start) * 1000.0
    return AdaptiveOctreeHierarchy(
        order=order,
        inverse_order=inverse_order,
        sorted_grid=sorted_grid,
        sorted_batch=sorted_batch,
        coordinate_offsets=coordinate_offsets,
        levels=[level for level in levels if level is not None],
        construction_ms=construction_ms,
    )


def select_adaptive_levels(
    hierarchy: AdaptiveOctreeHierarchy,
    k: int,
    min_candidates_factor: int = 2,
    max_candidates_factor: int = 8,
) -> torch.Tensor:
    """Select a level per point by coarsest-to-finest candidate traversal.

    A node descends while its neighborhood contains more than
    ``max_candidates_factor * k`` points.  If its child neighborhood contains
    fewer than ``min_candidates_factor * k`` points, traversal falls back to
    that coarser parent.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if min_candidates_factor < 1:
        raise ValueError("min_candidates_factor must be at least one")
    if max_candidates_factor < min_candidates_factor:
        raise ValueError("max_candidates_factor must not be smaller than min")
    point_count = hierarchy.order.numel()
    if point_count < k:
        raise ValueError(f"point cloud contains {point_count} points, fewer than k={k}")

    coarsest = len(hierarchy.levels) - 1
    selected = torch.full(
        (point_count,), coarsest, dtype=torch.int16,
        device=hierarchy.order.device,
    )
    coarse = hierarchy.levels[coarsest]
    coarse_count = coarse.candidate_counts[coarse.point_to_node.long()]
    active = coarse_count > max_candidates_factor * k

    for shift in range(coarsest - 1, -1, -1):
        level = hierarchy.levels[shift]
        child_count = level.candidate_counts[level.point_to_node.long()]
        can_use_child = active & (child_count >= min_candidates_factor * k)
        selected[can_use_child] = shift
        active = can_use_child & (child_count > max_candidates_factor * k)
    return selected


def _ragged_point_indices(
    starts: torch.Tensor, lengths: torch.Tensor,
) -> torch.Tensor:
    """Expand contiguous ranges without a Python loop."""

    lengths = lengths.long()
    destination_starts = F.pad(torch.cumsum(lengths, dim=0), (1, 0))[:-1]
    total = int(lengths.sum().item())
    owners = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=lengths.device), lengths
    )
    destination = torch.arange(total, device=lengths.device)
    return starts.long()[owners] + destination - destination_starts[owners]


def build_compatible_query_inputs(
    hierarchy: AdaptiveOctreeHierarchy,
    selected: torch.Tensor,
    sorted_xyz: torch.Tensor,
    min_candidates: int,
    max_candidates: int,
) -> CompatibleQueryInputs:
    """Flatten selected levels into the unchanged self-query kernel inputs.

    Query groups occupy the first rows of ``parent_steps``.  Referenced support
    nodes follow them as descriptor-only rows.  The CUDA kernel iterates only
    over ``neighbors.shape[0]`` query groups, while neighbor ids can reference
    the appended support descriptors through the same Parent2Child array.
    """

    query_chunks: list[torch.Tensor] = []
    query_count_chunks: list[torch.Tensor] = []
    neighbor_chunks: list[torch.Tensor] = []
    support_start_chunks: list[torch.Tensor] = []
    support_length_chunks: list[torch.Tensor] = []
    level_stats: list[dict[str, Any]] = []
    support_descriptor_base = 0

    for shift, level in enumerate(hierarchy.levels):
        query_positions = torch.nonzero(selected == shift).flatten()
        if query_positions.numel() == 0:
            continue
        query_parent = level.point_to_node[query_positions].long()
        active_parents, queries_per_parent = torch.unique_consecutive(
            query_parent, return_counts=True
        )
        local_neighbors = level.neighbors[active_parents].long()
        referenced_nodes = torch.unique(
            local_neighbors[local_neighbors >= 0], sorted=True
        )
        node_to_descriptor = torch.full(
            (level.node_count,), -1, dtype=torch.int32,
            device=selected.device,
        )
        node_to_descriptor[referenced_nodes] = torch.arange(
            support_descriptor_base,
            support_descriptor_base + referenced_nodes.numel(),
            dtype=torch.int32,
            device=selected.device,
        )
        mapped_neighbors = node_to_descriptor[local_neighbors.clamp_min(0)]
        mapped_neighbors[local_neighbors < 0] = -1

        query_chunks.append(query_positions)
        query_count_chunks.append(queries_per_parent.to(torch.int32))
        neighbor_chunks.append(mapped_neighbors)
        support_start_chunks.append(level.steps[referenced_nodes])
        support_length_chunks.append(level.counts[referenced_nodes])
        selected_counts = level.candidate_counts[query_parent]
        level_stats.append({
            "shift": shift,
            "query_points": int(query_positions.numel()),
            "active_nodes": int(active_parents.numel()),
            "referenced_support_nodes": int(referenced_nodes.numel()),
            "candidate_mean": float(selected_counts.float().mean().item()),
            "candidate_min": int(selected_counts.min().item()),
            "candidate_max": int(selected_counts.max().item()),
            "below_band_points": int(
                (selected_counts < min_candidates).sum().item()
            ),
            "within_band_points": int((
                (selected_counts >= min_candidates)
                & (selected_counts <= max_candidates)
            ).sum().item()),
            "above_band_points": int(
                (selected_counts > max_candidates).sum().item()
            ),
        })
        support_descriptor_base += referenced_nodes.numel()

    query_order = torch.cat(query_chunks).long().contiguous()
    query_counts = torch.cat(query_count_chunks).to(torch.int32)
    mapped_neighbors = torch.cat(neighbor_chunks).to(torch.int32)
    support_starts = torch.cat(support_start_chunks).long()
    support_lengths = torch.cat(support_length_chunks).to(torch.int32)
    group_count = int(query_counts.numel())
    if query_order.numel() != hierarchy.order.numel():
        raise RuntimeError("adaptive level selection did not cover every query")

    # Support descriptors are offset by the query-group descriptor count.
    neighbors = torch.where(
        mapped_neighbors >= 0,
        mapped_neighbors + group_count,
        mapped_neighbors,
    ).contiguous()
    padded_support_lengths = F.pad(support_lengths, (0, 1))
    per_neighbor = padded_support_lengths[mapped_neighbors.long()]
    cumulative_counts = F.pad(
        torch.cumsum(per_neighbor, dim=1), (1, 0)
    ).to(torch.int32).contiguous()

    descriptor_lengths = torch.cat((query_counts, support_lengths))
    parent_steps = F.pad(
        torch.cumsum(descriptor_lengths, dim=0), (1, 0)
    ).to(torch.int32).contiguous()
    support_point_indices = _ragged_point_indices(
        support_starts, support_lengths
    )
    compat_to_sorted = torch.cat(
        (query_order, support_point_indices)
    ).long().contiguous()
    coordinates = sorted_xyz[compat_to_sorted].contiguous()

    return CompatibleQueryInputs(
        coordinates=coordinates,
        parent_steps=parent_steps,
        neighbors=neighbors,
        cumulative_counts=cumulative_counts,
        query_order=query_order,
        compat_to_sorted=compat_to_sorted,
        level_stats=level_stats,
        group_count=group_count,
        support_descriptor_count=int(support_lengths.numel()),
        support_copy_count=int(support_point_indices.numel()),
    )


class AdaptiveNeighborhoodFlashKNN:
    """FlashKNN using an independently selected octree level per query point."""

    def __init__(
        self,
        num_nbr: int = 32,
        min_candidates_factor: int = 2,
        max_candidates_factor: int = 8,
    ):
        self.num_nbr = num_nbr
        self.min_candidates_factor = min_candidates_factor
        self.max_candidates_factor = max_candidates_factor
        self.last_stats: dict[str, Any] = {}

    @torch.no_grad()
    def query(
        self,
        grid_coord: torch.Tensor,
        batch_idx: torch.Tensor,
        xyz: torch.Tensor | None = None,
        *,
        max_shift: int | None = None,
        batch_for_prune: int = 1,
        cut_radius: float = torch.inf,
        return_distances: bool = False,
    ):
        hierarchy = build_adaptive_octree(grid_coord, batch_idx, max_shift)
        torch.cuda.synchronize(grid_coord.device)
        selection_start = time.perf_counter()
        selected = select_adaptive_levels(
            hierarchy,
            self.num_nbr,
            self.min_candidates_factor,
            self.max_candidates_factor,
        )
        torch.cuda.synchronize(grid_coord.device)
        selection_ms = (time.perf_counter() - selection_start) * 1000.0
        sorted_xyz = (
            hierarchy.sorted_grid.float().contiguous()
            if xyz is None else xyz[hierarchy.order].float().contiguous()
        )
        torch.cuda.synchronize(grid_coord.device)
        compatibility_start = time.perf_counter()
        compatible = build_compatible_query_inputs(
            hierarchy,
            selected,
            sorted_xyz,
            self.min_candidates_factor * self.num_nbr,
            self.max_candidates_factor * self.num_nbr,
        )
        torch.cuda.synchronize(grid_coord.device)
        compatibility_ms = (time.perf_counter() - compatibility_start) * 1000.0

        point_count = hierarchy.order.numel()
        compat_indices = torch.empty(
            (point_count, self.num_nbr), dtype=torch.int32,
            device=grid_coord.device,
        )
        compat_distances = torch.empty(
            (point_count, self.num_nbr), dtype=torch.float32,
            device=grid_coord.device,
        )
        torch.cuda.synchronize(grid_coord.device)
        query_start = time.perf_counter()
        FlashKNN_Query_Dynamic_Load(
            compatible.coordinates,
            compatible.parent_steps,
            compatible.neighbors,
            compatible.cumulative_counts,
            self.num_nbr,
            compat_indices,
            compat_distances,
            batch_for_prune,
            cut_radius * cut_radius,
        )
        mapped_neighbors = compatible.compat_to_sorted[compat_indices.long()]
        sorted_indices = torch.empty_like(compat_indices, dtype=torch.long)
        sorted_distances = torch.empty_like(compat_distances)
        sorted_indices[compatible.query_order] = mapped_neighbors
        sorted_distances[compatible.query_order] = compat_distances
        output_indices = hierarchy.order[
            sorted_indices[hierarchy.inverse_order]
        ]
        output_distances = sorted_distances[hierarchy.inverse_order]
        torch.cuda.synchronize(grid_coord.device)
        query_ms = (time.perf_counter() - query_start) * 1000.0
        self.last_stats = {
            "construction_ms": hierarchy.construction_ms,
            "selection_ms": selection_ms,
            "compatibility_ms": compatibility_ms,
            "query_ms": query_ms,
            "total_ms": (
                hierarchy.construction_ms + selection_ms
                + compatibility_ms + query_ms
            ),
            "k": self.num_nbr,
            "min_candidates_factor": self.min_candidates_factor,
            "max_candidates_factor": self.max_candidates_factor,
            "octree_levels": len(hierarchy.levels),
            "level_node_counts": [level.node_count for level in hierarchy.levels],
            "selected_levels": compatible.level_stats,
            "selection_band_points": {
                key: sum(level[key] for level in compatible.level_stats)
                for key in (
                    "below_band_points", "within_band_points",
                    "above_band_points",
                )
            },
            "query_kernel_launches": 1,
            "compatible_group_count": compatible.group_count,
            "compatible_support_descriptor_count": (
                compatible.support_descriptor_count
            ),
            "compatible_support_copy_count": compatible.support_copy_count,
            "compatible_point_count": int(compatible.coordinates.shape[0]),
            "compatible_point_ratio": (
                float(compatible.coordinates.shape[0]) / point_count
            ),
        }
        if return_distances:
            return output_indices, output_distances
        return output_indices
