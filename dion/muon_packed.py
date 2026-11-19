"""
Plan for Muon on FSDP-sharded matrices with every whole matrix on one owner rank.

Batches of one matrix per rank, padded with dummy matrices to the shard group size, cost host
work and traffic in proportion to the group size. Instead, each matrix goes to one owner rank
(balanced by Newton-Schulz cost): one all-to-all brings every owner the shards of its matrices
and one brings the orthogonalized shards back. Each matrix runs the same momentum update,
Newton-Schulz and weight update as in a batch, so with eager Newton-Schulz the updates are
bitwise those of the batched path.
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from functools import cached_property
from torch import Tensor
from typing import List, Optional, Sequence, Tuple

FOREACH_CHUNK = 16


@dataclass(frozen=True)
class PackedPlan:
    """
    Who owns which matrix, and the size of every rank's shard of it along the sharded dim.
    """

    owners: Tuple[int, ...]
    shard_sizes: Tuple[Tuple[int, ...], ...]
    padded_sizes: Tuple[int, ...]
    shard_dims: Tuple[int, ...]

    @cached_property
    def by_owner(self) -> Tuple[Tuple[int, ...], ...]:
        """
        The matrices of every rank, in index order.
        """
        owned = [[] for _ in range(max(self.owners, default=-1) + 1)]
        for index, owner in enumerate(self.owners):
            owned[owner].append(index)
        return tuple(tuple(indices) for indices in owned)

    def owned_by(self, rank: int) -> Tuple[int, ...]:
        return self.by_owner[rank] if rank < len(self.by_owner) else ()


def chunk_sizes(size: int, world_size: int) -> Tuple[int, ...]:
    """
    Each rank's length of a dim of `size` sharded with torch.chunk's layout (DTensor's Shard).
    """
    padded = -(-size // world_size)
    return tuple(max(0, min(padded, size - rank * padded)) for rank in range(world_size))


def newton_schulz_cost(shape: torch.Size, partitions: Optional[Tuple[int, ...]]) -> float:
    """
    Relative Newton-Schulz time of one tensor (a batch of matrices along its leading dims):
    its FLOPs plus a per-call floor, so small matrices are not free.
    """
    fan_in = shape[-1]
    blocks = partitions if partitions is not None else (shape[-2],)
    flops = math.prod(shape[:-2]) * sum(rows * fan_in * min(rows, fan_in) for rows in blocks)
    return flops / 6e14 + 3e-4 * len(blocks)


def make_plan(
    params: Sequence[Tensor],
    shard_dims: Sequence[int],
    partitions: Sequence[Optional[Tuple[int, ...]]],
    world_size: int,
) -> PackedPlan:
    """
    Owners by longest-processing-time-first over the Newton-Schulz costs.
    Every rank computes the same plan.
    """
    costs = [newton_schulz_cost(p.shape, parts) for p, parts in zip(params, partitions)]
    loads = [0.0] * world_size
    owners = [0] * len(params)
    for index in sorted(range(len(params)), key=lambda i: (-costs[i], i)):
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        owners[index] = owner
        loads[owner] += costs[index]
    return PackedPlan(
        owners=tuple(owners),
        shard_sizes=tuple(
            chunk_sizes(p.size(dim), world_size) for p, dim in zip(params, shard_dims)
        ),
        padded_sizes=tuple(-(-p.size(dim) // world_size) for p, dim in zip(params, shard_dims)),
        shard_dims=tuple(shard_dims),
    )


def foreach_chunks(indices: List[int]) -> List[List[int]]:
    """
    `indices` cut into lists of at most FOREACH_CHUNK entries.
    """
    return [indices[start : start + FOREACH_CHUNK] for start in range(0, len(indices), FOREACH_CHUNK)]


def pad_list(tensors: List[Tensor]) -> List[Tensor]:
    """
    `tensors`, all of one shape and dtype, filled up to FOREACH_CHUNK entries with fresh
    uninitialized tensors like the first, so compiled list functions see lists of one length
    and one shape whatever the number of real entries.
    """
    return tensors + [torch.empty_like(tensors[0]) for _ in range(FOREACH_CHUNK - len(tensors))]


def own_storage(tensor: Tensor) -> Tensor:
    """
    `tensor`, copied when it is a view at a nonzero offset. Dynamo guards whether each list
    entry's storage offset is 0, so views into a shared buffer next to fresh dummies would
    compile a graph per pattern of offsets.
    """
    return tensor if tensor.storage_offset() == 0 else tensor.clone()


def shard_shape(param: Tensor, dim: int, size: int) -> List[int]:
    shape = list(param.shape)
    shape[dim] = size
    return shape


def pad_dim(tensor: Tensor, dim: int, size: int) -> Tensor:
    missing = size - tensor.size(dim)
    if missing == 0:
        return tensor
    return F.pad(tensor, [0, 0] * (tensor.ndim - 1 - dim) + [0, missing])
