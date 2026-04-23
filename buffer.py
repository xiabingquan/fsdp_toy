import math
from typing import List

import torch
import torch.distributed as dist


class ShardedBuffer:
    """Sharded storage and temporary full storage for a group of tensors.

    Tensors are flattened, concatenated, and padded to be evenly divisible
    by world_size.  Two pieces of memory are managed:

    - Permanent shard (P / N elements) -- always resident.
    - Temporary full buffer (P elements) -- allocated on demand for
      all-gather / reduce-scatter, freed after use.

    Views into the full buffer are created once at init and survive
    alloc / free cycles.

    Args:
        tensors: Tensors whose shapes define the layout inside the buffer.
            Only shapes and numels are read; data is not copied.
        rank: Local rank in the distributed group.
        world_size: Total number of ranks.
        dtype: Element type for both the shard and the full buffer.
        device: Target CUDA device.  Defaults to cuda:<rank>.
    """

    def __init__(
        self,
        tensors: List[torch.Tensor],
        rank: int,
        world_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self.device = device or torch.device("cuda", rank)

        shapes = [t.shape for t in tensors]
        numels = [t.numel() for t in tensors]
        total_numel = sum(numels)

        # Pad total numel so that it is evenly divisible by world_size.
        self.padded_numel = math.ceil(total_numel / world_size) * world_size
        self.shard_numel = self.padded_numel // world_size
        self._padding = self.padded_numel - total_numel

        # Permanent shard storage (always resident).
        self._shard = torch.zeros(self.shard_numel, dtype=dtype, device=self.device)

        # Temporary full buffer and per-tensor views.
        self._full_buffer = torch.zeros(
            self.padded_numel, dtype=dtype, device=self.device
        )

        self._views: List[torch.Tensor] = []
        self._shapes: List[torch.Size] = []
        offset = 0
        for shape, numel in zip(shapes, numels):
            view = self._full_buffer[offset : offset + numel].view(shape)
            self._views.append(view)
            self._shapes.append(shape)
            offset += numel

        # Record byte size, then free storage.  Views stay alive as Python
        # objects; they become usable again after alloc_full_buffer.
        self._full_buffer_nbytes = self._full_buffer.untyped_storage().nbytes()
        self._full_buffer.untyped_storage().resize_(0)

    def get_views(self) -> List[torch.Tensor]:
        """Return pre-created views into the full buffer.

        The views remain as Python objects after free_full_buffer but
        accessing their data is only valid while the full buffer is
        allocated.

        Returns:
            A list of tensors, one per input tensor, shaped to match the
            originals.
        """
        return self._views

    def get_local_shard(self) -> torch.Tensor:
        """Return the permanent local shard (always valid, P / N elements)."""
        return self._shard

    def copy_shard_from_full(self) -> None:
        """Copy this rank's slice from the full buffer into the permanent shard."""
        shard_start = self.rank * self.shard_numel
        shard_end = shard_start + self.shard_numel
        self._shard.copy_(self._full_buffer[shard_start:shard_end])

    def alloc_full_buffer(self) -> None:
        """Reallocate the underlying storage of the full buffer.

        After this call every view returned by get_views points to valid
        (but uninitialised) memory.
        """
        self._full_buffer.untyped_storage().resize_(self._full_buffer_nbytes)

    def free_full_buffer(self) -> None:
        """Release the underlying storage of the full buffer.

        The tensor object and all views remain alive; only the storage is
        returned to the allocator (nbytes becomes 0).
        """
        self._full_buffer.untyped_storage().resize_(0)

    def all_gather(self, group: dist.ProcessGroup = None) -> None:
        """All-gather shards from every rank into the full buffer.

        Prerequisite: alloc_full_buffer must have been called.

        Args:
            group: Process group to communicate over.  None uses the
                default group.
        """
        dist.all_gather_into_tensor(self._full_buffer, self._shard, group=group)

    def reduce_scatter(self, group: dist.ProcessGroup = None) -> None:
        """Reduce-scatter the full buffer into the permanent local shard.

        The full buffer is summed element-wise across ranks and the result
        is scattered so that each rank's portion is written directly into
        its permanent shard, mirroring the symmetry with all_gather.

        Prerequisite: alloc_full_buffer must have been called.

        Args:
            group: Process group to communicate over.  None uses the
                default group.
        """
        dist.reduce_scatter_tensor(self._shard, self._full_buffer, group=group)
