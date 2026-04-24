from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from buffer import ShardedBuffer


class FSDPUnit:
    """Wraps a single nn.Module as one FSDP unit.

    Manages three ShardedBuffers (model_weight bf16, main_weight fp32,
    main_grad fp32), performs the initial parameter remapping, and registers
    forward/backward hooks for dynamic all-gather and reduce-scatter.

    Param buffer lifecycle within a single training step:
        forward pre-hook   -- alloc + all-gather
        forward            -- compute with full params
        forward post-hook  -- free param buffer
        backward pre-hook  -- alloc + all-gather
        backward           -- compute gradients with full params
        backward post-hook -- reduce-scatter grads, free param buffer

    Args:
        module: The module whose parameters will be sharded.
        rank: Local rank in the distributed group.
        world_size: Total number of ranks.
        process_group: Process group for collective ops.  None uses the
            default group.
        device: Target CUDA device.  Defaults to cuda:<rank>.
    """

    def __init__(
        self,
        module: nn.Module,
        rank: int,
        world_size: int,
        process_group: Optional[dist.ProcessGroup] = None,
        device: torch.device = None,
    ):
        self.module = module
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.device = device or torch.device("cuda", rank)

        self.params: List[nn.Parameter] = list(module.parameters())
        if not self.params:
            raise ValueError("FSDPUnit requires a module with at least one parameter")

        # bf16 buffer for model weights used in forward / backward.
        self.model_weight_buffer = ShardedBuffer(
            self.params,
            rank,
            world_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        # fp32 buffer for optimizer master copy.
        self.main_weight_buffer = ShardedBuffer(
            self.params,
            rank,
            world_size,
            dtype=torch.float32,
            device=self.device,
        )
        # fp32 buffer for accumulated reduced gradients.
        self.main_grad_buffer = ShardedBuffer(
            self.params,
            rank,
            world_size,
            dtype=torch.float32,
            device=self.device,
        )

        self._init_param_remapping()
        self._register_hooks()

    def _init_param_remapping(self) -> None:
        """Copy initial parameter values into buffers and remap param.data.

        After this method the model_weight_buffer and main_weight_buffer
        permanent shards hold the initial values (bf16 and fp32
        respectively), each param.data points to its view inside
        model_weight_buffer, and the full buffer is freed so that
        param.data becomes an empty shell.
        """
        # Allocate the temporary full buffer so we can write into it.
        self.model_weight_buffer.alloc_full_buffer()
        model_views = self.model_weight_buffer.get_views()

        for param, view in zip(self.params, model_views):
            # Copy the original bf16 param value into the buffer view, and
            # remap param.data to point to this view instead of the
            # original storage.
            view.copy_(param.data)
            param.data = view

        # Extract this rank's shard from the full buffer into the
        # permanent bf16 shard, then upcast to fp32 for the master copy.
        self.model_weight_buffer.copy_shard_from_full()
        self.main_weight_buffer.get_local_shard().copy_(
            self.model_weight_buffer.get_local_shard().float()
        )

        # Release the full buffer.  param.data views are now empty shells
        # whose storage will be restored by the forward pre-hook.
        self.model_weight_buffer.free_full_buffer()

    def _register_hooks(self) -> None:
        """Register forward/backward hooks on self.module."""
        self.module.register_forward_pre_hook(self._forward_pre_hook)
        self.module.register_forward_hook(self._forward_post_hook)
        self.module.register_full_backward_pre_hook(self._backward_pre_hook)
        self.module.register_full_backward_hook(self._backward_post_hook)

    def _forward_pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Unshard model weights before forward computation."""
        self.model_weight_buffer.alloc_full_buffer()
        self.model_weight_buffer.all_gather(self.process_group)

    def _forward_post_hook(
        self,
        module: nn.Module,
        args: tuple,
        output: torch.Tensor,
    ) -> None:
        """Free param buffer after forward to reclaim memory."""
        self.model_weight_buffer.free_full_buffer()

    def _backward_pre_hook(
        self,
        module: nn.Module,
        grad_output: tuple,
    ) -> None:
        """Unshard model weights before backward computation. The same as _forward_pre_hook"""
        self.model_weight_buffer.alloc_full_buffer()
        self.model_weight_buffer.all_gather(self.process_group)

    def _backward_post_hook(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        """Collect gradients, reduce-scatter, and release buffers.

        After the backward computation, param.grad holds the local bf16
        gradient.  This hook upcasts it to fp32, performs a reduce-scatter
        (SUM) across ranks, and writes the result directly into the
        permanent fp32 grad shard.  No averaging is done here -- the
        caller is responsible for scaling the loss appropriately.
        """
        # Allocate fp32 temporary full grad buffer.
        self.main_grad_buffer.alloc_full_buffer()
        grad_views = self.main_grad_buffer.get_views()

        # Upcast each param's bf16 gradient to fp32 and copy into the
        # contiguous grad buffer, then clear param.grad.
        for param, grad_view in zip(self.params, grad_views):
            if param.grad is not None:
                grad_view.copy_(param.grad.float())
                param.grad = None

        # Reduce-scatter the full grad buffer (SUM) across all ranks.
        # The result is written directly into the permanent fp32 grad
        # shard.  Accumulation is in fp32 to preserve numerical precision.
        self.main_grad_buffer.reduce_scatter(self.process_group)

        # Release temporary buffers.
        self.main_grad_buffer.free_full_buffer()
        self.model_weight_buffer.free_full_buffer()


def apply_fsdp(
    model: nn.Module,
    rank: int,
    world_size: int,
    process_group: Optional[dist.ProcessGroup] = None,
    device: torch.device = None,
) -> List[FSDPUnit]:
    """Wrap each layer of model.layers as an independent FSDP unit.

    Args:
        model: A model with a "layers" attribute (nn.ModuleList).
        rank: Local rank in the distributed group.
        world_size: Total number of ranks.
        process_group: Process group for collective ops.
        device: Target CUDA device.

    Returns:
        A list of FSDPUnit instances, one per layer.

    Raises:
        ValueError: If model has no "layers" attribute.
    """
    if not hasattr(model, "layers"):
        raise ValueError(
            "apply_fsdp expects model to have a 'layers' attribute (nn.ModuleList)"
        )
    units = []
    for layer in model.layers:
        unit = FSDPUnit(layer, rank, world_size, process_group, device)
        units.append(unit)
    return units
