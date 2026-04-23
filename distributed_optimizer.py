from typing import List, Tuple

import torch

from fsdp_wrapper import FSDPUnit


class DistributedOptimizer:
    """Adam optimizer that operates on fp32 main_weight shards.

    Args:
        fsdp_units: FSDP units whose shards will be optimised.
        lr: Learning rate.
        betas: Coefficients for computing running averages of gradient and
            its square.
        eps: Term added to the denominator for numerical stability.
    """

    def __init__(
        self,
        fsdp_units: List[FSDPUnit],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.fsdp_units = fsdp_units

        # Wrap each unit's fp32 main_weight shard as an nn.Parameter so
        # that torch.optim.Adam can manage it.  The Parameter shares
        # storage with the shard, so Adam updates it in-place.
        self.fp32_shards: List[torch.nn.Parameter] = []
        for unit in fsdp_units:
            shard = unit.main_weight_buffer.get_local_shard()
            param = torch.nn.Parameter(shard, requires_grad=False)
            self.fp32_shards.append(param)

        self.adam = torch.optim.Adam(
            self.fp32_shards,
            lr=lr,
            betas=betas,
            eps=eps,
        )

    def step(self) -> None:
        """Run one optimiser step.

        Reads accumulated fp32 gradients, runs Adam, writes back to bf16
        model weight shards, and clears the gradient buffer.
        """
        # Set each fp32 param's .grad from the accumulated fp32 grad
        # shard.  The main_grad shard is already fp32; .float() is a
        # no-op here but guards against dtype mismatches if the buffer
        # dtype ever changes.
        for fp32_param, unit in zip(self.fp32_shards, self.fsdp_units):
            fp32_param.grad = unit.main_grad_buffer.get_local_shard().float()

        # Adam updates fp32 main_weight shards in-place.
        self.adam.step()

        # Downcast updated fp32 shards to bf16 and write back to
        # model_weight shards, then zero out the grad shard for the next
        # training step.
        for fp32_param, unit in zip(self.fp32_shards, self.fsdp_units):
            unit.model_weight_buffer.get_local_shard().copy_(fp32_param.data.bfloat16())
            unit.main_grad_buffer.get_local_shard().zero_()

    def zero_grad(self) -> None:
        """Zero out all main_grad shards."""
        for unit in self.fsdp_units:
            unit.main_grad_buffer.get_local_shard().zero_()
