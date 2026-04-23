import os
import socket
import unittest
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from distributed_optimizer import DistributedOptimizer
from fsdp_wrapper import apply_fsdp
from model import ToyModel


def find_free_port() -> int:
    """Return an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(
    rank: int,
    world_size: int,
    port: int,
    fn: Callable,
    result_dict: Dict[int, Any],
    args_tuple: tuple,
) -> None:
    """Generic worker spawned by run_distributed.

    Args:
        rank: Local rank assigned to this process.
        world_size: Total number of processes.
        port: TCP port for the rendezvous master.
        fn: The user-supplied test function to execute.
        result_dict: Shared dict for collecting per-rank return values.
        args_tuple: Extra positional arguments forwarded to fn.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        result = fn(rank, world_size, *args_tuple)
        if result is not None:
            result_dict[rank] = result
    finally:
        dist.destroy_process_group()


def run_distributed(
    fn: Callable,
    world_size: int = 2,
    args: tuple = (),
) -> Dict[int, Any]:
    """Spawn world_size processes each running fn(rank, world_size, *args).

    fn must be a top-level (picklable) function.

    Args:
        fn: Worker function with signature (rank, world_size, *args) -> Any.
        world_size: Number of processes to spawn.
        args: Extra positional arguments forwarded to fn.

    Returns:
        A dict mapping rank to the value returned by fn on that rank
        (only ranks that returned a non-None value are included).
    """
    port = find_free_port()
    ctx = mp.get_context("spawn")
    result_dict = ctx.Manager().dict()
    mp.spawn(
        _worker,
        args=(world_size, port, fn, result_dict, args),
        nprocs=world_size,
        join=True,
    )
    return dict(result_dict)


class ReferenceTrainer:
    """Single-process trainer that mimics the same precision path as FSDP.

    Precision path:
        bf16 model params -> bf16 forward/backward -> upcast grad to fp32
        -> fp32 Adam step -> downcast updated params back to bf16.

    Args:
        model: The model to train (will be moved to CUDA and cast to bf16).
        lr: Learning rate for Adam.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        self.model = model.cuda().bfloat16()
        self.fp32_params: List[torch.nn.Parameter] = [
            torch.nn.Parameter(p.data.float().clone(), requires_grad=False)
            for p in self.model.parameters()
        ]
        self.adam = torch.optim.Adam(self.fp32_params, lr=lr)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """Run one training step: forward, backward, and optimiser update.

        Args:
            x: Input tensor (bf16, on CUDA).

        Returns:
            Detached forward output.
        """
        # Downcast fp32 master params to bf16 for forward / backward.
        for p, fp32 in zip(self.model.parameters(), self.fp32_params):
            p.data.copy_(fp32.data.bfloat16())

        # Forward and backward in bf16.
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        # Upcast bf16 gradients to fp32 and feed to Adam.
        for fp32_param, model_param in zip(self.fp32_params, self.model.parameters()):
            fp32_param.grad = model_param.grad.float()
            model_param.grad = None

        # Adam step on fp32 master params.
        self.adam.step()
        self.adam.zero_grad()

        return output.detach()

    def get_bf16_params(self) -> List[torch.Tensor]:
        """Return bf16-downcast copies of the current fp32 master parameters."""
        return [p.data.bfloat16() for p in self.fp32_params]


def _multi_step_worker(
    rank: int,
    world_size: int,
    state_dict_cpu: dict,
    num_steps: int,
    hidden_dim: int,
    num_layers: int,
) -> List[List[torch.Tensor]]:
    """Worker for test_multi_step_*: train N steps, collect params after each.

    Args:
        rank: Local rank.
        world_size: Total number of ranks.
        state_dict_cpu: Initial model state dict (on CPU).
        num_steps: Number of training steps to run.
        hidden_dim: Hidden dimension for ToyModel.
        num_layers: Number of MLP layers.

    Returns:
        A list of length num_steps, where each element is a list of
        all-gathered bf16 param tensors (on CPU) after that step.
    """
    model = ToyModel(hidden_dim=hidden_dim, num_layers=num_layers).cuda().bfloat16()
    model.load_state_dict({k: v.cuda() for k, v in state_dict_cpu.items()})
    units = apply_fsdp(model, rank, world_size)
    optimizer = DistributedOptimizer(units, lr=1e-3)

    collected: List[List[torch.Tensor]] = []
    for step in range(num_steps):
        # Use the same random seed across ranks so every rank sees the
        # same input data.  Divide loss by world_size so that after
        # reduce-scatter(SUM) the gradient matches single-process training.
        torch.manual_seed(step * 100 + 1)
        x = torch.randn(
            4, hidden_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        output = model(x)
        loss = output.sum() / world_size
        loss.backward()
        optimizer.step()

        # All-gather full params from shards for comparison.
        all_params: List[torch.Tensor] = []
        for unit in units:
            unit.model_weight_buffer.alloc_full_buffer()
            unit.model_weight_buffer.all_gather()
            for view in unit.model_weight_buffer.get_views():
                all_params.append(view.cpu().clone())
            unit.model_weight_buffer.free_full_buffer()
        collected.append(all_params)

    return collected


class TestFSDPTraining(unittest.TestCase):
    """End-to-end multi-step training correctness tests.

    Verifies that FSDP-wrapped training produces bit-exact identical
    parameters as single-process reference training that follows the same
    precision path (bf16 forward/backward, fp32 Adam, bf16 writeback).
    """

    def _run_multi_step(
        self,
        world_size: int,
        num_steps: int = 5,
        hidden_dim: int = 32,
        num_layers: int = 3,
    ) -> None:
        """Run num_steps of training with both reference and FSDP, compare
        parameters after every step.

        Args:
            world_size: Number of FSDP ranks.
            num_steps: Number of training iterations.
            hidden_dim: Hidden dimension for ToyModel.
            num_layers: Number of MLP layers.
        """
        # Build reference model and save initial state for FSDP workers.
        torch.manual_seed(42)
        ref_model = ToyModel(hidden_dim=hidden_dim, num_layers=num_layers)
        state_cpu = {k: v.cpu().clone() for k, v in ref_model.state_dict().items()}
        ref_trainer = ReferenceTrainer(ref_model, lr=1e-3)

        # Run reference training and record params after each step.
        ref_params_per_step: List[List[torch.Tensor]] = []
        for step in range(num_steps):
            torch.manual_seed(step * 100 + 1)
            x = torch.randn(
                4, hidden_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            ref_trainer.train_step(x)
            ref_params_per_step.append(
                [p.cpu().clone() for p in ref_trainer.get_bf16_params()]
            )

        # Run FSDP training in spawned processes.
        results = run_distributed(
            _multi_step_worker,
            world_size=world_size,
            args=(state_cpu, num_steps, hidden_dim, num_layers),
        )

        # Compare FSDP params against reference at every step.
        for step in range(num_steps):
            ref_params = ref_params_per_step[step]
            for rank in range(world_size):
                fsdp_params = results[rank][step]
                self.assertEqual(len(fsdp_params), len(ref_params))
                for i, (fp, rp) in enumerate(zip(fsdp_params, ref_params)):
                    torch.testing.assert_close(
                        fp,
                        rp,
                        atol=0,
                        rtol=0,
                        msg=f"Step {step}, rank {rank}, param {i}: mismatch",
                    )

    def test_multi_step_2gpu(self) -> None:
        """Multi-step training with 2 GPUs matches the reference."""
        self._run_multi_step(world_size=2)

    def test_multi_step_4gpu(self) -> None:
        """Multi-step training with 4 GPUs matches the reference."""
        if torch.cuda.device_count() < 4:
            self.skipTest("Need at least 4 GPUs")
        self._run_multi_step(world_size=4)

    def test_padding_multi_step(self) -> None:
        """Parameters not evenly divisible by world_size still work correctly."""
        self._run_multi_step(world_size=2, hidden_dim=13, num_layers=2)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    unittest.main()
