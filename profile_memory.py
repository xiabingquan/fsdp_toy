"""Profile GPU memory usage: single-GPU baseline vs 4-GPU FSDP.

Usage:
    python profile_memory.py

Runs 100 training iterations with a 10-layer, hidden_size=4096 model under
two configurations, collects per-iteration memory usage, and saves a
comparison plot to memory_profile.png.

Requires >= 4 CUDA GPUs.
"""

import os
import socket
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from distributed_optimizer import DistributedOptimizer
from fsdp import apply_fsdp
from model import ToyModel

HIDDEN_DIM = 4096
NUM_LAYERS = 10
BATCH_SIZE = 4
NUM_STEPS = 100


def find_free_port() -> int:
    """Return an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_baseline(
    num_steps: int,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
) -> List[float]:
    """Run single-GPU training without FSDP and collect per-step memory usage.

    Uses the same precision path as FSDP: bf16 forward/backward, fp32 Adam,
    bf16 writeback.

    Args:
        num_steps: Number of training iterations.
        hidden_dim: Model hidden dimension.
        num_layers: Number of MLP layers.
        batch_size: Batch size.

    Returns:
        A list of per-step GPU memory usage in MB.
    """
    torch.cuda.set_device(0)
    torch.manual_seed(42)

    # Build bf16 model and fp32 master params.
    model = ToyModel(hidden_dim=hidden_dim, num_layers=num_layers).cuda().bfloat16()
    fp32_params = [
        nn.Parameter(p.data.float().clone(), requires_grad=False)
        for p in model.parameters()
    ]
    adam = torch.optim.Adam(fp32_params, lr=1e-3)

    # Warm-up: run one step so CUDA context and allocator are initialised.
    torch.cuda.synchronize()
    x = torch.randn(
        batch_size, hidden_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    for p, fp32 in zip(model.parameters(), fp32_params):
        p.data.copy_(fp32.data.bfloat16())
    output = model(x)
    output.sum().backward()
    for fp32_param, model_param in zip(fp32_params, model.parameters()):
        fp32_param.grad = model_param.grad.float()
        model_param.grad = None
    adam.step()
    adam.zero_grad()

    # Collect per-step memory usage.
    memories: List[float] = []
    for step in range(num_steps):
        torch.cuda.synchronize()

        # Downcast fp32 master params to bf16.
        for p, fp32 in zip(model.parameters(), fp32_params):
            p.data.copy_(fp32.data.bfloat16())

        # Forward and backward in bf16.
        torch.manual_seed(step)
        x = torch.randn(
            batch_size,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Upcast grads to fp32 and run Adam.
        for fp32_param, model_param in zip(fp32_params, model.parameters()):
            fp32_param.grad = model_param.grad.float()
            model_param.grad = None
        adam.step()
        adam.zero_grad()

        mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        memories.append(mem_mb)

    return memories


def _fsdp_worker(
    rank: int,
    world_size: int,
    port: int,
    num_steps: int,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    result_dict: Dict[int, List[float]],
) -> None:
    """FSDP training worker. Rank 0 writes per-step memory usage to result_dict.

    Args:
        rank: Local rank.
        world_size: Total number of ranks.
        port: TCP port for rendezvous.
        num_steps: Number of training iterations.
        hidden_dim: Model hidden dimension.
        num_layers: Number of MLP layers.
        batch_size: Batch size.
        result_dict: Shared dict for collecting rank 0's memory trace.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        model = ToyModel(hidden_dim=hidden_dim, num_layers=num_layers).cuda().bfloat16()
        units = apply_fsdp(model, rank, world_size)
        optimizer = DistributedOptimizer(units, lr=1e-3)

        # Warm-up step.
        torch.cuda.synchronize()
        x = torch.randn(
            batch_size,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output = model(x)
        (output.sum() / world_size).backward()
        optimizer.step()

        # Collect per-step memory usage on rank 0.
        memories: List[float] = []
        for step in range(num_steps):
            torch.cuda.synchronize()

            torch.manual_seed(step)
            x = torch.randn(
                batch_size,
                hidden_dim,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            output = model(x)
            loss = output.sum() / world_size
            loss.backward()
            optimizer.step()

            if rank == 0:
                mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                memories.append(mem_mb)

        if rank == 0:
            result_dict[0] = memories

    finally:
        dist.destroy_process_group()


def run_fsdp(
    num_steps: int,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    world_size: int = 4,
) -> List[float]:
    """Spawn FSDP workers and return rank 0's per-step memory usage.

    Args:
        num_steps: Number of training iterations.
        hidden_dim: Model hidden dimension.
        num_layers: Number of MLP layers.
        batch_size: Batch size.
        world_size: Number of GPUs to use.

    Returns:
        A list of per-step GPU memory usage in MB (from rank 0).
    """
    port = find_free_port()
    ctx = mp.get_context("spawn")
    result_dict = ctx.Manager().dict()
    mp.spawn(
        _fsdp_worker,
        args=(
            world_size,
            port,
            num_steps,
            hidden_dim,
            num_layers,
            batch_size,
            result_dict,
        ),
        nprocs=world_size,
        join=True,
    )
    return list(result_dict[0])


def plot_memory(
    baseline: List[float],
    fsdp: List[float],
    output_path: str = "memory_profile.png",
) -> None:
    """Plot baseline vs FSDP memory curves and save to file.

    Args:
        baseline: Per-step memory usage (MB) for single-GPU baseline.
        fsdp: Per-step memory usage (MB) for FSDP.
        output_path: Path to save the plot image.
    """
    steps = list(range(1, len(baseline) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, baseline, label="No FSDP (1 GPU)", linewidth=1.5)
    ax.plot(steps, fsdp, label="FSDP (4 GPUs)", linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("GPU Memory Allocated (MB)")
    ax.set_title(
        f"GPU Memory: No FSDP vs FSDP\n"
        f"(hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, batch={BATCH_SIZE})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main():
    """Run baseline and FSDP profiling, then plot the comparison."""
    assert torch.cuda.device_count() >= 4, (
        f"Need at least 4 GPUs, found {torch.cuda.device_count()}"
    )

    print(
        f"Config: hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}, "
        f"batch_size={BATCH_SIZE}, num_steps={NUM_STEPS}"
    )

    # Run single-GPU baseline.
    print("\n[1/2] Running baseline (1 GPU, no FSDP)...")
    baseline_mem = run_baseline(NUM_STEPS, HIDDEN_DIM, NUM_LAYERS, BATCH_SIZE)
    print(f"  Baseline mem: {max(baseline_mem):.1f} MB")

    # Run 4-GPU FSDP.
    print("\n[2/2] Running FSDP (4 GPUs)...")
    fsdp_mem = run_fsdp(NUM_STEPS, HIDDEN_DIM, NUM_LAYERS, BATCH_SIZE, world_size=4)
    print(f"  FSDP rank 0 mem: {max(fsdp_mem):.1f} MB")

    # Plot comparison.
    print("\nPlotting...")
    plot_memory(baseline_mem, fsdp_mem)

    print(
        f"\nSavings: {max(baseline_mem):.1f} MB -> {max(fsdp_mem):.1f} MB "
        f"({(1 - max(fsdp_mem) / max(baseline_mem)) * 100:.1f}% reduction)"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
