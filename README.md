# fsdp_toy

A minimal, from-scratch implementation of FSDP (Fully Sharded Data Parallel, ZeRO-3) for educational purposes.

[中文版](README_zh.md)

## Architecture

Three-layer design following the real FSDP stack:

```
Buffer  →  FSDP Wrapper  →  Distributed Optimizer
```

| Layer | File | Responsibility |
|---|---|---|
| **Buffer** | `buffer.py` | Flat sharded storage, storage-resize memory management, all-gather / reduce-scatter |
| **FSDP Wrapper** | `fsdp_wrapper.py` | Per-layer param remapping, 4 forward/backward hooks for dynamic unshard/reshard |
| **Distributed Optimizer** | `distributed_optimizer.py` | fp32 Adam on shards, bf16 writeback |

### Param lifecycle in one training step

```
forward pre-hook    alloc + all-gather          param usable (bf16, full)
forward             compute                     autograd saves references
forward post-hook   free storage                param empty shell (0 bytes)
backward pre-hook   alloc + all-gather          param usable again
backward            compute grads               param.grad populated
backward post-hook  reduce-scatter + free       grad shard in fp32, param freed
optimizer step      fp32 Adam → bf16 writeback  shard updated
```

### Mixed precision flow

```
model_weight shard (bf16, P/N)
    → all-gather → full param (bf16, P) → forward/backward
    → param.grad (bf16) → upcast to fp32 → reduce-scatter
    → main_grad shard (fp32, P/N) → Adam → main_weight shard (fp32, P/N)
    → downcast → model_weight shard (bf16, P/N)
```

## Files

```
model.py                  Simple multi-layer MLP (test model)
buffer.py                 ShardedBuffer (storage-resize based)
fsdp_wrapper.py           FSDPUnit + apply_fsdp
distributed_optimizer.py  DistributedOptimizer (fp32 Adam on shards)
test_fsdp.py              Multi-step training correctness tests
DESIGN.md                 Detailed design document (Chinese)
```

## Running tests

Requires multiple CUDA GPUs and the nccl backend:

```bash
# 2-GPU test
python -m pytest test_fsdp.py -v

# or directly
python test_fsdp.py
```

The tests verify that FSDP training produces **bit-exact identical** parameters as single-process reference training following the same precision path.

## Requirements

- Python >= 3.8
- PyTorch >= 2.1 (for `register_full_backward_pre_hook`)
- Multiple CUDA GPUs
