# fsdp_toy

从零手写的最小化 FSDP（Fully Sharded Data Parallel, ZeRO-3）实现，用于教学目的。

[English](README.md)

## 架构

三层设计，对应真实 FSDP 的核心分层：

```
Buffer  →  FSDP Wrapper  →  Distributed Optimizer
```

| 层 | 文件 | 职责 |
|---|---|---|
| **Buffer** | `buffer.py` | 扁平化分片存储，基于 storage-resize 的显存管理，all-gather / reduce-scatter |
| **FSDP Wrapper** | `fsdp_wrapper.py` | 逐层参数重映射，4 个 forward/backward hook 实现动态 unshard/reshard |
| **Distributed Optimizer** | `distributed_optimizer.py` | 在 fp32 shard 上做 Adam，更新后 downcast 写回 bf16 |

### 单个训练 step 中的参数生命周期

```
forward pre-hook    alloc + all-gather          参数可用（bf16，完整）
forward             前向计算                     autograd 保存引用
forward post-hook   释放 storage                参数变为空壳（0 字节）
backward pre-hook   alloc + all-gather          参数重新可用
backward            反向计算                     param.grad 被填充
backward post-hook  reduce-scatter + 释放       梯度 shard（fp32），参数释放
optimizer step      fp32 Adam → bf16 写回       shard 更新完成
```

### 混合精度数据流

```
model_weight shard (bf16, P/N)
    → all-gather → 完整参数 (bf16, P) → forward/backward
    → param.grad (bf16) → upcast fp32 → reduce-scatter
    → main_grad shard (fp32, P/N) → Adam → main_weight shard (fp32, P/N)
    → downcast → model_weight shard (bf16, P/N)
```

## 文件结构

```
model.py                  简单多层 MLP（测试用模型）
buffer.py                 ShardedBuffer（基于 storage-resize）
fsdp_wrapper.py           FSDPUnit + apply_fsdp
distributed_optimizer.py  DistributedOptimizer（fp32 Adam）
test_fsdp.py              多步训练正确性测试
DESIGN.md                 详细设计文档
```

## 运行测试

需要多张 CUDA GPU 和 nccl 后端：

```bash
# 2 卡测试
python -m pytest test_fsdp.py -v

# 或直接运行
python test_fsdp.py
```

测试验证 FSDP 训练与单进程 reference 训练在相同精度路径下产出 **bit-exact 一致** 的参数。

## 依赖

- Python >= 3.8
- PyTorch >= 2.1（需要 `register_full_backward_pre_hook`）
- 多张 CUDA GPU
