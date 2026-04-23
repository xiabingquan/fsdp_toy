# FSDP 设计文档

## 1. 基本信息

### 概述

FSDP（Fully Sharded Data Parallel）在 ZeRO-1（分布式 Optimizer）的基础上，进一步切分梯度（ZeRO-2）和参数（ZeRO-3）。核心思想是将原本集中在训练流程末尾的通信操作**分散到每个 FSDP unit 的 forward/backward 中**，通过动态 all-gather 和 release 参数，以更多通信换取更低的显存峰值。

### 四种分片策略

| 策略 | 参数 | 梯度 | Optimizer State | 等价 |
|---|---|---|---|---|
| `no_shard` | 完整 | 完整 | 完整 | 普通 DDP |
| `optim` | 完整 | 完整 | 分片 | ZeRO-1 |
| `optim_grads` | 完整 | 分片 | 分片 | ZeRO-2 |
| `optim_grads_params` | 分片 | 分片 | 分片 | ZeRO-3 |

### 从 ZeRO-1 到 ZeRO-3 的逐步演进

#### ZeRO-1 → ZeRO-2：梯度 buffer 行为变化

ZeRO-1 中，梯度的 reduce-scatter 发生在**整个 backward 结束之后**。在此之前，每个 rank 维护完整大小（P）的梯度 buffer 用于跨 microbatch 累积。

ZeRO-2 将梯度 reduce-scatter 提前到**每个 FSDP unit backward 结束后立即执行**。每个 microbatch 的完整梯度在 reduce-scatter 后立即释放，跨 microbatch 的累积发生在 shard（P/N）上而非完整 buffer（P）上。

| | ZeRO-1 | ZeRO-2 |
|---|---|---|
| reduce-scatter 时机 | 全部 backward 结束后，一次性 | 每个 FSDP unit backward 后，逐步 |
| 跨 microbatch 累积位置 | 完整 buffer（P） | reduced shard（P/N） |
| 完整梯度 buffer 生命周期 | 整个训练 step | 仅当前 unit 的 backward 期间 |

#### ZeRO-2 → ZeRO-3：参数 buffer 行为变化

ZeRO-2 中，每个 rank 始终持有完整参数。参数的 all-gather 发生在 optimizer step 之后（与 ZeRO-1 相同）。

ZeRO-3 将参数 all-gather 延迟到 **forward/backward 需要使用时才执行**，计算完成后立即释放完整参数，只保留 local shard（P/N）。

| | ZeRO-2 | ZeRO-3 |
|---|---|---|
| 参数 all-gather 时机 | optimizer step 后，一次性 | forward/backward 每个 unit 前，逐步 |
| 完整参数生命周期 | 整个训练 step | 仅当前 unit 的 forward 或 backward 期间 |
| 参数常驻显存 | P（完整） | P/N（shard） |

### 通信量对比

设参数总量为 P，M 个 microbatch 做梯度累积。

| | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|
| 梯度 reduce-scatter | 1 次，~P | M 次，~M×P | M 次，~M×P |
| 参数 all-gather | 1 次，~P | 1 次，~P | 2M 次（每个 microbatch 的 forward + backward 各一次），~2M×P |
| **单 step 总计** | **~2P** | **~(M+1)×P** | **~3M×P** |

ZeRO-2/3 的通信量随 microbatch 数量线性增长。当 M 较大时，通信开销显著高于 ZeRO-1。

### 适用范围与局限性

**优势：**

- 对模型结构完全无感。TP 需要逐算子适配切分方案，FSDP 只操作扁平化的参数 buffer，新增算子零适配
- 适用于 TP 受限的场景：模型包含不支持 TP 的自定义模块、attention head 数量太少无法被 TP size 整除等
- 可与 PP 配合使用，通信域不冲突


**局限性：**

- 不减少计算量（每个 rank 仍做全量 forward/backward）
- 不减少 activation 显存
- 通信量显著高于 ZeRO-1，需要 DP 通信域有足够带宽

---

## 2. 基本实现思路

### FSDP Unit

FSDP unit 是参数释放和通信的最小粒度，通常是一个 TransformerLayer。同一 FSDP unit 内的参数被打包为一组，一起 all-gather、一起释放。

### 参数与梯度的生命周期

ZeRO-3 下，参数和梯度都不再常驻完整状态，而是按需申请和释放：

```
参数：shard(常驻) → all-gather → 完整(计算用) → release → shard(常驻)
梯度：申请临时 buffer → backward 写入 → reduce-scatter → 累积到 shard → 释放临时 buffer
```

### 两种存储

每个参数组维护：
- **永久存储**（P/N 大小）：local shard，始终存在
- **临时存储**（P 大小）：完整 buffer，按需分配和释放，用于实际的 forward/backward 计算和通信

---

## 3. 核心实现

以下描述最 naive 的实现，不包含任何通信-计算 overlap 优化。所有通信均为同步操作。

### 抽象层级

三层架构：**Buffer → FSDP Wrapper → 分布式 Optimizer**。

#### Buffer 层

Buffer 是 weight、gradient 等数据的通用底层容器，不感知上层语义（不区分参数还是梯度）。它负责管理一块连续内存的 shard 存储与临时完整存储，提供通用的分片和通信接口。

**分片规则：** 数据按顺序展平拼入连续 buffer，按地址均分为 N 个 shard，不尊重参数边界。

**Buffer 的通用接口：**

- `get_shard_range(param)`：返回 param 在当前 rank shard 中的 (start, end) offset
- `get_local_shard()`：返回当前 rank 的永久 shard 存储（P/N 大小，常驻）
- `alloc_full_buffer()`：申请临时完整存储（P 大小）
- `free_full_buffer()`：释放临时完整存储
- `all_gather()`：从各 rank 的 local shard gather 到临时完整存储
- `reduce_scatter()`：对临时完整存储做 reduce-scatter，输出到指定 shard

上层按（数据精度, 参数类型, fsdp_unit）将参数分组，为每组创建三个 Buffer 实例：

| 实例 | 精度 | 永久存储大小 | 用途 | ZeRO-1 是否有 |
|---|---|---|---|---|
| model_weight_buffer | 低精度(bf16) | P/N | forward/backward 使用的模型参数 shard | 有，但大小为 P（完整），不做动态申请/释放 |
| main_weight_buffer | fp32 | P/N | optimizer 持有的 fp32 主副本 shard | 有，大小 P/N，完全一致 |
| main_grad_buffer | fp32 | P/N | 跨 microbatch 累积 reduced gradient shard | 无，ZeRO-1 的梯度 buffer 是完整大小 P 的 grad_data，reduce-scatter 后直接从中读取 local shard |

**与 ZeRO-1 的差异：**

- ZeRO-1 中，model_weight_buffer 始终是完整大小 P，不需要动态 all-gather/release；grad_data 也是完整大小 P，reduce-scatter 的输出直接写入其中的 local shard 位置，不需要单独的 main_grad_buffer。
- ZeRO-3 由于参数和梯度都需要动态申请/释放完整存储，永久存储只保留 shard，因此需要独立的 main_grad_buffer（P/N）来跨 microbatch 累积 reduced gradient。

上层操作通过组合 Buffer 的通用接口实现：
- **参数 unshard**：`model_weight_buffer.alloc_full_buffer()` → `.all_gather()` → `param.data` 指向完整存储中的 view
- **参数 release**：`model_weight_buffer.free_full_buffer()` → `param.data` 变为空壳
- **梯度 reduce**：`main_grad_buffer.alloc_full_buffer()` → 写入梯度 → `.reduce_scatter()` → `.get_local_shard() += reduce 结果` → `.free_full_buffer()`
- **optimizer 读写**：`main_grad_buffer.get_local_shard()` 读梯度，`model_weight_buffer.get_local_shard()` 写参数

**临时完整存储的申请与释放机制：**

-  `alloc_full_buffer()` 和 `free_full_buffer()` 仅操作底层显存，不创建或销毁 Python 层的 tensor 对象。初始化时，临时完整存储对应的 tensor 变量及其与 `param.data` 之间的 view 引用关系一次性建立后始终保留。
- `free_full_buffer()` 只是将该 tensor 的底层 storage 归还给 CUDA allocator（显存占用变为 0），但 tensor 变量本身仍然存在，`param.data` 的 view 关系也不会断裂。`alloc_full_buffer()` 则重新为同一个 tensor 分配显存。
- 这种机制避免了反复创建/销毁 Python 对象的开销，也保证了 `param.data` 无需在每次 unshard/release 时重新映射。

#### FSDP Wrapper 层

**职责：**
1. 模型初始化时，完成参数和梯度与 Buffer 之间的重映射：将低精度模型参数（`param.data`）映射到 model_weight_buffer 的临时完整存储中的 view，将低精度梯度（`param.main_grad`）映射到 main_grad_buffer 的临时完整存储中的 view，并将参数初始值的 fp32 副本写入 main_weight_buffer。初始化完成后释放临时存储。
2. 通过 hook 管理参数的 all-gather/release 时机和梯度的 reduce-scatter 时机。注册的 hook：

  1. **forward pre-hook**：all-gather 当前 FSDP unit 的参数
  2. **forward post-hook**：release 当前 FSDP unit 的参数
  3. **backward pre-hook**：all-gather 当前 FSDP unit 的参数
  4. **backward post-hook**：将 `param.grad` 写入临时 grad buffer → reduce-scatter → 累积到 shard → release 参数和临时 grad buffer
#### 分布式 Optimizer 层

**职责：** 与 ZeRO-1 基本一致，区别是通过 FSDP Buffer 的接口读写 shard 而非直接操作完整 buffer。

- 从 main_grad shard 读取梯度 → 设为 fp32 主副本的 `.grad`
- Adam.step() 更新 fp32 main_weight shard
- 将 fp32 main_weight shard 写回 model_weight shard


**与 ZeRO-1 的行为差异：** ZeRO-3 下，梯度的 reduce-scatter 和参数的 all-gather 已由 FSDP Wrapper 层在 backward hook 和 forward hook 中完成，因此 optimizer 层不再需要处理这两步。具体对比：

| 操作 | ZeRO-1 optimizer 负责 | ZeRO-3 optimizer 负责 |
|---|---|---|
| 梯度 reduce-scatter | 是（ZERO1 或者 DDP 完成） | 否（FSDP backward hook 已完成） |
| 梯度读取 | 从 grad buffer 的 local shard 读取并 upcast 到 fp32 | 同左，直接从 main_grad shard 读取并 upcast |
| Adam.step() | 更新 fp32 shard | 同左，但是不需要创建shard |
| 参数写回 | fp32 shard → param buffer shard | 同左，fp32 shard → model_weight shard |
| 参数 all-gather | 是（optimizer step 后触发） | 否（下一次 forward pre-hook 触发） |

### 执行流程

#### 初始化

```
1. 按 (dtype, is_expert, fsdp_unit) 对参数分组

2. 为每组创建三个 buffer
   ├─ model_weight_buffer: 分配 P/N 永久存储
   ├─ main_weight_buffer:  分配 P/N 永久存储
   └─ main_grad_buffer:    分配 P/N 永久存储

3. 参数重映射
   ├─ 为 model_weight_buffer 申请临时完整存储（bf16，大小 P）
   ├─ 将 param 初始值拷贝到该临时完整存储中
   ├─ 将 param 初始值中属于当前 rank 的 shard 部分拷贝到 model_weight_buffer 的永久存储（P/N）
   ├─ 将 param 初始值 upcast 为 fp32 后，将当前 rank 的 shard 部分写入 main_weight_buffer 的永久存储（P/N）
   ├─ param.data 指向 model_weight_buffer 临时完整存储中的 view
   └─ 释放 model_weight_buffer 的临时完整存储 → param.data 变为空壳

4. 创建底层 Adam，替换 param_groups 为 fp32 shard 版本

5. 注册 forward/backward hook
```

#### 训练 Step（以 ZeRO-3 为例）

```
=== Forward ===

Layer i pre-hook:
  ├─ 申请临时完整存储（大小 P）
  ├─ all_gather(model_weight shard → 临时存储)
  └─ param.data 指向临时存储中的 view → 参数恢复完整

Layer i forward 计算

Layer i post-hook:
  └─ 释放临时存储 → param.data 变为空壳

=== Backward ===

Layer i backward pre-hook:
  ├─ 申请临时完整存储
  ├─ all_gather(model_weight shard → 临时存储)
  └─ param.data 恢复完整

Layer i backward 计算 → 产出 param.grad

Layer i backward post-hook:
  ├─ 申请临时 grad buffer（完整大小 P）
  ├─ 临时 grad buffer.copy_(param.grad)
  ├─ del param.grad
  ├─ reduce_scatter(临时 grad buffer → grad shard)
  ├─ main_grad_buffer(P/N) += grad shard    ← 跨 microbatch 累积
  ├─ 释放临时 grad buffer
  └─ 释放参数临时存储 → param.data 变为空壳

=== Optimizer Step ===

  ├─ 从 main_grad shard 读取梯度 → fp32 主副本.grad
  ├─ Adam.step() → 更新 fp32 main_weight shard
  ├─ 将 fp32 shard 写回 model_weight shard
  └─ main_grad shard 清零，准备下一个 step
```

### 数据精度流

```
model_weight shard (bf16, P/N)
         │
    all-gather（临时存储）
         │
param.data (bf16, 完整 P) ──── forward/backward ──── param.grad (bf16/fp32)
         │                                                │
    release                                       copy 到临时 grad buffer
         │                                                │
model_weight shard (bf16, P/N)                    reduce-scatter
                                                          │
                                              main_grad shard (fp32, P/N)
                                                          │  ← 跨 microbatch 累积
                                                     .float() 
                                                          │
                                              main_weight.grad (fp32, P/N)
                                                          │
                                                     Adam.step()
                                                          │
                                              main_weight shard (fp32, P/N)
                                                          │
                                                     .to(bf16)
                                                          │
                                              model_weight shard (bf16, P/N)
```

---

## 4. 拓展实现

以下为可选优化，不影响核心功能的正确性。

### 参数 all-gather 与计算的 overlap

**思想：** 将同步的参数all-gather改为异步的，即在当前层计算时，预先发起下一层参数的异步 all-gather（prefetch），使通信与计算重叠。

**大致流程：**

- Forward 时：all-gather 当前层参数的同时，异步发起下一层（正向顺序）的 all-gather
- Backward 时：同理，按反向顺序 prefetch 下一层的 all-gather
- 每层计算前 wait 自己的 all-gather 完成即可


### 梯度 reduce-scatter 与计算的 overlap

**思想：** 当一层的梯度 ready 后，异步发起 reduce-scatter，不用等待通信完成即可开始前一层的 backward，从而让本层头的 reduce-scatter 通信与前一层的 backward 计算重叠。

**大致流程：**

- 每个 FSDP unit backward 完成后，立即将梯度写入临时 buffer 并发起异步 reduce-scatter
- 异步通信在独立 CUDA stream 上执行，不阻塞 backward 计算

### 动态显存管理

**思想：** 临时存储不通过 `del` 或 `= None` 释放，而是将底层 storage 归还 CUDA allocator，Python 层保留 tensor 对象和引用关系，下次使用时重新分配 storage 即可。

**优势：**

- 不破坏 param.data → buffer view 的引用链，参数的重映射得以保留
- 不影响 autograd 计算图
- 避免反复创建/销毁 Python 对象

> 进一步的优化包括：buffer 池化复用以减少显存碎片。

### Activation Recomputation 适配

开启 activation recomputation 时，每层的 forward 会在 backward 阶段重新执行一次。naive 实现下，recompute forward 和 backward 各触发一次参数 all-gather，同一层参数被 gather 了两次。

优化方式：对于重计算的层，可选择性地在 forward 之后不释放参数，这样的话 backward 做 recompute forward 时不用重新 gather 参数。空间换时间。

### Checkpoint 保存与加载

- 保存时直接保存各 rank 的 local shard，无需 gather 到 rank 0
- 加载时将 shard 写入对应 rank 的永久存储即可
- 需处理 DP world size 变化时的 shard 重分布

---

## 5. 拓展思考

### FSDP vs TP+PP 的全面对比

| | FSDP | TP | PP |
|---|---|---|---|
| 切分对象 | 参数 buffer（扁平化） | 单个算子的矩阵 | 模型的层 |
| 是否减少单卡的计算量 | 否 | 是（1/N） | 是（只算自己的层） |
| 是否减少单卡的 activation 显存 | 否 | 是（hidden dim 切分后中间结果更小） | 是（更少的层意味着更少的 activation） |
| 通信模式 | 每层 all-gather + reduce-scatter | 每个 matmul 后 all-reduce | 仅层间传 activation |
| 对模型结构的要求 | 无 | 需要逐算子适配 | 需要按层划分 |
| 对通信带宽的要求 | 中等 | 极高（NVLink） | 低 |

**能用 TP+PP 时优先 TP+PP**：TP 不仅切分参数显存，还切分 activation 显存，在相同的 GPU 组上效率严格优于 FSDP。PP 通信量极小（仅传 activation），适合跨节点。

**FSDP 的不可替代场景：**

- 模型包含不支持 TP 的自定义模块，比如某些非矩阵乘法
- Attention head 数量无法被 TP size 整除，比如 attention head 极少时
- 希望减少 TP size 以增加 DP 并行度和吞吐量
- 没有 NVLink 的集群，TP 通信成为瓶颈

### FSDP 的通信域选择

梯度和参数的通信域有本质区别：

**梯度 reduce-scatter 必须在 DP-CP 域内进行。** 因为 reduce-scatter 的语义是对不同数据产出的梯度做归约，只有处理不同数据的 rank（即 DP-CP 域内的 rank）之间才需要归约。这一点无法改变。

**参数 all-gather 不挑通信域。** 参数 all-gather 的语义是从 shard 恢复完整参数，只要各 rank 持有的 shard 拼起来是完整的即可。参数可以在任意自定义的分片域内做 all-gather，不必须是 DP-CP 域。

这为 **HSDP（Hybrid Sharded Data Parallel）** 提供了可能：

```
例：64 GPU，TP=4，PP=1，DP=16

传统 FSDP：参数在全部 16 个 DP rank 上分片
  → all-gather 跨 16 个 rank（跨节点，带宽低）

HSDP：参数只在 4 个 rank 上分片（节点内）
  → all-gather 只在 4 个 rank 内进行（节点内，带宽高）
  → 梯度仍在全部 16 个 DP rank 上 reduce-scatter（不可改变）
  → 代价：每个 rank 存储 P/4 而非 P/16 的参数 shard
```

HSDP 将参数分片域限定在高带宽的子域内（如节点内），将梯度归约留在完整 DP-CP 域内，在通信效率和显存节省之间取得平衡。HSDP 可以和 PP 搭配使用，比如机间开 PP，机内开 HSDP。
