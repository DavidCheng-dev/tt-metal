# simple_text_demo.py 运行机制：Tracing 与 Batching 深度解析

> **目标读者**：编译器工程师。以 **Llama-3.1-8B** 为贯穿全文的运行示例。
> 所有论述均附 `file:line` 源码锚点，便于跳读验证。代码标识符、API、文件路径、pytest id 保留英文。

---

## 目录

1. [概述](#1-概述)
2. [如何运行](#2-如何运行以-llama-31-8b-为例)
3. [整体控制流](#3-整体控制流)
4. [Batching](#4-batching)
5. [Tracing](#5-tracing)
6. [端到端时间线：Llama-3.1-8B batch-1 一次完整运行](#6-端到端时间线llama-31-8b-batch-1-一次完整运行)
7. [编译器工程师 FAQ](#7-编译器工程师-faq)
8. [源码索引表](#8-源码索引表)

---

## 1. 概述

[`simple_text_demo.py`](simple_text_demo.py) 是 TT-Transformers 框架的文本生成端到端 demo。
它接受一组 JSON 格式的用户 prompt，在 Tenstorrent 硬件（N150 / T3K / TG 等）上运行自回归文本生成，并输出生成文本与性能指标。

**编译器工程师为什么关心这个 demo？**

TT-Metal 的推理路径采用了两项对性能至关重要的技术：

- **Tracing（本文重点之一）**：把一次 host 录制的 device 计算图（含所有 op dispatch、数据依赖、设备内存布局）以 `trace_id` 固化，之后每个 decode step 只执行一次 `execute_trace`，彻底消除逐 op 的 host → device dispatch 开销。这是 decode 阶段达到硬件峰值性能的关键。

- **Batching + Data Parallelism（本文重点之二）**：多用户、多设备的 batch 组织方式直接影响内存布局、KV cache 管理与 tensor 切片方式。理解这两层概念是分析 trace capture 边界的前提。

**总览图（时间从左到右）**：

```
pytest 启动
    │
    ├─ prepare_generator_args()        # 建模、分 DP submesh
    │
    ├─ [PREFILL 阶段]
    │   ├─ prefill_forward_text()  ← warmup（compile）
    │   └─ prefill_forward_text()  ← 实跑（若 enable_trace，此处内部做 trace capture+replay）
    │
    └─ [DECODE 循环]
        ├─ iteration=0: decode_forward()  ← compile_decode（含 trace capture）
        ├─ iteration=1: decode_forward()  ← trace REPLAY  ←── 纯 replay 从这里开始
        ├─ iteration=2: decode_forward()  ← trace REPLAY
        ├─ ...
        └─ iteration=N: decode_forward()  ← trace REPLAY（EoS 或 max_generated_tokens 终止）
```

---

## 2. 如何运行（以 Llama-3.1-8B 为例）

> **注意**：所有命令必须在 `ct_metal` Docker 容器内执行。
> 从 repo 根目录进入容器：`./start_container.sh`

### 2.1 环境变量

| 变量 | 作用 | 示例 |
|------|------|------|
| `HF_MODEL` | 指定模型路径或 HuggingFace 模型名 | `export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct` |
| `MESH_DEVICE` | 覆盖 mesh 拓扑（可选；不设则自动检测） | `export MESH_DEVICE=N150` |

### 2.2 常用 pytest 命令

```bash
# 单用户延迟测试（batch-1, performance 精度）
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"

# 32 用户吞吐测试
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"

# 长上下文 64k
pytest models/tt_transformers/demo/simple_text_demo.py -k "long-context-64k"

# Data-parallel 4 组，每组 batch-1
pytest models/tt_transformers/demo/simple_text_demo.py -k "DP-4-b1"

# 命令行覆盖参数（优先级高于 parametrize 默认值）
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" \
    --batch_size 4 --max_generated_tokens 512 --max_seq_len 4096
```

### 2.3 常用 parametrize id 速查

| pytest id | batch_size | data_parallel | max_seq_len | 用途 |
|-----------|-----------|---------------|-------------|------|
| `batch-1` | 1 | 1 | 1024 | 单用户延迟基准 |
| `batch-32` | 32 | 4 | 1024 | 多用户吞吐基准 |
| `long-context-64k` | 1 | 1 | 131072 | 长上下文 |
| `DP-4-b1` | 1 | 4 | 1024 | DP=4 延迟 |
| `DP-4-b32` | 32 | 4 | 1024 | DP=4 吞吐（global=128） |
| `device-perf` | 1 | 1 | 1024 | 设备性能剖析（仅 prefill） |
| `ci-token-matching` | 1 | 1 | 1024 | token 精度验证（**关闭 trace**） |

完整 22 个 id 见 [`simple_text_demo.py:808-831`](simple_text_demo.py#L808-L831)。

---

## 3. 整体控制流

### 3.1 调用链（带行号）

```
test_demo_text(...)                     # simple_text_demo.py:869
│
├─ 参数覆盖（CLI > parametrize 默认值）  # L916-939
│   --batch_size, --max_seq_len, --enable_trace, ...
│
├─ prepare_generator_args(...)          # L265-340
│   ├─ create_submeshes(mesh, DP)       # 切分 DP submesh
│   └─ 循环每个 submesh:
│       └─ create_tt_model(submesh, ..., max_batch_size, optimizations, ...)
│           → 返回 (model_args, model, tt_kv_cache, state_dict)
│
├─ Generator(model, model_args, ...)    # L1067
│
├─ preprocess_inputs_prefill(...)       # L1093-1101
│   → input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens
│
├─ [PREFILL]
│   ├─ generator.prefill_forward_text(..., warmup_prefill=True)   # L1166 warmup
│   └─ generator.prefill_forward_text(..., warmup_prefill=True)   # L1180 实跑
│       → prefilled_token（每用户第一个生成 token）
│
├─ [DECODE 循环] while users_decoding:  # L1238
│   ├─ generator.decode_forward(out_tok, current_pos, ...)        # L1248
│   ├─ sample_host / device sampling → out_tok                    # L1261-1271
│   ├─ current_pos += 1                                           # L1287
│   └─ 检查 EoS / max_generated_tokens → 更新 users_decoding
│
└─ 输出 + 性能统计（TTFT、decode tok/s、verify_perf）            # L1320+
```

### 3.2 关键对象说明

| 对象 | 类型 | 含义 |
|------|------|------|
| `model_args` | `List[ModelArgs]` | 每个 DP 组一份；含 tokenizer、精度配置、mesh_device |
| `model` | `List[Transformer]` | 每个 DP 组一份权重实例（同一 state_dict） |
| `tt_kv_cache` | `List[List[Tensor]]` | 每个 DP 组的 KV cache，shape `[batch, heads, seq, head_dim]` |
| `page_table` | `Tensor` | shape `[global_batch, blocks_per_user]`，虚拟→物理块映射 |
| `generator` | `Generator` | 封装 prefill/decode 调用、trace 状态机（`trace_ids_decode` 等） |

---

## 4. Batching

### 4.1 三层 batch 概念

```
global_batch_size = batch_size × data_parallel
                    ↑每 DP 组用户数  ↑DP 副本数

示例（batch-32 with DP-4）：
  batch_size     = 32   # 每个 DP 组处理 32 个用户
  data_parallel  = 4    # 4 个 DP 副本（4 个 submesh）
  global_batch_size = 128
```

源码：[`simple_text_demo.py:944`](simple_text_demo.py#L944)

```python
global_batch_size = batch_size * data_parallel  # L944
```

每个 submesh 持有**独立的模型副本**，但共享同一份 `state_dict`（第一个 submesh 加载，后续复用；[`simple_text_demo.py:310-326`](simple_text_demo.py#L310-L326)）。

### 4.2 Prefill vs Decode 的 batch 语义差异

这是理解 trace 结构的关键：两个阶段对 batch 的处理方式完全不同。

#### Prefill：面向序列长度，逐用户（或同长批量）

- `preprocess_inputs_prefill()` 对每个 prompt 独立 tokenize，将各自 token 填入 `[1, max_prompt_len]` 的 pad 张量（[`common.py`](../tt/common.py)），记录 `decoding_pos[i]`（即该用户 prompt 的实际 token 数）。
- 若**所有用户的 padded seq_len 相同且 `data_parallel==1`**，可走**批量 prefill**路径（一次 forward 处理整个 batch）；否则逐用户串行 prefill（[`generator.py`](../tt/generator.py) `use_batched_prefill` 判断）。
- prefill 输入 shape：`[batch_size, max_prompt_len]`（批量）或 `[1, seq_len]`（单用户）。
- prefill 结束后每用户得到**第一个生成 token**（`prefilled_token`）。

#### Decode：面向 batch 宽度，每步生成 1 token

- 每个 decode step，全部 `global_batch_size` 个用户**同时**做一次 forward，每用户生成 1 个 token。
- 输入：`out_tok` shape `[global_batch_size, 1]`（上一步生成的 token），`current_pos` shape `[global_batch_size]`（各用户当前写入 KV cache 的位置）。
- `current_pos` 初值 = `decoding_pos`（用户 prompt 长度，[`L1216`](simple_text_demo.py#L1216)），每步 `+1`（[`L1287`](simple_text_demo.py#L1287)）。
- decode 输入 shape 固定（batch 固定、seq_len=1），这正是 decode 适合 trace 的根本原因。

```
                  ┌── Prefill ──────────────────────────────────────────┐
  User 0: [tok_0, tok_1, ..., tok_127] → prefilled_token_0
  User 1: [tok_0, tok_1, ..., tok_95]  → prefilled_token_1
  User 2: [tok_0, tok_1, ..., tok_63]  → prefilled_token_2
  ...（各用户序列长度可不同，逐用户处理）
                  └─────────────────────────────────────────────────────┘

                  ┌── Decode（每步）────────────────────────────────────┐
  step 0: [prev_tok_0, prev_tok_1, ..., prev_tok_{B-1}]  →  [new_tok_0, ..., new_tok_{B-1}]
  step 1: 同上（B 个用户并行，shape 固定）
  step N: 同上（直到 EoS 或 max_generated_tokens）
                  └─────────────────────────────────────────────────────┘
```

### 4.3 Data Parallel：submesh 切分与 batch 分发

`create_submeshes()` 把整个 mesh 按行切分为 `data_parallel` 个子 mesh，每个子 mesh 运行一份独立的模型副本（[`generator.py:2626-2643`](../tt/generator.py#L2626-L2643)）：

```
T3K (1×8 devices), data_parallel=4:
  submesh[0] = devices [0,1]
  submesh[1] = devices [2,3]
  submesh[2] = devices [4,5]
  submesh[3] = devices [6,7]
  每个 submesh 内 Tensor Parallel 处理 batch_size=32 个用户
```

`decode_forward()` 用 `torch.chunk` 把 token/pos/page_table 按 DP 维度切分后分发（[`generator.py:1194-1198`](../tt/generator.py#L1194-L1198)）：

```python
tokens     = torch.chunk(tokens,     self.data_parallel, 0)
current_pos = torch.chunk(current_pos, self.data_parallel, 0)
page_table  = torch.chunk(page_table,  self.data_parallel, 0)
```

### 4.4 Paged Attention：KV cache 的虚拟/物理块映射

**为什么需要 paged attention？** vLLM 等推理框架要求将 KV cache 组织为固定大小的内存页，以支持动态 context 长度和跨请求共享。

**page table 的构造**（[`simple_text_demo.py:197-208`](simple_text_demo.py#L197-L208)）：

```python
permutation = torch.randperm(max_num_blocks)          # 打乱物理块顺序（模拟随机分配）
reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
page_table = reverse_permutation.reshape(
    global_batch_size,
    max_num_blocks // (global_batch_size // data_parallel)  # 每用户分配块数
)
# page_table[user_id, virtual_block_idx] = physical_block_idx
```

**关键配置参数**：

| 参数 | 含义 | 示例值 |
|------|------|--------|
| `page_block_size` | 每块存 N 个 token 的 KV | 32（短 ctx）/ 64（长 ctx） |
| `page_max_num_blocks_per_dp` | 每个 DP 组的总物理块数 | 1024 / 2048 |
| 每用户可用 seq_len | `block_size × max_blocks / batch_size` | 32×1024/32 = 1024 |

**repeat-batch 间清零 KV cache**（[`simple_text_demo.py:1118-1125`](simple_text_demo.py#L1118-L1125)）：

```python
if batch_idx != 0:
    for layer in model[i].layers:
        k_cache, v_cache = layer.attention.layer_past
        k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)  # 就地清零
        v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)
    generator.prev_page_table = None
```

这防止多轮 repeat-batch 之间的 KV 内容串话。

### 4.5 采样：device vs host

- **Device sampling**（on-chip）：模型支持时（`model._supports_on_device_sampling`），top-k/top-p 在 Tenstorrent 芯片上完成，返回已采样的 token id，省去 host 传输 logits 的开销。
- **Host sampling**：`sample_host()` 把 logits 从 device 拉回 host 做采样。

`batch-32` 配置使用**逐用户不同的采样参数**（[`L402-409`](simple_text_demo.py#L402-L409)）：

```python
"temperature": torch.linspace(0.0, 1.0, steps=32).tolist(),  # 每用户不同
"top_p":       torch.linspace(0.08, 1.0, steps=32).tolist(),
"top_k":       torch.arange(1, 33).tolist(),
```

这要求 device sampling 模块能处理 per-batch-entry 的采样参数。

---

## 5. Tracing

### 5.1 为什么需要 Trace

不使用 trace 时，每个 decode step 的执行路径是：

```
host: 准备输入 tensor → dispatch op_0 → dispatch op_1 → ... → dispatch op_N → 等待输出
```

对于 Transformer decode（batch=1，seq=1，layer 数十层），每层包含 QKV proj、attention、MLP 等数十个算子。逐 op 的 host-to-device dispatch 延迟（内核启动、参数传输）会占据 decode step 总时间的相当比例。

**Tracing 的工作原理**（类似 CUDA Graph）：

1. **Capture**：在一次真实 forward 过程中，ttnn 将所有 device op（含 op 类型、参数、输入输出 buffer 地址）录制为一个 `trace_id`。
2. **Replay**：后续每步只调用 `execute_trace(trace_id)`，device 端直接重放整条 op 序列，host 侧仅更新输入 buffer 的数值，**不再逐 op dispatch**。

**设备前提**：`device_params` 必须包含 `trace_region_size`（在 device DRAM 中预留 trace 录制区域）。demo 固定传入此参数（[`simple_text_demo.py:846-850`](simple_text_demo.py#L846-L850)）：

```python
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": _trace_region_size, "num_command_queues": 1}],
    indirect=True,
)
```

`_trace_region_size` 默认 50 MB，部分大模型（Qwen2.5-72B/32B + Blackhole）扩展到 100 MB（[`L364-368`](simple_text_demo.py#L364-L368)）。

### 5.2 三段式生命周期

Decode trace 的完整生命周期由两个方法实现：

```
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 1：Compile / Warmup                                            │
│  _capture_decode_trace_text() → _decode_forward_no_trace_text()     │
│  作用：触发 ttnn kernel 编译，填充 op metadata 缓存                   │
│  对应日志："Done Compiling Model"                                     │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 2：Capture                                                     │
│  ttnn.begin_trace_capture(mesh_device, cq_id=0)                     │
│    ttnn_decode_forward(*device_inputs, ...)   ← 录制这一次 forward   │
│  ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)             │
│  作用：把 op 序列 + buffer 地址固化为 trace_id                        │
│  对应日志："Done Capturing Decode Trace"                              │
├─────────────────────────────────────────────────────────────────────┤
│  阶段 3：Replay（每个 decode step 执行一次）                          │
│  copy_host_to_device(host_inputs, device_tensors=<persistent>)      │
│    → ttnn.copy_host_to_device_tensor(host, device)  ← 就地更新值    │
│  ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False) │
│  作用：仅更新输入 buffer，重放 op 序列，不重新 dispatch 任何算子      │
└─────────────────────────────────────────────────────────────────────┘
```

源码对应关系：

| 阶段 | 函数 | 行号 |
|------|------|------|
| Compile | `_decode_forward_no_trace_text()` | [`generator.py:1333`](../tt/generator.py#L1333) |
| Capture | `_capture_decode_trace_text()` | [`generator.py:1320-1380`](../tt/generator.py#L1320-L1380) |
| Replay | `_decode_forward_trace_text()` | [`generator.py:1382-1438`](../tt/generator.py#L1382-L1438) |

`_decode_forward_trace_text()` 的首次调用触发 capture（通过 `if not self.trace_ids_decode[sampling_on_device]` 判断），后续调用直接进入 replay。

### 5.3 Persistent 输入张量与就地更新

这是 trace 机制的核心约束：**trace 固化的是 buffer 地址，不是数据**。重放时只能修改 buffer 内容，不能改变 tensor 的 shape 或分配新 buffer。

`copy_host_to_device()` 实现了两种模式（[`common.py:544-571`](../tt/common.py#L544-L571)）：

```python
def copy_host_to_device(host_tensors, device_tensors=None, mesh_device=None):
    if device_tensors is None:
        # 首次：分配新 device buffer，返回持久化 tensor 列表
        ret = [ttnn.to_device(t, device=mesh_device) for t in host_tensors]
        return ret
    else:
        # 重放：就地写入已有 device buffer（地址不变）
        for i in range(len(host_tensors)):
            ttnn.copy_host_to_device_tensor(host_tensors[i], device_tensors[i])
        return device_tensors  # 返回同一个列表（地址未变）
```

在 capture 阶段，`device_inputs` 被创建并持有：

```python
# generator.py:1353
device_inputs_i = copy_host_to_device(host_inputs, mesh_device=...)  # 首次分配
```

在每次 replay 前，只更新 buffer 内容：

```python
# generator.py:1416-1419
copy_host_to_device(
    host_tensors=host_inputs_i,
    device_tensors=self.trace_inputs_decode[sampling_on_device][i],  # 同一个持久化列表
)
```

### 5.4 何时必须刷新/重置 trace 输入

并非每步 decode 都需要完整刷新所有输入 tensor。`_decode_forward_trace_text()` 通过
`reset_inputs` 标志控制（[`generator.py:1399-1419`](../tt/generator.py#L1399-L1419)）：

| 触发条件 | 说明 |
|----------|------|
| `reset_batch=True`（首个 decode step）| prefill 刚结束，进入 decode 模式，token/pos 全新 |
| `sampling_on_device` 发生切换 | trace 结构不同（含/不含 on-chip 采样算子），需重新对齐 |
| `page_table` 发生变化 | 物理块映射更新（如新分配块或 batch shuffle），需刷新 page table buffer |

```python
# generator.py:1402-1409
reset_inputs = reset_batch or not sampling_on_device or sampling_mode_changed
if self.prev_page_table is None or any(
    not torch.equal(prev, curr) for prev, curr in zip(self.prev_page_table, page_table)
):
    reset_inputs = True
    self.prev_page_table = tuple(pt.clone() for pt in page_table)
```

注意：`not sampling_on_device`（host 采样模式）时**每步都刷新**，因为 host sampling 路径不在 trace 内，host 需要每步显式准备正确的 token 输入。

### 5.5 Trace 缓存键

Generator 以字典形式缓存多份 trace，避免重复 capture：

**Decode trace**（[`generator.py:96-98`](../tt/generator.py#L96-L98)）：

```python
self.trace_ids_decode   = defaultdict(lambda: None)  # key: sampling_on_device (bool)
self.trace_inputs_decode = defaultdict(lambda: None)
self.trace_output_decode = defaultdict(lambda: None)
```

缓存键是 `sampling_on_device`（`True`/`False`），即 device sampling 与 host sampling 各存一份 trace（因为两种模式下 trace 内的 op 序列不同）。

**Prefill trace**（`_easy_trace_prefill()`）：

缓存键为 `f"{prefill_seq_len}_{model_id}_{batch_size}_{use_start_pos}"`。由于 prefill 输入 shape 取决于序列长度，不同 padded seq_len 会 capture 不同的 trace，这是 prefill trace 比 decode trace 复杂得多的原因（详见 [FAQ 7.1](#71-trace-与动态-shape)）。

### 5.6 Demo 视角：trace 在哪一步发生

**Decode iteration=0（`compile_decode` 计时段）**：

```python
# simple_text_demo.py:1239-1240
if iteration == 0:
    profiler.start(f"compile_decode", iteration=batch_idx)
```

这一拍调用 `decode_forward(reset_batch=True)` 时，`trace_ids_decode` 为空，因此触发：
1. `_decode_forward_no_trace_text()`（compile warmup）
2. `begin_trace_capture` → `ttnn_decode_forward` → `end_trace_capture`（capture）

因此 iteration=0 的耗时**包含编译 + capture 成本**，不能用于性能基准。

**Decode iteration≥1（`inference_decode_time_i` 计时段）**：

纯 `execute_trace` replay，是真实的 decode 延迟。

**Prefill warmup（两次调用）**：

[`L1166`](simple_text_demo.py#L1166) 和 [`L1180`](simple_text_demo.py#L1180) 分别标注 `compile_prefill` 和 `inference_prefill`。第一次（warmup）触发 prefill kernel 编译和 trace capture；第二次是真实推理（replay）。`is_device_perf_test` 为 True 时跳过第一次 warmup（专门用于设备性能剖析的 `device-perf` 配置）。

**反例：`ci-token-matching` 关闭 trace**（[`L727`](simple_text_demo.py#L727)）：

```python
True,   # ci_only
False,  # enable_trace -> Teacher forcing does not work if it is on
```

Token accuracy 测试使用 teacher forcing：在 decode 循环中，每步用参考 token 替换模型预测 token（[`L1244-1245`](simple_text_demo.py#L1244-L1245)）。但 trace 在 capture 时固化了 token 输入 buffer 的操作流，无法支持从 host 动态注入替换 token 的模式，因此必须关闭 trace。

---

## 6. 端到端时间线：Llama-3.1-8B batch-1 一次完整运行

**配置**：`HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`，`MESH_DEVICE=N150`，`-k "performance and batch-1"`

```
batch_size=1, data_parallel=1, global_batch_size=1
max_seq_len=1024, max_generated_tokens=200
paged_attention=True, block_size=32, max_num_blocks=1024
enable_trace=True, optimizations=DecodersPrecision.performance
dtype=bfloat8_b
```

```
T=0  pytest 启动
     │
     ├─ create_tt_model(N150 submesh, max_batch=1, optimizations=performance)
     │   - ModelArgs 从 HF config 加载模型结构（n_layers=32, dim=4096, ...）
     │   - 权重从 HF 下载/缓存，量化为 bfloat8_b，写入 tensor cache
     │   - Transformer 实例化，weight 上传到 N150 DRAM
     │   - tt_kv_cache 分配：32层 × [1, n_heads, 1024, head_dim]
     │   - page_table: shape [1, 1024]（1 用户，1024 个虚拟块）
     │
     ├─ Generator 实例化，初始化 trace 状态字典（均为 None）
     │
T=1  ├─ [profiler: loading_inputs]
     │   preprocess_inputs_prefill(["<prompt>"], tokenizer, ...)
     │   - tokenize prompt → 约 128 tokens
     │   - pad 到 128 → input_tokens_prefill_pt: [1, 128]
     │   - decoding_pos = [128]
     │
T=2  ├─ [profiler: compile_prefill]
     │   generator.prefill_forward_text(..., warmup_prefill=True, enable_trace=True)
     │   ▶ 内部：检查是否有 prefill trace 缓存（key="128_0_1_0"）→ 无
     │     ① _decode_forward_no_trace (prefill compile warmup)
     │     ② begin_trace_capture(N150, cq_id=0)
     │        ttnn_prefill_forward([1, 128] tokens)   ← 写入 KV cache pos 0..127
     │     ③ end_trace_capture → trace_id_prefill["128_0_1_0"] = trace_id
     │   ← warmup 调用返回（不使用此次输出）
     │
T=3  ├─ [profiler: inference_prefill]
     │   generator.prefill_forward_text(..., warmup_prefill=True, enable_trace=True)
     │   ▶ 内部：trace 已缓存 → 直接 replay
     │     copy_host_to_device(host_inputs, device_tensors=<persistent>)
     │     execute_trace(N150, trace_id_prefill[...], blocking=False)
     │   ← 返回 prefilled_token（第一个生成 token，如 "The"）
     │   TTFT = T3结束 - T=0（首 token 延迟）
     │
T=4  ├─ [DECODE 循环开始] current_pos = tensor([128])
     │
     ├─ [iteration=0, profiler: compile_decode]
     │   generator.decode_forward(out_tok=[prefilled_token], current_pos=[128],
     │                            enable_trace=True, reset_batch=True)
     │   ▶ _decode_forward_trace_text → trace_ids_decode[True] is None → capture!
     │     ① _decode_forward_no_trace_text(tokens=[1,1], pos=[128])  ← compile
     │     ② begin_trace_capture(N150, cq_id=0)
     │        ttnn_decode_forward(*device_inputs)   ← 录制 decode graph
     │     ③ end_trace_capture → trace_ids_decode[True] = {0: trace_id}
     │   current_pos → tensor([129])
     │
     ├─ [iteration=1, profiler: inference_decode_time_1]
     │   generator.decode_forward(out_tok=[tok_1], current_pos=[129], reset_batch=False)
     │   ▶ trace_ids_decode[True] 已有 → replay only
     │     copy_host_to_device(host_inputs=[tok_1, pos=129, page_table],
     │                         device_tensors=<persistent>)  ← 就地更新 3 个 buffer
     │     execute_trace(N150, trace_id, blocking=False)  ← 纯 replay
     │   current_pos → tensor([130])
     │
     ├─ [iteration=2..199]  重复上述 replay 模式
     │   每步：1 次 copy_host_to_device + 1 次 execute_trace
     │
T=5  └─ [profiler: inference_decode 结束]
         输出文本 + 性能统计：
         - TTFT: compile_prefill + inference_prefill
         - decode tok/s/user: 1 / mean(inference_decode_time_i)  (i≥1)
         - 总吞吐: decode tok/s/user × global_batch_size
```

---

## 7. 编译器工程师 FAQ

### 7.1 trace 与动态 shape

**Q：为什么 prefill 要建多份 trace，而 decode 只需一份？**

trace 在 capture 时固化了所有 tensor 的 shape 信息（包括用于 SIMD tile 计算的 padding）。decode 每步的输入 shape 固定（`[batch, 1]`），因此一份 trace 可覆盖所有 step。但 prefill 的输入 shape 取决于 prompt 长度（如 128、256、512...），不同长度需要 capture 不同的 trace。

这也是 demo 中 `_easy_trace_prefill()` 用 `(seq_len, batch_size, model_id, start_pos)` 作为 cache key 的原因——每个唯一 shape 组合对应一份独立 trace。

### 7.2 page_table 变化触发输入重置

**Q：什么情况下 page_table 会在 decode 过程中变化？**

当前 demo 中 page_table 在 batch 开始时创建，decode 期间不变（故通常只在 `reset_batch=True` 时触发一次完整刷新）。但在 vLLM 集成等生产场景中，新请求加入或页面重新分配会导致 page_table 变化，此时 `_decode_forward_trace_text()` 的 `prev_page_table` 比较逻辑（[`generator.py:1403-1409`](../tt/generator.py#L1403-L1409)）会检测到变化并刷新对应 buffer。

### 7.3 device sampling 改变 trace 结构

**Q：为什么 `sampling_on_device=True` 和 `False` 各有一份 trace？**

device sampling（on-chip top-k/top-p）将采样 op 融合进 decode graph，`capture_sampling_trace` 参数会在 trace 中附加 sampling op 序列（[`generator.py:1371-1374`](../tt/generator.py#L1371-L1374)）：

```python
# generator.py:1376-1377
if split_enabled:
    sampling_module.capture_trace(logits=tt_out_trace[i], ...)
```

host sampling 则不包含这些 op，两种 trace 结构不同，不能混用。

### 7.4 teacher forcing / token accuracy 必须关 trace

**Q：为什么 `ci-token-matching` 测试关闭 trace？**

Teacher forcing 在 decode 循环中每步用**参考 token** 替换模型预测 token（[`L1244-1245`](simple_text_demo.py#L1244-L1245)）。替换操作发生在 host 侧，trace 内的 op 序列只读取 device buffer（在 replay 时 host 无法动态插入替换逻辑）。因此 teacher forcing 与 trace 机制从根本上不兼容。

### 7.5 enable_split_sampling 与 trace 边界

`Generator` 默认开启 `enable_split_sampling=True`（[`generator.py:102`](../tt/generator.py#L102)），将 decode trace 分为两段：
1. 主 trace：从 token embedding 到 logits（或一直到 pre-softmax）
2. sampling trace：norm + lm_head + top-k/top-p

这允许在不重新 capture 主 trace 的情况下，单独调整采样参数（temperature 等在 device 侧参数化）。

### 7.6 vision/多模态与 vLLM

当前 demo（`simple_text_demo.py`）仅覆盖纯文本路径。vision 模型（Llama-3.2-11B/90B、Mistral-3.1-24B）走 `prefill_forward_vision()` / `decode_forward_vision()` 路径，cross-attention 处理方式不同但 trace 机制相同。vLLM 集成在 `models/tt_transformers/tt/generator.py` 的子类中实现，复用相同的 trace capture/replay 基础设施。

---

## 8. 源码索引表

| 概念 | 文件 | 行号 |
|------|------|------|
| **整体控制流** | | |
| `test_demo_text` 入口 | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L869](simple_text_demo.py#L869) |
| parametrize 配置列表（22 个 id）| [`demo/simple_text_demo.py`](simple_text_demo.py) | [L371-L832](simple_text_demo.py#L371-L832) |
| `trace_region_size` 配置 | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L364-368](simple_text_demo.py#L364-L368), [L846-850](simple_text_demo.py#L846-L850) |
| CLI 参数覆盖逻辑 | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L916-939](simple_text_demo.py#L916-L939) |
| `prepare_generator_args()` | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L265-340](simple_text_demo.py#L265-L340) |
| `Generator.__init__` + trace 状态 | [`tt/generator.py`](../tt/generator.py) | [L85-104](../tt/generator.py#L85-L104) |
| **Prefill** | | |
| `preprocess_inputs_prefill()` | [`tt/common.py`](../tt/common.py) | 函数入口 |
| prefill warmup + 实跑（两次调用）| [`demo/simple_text_demo.py`](simple_text_demo.py) | [L1163-1195](simple_text_demo.py#L1163-L1195) |
| `prefill_forward_text()` | [`tt/generator.py`](../tt/generator.py) | [L525](../tt/generator.py#L525) |
| prefill trace capture | [`tt/generator.py`](../tt/generator.py) | `_capture_trace_prefill()` |
| **Decode 循环** | | |
| decode while 循环 | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L1238-1318](simple_text_demo.py#L1238-L1318) |
| `decode_forward()` 调用 | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L1248](simple_text_demo.py#L1248) |
| `current_pos += 1` | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L1287](simple_text_demo.py#L1287) |
| `decode_forward()` 实现 | [`tt/generator.py`](../tt/generator.py) | [L1166](../tt/generator.py#L1166) |
| **Tracing 核心** | | |
| decode trace capture | [`tt/generator.py`](../tt/generator.py) | [L1320-1380](../tt/generator.py#L1320-L1380) |
| decode trace replay | [`tt/generator.py`](../tt/generator.py) | [L1382-1438](../tt/generator.py#L1382-L1438) |
| reset_inputs 判断逻辑 | [`tt/generator.py`](../tt/generator.py) | [L1399-1419](../tt/generator.py#L1399-L1419) |
| trace 缓存字典 | [`tt/generator.py`](../tt/generator.py) | [L96-98](../tt/generator.py#L96-L98) |
| `enable_split_sampling` | [`tt/generator.py`](../tt/generator.py) | [L102](../tt/generator.py#L102) |
| **Persistent 张量更新** | | |
| `copy_host_to_device()` | [`tt/common.py`](../tt/common.py) | [L544-571](../tt/common.py#L544-L571) |
| `ttnn.copy_host_to_device_tensor` | [`tt/common.py`](../tt/common.py) | [L569](../tt/common.py#L569) |
| **Batching** | | |
| `global_batch_size` 公式 | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L944](simple_text_demo.py#L944) |
| `current_pos` 初始化 | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L1216](simple_text_demo.py#L1216) |
| `torch.chunk` DP 分发 | [`tt/generator.py`](../tt/generator.py) | [L1194-1198](../tt/generator.py#L1194-L1198) |
| `create_submeshes()` | [`tt/generator.py`](../tt/generator.py) | [L2626-2643](../tt/generator.py#L2626-L2643) |
| **Paged Attention** | | |
| `create_tt_page_table()` | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L197-208](simple_text_demo.py#L197-L208) |
| `PagedAttentionConfig` | [`tt/common.py`](../tt/common.py) | `class PagedAttentionConfig` |
| KV cache 清零（repeat-batch）| [`demo/simple_text_demo.py`](simple_text_demo.py) | [L1118-1125](simple_text_demo.py#L1118-L1125) |
| **采样** | | |
| batch-32 逐用户采样参数 | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L402-409](simple_text_demo.py#L402-L409) |
| device sampling 判断 | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L1129-1148](simple_text_demo.py#L1129-L1148) |
| **其他** | | |
| `ci-token-matching` 关闭 trace | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L727](simple_text_demo.py#L727) |
| `DecodersPrecision.performance/accuracy` | [`tt/model_config.py`](../tt/model_config.py) | `DecodersPrecision` 类 |
| 权重 dtype `bfloat8_b` | [`demo/simple_text_demo.py`](simple_text_demo.py) | [L318](simple_text_demo.py#L318) |
