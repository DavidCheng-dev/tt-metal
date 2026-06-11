# tt-metal Paged Attention Tracing 技术解析

## 1. 背景：Trace 机制与 Paged Attention 的根本矛盾

### 1.1 Trace 机制（CUDA Graph 的类比）

tt-metal 的 Trace（类似 CUDA Graph）解决了 LLM decode 阶段的**主机端分发瓶颈**问题。每个 decode step 有数十~数百个微小算子，如果逐一由主机下发到设备命令队列，延迟会大幅超过设备实际计算时间。

Trace 工作方式：

```
阶段 1 Compile：  触发 kernel 编译，预热 op metadata 缓存
阶段 2 Capture：  begin_trace_capture → 真实执行一次 forward（记录所有 op + 设备内存地址）→ end_trace_capture
阶段 3 Replay：   每步仅调用 execute_trace，设备端自行重放整条 op 链
```

**核心约束**：Trace 将**设备端物理缓冲区地址（Device Physical Address）**硬编码进捕获的指令图。Replay 时读写的永远是 Capture 时绑定的那块内存，无法容纳地址变化。

### 1.2 Paged Attention 的动态性

Paged Attention（vLLM 等框架的标配）将 KV Cache 按固定大小的 Block 管理，用 Page Table 记录"虚拟块→物理块"的映射：

- 每次新 token 生成时，调度器可能分配新物理块 → Page Table 追加新 Block ID
- 请求调度/抢占/置换 → Page Table 整行重排
- 混合注意力模型（如 Gemma3）每层独立 Page Table，形状各异

**问题**：如果在 Trace 内部用 `ttnn.from_torch(page_table)` 动态创建张量，每次调用都会分配新的设备缓冲区，地址发生变化，与 Trace 的静态地址约束直接冲突。

---

## 2. 核心设计：持久化缓冲区 + 原位覆写

tt-metal 的解决方案是：**在 Trace 外部预先分配 Page Table 的设备缓冲区（固定物理地址），每次推理只原位覆写其数据内容，地址永远不变**。

```
Host (Python)                     Device DRAM (固定地址 A)           Trace Graph
    │                                      │                              │
    │── Warmup Compile ──────────────────> │ 分配 Page Table 缓冲区        │
    │── begin_trace_capture ──────────────>│                              │ 开始录制
    │   ttnn_forward(page_table@addr_A) ──>│ <── 录制"读取 addr_A"指令 ───>│
    │── end_trace_capture ────────────────>│                              │ 录制完成
    │                                      │                              │
    │  [每次 decode step]                  │                              │
    │── copy_host_to_device_tensor ───────>│ 覆写 addr_A 的内容（新Block IDs）
    │── execute_trace ──────────────────── │ ────────────────────────────>│ 重放
    │                                      │ <── 读取 addr_A（已更新内容）─│
```

关键点：
1. **静态地址，动态内容**：Page Table 的设备地址在整个推理生命周期中不变
2. **Trace 外覆写**：`copy_host_to_device_tensor` 操作发生在 Trace 指令流之外，不被录制
3. **无需重新 Capture**：无论 Block ID 如何更新，都不需要重新录制 Trace

---

## 3. 分层代码解析

### 3.1 KV Cache 结构与 Page Table 定义

**配置入口** [models/tt_transformers/tt/common.py:74-78](../models/tt_transformers/tt/common.py#L74-L78)：

```python
class PagedAttentionConfig:
    def __init__(self, block_size=32, max_num_blocks=1024):
        self.block_size = block_size      # 每个物理 Block 存 N 个 token 的 KV
        self.max_num_blocks = max_num_blocks  # 全局物理 Block 池大小
```

**Page Table 构造**（示例来自 demo）[models/tt_transformers/demo/simple_text_demo.py:197-208](../models/tt_transformers/demo/simple_text_demo.py#L197-L208)：

```python
permutation = torch.randperm(max_num_blocks)          # 模拟随机物理块分配
page_table = reverse_permutation.reshape(
    global_batch_size,
    max_num_blocks // (global_batch_size // data_parallel)
)
# page_table[user_id, virtual_block_idx] = physical_block_idx
# shape: [global_batch_size, blocks_per_user]
```

**KV Cache 物理布局**：
- Paged 模式：`[num_total_blocks, n_kv_heads, block_size, head_dim]`（按物理 Block 索引）
- 非 Paged 模式：`[batch, n_kv_heads, max_seq_len, head_dim]`（连续布局）

---

### 3.2 Decode 阶段的 Paged Attention 核算子

**文件** [models/tt_transformers/tt/attention.py:687-717](../models/tt_transformers/tt/attention.py#L687-L717)

**写入 KV Cache（两种路径）**：

```python
# 路径 1：Q+K 融合更新（use_qk_fused=True 时）
if self.use_qk_fused:
    ttnn.experimental.paged_fused_update_cache(
        keys, k_heads_1BKD, values, v_heads_1BKD,
        update_idxs_tensor=current_pos,   # 当前各用户写入位置
        page_table=page_table             # 虚拟→物理块映射
    )
else:
    # 路径 2：K 和 V 分别更新
    ttnn.experimental.paged_update_cache(
        keys, k_heads_1BKD,
        update_idxs_tensor=current_pos,
        page_table=page_table
    )
    ttnn.experimental.paged_update_cache(
        values, v_heads_1BKD,
        update_idxs_tensor=current_pos,
        page_table=page_table
    )
```

**执行分页 SDPA 注意力计算**：

```python
if page_table is not None:
    attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_heads_1BQD,
        keys,
        values,
        page_table_tensor=page_table,    # 指向持久化缓冲区（固定地址）
        cur_pos_tensor=current_pos,
        scale=self.scale,
        sliding_window_size=self.sliding_window,
        program_config=sdpa_decode_prog_cfg,
        compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
```

---

### 3.3 Prefill 阶段的 KV Cache 填充

**文件** [models/tt_transformers/tt/attention.py:1006-1062](../models/tt_transformers/tt/attention.py#L1006-L1062)

```python
block_size = keys_BKSD.shape[2]    # 从 cache tensor 形状推断 block 大小
fill_page_table = chunk_page_table if chunk_page_table is not None else page_table

if batch_size > 1:
    # 批量 Prefill：逐用户调用 paged_fill_cache
    # 原因：paged_fill_cache 内核每次只能处理一个 batch_idx
    for slot_idx in valid_slots:
        k_user = k_fill[slot_idx:slot_idx+1, :, :page_len, :]
        v_user = v_fill[slot_idx:slot_idx+1, :, :page_len, :]
        ttnn.experimental.paged_fill_cache(keys_BKSD, k_user_sliced,
                                           fill_page_table, batch_idx=slot_idx)
        ttnn.experimental.paged_fill_cache(values_BKSD, v_user_sliced,
                                           fill_page_table, batch_idx=slot_idx)
elif page_table is not None:
    # 单用户路径
    page_len = fill_page_table.shape[1] * block_size
    ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill_sliced,
                                       fill_page_table, batch_idx=user_id)
    ttnn.experimental.paged_fill_cache(values_BKSD, v_fill_sliced,
                                       fill_page_table, batch_idx=user_id)
```

---

### 3.4 Decode Trace 三阶段：Compile → Capture → Replay

**文件** [models/tt_transformers/tt/generator.py:1320-1438](../models/tt_transformers/tt/generator.py#L1320-L1438)

**阶段 1 & 2：Compile + Capture**（`_capture_decode_trace_text`，L1320）：

```python
# 阶段 1：Compile Warmup（触发 kernel 编译，不录制）
self._decode_forward_no_trace_text(tokens, current_pos, page_table=page_table, ...)

# 阶段 2：分配持久化输入缓冲区 + 录制 Trace
for i in range(self.data_parallel):
    device_inputs_i = copy_host_to_device(host_inputs, mesh_device=...)  # 分配静态缓冲区
    device_inputs.append(device_inputs_i)                                 # 保存引用（地址固定）

for i in range(self.data_parallel):
    trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
    tt_out_trace.append(
        self.model[i].ttnn_decode_forward(*device_inputs[i], kv_cache=..., ...)
    )
    ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)
```

**阶段 3：Replay**（`_decode_forward_trace_text`，L1382）：

```python
# 检测 Page Table 是否变化（分配新块 or batch shuffle）
if self.prev_page_table is None or any(
    not torch.equal(prev, curr) for prev, curr in zip(self.prev_page_table, page_table)
):
    reset_inputs = True
    self.prev_page_table = tuple(pt.clone() for pt in page_table)

# 若需要刷新，在 Trace 外侧原位覆写持久化缓冲区
if reset_inputs:
    for i in range(self.data_parallel):
        host_inputs_i = self.model[i].prepare_decode_inputs_host(
            tokens[i], current_pos[i], page_table[i]
        )
        copy_host_to_device(                          # 调用 ttnn.copy_host_to_device_tensor
            host_tensors=host_inputs_i,
            device_tensors=self.trace_inputs_decode[sampling_on_device][i]  # 同一个持久化列表
        )

# 触发 Trace 回放（blocking=False 允许主机继续执行，异步提升吞吐）
for i, trace_id in self.trace_ids_decode[sampling_on_device].items():
    ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
```

---

### 3.5 原位覆写的底层实现

**文件** [models/tt_transformers/tt/common.py:544-570](../models/tt_transformers/tt/common.py#L544-L570)：

```python
def copy_host_to_device(host_tensors, device_tensors=None, mesh_device=None, shard_specs=None):
    if device_tensors is None:
        # 首次分配：创建新设备 tensor（地址确定）
        ret = []
        for i in range(len(host_tensors)):
            on_device = ttnn.to_device(host_tensors[i], device=mesh_device)
            ret.append(on_device)
        return ret
    else:
        # Replay 路径：原位覆写，地址不变
        for i in range(len(host_tensors)):
            ttnn.copy_host_to_device_tensor(host_tensors[i], device_tensors[i])
        return device_tensors
```

**关键**：`device_tensors` 参数非空时，调用 `ttnn.copy_host_to_device_tensor` 将数据写入已有 tensor 的原始物理地址，不触发任何新分配。

---

### 3.6 混合注意力（Hybrid Attention）的每层独立 Page Table

对于 Gemma3 等混合模型，每层注意力类型不同（全局/局部），Block 大小也不同，需要独立的 Page Table。

**持久化每层 Page Table 的延迟分配** [models/tt_transformers/tt/model.py:667-712](../models/tt_transformers/tt/model.py#L667-L712)：

```python
def _page_tables_to_ttnn(self, page_tables_per_layer):
    persistent = getattr(self, "_persistent_per_layer_page_tables", None)
    if persistent is None or len(persistent) != n:
        persistent = []
        for pt in page_tables_per_layer:
            # 首次（Warmup Compile）：在设备上分配静态缓冲区
            persistent.append(
                ttnn.from_torch(pt, device=self.mesh_device, dtype=ttnn.int32, ...)
            )
        self._persistent_per_layer_page_tables = persistent  # 缓存，后续不再重新分配
    return persistent
```

**每步推理前更新 Page Table 内容** [models/tt_transformers/tt/model.py:714-739](../models/tt_transformers/tt/model.py#L714-L739)：

```python
def update_persistent_per_layer_page_tables(self, page_tables_per_layer):
    persistent = getattr(self, "_persistent_per_layer_page_tables", None)
    for i, pt in enumerate(page_tables_per_layer):
        host_pt = ttnn.from_torch(pt, device=None, dtype=ttnn.int32, ...)  # 仅主机端打包
        ttnn.copy_host_to_device_tensor(host_pt, persistent[i])            # 原位覆写设备端
```

**在 vLLM 桥接层触发更新**（必须在 `execute_trace` 之前）[models/tt_transformers/tt/generator_vllm.py](../models/tt_transformers/tt/generator_vllm.py)：

```python
def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
    # 步骤 1：把最新的 Block IDs 写入持久化缓冲区（Trace 执行前）
    for m, pt_for_submesh in zip(self.model, per_submesh):
        m.update_persistent_per_layer_page_tables(pt_for_submesh)
    # 步骤 2：路由到 Generator 执行 Trace Replay
    return super(HybridAttentionForCausalLM, self).decode_forward(*args, **kwargs)
```

---

### 3.7 Prefill Trace 的三阶段（与 Decode 的差异）

**文件** [models/tt_transformers/tt/generator.py:238-343](../models/tt_transformers/tt/generator.py#L238-L343)

Prefill 因序列长度可变，每种 padded seq_len 需要独立的 Trace：

```python
# Trace 缓存键：序列长度 + model_id + batch_size + start_pos 四元组
trace_key = f"{prefill_seq_len}_{model_id}_{batch_size}_{use_start_pos}"
```

Capture 流程（单用户路径，L300-343）：

```python
# Warmup（不录制）
device_inputs = copy_host_to_device(host_inputs, mesh_device=...)
tt_out_trace = self.model[model_id].ttnn_prefill_forward(...)
ttnn.synchronize_device(...)

# Capture
device_inputs = copy_host_to_device(host_inputs, mesh_device=...)   # 重新分配（新地址）
trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
tt_out_trace = self.model[model_id].ttnn_prefill_forward(
    x=transformed_inputs[0],
    page_table=transformed_inputs[1],      # 此时传入的是持久化 Page Table 引用
    chunk_page_table=transformed_inputs[2],
    ...
)
ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
```

Replay（`_prefill_forward_trace`，L489-522）：

```python
# 原位覆写输入缓冲区（含 Page Table）
device_inputs = copy_host_to_device(host_inputs, device_tensors=device_inputs, ...)
# 异步触发 Trace 回放
ttnn.execute_trace(self.model_args[model_id].mesh_device, trace_id, cq_id=0, blocking=False)
return tt_out_trace
```

---

### 3.8 Host Inputs 打包：Page Table 的设备端格式

**文件** [models/tt_transformers/tt/model.py:471-519](../models/tt_transformers/tt/model.py#L471-L519)

```python
def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
    ...
    if page_table is not None:
        page_table = ttnn.from_torch(
            page_table,
            device=None,                   # 仅打包为主机端 ttnn tensor，不上传到设备
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(   # Galaxy 多卡按 batch 维度切分
                self.mesh_device,
                dims=(None, -2) if (self.args.is_galaxy and B > 1) else (None, None),
                mesh_shape=self.args.cluster_shape,
            ),
        )
    return tokens, current_pos_tt, rope_idxs, page_table
```

返回的四元组（tokens, current_pos, rope_idxs, page_table）将被传入 `copy_host_to_device`，其中 page_table 会被原位覆写进持久化缓冲区。

---

## 4. 数据流全景图

```
[外部调度器 / vLLM / simple_text_demo]
    │
    │ 每次 decode step：更新 Block IDs (torch.Tensor)
    ▼
copy_host_to_device / update_persistent_per_layer_page_tables
    │
    │ ttnn.copy_host_to_device_tensor()    ← Trace 外部，原位覆写
    ▼
[Device DRAM - 固定物理地址 A]
    Page Table Buffer: [batch, blocks_per_user] = 最新 Block IDs
    ▲
    │ execute_trace(trace_id)
    │
[Trace Graph - 已录制的指令序列]
    │
    ├─ paged_update_cache(keys, k_heads, page_table@addr_A, cur_pos)
    │      → 写入 KV Cache 对应物理 Block
    │
    ├─ paged_update_cache(values, v_heads, page_table@addr_A, cur_pos)
    │      → 写入 KV Cache 对应物理 Block
    │
    └─ paged_scaled_dot_product_attention_decode(Q, K, V, page_table@addr_A)
           → 读取散布在 DRAM 各处的物理 Block，执行分页注意力计算
```

---

## 5. Trace 缓存键设计

| Trace 类型 | 缓存键 | 代码位置 |
|-----------|--------|----------|
| Decode | `sampling_on_device` (bool) | [generator.py:96-98](../models/tt_transformers/tt/generator.py#L96-L98) |
| Prefill | `f"{seq_len}_{model_id}_{batch_size}_{use_start_pos}"` | [generator.py:420](../models/tt_transformers/tt/generator.py#L420) |
| Prefill Sampling | `f"sampling_{seq_len}_{model_id}_{batch}_{dp}"` | [generator.py:862](../models/tt_transformers/tt/generator.py#L862) |

Decode Trace 只需两份（含/不含 on-device sampling），因为 batch size 和 seq_len=1 固定不变；Prefill Trace 则因 padded 序列长度可变而需要多份。

---

## 6. 何时触发 reset_inputs

**文件** [models/tt_transformers/tt/generator.py:1397-1410](../models/tt_transformers/tt/generator.py#L1397-L1410)

```python
reset_inputs = reset_batch or not sampling_on_device or sampling_mode_changed
# Page Table 变化检测
if self.prev_page_table is None or any(
    not torch.equal(prev, curr) for prev, curr in zip(self.prev_page_table, page_table)
):
    reset_inputs = True
    self.prev_page_table = tuple(pt.clone() for pt in page_table)
```

| 触发条件 | 说明 |
|---------|------|
| `reset_batch=True` | 从 prefill 切换进入 decode 的第一步，token/pos 全新 |
| `not sampling_on_device` | Host sampling 模式每步都刷新（因 sampling 不在 Trace 内） |
| `sampling_mode_changed` | on-device / host 采样模式切换，Trace 结构不同 |
| `page_table` 内容变化 | 分配新块或 batch shuffle 导致 Block IDs 更新 |

批间清零（防止多轮 batch 之间 KV 内容串话）[models/tt_transformers/demo/simple_text_demo.py:1118-1125](../models/tt_transformers/demo/simple_text_demo.py#L1118-L1125)：

```python
if batch_idx != 0:
    for layer in model[i].layers:
        k_cache, v_cache = layer.attention.layer_past
        ttnn.mul(k_cache, 0, output_tensor=k_cache)  # 就地清零，不改变地址
        ttnn.mul(v_cache, 0, output_tensor=v_cache)
    generator.prev_page_table = None     # 强制下次 decode 重置所有输入缓冲区
```

---

## 7. 关键文件索引

| 文件 | 核心职责 |
|------|---------|
| [models/tt_transformers/tt/common.py:74-78](../models/tt_transformers/tt/common.py#L74-L78) | `PagedAttentionConfig`：block_size、max_num_blocks 配置 |
| [models/tt_transformers/tt/common.py:544-571](../models/tt_transformers/tt/common.py#L544-L571) | `copy_host_to_device`：持久化缓冲区原位覆写实现 |
| [models/tt_transformers/tt/attention.py:687-717](../models/tt_transformers/tt/attention.py#L687-L717) | Decode 阶段：`paged_update_cache` + `paged_scaled_dot_product_attention_decode` |
| [models/tt_transformers/tt/attention.py:1006-1062](../models/tt_transformers/tt/attention.py#L1006-L1062) | Prefill 阶段：`paged_fill_cache`（单用户 + 批量路径） |
| [models/tt_transformers/tt/model.py:471-519](../models/tt_transformers/tt/model.py#L471-L519) | `prepare_decode_inputs_host`：主机端打包 Page Table 为 ttnn tensor |
| [models/tt_transformers/tt/model.py:667-712](../models/tt_transformers/tt/model.py#L667-L712) | `_page_tables_to_ttnn`：持久化每层 Page Table 延迟分配 |
| [models/tt_transformers/tt/model.py:714-739](../models/tt_transformers/tt/model.py#L714-L739) | `update_persistent_per_layer_page_tables`：每步推理前更新内容 |
| [models/tt_transformers/tt/generator.py:1320-1380](../models/tt_transformers/tt/generator.py#L1320-L1380) | `_capture_decode_trace_text`：Decode Trace Compile + Capture |
| [models/tt_transformers/tt/generator.py:1382-1438](../models/tt_transformers/tt/generator.py#L1382-L1438) | `_decode_forward_trace_text`：Decode Trace Replay + Page Table 变化检测 |
| [models/tt_transformers/tt/generator.py:238-343](../models/tt_transformers/tt/generator.py#L238-L343) | `_capture_trace_prefill`：Prefill Trace Compile + Capture |
| [models/tt_transformers/tt/generator.py:489-522](../models/tt_transformers/tt/generator.py#L489-L522) | `_prefill_forward_trace`：Prefill Trace Replay |
| [models/tt_transformers/tt/generator_vllm.py](../models/tt_transformers/tt/generator_vllm.py) | vLLM 桥接：Trace 执行前调用 `update_persistent_per_layer_page_tables` |

---

## 8. 设计总结

tt-metal 中 Paged Attention 与 Trace 的共存，依赖一个精巧的**地址稳定性契约**：

1. **Warmup Compile 阶段**：为 Page Table 分配设备端持久化缓冲区（物理地址固定）
2. **Capture 阶段**：将指向该固定地址的 tensor 引用传入 paged attention 算子，地址被硬编码进 Trace 图
3. **每步 Replay 前**：在 Trace 图外部，用 `ttnn.copy_host_to_device_tensor` 将最新的 Block IDs 写入同一物理地址
4. **Replay 阶段**：`execute_trace` 触发设备端重放，读取到的始终是刚刚写入的最新 Block IDs

这一设计使得动态的内存页调度与静态的硬件指令图能够无缝配合，是 tt-metal 在 LLM 推理路径上实现高吞吐的关键工程设计之一。
