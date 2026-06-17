# PR #45041 分析: MoE 可配置的 e2e decode 模块

## 基本信息

| 项目 | 详情 |
|------|------|
| **标题** | Feature: MoE: (towards) a configurable e2e decode module |
| **作者** | amorrisonTT |
| **分支** | `amorrison/tt-moe-decode` |
| **状态** | ✅ **已合并** (2026-06-04) |
| **改动规模** | +2,777 / -48, 30 个文件 |
| **标签** | `feature`, `Mixture of Experts (MoE)` |

---

## 核心目标

> [!IMPORTANT]
> **这个 PR 并不是专门为了支持 DeepSeek V4 运行而提交的**。它的目标是构建一个**通用的、可配置的 MoE decode 模块** (`TTMoEDecode`)，可以通过 YAML 配置文件支持多种 MoE 模型。DeepSeek V4 只是其中的一个配置目标。

该 PR 做了三件事：
1. **封装 MoE decode 流程**：将 gate 之后的整个 MoE decode 流水线封装为一个独立的 Python 模块 `TTMoEDecode`
2. **YAML 配置化**：为所有已支持（及即将支持）的 MoE 模型创建 YAML 配置文件
3. **模块测试**：创建了覆盖所有配置的测试，可以灵活测试完整模型或单列

---

## 涵盖的模型配置

该 PR 新增了 **16 个模型的 YAML 配置**：

| 模型 | mesh_shape | hidden_size | experts | 状态 |
|------|-----------|-------------|---------|------|
| DeepSeek V3 | 16x8 | 7168 | 256 routed + 1 shared | ✅ 16x1 PASSED |
| **DeepSeek V4 Flash** | 16x8 | 4096 | 256 routed + 1 shared | ❌ 16x1 FAILED |
| **DeepSeek V4 Pro** | 16x8 | 7168 | 384 routed + 1 shared | ⚠️ SKIPPED (known failure) |
| DeepSeek OCR | 8x4 | 1280 | 64 routed + 2 shared | ⚠️ SKIPPED (known failure) |
| GLM-5 | 16x8 | 6144 | 256 routed + 1 shared | ✅ 16x1 PASSED |
| GLM-4.7 | 8x4 | 5120 | 160 routed + 1 shared | ✅ 8x4 PASSED |
| Kimi K2.5 | 16x8 | 7168 | 384 routed + 1 shared | ✅ 16x1 PASSED |
| Qwen3 235B | 8x4 | 4096 | 128 routed | ✅ 8x4 PASSED |
| Qwen3.5 35B | 16x4 | 2048 | 256 routed + 1 shared | ✅ 16x1 PASSED |
| Qwen3.5 397B | 16x8 | 4096 | 512 routed + 1 shared | ❌ 16x1 FAILED |
| Qwen3-Omni Thinker | 16x4 | 2048 | 128 routed | ✅ 16x1 PASSED |
| Qwen3-Omni Talker | 16x4 | 1024 | 128 routed + 1 shared | ❌ 16x1 FAILED |
| GPT-OSS | 8x4 | 2880 | 128 routed | ✅ 8x4 PASSED |
| Gemma 4 26B | 8x4 | 2816 | 64 routed | ✅ 8x4 PASSED |
| Ling 1T | 16x8 | 8192 | 256 routed + 1 shared | ⚠️ SKIPPED (known failure) |
| Mistral Large 3 | 16x4 | 7168 | 128 routed + 1 shared | ⚠️ SKIPPED (known failure) |

---

## DeepSeek V4 的具体状态

> [!WARNING]
> **DeepSeek V4 在此 PR 中尚未完全通过测试**

### DeepSeek V4 Flash
- 配置文件已创建：[deepseek_v4_flash.yaml](file:///home/stc/chengtao/tt-metal/models/common/modules/moe/configs/deepseek_v4_flash.yaml)
- **16x1 测试结果: ❌ FAILED**
  ```
  FAILED test_tt_moe_decode[wormhole_b0-deepseek_v4_flash-3-fabric_1D_ring-16x1]
  AssertionError: TTMoEDecode output verification failed for deepseek_v4_flash
  ```

### DeepSeek V4 Pro
- 配置文件已创建：[deepseek_v4_pro.yaml](file:///home/stc/chengtao/tt-metal/models/common/modules/moe/configs/deepseek_v4_pro.yaml)
- **被标记为 known failure，测试直接 SKIP**
- 在测试代码中被加入了 `SKIP_LIST`（会导致硬件 crash）

---

## 关键新增文件

### 核心模块
- **`models/common/modules/moe/tt_moe_decode.py`** (+971 行) — MoE decode 主模块
  - `TTMoEDecode` 类封装了整个 decode 时的 MoE 流水线
  - 流水线: `all_to_all_dispatch` → `moe_compute` → `fast_reduce` → `reduce_scatter`
  - 支持 shared experts 和 bias

- **`models/common/modules/moe/tt_moe_decode_config.py`** — 配置管理
  - `TTMoEDecodeConfig` 从 YAML 读取并推导运行参数
  - 支持 `with_mesh_shape()` 进行 mesh 切片

### 测试
- **`models/common/tests/modules/moe/test_tt_moe_decode.py`** (+556 行) — 集成测试
  - 覆盖所有 YAML 配置
  - 支持 16x1, 16x4, 8x4, 8x1 多种 mesh 形状
  - 对比 torch 参考实现验证输出正确性

---

## 结论

> [!CAUTION]
> **这个 PR 不等于"支持了 DeepSeek V4 运行"**。它做的是：
> 1. ✅ 创建了一个通用的 MoE decode 框架
> 2. ✅ 为 DeepSeek V4 (Flash 和 Pro) 添加了 YAML 配置
> 3. ❌ DeepSeek V4 Flash 的 MoE decode 测试**未通过**（输出验证失败）
> 4. ❌ DeepSeek V4 Pro 被标记为 known failure，测试直接**跳过**
> 5. ⚠️ 这只是 MoE 的 **gate 之后的部分**，gate 和其他前置步骤将在后续 PR 中添加

### 如果你要在 Wormhole Galaxy 上跑 DeepSeek V4：
- MoE decode 模块已经有框架，但 **V4 Flash 输出精度不通过，V4 Pro 会导致 crash**
- 这只是 MoE 层的一部分（gate 之后），完整的模型推理还需要更多工作
- 需要等待后续 PR 修复 V4 的精度问题、加入 gate 逻辑、并完成端到端模型集成
