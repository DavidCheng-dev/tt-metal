# Wormhole Galaxy 32卡 Mesh Device 系统健康检查与测试指南

本指南旨在介绍如何利用 `tt-metal` 仓库中现有的 C++ 和 Python 测试代码，对 **Tenstorrent Wormhole Galaxy**（32张卡，8×4 二维网格互联系统）进行全面的状态健康检查，重点关注 **Mesh Device (网格设备)** 的初始化、子网格划分、张量操作以及卡间 CCL (Collective Communication Library) 互联通信。

---

## 1. 硬件与系统环境准备

在开始测试前，请先确保硬件连接和基础驱动状态正常。

### 1.1 使用 `tt-smi` 重置与信息导出
Galaxy 32卡系统在运行前或出现错误时，通常需要进行整机重置：
* **整机重置**：参考 [TT-SMI](https://github.com/tenstorrent/tt-smi) 仓库指南，执行 `tt-smi` 重置指令确保所有 PCIe 和网络连接正常初始化。
* **信息导出**：执行系统状态快照导出，确认 32 张卡在 PCIe 拓扑上全部可见：
  ```bash
  tt-smi -g
  ```
  或者保存系统快照以便排查故障。

### 1.2 环境变量配置
大部分 Galaxy 测试依赖特定的架构环境变量。在运行测试前，确保设置了正确的架构名称：
```bash
export ARCH_NAME=wormhole_b0
```

---

## 2. C++ 层系统健康与通信基础测试

这些底层 C++ 单元测试主要验证以太网物理连接、指令队列通道（Command Queue）、Program API 以及底层的 Fabric 数据路由。

> [!NOTE]
> **关于 C++ 测试编译**：
> 默认编译不会生成 C++ 测试程序。如果您在执行时遇到 "No such file or directory" 错误，请先在 `ct_metal` 容器中编译测试：
> * **全量编译所有测试**：
>   ```bash
>   ./build_metal.sh --build-tests
>   ```
> * **仅编译 Metal 相关测试**（耗时较短，包含 `test_system_health` 等）：
>   ```bash
>   ./build_metal.sh --build-metal-tests
>   ```
> * **或者从宿主机一键触发编译**：
>   ```bash
>   docker exec -it -w /root/tt-metal ct_metal ./build_metal.sh --build-metal-tests
>   ```

### 2.1 物理以太网连接与链路状态检查 (`test_system_health`)
该测试会自动遍历所有以太网连接，打印物理链路状态（Link Status），并执行系统健全性校验：
```bash
./build/test/tt_metal/tt_fabric/test_system_health
```
* **期望输出**：成功时会打印出每张卡与相邻卡的以太网链路互联图（包括本地端口与对端端口），确保无断开或异常协商速率的链路。

### 2.2 单卡 Command Queue 与内存 Buffer 测试
在开启以太网核心容错的情况下，使用 `unit_tests_dispatch` 执行针对 Mesh 设备的单卡控制通路与内存读写测试：
```bash
# 1. 验证基本指令队列 API (Command Queue)
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardFixture.*"

# 2. 验证 Metal Program API 运行与调度
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardProgramFixture.*"

# 3. 验证 L1 / DRAM 内存大缓冲区读写 (Buffer Read/Writes)
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"
```

### 2.3 2D Fabric 网格互联测试
使用底层的 Fabric 2D 拓扑结构，验证 Galaxy 节点间的 2D 路由读写功能：
```bash
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
```

---

## 3. Python 层 Galaxy 专属健康检查套件 (`tests/galaxy_health_check`)

`tt-metal` 提供了专门针对 Galaxy 32卡系统的健康检查套件，位于 [tests/galaxy_health_check/](file:///home/stc/chengtao/tt-metal/tests/galaxy_health_check/)。

### 3.1 一键运行全部 Python 测试
在容器或宿主机环境中进入 `/root/tt-metal` 目录，通过以下一键脚本运行完整的健康检查测试流（包含单卡隔离、多卡并行、Mesh生命周期及通信）：
```bash
bash tests/galaxy_health_check/run_all.sh
```
测试结果与日志将自动输出至 `generated/test_reports/galaxy_health_check/`。

### 3.2 模块化运行说明

#### ① 全量 32 张卡逐一健康检查 (`test_all_cards.py`)
该测试会对 Galaxy 系统的 32 张卡（`device_id` 为 0-31）**逐个打开、执行精度校验测试、然后关闭**。这样能够实现硬件隔离，快速筛选出具体有损坏或计算异常的坏卡。
* **覆盖算法**：Softmax Correctness & Matmul Correctness
* **运行命令**：
  ```bash
  # 测试全量卡
  pytest tests/galaxy_health_check/test_all_cards.py -v

  # 定向测试某张卡（如 device_id-5）
  pytest tests/galaxy_health_check/test_all_cards.py -v -k "device_id-5"
  ```

#### ② Mesh Device 开启与关闭测试 (`test_mesh_device.py`)
主要测试 Galaxy 系统下二维 Mesh 逻辑设备的生命周期管理，包括重复开关、子网格（Submesh）划分等。
* **覆盖内容**：
  * **逻辑形状**：8×4 Mesh（行优先）与 4×8 Mesh（列优先）的成功初始化与关闭。
  * **资源复用**：使用 `ttnn.create_mesh_device` 上下文管理器，或连续三次打开/关闭 8×4 Mesh，检查是否有句柄或物理资源泄漏。
  * **子网格创建 (`create_submesh`)**：从 8×4 全局 Mesh 中抽取子拓扑，验证 `(1, 4)`, `(2, 4)`, `(8, 1)`, `(4, 4)`, `(8, 4)` 等子网络形状是否能正常运行。
* **运行命令**：
  ```bash
  pytest tests/galaxy_health_check/test_mesh_device.py -v
  ```

#### ③ 子网格张量操作测试 (`test_mesh_ops.py`)
验证从大 Mesh 中切分出的各种小 Submesh（如 1x4、2x4、8x1、4x4 等）能否正常分发数据并完成基本运算。
* **覆盖算子**：
  * `ttnn.ReplicateTensorToMesh`：数据复制到全部子卡。
  * `ttnn.ShardTensorToMesh` & `ttnn.ConcatMeshToTensor`：对张量沿 dim=3 分片并恢复，验证数据一致性。
  * `ttnn.add` / `ttnn.matmul`：子网格上的并行矩阵乘法 and 逐元素加法。
* **运行命令**：
  ```bash
  pytest tests/galaxy_health_check/test_mesh_ops.py -v
  ```

#### ④ 原生 8×4 全局网格张量与 CCL 测试 (`test_native_mesh_ops.py`)
在完整不拆分的 8×4 Mesh 逻辑设备上，直接进行大规模多芯片并行计算和卡间互联通信（CCL）测试。
* **覆盖内容**：
  * 32 卡全局 Replicate / Shard & Concat / Add / Matmul 正确性与 PCC 校验。
  * **双轴 CCL 串联测试**：
    * **TestCCLAllGatherAxis1Ring**：先在 Axis 1（行方向，4卡环形 Ring）做 AllGather，再在 Axis 0（列方向，8卡线性 Linear）做 AllGather，验证 32 卡上数据重新组装的完整性和精确度。
    * **TestCCLAllGatherAxis0Linear**：先在 Axis 0 线性收集，再在 Axis 1 环形收集。
  * **分布式 AllReduce 测试**（ReduceScatter + AllGather）：
    * **TestCCLAllReduceAxis1Ring**：各行 4 卡独立进行 Ring AllReduce。
    * **TestCCLAllReduceAxis0Linear**：各列 8 卡独立进行 Linear AllReduce。
* **运行命令**：
  ```bash
  pytest tests/galaxy_health_check/test_native_mesh_ops.py -v
  ```

#### ⑤ 独立 CCL 算子通信深度校验 (`test_ccl_all_gather.py`)
针对大模型（例如 Llama-70B）在 Galaxy 系统中的关键网络通信拓扑（Axis 1 的 Ring 和 Axis 0 的 Linear），进行深度以太网带宽通信检查。
* **关键配置与拓扑**：
  * **Axis 1 (行方向，4卡 Ring)**：模拟 Llama SDPA/MLP 时的 `line_all_gather` 或 `ring_all_gather`，使用 `FABRIC_1D_RING` 协议。
  * **Axis 0 (列方向，8卡 Linear)**：模拟 AllReduce 第一阶段的 `all_gather`，使用 `FABRIC_1D` 协议。
  * **全网并发压测 (8x4 Axis1 & Axis0)**：验证 8 行并行进行 Ring AllGather，或 4 列同时进行 Linear AllGather 时的以太网稳定性。
* **运行命令**：
  ```bash
  # 运行完整的 CCL AllGather 测试
  pytest tests/galaxy_health_check/test_ccl_all_gather.py -v

  # 分别过滤行、列方向测试
  pytest tests/galaxy_health_check/test_ccl_all_gather.py -k TestAllGatherAxis1Ring -v
  pytest tests/galaxy_health_check/test_ccl_all_gather.py -k TestAllGatherAxis0Linear -v
  ```

---

## 4. 以太网链路吞吐量与时延基准测试 (Microbenchmarks)

除功能和精度测试外，以太网性能微基准测试（Microbenchmarks）能够揭示是否存在特定的以太网端口故障、重新协商超时或链路速率降级。
这些测试会遍历所有激活的以太网连接并打印实测指标，**需要确保编译时开启了 Tracy 分析（默认已开启）**。

### 4.1 编译测试集
```bash
./build_metal.sh --build-tests
```

### 4.2 吞吐量性能测试
```bash
pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py
```
* **注意**：该微基准测试会全面压测以太网总线，运行时间可能会持续数小时，但能精确诊断出带宽不达标或频繁发生数据重传的以太网链路端口。

### 4.3 时延性能测试
```bash
pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py
```
* 该测试可快速测试每一对物理链路的数据往返时延（Latency），用以定位响应过慢或连接抖动的网卡节点。

---

## 5. 常见故障分析与定位指南

| 现象 | 可能原因 | 解决办法 |
| --- | --- | --- |
| 测试抛出 `TypeError: l1_small_size expects int...` | 旧版 API 调用参数位置冲突（属于已知测试 Bug，非硬件问题） | 更新代码，或排查使用 `pytest` 运行 `test_multi_device.py` 时的具体堆栈，忽略或修复该特定 API 参数冲突。 |
| 单个 device_id 出现 Softmax/Matmul PCC 偏低或超时 | 该特定卡存在芯片制造缺陷、电压不稳或散热故障 | 使用 `pytest tests/galaxy_health_check/test_all_cards.py -v` 进行排查，精确定位故障卡号，通知硬件维护人员或使用 `tt-smi` 对该卡进行单卡重置。 |
| CCL 互联测试报错或数值校验失败 (PCC 不为 1.0) | 卡间以太网跳线松动、端口损坏，或以太网内核 Retrain 协商失败 | 1. 运行 `./build/test/tt_metal/tt_fabric/test_system_health` 确认是否有以太网端口状态为 Down。<br>2. 运行 `tt-smi` 执行物理整机以太网复位并重新开始测试。 |
| 检测到设备数量不足 32 卡被 Skip | 当前机器不是 6U Galaxy 或未完全识别 32 张卡 | 运行 `tt-smi -g` 查看实际检测到的物理设备数量。若物理可见卡数小于32，需检查 PCIe 物理连接或进行驱动重新加载。 |
