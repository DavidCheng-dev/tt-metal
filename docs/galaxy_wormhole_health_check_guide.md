# Galaxy Wormhole 6U 系统健康检测指南

本文档介绍如何利用 tt-metal 仓库中已有的测试代码来全面验证 Galaxy Wormhole 6U（32 卡）系统是否正常运行。

测试按照从**底层硬件与发现到高层应用与性能压测**的顺序组织为 7 个层级，每个层级验证系统的一个关键方面。

---

## 目录

- [系统概述](#系统概述)
- [前置条件](#前置条件)
- [层级 1：系统健康与物理连接检测](#层级-1系统健康与物理连接检测)
- [层级 2：Fabric 数据通路与发现测试](#层级-2fabric-数据通路与发现测试)
- [层级 3：MeshDevice 初始化与管理](#层级-3meshdevice-初始化与管理)
- [层级 4：多设备基本计算](#层级-4多设备基本计算)
- [层级 5：Trace 程序追踪与回放](#层级-5trace-程序追踪与回放)
- [层级 6：集合通信 CCL 与大模型融合算子](#层级-6集合通信-ccl-与大模型融合算子)
- [层级 7：Nightly 与长期性能压力测试](#层级-7nightly-与长期性能压力测试)
- [快速健康检查流程](#快速健康检查流程)
- [完整测试流程](#完整测试流程)
- [故障排查](#故障排查)

---

## 系统概述

Galaxy Wormhole 6U 系统的核心参数：

| 参数 | 值 |
|------|-----|
| 芯片数量 | 32 |
| Mesh 拓扑 | 8 行 × 4 列 |
| MMIO（PCIe）芯片 | 32（每个芯片都有 PCIe 连接） |
| 芯片间连接 | 以太网，每对相邻芯片 4 条链路 |
| ClusterType | `GALAXY`（通过 `is_6u()` 判断） |
| 芯片架构 | Wormhole B0 |
| 计算网格 | 7×10 (无 harvest) |

tt-metal 代码中，Galaxy 6U 系统被识别为 `tt::tt_metal::ClusterType::GALAXY`，与 4U TG（`ClusterType::TG`）有所不同。

---

## 前置条件

### 1. 进入容器

所有测试必须在 `ct_metal` Docker 容器内执行：

```bash
# 从仓库根目录
./start_container.sh
```

容器内工作目录为 `/root/tt-metal`。

### 2. 确认编译完成

C++ GTest 测试需要预先编译。确认以下二进制文件存在：

```bash
# 系统健康检测
ls build/test/tt_metal/tt_fabric/test_system_health
# 物理拓扑发现
ls build/test/tt_metal/tt_fabric/test_physical_discovery
# Fabric 烟雾与地址读写测试
ls build/test/tt_metal/tt_fabric/fabric_smoke_tests
ls build/test/tt_metal/tt_fabric/test_addrgen_write
# Fabric 基准测试
ls build/test/tt_metal/tt_fabric/bench_unicast
ls build/test/tt_metal/tt_fabric/test_bandwidth_telemetry_validation
# Fabric 2D 单元测试
ls build/test/tt_metal/tt_fabric/fabric_unit_tests
# Command Queue / Program / Buffer 单元测试
ls build/test/tt_metal/unit_tests_dispatch
# 分布式单元测试
ls build/test/tt_metal/distributed/distributed_unit_tests
```

如果二进制文件不存在，需要先编译：

```bash
cmake --build build --target test_system_health test_physical_discovery fabric_smoke_tests test_addrgen_write bench_unicast test_bandwidth_telemetry_validation distributed_unit_tests fabric_unit_tests unit_tests_dispatch -j$(nproc)
```
或者编译所有测试
```bash
./build_metal.sh --build-tests
```

> [!TIP]
> 也可以直接运行 Tenstorrent 官方组件测试套件脚本（需先完成上述编译）：
> ```bash
> dockerfile/upstream_test_images/run_upstream_tests_vanilla.sh
> ```
> 该脚本包含 `Galaxy_WH_6U_SW_Guide.md` 中推荐的所有 Component Tests（`test_system_health`、`unit_tests_dispatch`、`fabric_unit_tests`）以及以太网带宽测试，适合系统整体验收时一键运行。

### 3. 确认环境变量

```bash
# 确认检测到 32 个设备 [已通过 ✅]
python3 -c "import ttnn; print(f'设备数量: {ttnn.get_num_devices()}')"

# 确认是 Galaxy 集群类型 [已通过 ✅]
python3 -c "import ttnn; print(f'集群类型: {ttnn.cluster.get_cluster_type()}')"
```

输出：
```
设备数量: 32
集群类型: ClusterType.GALAXY
```

---

## 层级 1：系统健康与物理连接检测

> **目的**：验证 32 块芯片的物理以太网链路是否全部正常连通，检测 CRC 错误和 retrain 计数；同时验证每块芯片上基础 Metal API（Command Queue、Program、Buffer）的正确性，以及以太网链路的实际带宽和延迟（对应 `Galaxy_WH_6U_SW_Guide.md` 中的 Component Tests 和 Ethernet Bandwidth Tests）。

### 测试 1.1：系统健康报告（ReportSystemHealth）

- **源码**：`tests/tt_metal/tt_fabric/system_health/test_system_health.cpp`
- **二进制**：`build/test/tt_metal/tt_fabric/test_system_health`
- **类型**：C++ GTest（信息报告，不会断言失败）

**运行命令**：

```bash
./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportSystemHealth"
```

**预期输出**：
```
Found 32 chips in cluster:
Chip: 0 PCIe: 0 Unique ID: xxxx Tray: 0 N1
 eth channel 0 core (1,0) link UP (internal trace), connected to Chip 1 ...
   Retrain count: 0 CRC Errors: 0x0 Corrected Codewords: 0x0 Uncorrected Codewords: 0x0
 ...
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **硬件状态**：成功在集群中检测到 32 块芯片，所有 512 个以太网通道连接均正常连通（`link UP`），Retrain count 均为 0，无 CRC 错误。
- **输出日志片段**：
```
Running main() from gmock_main.cc
Note: Google Test filter = Cluster.ReportSystemHealth
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Cluster
[ RUN      ] Cluster.ReportSystemHealth
...
2026-06-11 13:19:37.769 | info     |            Test | Found 32 chips in cluster:

Chip: 0 PCIe: 16 Unique ID: 135343630303447 Tray: 3 N1
 eth channel 0 core 0-0 link UP (QSFP), connected to Chip 8 Tray: 4 N1 core 0-0
	Retrain count: 0 CRC Errors: 0x0 Corrected Codewords: 0x0 Uncorrected Codewords: 0x0
 ...
 eth channel 15 core 0-15 link UP (internal trace), connected to Chip 4 Tray: 3 N5 core 0-3
	Retrain count: 0 CRC Errors: 0x0 Corrected Codewords: 0x0 Uncorrected Codewords: 0x0

... (其余 30 个芯片信息均正常连通且无 CRC 错误，篇幅原因省略) ...

Chip: 31 PCIe: 7 Unique ID: 835343630303236 Tray: 1 N8
 eth channel 0 core 0-0 link UP (internal trace), connected to Chip 27 Tray: 1 N4 core 0-12
	Retrain count: 0 CRC Errors: 0x0 Corrected Codewords: 0x0 Uncorrected Codewords: 0x0
 ...
 eth channel 15 core 0-15 link UP (linking board 2 type A), connected to Chip 23 Tray: 2 N8 core 0-15
	Retrain count: 0 CRC Errors: 0x0 Corrected Codewords: 0x327 Uncorrected Codewords: 0x0

[       OK ] Cluster.ReportSystemHealth (432 ms)
[----------] 1 test from Cluster (432 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (432 ms total)
[  PASSED  ] 1 test.
```


**通过标准**：
- 所有预期的以太网通道显示 `link UP`
- Retrain count 均为 0
- 无 CRC 错误
- 末尾无 warning 消息

---

### 测试 1.2：Mesh 完整连接性验证（TestMeshFullConnectivity）

- **源码**：`tests/tt_metal/tt_fabric/system_health/test_system_health.cpp`
- **类型**：C++ GTest（断言测试，失败会报错）

**运行命令**：

```bash
# 基本连接性测试（自动检测集群类型）
./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.TestMeshFullConnectivity"

# 显式验证 TORUS 拓扑
./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.TestMeshFullConnectivity" -- --cluster-type GALAXY --system-topology TORUS_XY

# 显式验证 MESH 拓扑
./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.TestMeshFullConnectivity" -- --cluster-type GALAXY --system-topology MESH
```

**Galaxy 6U 专属参数**：
- 预期芯片数 `num_expected_chips = 32`
- 预期 MMIO 芯片数 `num_expected_mmio_chips = 32`
- 每侧连接数 `num_connections_per_side = 4`

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **硬件状态**：成功验证 32 块芯片的 Mesh 完整连接性，自动检测、TORUS_XY 拓扑以及 MESH 拓扑验证均成功通过。
- **输出日志片段（基本连接性测试）**：
```bash
Running main() from gmock_main.cc
Note: Google Test filter = Cluster.TestMeshFullConnectivity
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Cluster
[ RUN      ] Cluster.TestMeshFullConnectivity
2026-06-11 13:22:52.603 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:228)
2026-06-11 13:22:52.603 | info     |             UMD | Cluster constructor started. (cluster.cpp:335)
2026-06-11 13:22:52.618 | info     |             UMD | Creating TopologyDiscovery for architecture: wormhole_b0 (topology_discovery.cpp:91)
2026-06-11 13:22:52.618 | info     |             UMD | Starting topology discovery. (topology_discovery.cpp:110)
2026-06-11 13:22:52.627 | info     |             UMD | Established firmware bundle version: 19.5.0 (topology_discovery.cpp:567)
2026-06-11 13:22:52.690 | info     |             UMD | Completed topology discovery. (topology_discovery.cpp:114)
2026-06-11 13:22:52.766 | info     |             UMD | Opening local chip ids/PCIe ids: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}/[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7] and remote chip ids {} (cluster.cpp:168)
2026-06-11 13:22:52.766 | info     |             UMD | IOMMU: disabled (cluster.cpp:142)
2026-06-11 13:22:52.766 | info     |             UMD | KMD version: 2.8.1 (cluster.cpp:145)
2026-06-11 13:22:52.766 | info     |             UMD | Cluster constructor completed. (cluster.cpp:507)
2026-06-11 13:22:52.787 | info     |             UMD | Starting devices in cluster (cluster.cpp:999)
2026-06-11 13:22:52.841 | info     |             UMD | Starting devices in cluster completed. (cluster.cpp:1007)
[       OK ] Cluster.TestMeshFullConnectivity (371 ms)
[----------] 1 test from Cluster (371 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (371 ms total)
[  PASSED  ] 1 test.
2026-06-11 13:22:52.959 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:512)
2026-06-11 13:22:52.959 | info     |             UMD | Closing devices in cluster (cluster.cpp:1012)
2026-06-11 13:23:10.705 | info     |             UMD | Closing devices in cluster completed. (cluster.cpp:1021)
2026-06-11 13:23:10.706 | info     |             UMD | Cluster destructor started. (cluster.cpp:729)
2026-06-11 13:23:10.706 | info     |             UMD | Cluster destructor completed. (cluster.cpp:732)
```

---

### 测试 1.3：跨 Mesh 链路报告（ReportIntermeshLinks）

- **源码**：`tests/tt_metal/tt_fabric/system_health/test_system_health.cpp`

报告 UBB Galaxy 系统中跨 tray 的 inter-mesh 链路配置（通过 linking board 连接的以太网通道）。

```bash
./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks"
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **状态说明**：集群当前不支持/未配置跨 Mesh 链路（在单机 32 卡 UBB Galaxy 系统上，由于不存在连接其他系统的外部跨 Mesh 链路，因此输出 `Cluster does not support intermesh links` 为预期内的正常状态，测试通过）。
- **输出日志**：
```
Running main() from gmock_main.cc
Note: Google Test filter = Cluster.ReportIntermeshLinks
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Cluster
[ RUN      ] Cluster.ReportIntermeshLinks
2026-06-11 13:28:46.270 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:228)
2026-06-11 13:28:46.270 | info     |             UMD | Cluster constructor started. (cluster.cpp:335)
2026-06-11 13:28:46.285 | info     |             UMD | Creating TopologyDiscovery for architecture: wormhole_b0 (topology_discovery.cpp:91)
2026-06-11 13:28:46.285 | info     |             UMD | Starting topology discovery. (topology_discovery.cpp:110)
2026-06-11 13:28:46.293 | info     |             UMD | Established firmware bundle version: 19.5.0 (topology_discovery.cpp:567)
2026-06-11 13:28:46.355 | info     |             UMD | Completed topology discovery. (topology_discovery.cpp:114)
2026-06-11 13:28:46.429 | info     |             UMD | Opening local chip ids/PCIe ids: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}/[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7] and remote chip ids {} (cluster.cpp:168)
2026-06-11 13:28:46.429 | info     |             UMD | IOMMU: disabled (cluster.cpp:142)
2026-06-11 13:28:46.429 | info     |             UMD | KMD version: 2.8.1 (cluster.cpp:145)
2026-06-11 13:28:46.429 | info     |             UMD | Cluster constructor completed. (cluster.cpp:507)
2026-06-11 13:28:46.450 | info     |             UMD | Starting devices in cluster (cluster.cpp:999)
2026-06-11 13:28:46.504 | info     |             UMD | Starting devices in cluster completed. (cluster.cpp:1007)
2026-06-11 13:28:46.624 | info     |            Test | Cluster does not support intermesh links (test_system_health.cpp:217)
[       OK ] Cluster.ReportIntermeshLinks (369 ms)
[----------] 1 test from Cluster (369 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (369 ms total)
[  PASSED  ] 1 test.
2026-06-11 13:28:46.624 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:512)
2026-06-11 13:28:46.624 | info     |             UMD | Closing devices in cluster (cluster.cpp:1012)
2026-06-11 13:28:46.724 | info     |             UMD | Closing devices in cluster completed. (cluster.cpp:1021)
2026-06-11 13:28:46.724 | info     |             UMD | Cluster destructor started. (cluster.cpp:729)
2026-06-11 13:28:46.724 | info     |             UMD | Cluster destructor completed. (cluster.cpp:732)
```

---

### 测试 1.4：物理拓扑与发现验证（TestPhysicalDiscovery）

- **源码**：`tests/tt_metal/tt_fabric/physical_discovery/test_physical_system_descriptor.cpp`
- **二进制**：`build/test/tt_metal/tt_fabric/test_physical_discovery`
- **类型**：C++ GTest

**功能说明**：
运行物理拓扑发现算法，验证 hostnames/ASICs 的拓扑图节点和物理以太网链路映射是否符合预期。

```bash
# 验证物理系统描述符的生成与合并
./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.TestPhysicalSystemDescriptor"

# 打印并导出完整的 Host 拓扑连接状态（排查首选）
./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.PrintHostTopology"
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **硬件状态**：成功生成并验证物理系统描述符，包含 32 块 ASIC 设备信息及其在 4 个 Tray（Tray 1-4）上的分布、PCIe 设备 ID 和 ASIC 物理位置的映射关系。主机拓扑结构发现和打印过程均正常完成。
- **输出日志片段（验证物理系统描述符的生成与合并）**：
```
Running main() from gmock_main.cc
Note: Google Test filter = PhysicalDiscovery.TestPhysicalSystemDescriptor
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from PhysicalDiscovery
[ RUN      ] PhysicalDiscovery.TestPhysicalSystemDescriptor
2026-06-11 13:31:12.862 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:228)
2026-06-11 13:31:12.862 | info     |             UMD | Cluster constructor started. (cluster.cpp:335)
2026-06-11 13:31:12.877 | info     |             UMD | Creating TopologyDiscovery for architecture: wormhole_b0 (topology_discovery.cpp:91)
2026-06-11 13:31:12.877 | info     |             UMD | Starting topology discovery. (topology_discovery.cpp:110)
2026-06-11 13:31:12.886 | info     |             UMD | Established firmware bundle version: 19.5.0 (topology_discovery.cpp:567)
2026-06-11 13:31:12.970 | info     |             UMD | Completed topology discovery. (topology_discovery.cpp:114)
2026-06-11 13:31:13.051 | info     |             UMD | Opening local chip ids/PCIe ids: {0, 1, ... 31}/[16, 17, ... 7] and remote chip ids {} (cluster.cpp:168)
2026-06-11 13:31:13.051 | info     |             UMD | IOMMU: disabled (cluster.cpp:142)
2026-06-11 13:31:13.051 | info     |             UMD | KMD version: 2.8.1 (cluster.cpp:145)
2026-06-11 13:31:13.051 | info     |             UMD | Cluster constructor completed. (cluster.cpp:507)
2026-06-11 13:31:13.068 | info     |             UMD | Starting devices in cluster (cluster.cpp:999)
2026-06-11 13:31:13.122 | info     |             UMD | Starting devices in cluster completed. (cluster.cpp:1007)
2026-06-11 13:31:13.250 | info     |            Test | Dumping Physical System Descriptor to YAML (test_physical_system_descriptor.cpp:151)
compute_node_specs:
  2930fbadc12d:
    motherboard: S7T-MB
    asic_info:
      - tray_id: 1
        board_type: UBB
        asics:
          - asic_location: 1
            asic_id: 87033175144935990
          - asic_location: 2
            asic_id: 159090769182863926
          - asic_location: 3
            asic_id: 231148363220791862
          - asic_location: 4
            asic_id: 303205957258719798
          - asic_location: 5
            asic_id: 375263551296647734
          - asic_location: 6
            asic_id: 447321145334575670
          - asic_location: 7
            asic_id: 519378739372503606
          - asic_location: 8
            asic_id: 591436333410431542
      - tray_id: 2
        board_type: UBB
        asics:
          - asic_location: 1
            asic_id: 87033175144936025
... (其余 Tray 3 和 Tray 4 等 32 个 ASIC 的物理节点分布正常生成) ...
local_eth_connections:
  - [{host_name: 2930fbadc12d, tray_id: 3, asic_location: 7, chan_id: 11}, {host_name: 2930fbadc12d, tray_id: 3, asic_location: 8, chan_id: 7}]
... (共发现并解析出所有内部以太网连接) ...
[       OK ] PhysicalDiscovery.TestPhysicalSystemDescriptor (383 ms)
[----------] 1 test from PhysicalDiscovery (383 ms total)
[==========] 1 test from 1 test suite ran. (383 ms total)
[  PASSED  ] 1 test.
```
- **输出日志片段（打印主机拓扑连接状态）**：
```
Running main() from gmock_main.cc
Note: Google Test filter = PhysicalDiscovery.PrintHostTopology
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from PhysicalDiscovery
[ RUN      ] PhysicalDiscovery.PrintHostTopology
...
2026-06-11 13:31:18.646 | info     |            Test | === Host Topology === (test_physical_system_descriptor.cpp:196)
2026-06-11 13:31:18.646 | info     |            Test | 2930fbadc12d: {} (test_physical_system_descriptor.cpp:207)
2026-06-11 13:31:18.646 | info     |            Test | === End Host Topology === (test_physical_system_descriptor.cpp:209)
[       OK ] PhysicalDiscovery.PrintHostTopology (423 ms)
[----------] 1 test from PhysicalDiscovery (423 ms total)
[==========] 1 test from 1 test suite ran. (423 ms total)
[  PASSED  ] 1 test.
```

---

### 测试 1.5：Command Queue API 验证（UnitMeshCQSingleCardFixture）

- **来源**：`Galaxy_WH_6U_SW_Guide.md` 官方 Component Tests
- **二进制**：`build/test/tt_metal/unit_tests_dispatch`
- **类型**：C++ GTest（逐卡循环验证基础 Command Queue API 正确性）

**功能说明**：
在每块芯片上验证基础 Command Queue API 的正确性，包括命令入队、执行和同步机制。

**运行命令**：

```bash
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="UnitMeshCQSingleCardFixture.*"
```

**通过标准**：
- 所有 `UnitMeshCQSingleCardFixture.*` 用例显示 `PASSED`
- 无 Command Queue 超时或同步错误

---

### 测试 1.6：Metal Program API 验证（UnitMeshCQSingleCardProgramFixture）

- **二进制**：`build/test/tt_metal/unit_tests_dispatch`
- **类型**：C++ GTest（逐卡循环验证 Metal Program API 正确性）

**功能说明**：
在每块芯片上验证 Metal Program API 的正确性，确保程序编译和执行流程正常。

**运行命令**：

```bash
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="UnitMeshCQSingleCardProgramFixture.*"
```

**通过标准**：
- 所有 `UnitMeshCQSingleCardProgramFixture.*` 用例显示 `PASSED`
- 无程序编译或执行异常

---

### 测试 1.7：内存 Buffer 读写验证（ShardedBufferLargeReadWrites）

- **二进制**：`build/test/tt_metal/unit_tests_dispatch`
- **类型**：C++ GTest（验证 L1 和 DRAM 内存 Buffer 分片大块读写）

**功能说明**：
验证每块芯片上的 L1 和 DRAM 内存 Buffer 分片大块读写是否正确，确保内存访问路径无误。

**运行命令**：

```bash
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="UnitMeshCQSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"
```

**通过标准**：
- L1 和 DRAM 的大块分片 Buffer 读写均返回正确数据
- 无内存访问错误

---

### 以太网带宽与延迟微基准

> [!NOTE]
> 以下以太网微基准测试需要在启用 Tracy 的构建下运行（`./build_metal.sh --build-tests` 默认已启用）。带宽测试可能**需要数小时**才能完成，建议在非生产时段运行。

### 测试 1.8：以太网链路带宽测试（test_all_ethernet_links_bandwidth）

- **源码**：`tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py`
- **类型**：Python pytest 微基准测试

**功能说明**：
遍历所有激活的以太网链路，逐一测量实际发送/接收带宽，打印每条链路的测量带宽并识别带宽不匹配的链路。

**运行命令**：

```bash
pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py
```

> [!WARNING]
> 此测试可能需要数小时完成，请勿在生产任务运行期间执行。

**通过标准**：
- 所有以太网链路带宽均在预期范围内
- 无严重的带宽不匹配链路

---

### 测试 1.9：以太网链路延迟测试（test_all_ethernet_links_latency）

- **源码**：`tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py`
- **类型**：Python pytest 微基准测试

**功能说明**：
测量所有以太网链路的往返延迟，打印每条链路的延迟值，识别高延迟链路。

**运行命令**：

```bash
pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py
```

**通过标准**：
- 所有链路延迟在正常范围内（通常为个位数微秒量级）
- 无异常高延迟链路

---

## 层级 2：Fabric 数据通路与发现测试

> **目的**：验证 Fabric 路由器层的基本数据传输和寻址机制，评估芯片间 Fabric 报文的通过率与性能。

### 测试 2.1：2D Fabric 单播烟雾测试 [未通过 ❌]

- **源码**：`tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke.cpp`
- **二进制**：`build/test/tt_metal/tt_fabric/fabric_smoke_tests`

```bash
./build/test/tt_metal/tt_fabric/fabric_smoke_tests --gtest_filter="Fabric2DFixture.TestUnicastConnAPI2DSmoke"
```

---

### 测试 2.2：1D Fabric 单播烟雾测试 [未通过 ❌]

```bash 
./build/test/tt_metal/tt_fabric/fabric_smoke_tests --gtest_filter="Fabric1DFixture.TestUnicastConnAPI1DSmoke"
```

---

### 测试 2.3：Fabric 地址生成器写入验证（TestAddrgenWrite）

- **源码**：`tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/test_main.cpp`
- **二进制**：`build/test/tt_metal/tt_fabric/test_addrgen_write`

**功能说明**：
验证 Fabric 路由下多播与单播地址生成器（AddrGen）写入机制是否正确，测试在高并发路由下的传输。

```bash
./build/test/tt_metal/tt_fabric/test_addrgen_write
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **功能验证**：成功通过了地址生成器综合测试中所有变体与数据大小的写入验证（共 216 个测试），涵盖了 L1、DRAM 以及不同写入状态设置（ScatterWrite、ScatterWriteWithState、ScatterWriteSetState）。
- **输出日志片段**：
```
[ RUN      ] AllVariantsAndSizes/AddrgenComprehensiveTest.Write/ScatterWrite_100B_L1
[       OK ] AllVariantsAndSizes/AddrgenComprehensiveTest.Write/ScatterWrite_100B_L1 (1046 ms)
...
[ RUN      ] AllVariantsAndSizes/AddrgenComprehensiveTest.Write/ScatterWriteSetState_99999B_DRAM
[       OK ] AllVariantsAndSizes/AddrgenComprehensiveTest.Write/ScatterWriteSetState_99999B_DRAM (1122 ms)
[----------] 216 tests from AllVariantsAndSizes/AddrgenComprehensiveTest (182175 ms total)

[----------] Global test environment tear-down
[==========] 216 tests from 1 test suite ran. (198143 ms total)
[  PASSED  ] 216 tests.
```

---

### 测试 2.4：Fabric 性能吞吐基准（BenchUnicast）

- **二进制**：`build/test/tt_metal/tt_fabric/bench_unicast`

**功能说明**：
测试 Fabric 下芯片间单播的极限传输带宽与延迟。

```bash
./build/test/tt_metal/tt_fabric/bench_unicast
```

---

### 测试 2.5：带宽遥测验证（TestBandwidthTelemetry）

- **二进制**：`build/test/tt_metal/tt_fabric/test_bandwidth_telemetry_validation`

**功能说明**：
验证 Fabric 中的遥测与性能计数器是否能正确监控物理链路带宽。

```bash
TT_METAL_FABRIC_TELEMETRY=1 TT_METAL_FABRIC_BW_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/test_bandwidth_telemetry_validation
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **运行命令**：
```bash
TT_METAL_FABRIC_TELEMETRY=1 TT_METAL_FABRIC_BW_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/test_bandwidth_telemetry_validation
```
- **输出日志**：
```
2026-06-11 14:07:39.109 | info     |            Test | 
--- Testing 1 MB transfers --- (test_bandwidth_telemetry_validation.cpp:232)
2026-06-11 14:07:39.109 | info     |            Test | Config: 1 MB × 100 iters × 10 trace = 1.05 GB payload (test_bandwidth_telemetry_validation.cpp:147)
2026-06-11 14:07:39.109 | info     |            Test | Running warmup transfer... (test_bandwidth_telemetry_validation.cpp:150)
2026-06-11 14:07:40.267 | info     |            Test | Warmup complete (test_bandwidth_telemetry_validation.cpp:152)
2026-06-11 14:07:40.267 | info     |            Test | Reading baseline counters... (test_bandwidth_telemetry_validation.cpp:155)
2026-06-11 14:07:40.267 | info     |            Test | Using AICLK = 1000.0 MHz (test_bandwidth_telemetry_validation.cpp:159)
2026-06-11 14:07:40.267 | info     |            Test | Running 100 measured transfers... (test_bandwidth_telemetry_validation.cpp:162)
2026-06-11 14:07:40.449 | info     |            Test |   Completed 25/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:40.627 | info     |            Test |   Completed 50/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:40.813 | info     |            Test |   Completed 75/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:40.991 | info     |            Test |   Completed 100/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:40.991 | info     |            Test | Transfers complete (test_bandwidth_telemetry_validation.cpp:170)
2026-06-11 14:07:40.991 | info     |            Test | Results: (test_bandwidth_telemetry_validation.cpp:192)
2026-06-11 14:07:40.991 | info     |            Test |   Counted:  1.18 GB (112.6% of payload) (test_bandwidth_telemetry_validation.cpp:197)
2026-06-11 14:07:40.991 | info     |            Test |   Expected: 1.05 GB payload (test_bandwidth_telemetry_validation.cpp:198)
2026-06-11 14:07:40.991 | info     |            Test |   Telemetry BW: 7132.13 MB/s (test_bandwidth_telemetry_validation.cpp:199)
2026-06-11 14:07:40.991 | info     |            Test |   Bench BW:     6408.99 MB/s (test_bandwidth_telemetry_validation.cpp:200)
2026-06-11 14:07:40.991 | info     |            Test |   Error:        11.3% (test_bandwidth_telemetry_validation.cpp:201)
2026-06-11 14:07:40.991 | info     |            Test | 
--- Testing 5 MB transfers --- (test_bandwidth_telemetry_validation.cpp:232)
2026-06-11 14:07:40.991 | info     |            Test | Config: 5 MB × 100 iters × 10 trace = 5.24 GB payload (test_bandwidth_telemetry_validation.cpp:147)
2026-06-11 14:07:40.991 | info     |            Test | Running warmup transfer... (test_bandwidth_telemetry_validation.cpp:150)
2026-06-11 14:07:41.710 | info     |            Test | Warmup complete (test_bandwidth_telemetry_validation.cpp:152)
2026-06-11 14:07:41.710 | info     |            Test | Reading baseline counters... (test_bandwidth_telemetry_validation.cpp:155)
2026-06-11 14:07:41.711 | info     |            Test | Using AICLK = 1000.0 MHz (test_bandwidth_telemetry_validation.cpp:159)
2026-06-11 14:07:41.711 | info     |            Test | Running 100 measured transfers... (test_bandwidth_telemetry_validation.cpp:162)
2026-06-11 14:07:42.411 | info     |            Test |   Completed 25/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:43.129 | info     |            Test |   Completed 50/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:43.826 | info     |            Test |   Completed 75/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:44.528 | info     |            Test |   Completed 100/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:44.528 | info     |            Test | Transfers complete (test_bandwidth_telemetry_validation.cpp:170)
2026-06-11 14:07:44.528 | info     |            Test | Results: (test_bandwidth_telemetry_validation.cpp:192)
2026-06-11 14:07:44.528 | info     |            Test |   Counted:  5.90 GB (112.6% of payload) (test_bandwidth_telemetry_validation.cpp:197)
2026-06-11 14:07:44.528 | info     |            Test |   Expected: 5.24 GB payload (test_bandwidth_telemetry_validation.cpp:198)
2026-06-11 14:07:44.528 | info     |            Test |   Telemetry BW: 7306.32 MB/s (test_bandwidth_telemetry_validation.cpp:199)
2026-06-11 14:07:44.528 | info     |            Test |   Bench BW:     6727.42 MB/s (test_bandwidth_telemetry_validation.cpp:200)
2026-06-11 14:07:44.528 | info     |            Test |   Error:        8.6% (test_bandwidth_telemetry_validation.cpp:201)
2026-06-11 14:07:44.528 | info     |            Test | 
--- Testing 10 MB transfers --- (test_bandwidth_telemetry_validation.cpp:232)
2026-06-11 14:07:44.528 | info     |            Test | Config: 10 MB × 100 iters × 10 trace = 10.49 GB payload (test_bandwidth_telemetry_validation.cpp:147)
2026-06-11 14:07:44.528 | info     |            Test | Running warmup transfer... (test_bandwidth_telemetry_validation.cpp:150)
2026-06-11 14:07:45.278 | info     |            Test | Warmup complete (test_bandwidth_telemetry_validation.cpp:152)
2026-06-11 14:07:45.278 | info     |            Test | Reading baseline counters... (test_bandwidth_telemetry_validation.cpp:155)
2026-06-11 14:07:45.278 | info     |            Test | Using AICLK = 1000.0 MHz (test_bandwidth_telemetry_validation.cpp:159)
2026-06-11 14:07:45.278 | info     |            Test | Running 100 measured transfers... (test_bandwidth_telemetry_validation.cpp:162)
2026-06-11 14:07:46.572 | info     |            Test |   Completed 25/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:47.868 | info     |            Test |   Completed 50/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:49.164 | info     |            Test |   Completed 75/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:50.461 | info     |            Test |   Completed 100/100 (test_bandwidth_telemetry_validation.cpp:167)
2026-06-11 14:07:50.461 | info     |            Test | Transfers complete (test_bandwidth_telemetry_validation.cpp:170)
2026-06-11 14:07:50.461 | info     |            Test | Results: (test_bandwidth_telemetry_validation.cpp:192)
2026-06-11 14:07:50.461 | info     |            Test |   Counted:  11.80 GB (112.6% of payload) (test_bandwidth_telemetry_validation.cpp:197)
2026-06-11 14:07:50.461 | info     |            Test |   Expected: 10.49 GB payload (test_bandwidth_telemetry_validation.cpp:198)
2026-06-11 14:07:50.461 | info     |            Test |   Telemetry BW: 7329.04 MB/s (test_bandwidth_telemetry_validation.cpp:199)
2026-06-11 14:07:50.461 | info     |            Test |   Bench BW:     6768.56 MB/s (test_bandwidth_telemetry_validation.cpp:200)
2026-06-11 14:07:50.461 | info     |            Test |   Error:        8.3% (test_bandwidth_telemetry_validation.cpp:201)
[       OK ] FabricBandwidthTelemetry.ValidateMultipleSizes (25600 ms)
[----------] 1 test from FabricBandwidthTelemetry (25601 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (25601 ms total)
[  PASSED  ] 1 test.
```

---

### 测试 2.6：2D Fabric 读写 API 验证（Fabric2D\*Fixture）

- **来源**：`Galaxy_WH_6U_SW_Guide.md` 官方 Component Tests
- **二进制**：`build/test/tt_metal/tt_fabric/fabric_unit_tests`
- **类型**：C++ GTest

**功能说明**：
将 Galaxy 配置为 2D Mesh Fabric，使用基础 Fabric 读写 API 在各芯片间进行数据传输，验证 2D 路由路径的正确性。与层级 2.1/2.2 的 `fabric_smoke_tests` 不同，本测试使用专门的 `fabric_unit_tests` 二进制，覆盖更完整的 2D Fabric API 用例（包括芯片间的读、写和多播操作）。

**运行命令**：

```bash
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
```

**通过标准**：
- 所有 `Fabric2D*Fixture.*` 用例显示 `PASSED`
- 芯片间 2D Fabric 读写数据正确无误
- 无路由错误或数据损坏

---

## 层级 3：MeshDevice 初始化与管理

> **目的**：验证能否正确初始化 Galaxy 的 32 卡 mesh 设备，各种 mesh 形状配置是否正常。

### 测试 3.1：Galaxy Mesh 打开/关闭

- **源码**：`tests/ttnn/unit_tests/base_functionality/test_multi_device.py`
- **类型**：Python pytest

测试在 Galaxy 32 卡系统上以不同 MeshShape 打开和关闭 mesh device：
`MeshShape(1, 4)`、`MeshShape(8, 1)`、`MeshShape(8, 4)` (32卡)、`MeshShape(3, 2)`。

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_open_close_galaxy_mesh -v
```

---

### 测试 3.2：Mesh Device 可视化

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_visualize_mesh_device -v
```

---

### 测试 3.3：C++ MeshDevice 初始化

- **源码**：`tests/tt_metal/distributed/test_mesh_device.cpp`
- **二进制**：`build/test/tt_metal/distributed/distributed_unit_tests`

```bash
# 1×1 Mesh 初始化
./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="MeshDeviceInitTest.Init1x1Mesh"

# Fabric Node ID 验证
./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="MeshDeviceTest.CheckFabricNodeIds"

# Submesh 创建测试
./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="MeshDevice2x4Test.*"
```

---

### 测试 3.4：Fabric 与 Submesh 兼容性

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_fabric_with_submeshes -v
```

---

## 层级 4：多设备基本计算

> **目的**：验证在 mesh 上的 tensor 分片、复制、设备间数据传输和基本算子是否正确工作。

### 测试 4.1：Tensor 分片与回环

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (12 passed, 4 skipped in 13.89s)
- **状态说明**：有 4 个关于 bfloat8_b 在 ROW_MAJOR_LAYOUT 下的测试排列组合因系统不支持而被跳过，其余 12 个关于 BFLOAT16 和 BFLOAT8_B 在 TILE 和 ROW_MAJOR_LAYOUT 上的分片测试均顺利通过。
- **输出日志片段**：
```
l     tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT8_B-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-device_params={'dispatch_core_axis': DispatchCoreAxis.ROW}]
0.00s call     tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT8_B-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
0.00s call     tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT8_B-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
0.00s call     tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT8_B-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-device_params={'dispatch_core_axis': DispatchCoreAxis.ROW}]
======================================= short test summary info ========================================
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-device_params={'dispatch_core_axis': DispatchCoreAxis.ROW}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-device_params={'dispatch_core_axis': DispatchCoreAxis.ROW}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-device_params={'dispatch_core_axis': DispatchCoreAxis.ROW}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-device_params={'dispatch_core_axis': DispatchCoreAxis.ROW}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT8_B-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-device_params={'dispatch_core_axis': DispatchCoreAxis.ROW}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT8_B-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT8_B-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-device_params={'dispatch_core_axis': DispatchCoreAxis.ROW}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_ttnn_to_and_from_multi_device_shard[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT8_B-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
SKIPPED [4] tests/ttnn/unit_tests/base_functionality/test_multi_device.py:134: Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT
==================================== 12 passed, 4 skipped in 13.89s ====================================
```

---

### 测试 4.2：Tensor 广播复制

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_replicate -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (16 passed in 17.13s)
- **状态说明**：多设备广播复制测试顺利通过，验证了多设备（MeshDevice）之间的 Tensor 复制与广播机制。在 DRAM 和 L1 缓冲类型下，TILE 和 ROW_MAJOR 布局的各种形状配置全部通过。
- **输出日志片段**：
```
0.00s call     tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_replicate[silicon_arch_name=wormhole_b0-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-shape=(1, 1, 16, 32)-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
======================================= short test summary info ========================================
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_replicate[silicon_arch_name=wormhole_b0-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-shape=(1, 1, 32, 128)-device_params={'dispatch_core_axis': DispatchCoreAxis.ROW}]
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_replicate[silicon_arch_name=wormhole_b0-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.TILE-shape=(1, 1, 32, 128)-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
... (其余测试通过日志省略) ...
PASSED tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_replicate[silicon_arch_name=wormhole_b0-memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)-layout=Layout.ROW_MAJOR-shape=(1, 1, 16, 32)-device_params={'dispatch_core_axis': DispatchCoreAxis.COL}]
========================================= 16 passed in 17.13s ==========================================
```

---

### 测试 4.3：多设备一元算子（GELU）

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_single_op_unary -v
```

---

### 测试 4.4：多设备二元算子（Add）

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_single_op_binary -v
```

---

### 测试 4.5：数据并行 MatMul

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_data_parallel_matmul_op -v
```

---

### 测试 4.6：多算子链式执行

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_multi_op -v
```

---

### 测试 4.7：Galaxy 8×4 2D 分片 MatMul

- **源码**：`tests/ttnn/distributed/test_multidevice_TG.py`

使用 Llama3-70B/405B 的 FF1/FF2 形状在 8×4 Galaxy mesh 上执行 2D 分片矩阵乘法，验证 PCC >= 0.99。

```bash
pytest tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (12 passed in 191.59s)
- **状态说明**：2D 分片矩阵乘法测试顺利通过，验证了 Llama3-70B 和 Llama3-405B 的 FF1/FF2 在 8x4 Grid 网格上的 prefill 及 decode 分片计算，所有 12 个测试用例全部通过。
- **输出日志片段**：
```
===================== short test summary info =====================
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-70B_decode_FF1-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-70B_decode_FF2-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-70B_prefill_seq512_FF1-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-70B_prefill_seq512_FF2-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-405B_decode_FF1-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-405B_decode_FF2-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-405B_prefill_seq128_FF1-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-405B_prefill_seq128_FF2-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-405B_prefill_seq256_FF1-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-405B_prefill_seq256_FF2-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-405B_prefill_seq512_FF1-8x4_grid]
PASSED tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_matmul_2d_fracture[silicon_arch_name=wormhole_b0-Llama3-405B_prefill_seq512_FF2-8x4_grid]
================= 12 passed in 191.59s (0:03:11) ==================
```

---

### 测试 4.8：Galaxy 8×4 逐元素运算

```bash
# 2D 分片逐元素乘法
pytest tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_eltwise_mul_2d_fracture -v

# 残差加法
pytest tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_eltwise_add -v
```

---

### 测试 4.9：Galaxy 注意力相关 MatMul

```bash
pytest tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_attn_matmul -v
```

---

### 测试 4.10：Galaxy NLP Create Heads

```bash
pytest tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_nlp_create_heads_decode -v
```

---

### 测试 4.11：Galaxy Rotary MatMul

```bash
pytest tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_rotary_matmul -v
```

---

## 层级 5：Trace 程序追踪与回放

> **目的**：验证 Galaxy 上的 trace capture 和 replay 功能，测试程序能否被正确捕获和重复执行。

### 测试 5.1：Galaxy 单 Trace 捕获与回放

- **源码**：`tests/ttnn/unit_tests/base_functionality/test_multi_device_trace_TG.py`

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_trace_TG.py::test_multi_device_single_trace -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED（7 个参数组合全部通过）
- **状态说明**：在 8x4 Galaxy 上的单 Trace 捕获与回放测试全部通过，覆盖 `enable_multi_cq=True/False`、多种输入 shape（`(1,1,32,32)`、`(1,1,256,256)`、`(1,1,512,512)`、`(1,3,32,32)`、`(1,3,128,128)`、`(1,3,512,512)`）共 7 种参数组合，每次均完成 trace capture、多轮回放（Send → Execute → Read Back）验证。
- **输出日志片段**：
```
PASSED test_multi_device_single_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 1, 32, 32)]
PASSED test_multi_device_single_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 1, 256, 256)]
PASSED test_multi_device_single_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 1, 512, 512)]
PASSED test_multi_device_single_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 3, 32, 32)]
PASSED test_multi_device_single_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 3, 128, 128)]
PASSED test_multi_device_single_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 3, 512, 512)]
PASSED test_multi_device_single_trace[...enable_multi_cq=False-8x4_grid-shape=(1, 1, 512, 512)]
```

---

### 测试 5.2：Galaxy 多 Trace 捕获与回放

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_trace_TG.py::test_multi_device_multi_trace -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED（6 个参数组合全部通过）
- **状态说明**：多 Trace 捕获与回放测试全部通过，验证了在 8x4 mesh 上同时捕获 3 条不同 trace（Trace 0/1/2）并串行回放的能力，覆盖 `enable_multi_cq=True/False`、多种 shape 共 6 种参数组合，每次均完成 3 条 trace 的完整 Capture → Execute → Read Back 循环。
- **输出日志片段**：
```
PASSED test_multi_device_multi_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 1, 32, 32)]
PASSED test_multi_device_multi_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 1, 256, 256)]
PASSED test_multi_device_multi_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 1, 512, 512)]
PASSED test_multi_device_multi_trace[...enable_multi_cq=True-8x4_grid-shape=(1, 3, 32, 32)]
PASSED test_multi_device_multi_trace[...enable_multi_cq=False-8x4_grid-shape=(1, 1, 256, 256)]
PASSED test_multi_device_multi_trace[...enable_multi_cq=False-8x4_grid-shape=(1, 1, 512, 512)]
```

> **注**：测试 5.1 和 5.2 合并一次运行，共 **12 passed in 约 128s**。

---

### 测试 5.3：C++ MeshTrace 4×8 Sweep 测试

- **源码**：`tests/tt_metal/distributed/test_mesh_trace.cpp`
- **二进制**：`build/test/tt_metal/distributed/distributed_unit_tests`

```bash
./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="MeshTraceSweepTest4x8*"
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED（5 tests in 90139ms）
- **状态说明**：C++ MeshTrace 4×8 Sweep 测试全部通过，覆盖 5 种随机种子参数组合（Sweep/0 ~ Sweep/4），依次完成 trace capture 和多轮 replay，每轮测试结束后均打印 Seed 值和设备频率同步信息，所有 32 张卡的频率均约为 0.9855 GHz。
- **输出日志片段**：
```
[ RUN      ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/0
[       OK ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/0 (21597 ms)
[ RUN      ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/1
[       OK ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/1 (10565 ms)
[ RUN      ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/2
[       OK ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/2 (20330 ms)
[ RUN      ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/3
[       OK ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/3 (21063 ms)
[ RUN      ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/4
[       OK ] MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8.Sweep/4 (16583 ms)
[----------] 5 tests from MeshTraceSweepTest4x8Tests/MeshTraceSweepTest4x8 (90139 ms total)
[==========] 5 tests from 1 test suite ran. (90139 ms total)
[  PASSED  ] 5 tests.
```

---

## 层级 6：集合通信 CCL 与大模型融合算子

> **目的**：验证 Galaxy 上的 all-gather、reduce-scatter 等核心通信算子，特别是 6U 专属的环形拓扑 (`FABRIC_1D_RING`) 融合通信算子在计算/通信并发下的稳定性。

### 测试 6.1：All-Reduce（Galaxy 32 卡）

- **源码**：`tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py`

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py::test_all_reduce -v -k "bfloat8_b" --timeout 1500
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (9 passed, 9 deselected, 1 warning in 32.83s)
- **状态说明**：使用 `bfloat8_b` 数据类型的 All-Reduce 测试全部通过，覆盖了 `FABRIC_1D` 和 `FABRIC_1D_RING` 拓扑、`cluster_axis=0/1`、`num_links=1/3/4`、多种输出形状 (1280/2048/3584/16384) 的参数组合，每组均执行 100 次迭代（10 次 warmup）。
- **输出日志片段**：
```
PASSED test_all_reduce[...mesh_device=(8, 4)...FABRIC_1D_RING...trace_mode=True...bfloat8_b...output_shape=[1, 1, 32, 16384]...cluster_axis=1...num_links=3...]
PASSED test_all_reduce[...mesh_device=(8, 4)...FABRIC_1D_RING...trace_mode=True...bfloat8_b...output_shape=[1, 1, 32, 1280]...cluster_axis=1...num_links=1...]
... (共 9 个参数组合全部通过) ...
========= 9 passed, 9 deselected, 1 warning in 32.83s =========
```

---

### 测试 6.2：All-Reduce Loopback（Galaxy 32 卡）

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py::test_all_reduce_loopback -v --timeout 600
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (3 passed, 1 warning in 8.04s)
- **状态说明**：All-Reduce Loopback 测试通过，在 `FABRIC_1D`（Trace 模式，100 次迭代）下，覆盖了 `cluster_axis=0`（output_shape=2048）和 `cluster_axis=1`（output_shape=1280 和 3584）共 3 种参数组合，均成功回环验证。
- **输出日志片段**：
```
PASSED test_all_reduce_loopback[...FABRIC_1D...trace_mode=True...bfloat8_b...output_shape=[1, 1, 32, 1280]...cluster_axis=1...num_links=1...]
PASSED test_all_reduce_loopback[...FABRIC_1D...trace_mode=True...bfloat8_b...output_shape=[1, 1, 32, 3584]...cluster_axis=1...num_links=1...]
PASSED test_all_reduce_loopback[...FABRIC_1D...trace_mode=True...bfloat8_b...output_shape=[1, 1, 32, 2048]...cluster_axis=0...num_links=1...]
========= 3 passed, 1 warning in 8.04s =========
```

---

### 测试 6.3：6U RMS Norm + AllGather 融合（Ring 拓扑）

- **源码**：`tests/ttnn/unit_tests/operations/ccl/test_minimals.py`

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_6u_trace_rms_fuse -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **状态说明**：6U RMS Norm + AllGather 融合（Ring 拓扑, `FABRIC_1D_RING`, num_links=1）测试通过，在 4 设备子网格上执行 200 次迭代（20 次 warmup），`elements_per_batch=8192`，`fused_add=True`，Trace 模式验证成功。

---

### 测试 6.4：6U DeepSeek RMS 融合

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_6u_trace_rms_fuse_deepseek -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **状态说明**：6U DeepSeek RMS 融合（`FABRIC_1D_RING`, num_links=1）测试通过，在 8 设备子网格上执行 200 次迭代（20 次 warmup），`elements_per_batch=7168`，`fused_add=True`，Trace 模式验证成功。

---

### 测试 6.5：6U Qwen RMS 融合

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_6u_trace_rms_fuse_qwen -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **状态说明**：6U Qwen RMS 融合（`FABRIC_1D_RING`, num_links=1）测试通过，在 4 设备子网格上执行 200 次迭代（20 次 warmup），`elements_per_batch=5120`，`fused_add=True`，Trace 模式验证成功。

> **注**：以上测试 6.3/6.4/6.5 合并一次运行，共 **3 passed in 7.55s**。

---

### 测试 6.6：6U Reduce-Scatter（带 Trace）

- **源码**：`tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_async_6U.py`

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_async_6U.py::test_fabric_reduce_scatter_tg_trace_6u -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (2 passed)
- **状态说明**：6U Reduce-Scatter（带 Trace）测试通过，覆盖 `Topology.Ring` 下 `num_links=3` 和 `num_links=4` 两种配置，均使用 `FABRIC_1D_RING`、Trace 捕获与回放，验证成功。
- **输出日志片段**：
```
PASSED test_fabric_reduce_scatter_tg_trace_6u[...topology=Topology.Ring-num_links=3...trace_mode=True...FABRIC_1D_RING]
PASSED test_fabric_reduce_scatter_tg_trace_6u[...topology=Topology.Ring-num_links=4...trace_mode=True...FABRIC_1D_RING]
```

---

### 测试 6.7：6U Reduce-Scatter（无 Trace）

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_async_6U.py::test_fabric_reduce_scatter_tg_no_trace_6u -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (4 passed)
- **状态说明**：6U Reduce-Scatter（无 Trace）测试通过，覆盖 `Topology.Ring`（3/4 links）和 `Topology.Linear`（3/4 links）共 4 种配置，均使用 `FABRIC_1D_RING` 无 Trace 模式验证成功。
- **输出日志片段**：
```
PASSED test_fabric_reduce_scatter_tg_no_trace_6u[...topology=Topology.Ring-num_links=3...trace_mode=False...FABRIC_1D_RING]
PASSED test_fabric_reduce_scatter_tg_no_trace_6u[...topology=Topology.Ring-num_links=4...trace_mode=False...FABRIC_1D_RING]
PASSED test_fabric_reduce_scatter_tg_no_trace_6u[...topology=Topology.Linear-num_links=3...trace_mode=False...FABRIC_1D_RING]
PASSED test_fabric_reduce_scatter_tg_no_trace_6u[...topology=Topology.Linear-num_links=4...trace_mode=False...FABRIC_1D_RING]
========= 6 passed, 1 warning in 48.56s =========
```

---

### 测试 6.8：RMS Norm 融合功能测试（通用）

```bash
# DeepSeek 版本
pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_rms_fuse_deepseek -v

# 通用版本
pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_rms_fuse -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED（多个参数组合全部通过）
- **状态说明**：`test_rms_fuse_deepseek` 和 `test_rms_fuse` 功能测试（`Topology.Linear`，`FABRIC_1D`，num_links=1，num_iters=20）全部通过。`test_rms_fuse` 覆盖了 `BFLOAT16`/`BFLOAT8_B` 输入 dtype、`use_noc1_only=True/False`、`fused_add=True/False`、`output_shard_grid=None` 和指定 shard grid 等多种参数组合（共 16 个测试）。

---

### 测试 6.9：6U QKV All-Reduce Fused Create Heads（Ring 拓扑）

- **源码**：`tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py`

**功能说明**：
专门验证 6U Galaxy 的 QKV 融合通信，模拟 Llama 3 大模型在 32 卡下的算子负载，配置 `FABRIC_1D_RING` 和 3 links。

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py::test_all_reduce_qkv_heads_fuse_perf_6U -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **状态说明**：6U QKV All-Reduce Fused Create Heads（`FABRIC_1D_RING`, 3 links）测试通过，在 32 卡下模拟 Llama 3 大模型算子负载，成功完成 Trace 捕获与回放，数值精度验证通过。

---

### 测试 6.10：6U Reduce-Scatter Fused Create Heads（Ring 拓扑）

- **源码**：`tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_create_heads_async_TG.py`

**功能说明**：
在高并发的 6U 环形拓扑 (`FABRIC_1D_RING`, 4 links) 下执行 Fused Reduce-Scatter + Create Heads 捕获与 Trace 重放。

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_create_heads_async_TG.py::test_rs_create_heads_6u_trace -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED
- **状态说明**：6U Fused Reduce-Scatter + Create Heads（`FABRIC_1D_RING`, 4 links）在高并发环形拓扑下的 Trace 捕获与回放测试通过，验证了计算/通信并发的稳定性。

> **注**：以上测试 6.8/6.9/6.10 合并一次运行，共 **50 passed in 115.71s**。

---

### 测试 6.11：6U Line/Ring All-Gather Llama

- **源码**：`tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py`

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_gather_6u_llama -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (5 passed)
- **状态说明**：6U Line/Ring All-Gather Llama 测试通过，在 `FABRIC_1D_RING`（8x4 Grid，4 设备子网格，replication_factor=8，75 次迭代）下，覆盖 `sdpa`、`binary_mult`、`layernorm`、`sampling_values`、`sampling_indices` 共 5 种算子负载变体，全部通过。
- **输出日志片段**：
```
PASSED test_all_gather_6u_llama[...FABRIC_1D_RING...replication_factor=8-sdpa...num_iters=75...]
PASSED test_all_gather_6u_llama[...FABRIC_1D_RING...replication_factor=8-binary_mult...num_iters=75...]
PASSED test_all_gather_6u_llama[...FABRIC_1D_RING...replication_factor=8-layernorm...num_iters=75...]
PASSED test_all_gather_6u_llama[...FABRIC_1D_RING...replication_factor=8-sampling_values...num_iters=75...]
PASSED test_all_gather_6u_llama[...FABRIC_1D_RING...replication_factor=8-sampling_indices...num_iters=75...]
```

---

### 测试 6.12：6U Line/Ring All-Reduce Llama

- **源码**：`tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py`

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py::test_all_reduce_6U_llama -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (7 passed)
- **状态说明**：6U Line/Ring All-Reduce Llama 测试通过，在 `FABRIC_1D_RING`（8x4 mesh，Trace 模式，75 次迭代）下，覆盖 Llama 和 Qwen 系列的 `ff1`、`ff2`、`qkv`、`lm_head` 共 7 种算子变体，全部通过。
- **输出日志片段**：
```
PASSED test_all_reduce_6U_llama[...FABRIC_1D_RING...trace_mode=True...ff2_llama]
PASSED test_all_reduce_6U_llama[...FABRIC_1D_RING...trace_mode=True...ff2_qwen]
PASSED test_all_reduce_6U_llama[...FABRIC_1D_RING...trace_mode=True...qkv]
PASSED test_all_reduce_6U_llama[...FABRIC_1D_RING...trace_mode=True...ff1_llama]
PASSED test_all_reduce_6U_llama[...FABRIC_1D_RING...trace_mode=True...ff1_qwen]
PASSED test_all_reduce_6U_llama[...FABRIC_1D_RING...trace_mode=True...lm_head_llama]
PASSED test_all_reduce_6U_llama[...FABRIC_1D_RING...trace_mode=True...lm_head_qwen]
```

---

### 测试 6.13：6U Concat Heads 前 Concat 融合 (Trace)

- **源码**：`tests/ttnn/unit_tests/operations/ccl/test_minimals.py`

```bash
pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py::test_concat_fuse_6u -v
```

**运行结果**： [未通过 ❌]
- **测试状态**：FAILED (1 failed, 12 passed in 31.68s)
- **失败原因**：在 `Topology.Ring`、`num_links=4`、`trace_mode=True`、`dim=1`、`Layout.ROW_MAJOR`、`output_shape=[1, 32, 32, 128]` 的参数组合下，分布式张量重组时触发断言失败。错误信息：`TT_FATAL @ distributed_tensor.cpp:239: distribution_shape_.dims() == 1 || chunks.size() == sharded_mesh_size`，即分布维度与 sharded mesh 尺寸不匹配，疑似 ROW_MAJOR 布局在多 link Ring 拓扑下的分块逻辑存在边界 Bug。其余 12 个参数组合（包括不同 dim、layout、num_links=1/2/3 等）均通过。
- **输出日志片段**：
```
FAILED test_concat_fuse_6u[...FABRIC_1D_RING...trace_mode=True...num_links=4...output_shape=[1, 32, 32, 128]-dim=1-layout=Layout.ROW_MAJOR...HEIGHT_SHARDED]
RuntimeError: TT_FATAL @ /root/tt-metal/ttnn/core/distributed/distributed_tensor.cpp:239:
  distribution_shape_.dims() == 1 || chunks.size() == sharded_mesh_size
========= 1 failed, 12 passed, 1 warning in 31.68s =========
```

---

## 层级 7：Nightly 与长期性能压力测试

> **目的**：通过长周期、重负载以及大 Packet 的 Nightly 算子，验证 Galaxy 32卡系统的长期电气稳定性和性能一致性，暴露可能因瞬时高功耗跌压引起的 Hang 机故障。

### 测试 7.1：Nightly 6U 1D Matmul with Reduce-Scatter 性能测试

- **源码**：`tests/ttnn/nightly/unit_tests/operations/matmul/test_rs_matmul_1d_gather_in0.py`

**功能说明**：
高难度的 fused 1D matmul + reduce-scatter 大参数量压测，专门在 6U 环形 Fabric (`FABRIC_1D_RING`) 上测试，验证性能一致性（PCC）并跑足 50 轮迭代。

```bash
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_rs_matmul_1d_gather_in0.py::test_6U_matmul_1d_ring_llama_with_rs_perf -v
```

**运行结果**： [未通过 ❌]
- **测试状态**：FAILED
- **失败原因**：测试代码存在 Bug（`NameError: name 'device' is not defined`）。在 `run_multi_core_matmul_1d` 函数第 445 行，变量 `device` 未定义，应为 `mesh_device`。这是测试文件自身的代码问题，与硬件无关。
- **输出日志片段**：
```
>           assert (
                False
            ), "Please set HF_MODEL to a HuggingFace name e.g. meta-llama/Llama-3.1-8B-Instruct or LLAMA_DIR to a Meta-style checkpoint directory"
E           AssertionError: ...

# 使用 LLAMA_DIR 后出现第二个错误：
>           compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                ...
            )
E           NameError: name 'device' is not defined

tests/ttnn/nightly/unit_tests/operations/matmul/test_rs_matmul_1d_gather_in0.py:445: NameError
======================== 1 failed, 1 warning in 2.98s =========================
```

---

### 测试 7.2：Nightly CCL + MatMul Async

- **源码**：`tests/nightly/tg/ccl/test_minimal_all_gather_matmul_async_nightly.py`

**功能说明**：
在 8 设备子网格（Submesh）下运行异步 All-Gather 与矩阵乘法的融合流水线，使用 `FABRIC_1D_RING` 并强制加载 Trace。

```bash
pytest tests/nightly/tg/ccl/test_minimal_all_gather_matmul_async_nightly.py::test_all_gather_async -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (1 passed in 15.24s)
- **功能验证**：在 8 设备子网格（Submesh）下，使用 `FABRIC_1D_RING`（3 links）成功完成异步 All-Gather + MatMul 融合流水线的 Trace 捕获与回放，共执行 10 轮迭代，PCC 持续稳定在 0.9999934649。
- **输出日志片段**：
```
2026-06-11 16:12:22.124 | INFO | run_all_gather_impl:264 - Done capturing trace
2026-06-11 16:12:22.129 | INFO | run_all_gather_impl:273 - Done executing trace
2026-06-11 16:12:22.424 | INFO | run_all_gather_impl:294 - Max ATOL Delta: 1.0, Max RTOL Delta: 14784.0, PCC: 0.9999934649036921, iteration 0
2026-06-11 16:12:23.129 | INFO | run_all_gather_impl:304 - Max ATOL Delta: 0.0, Max RTOL Delta: 0.0, PCC: 1.0, iteration 0
... (共 10 次迭代，PCC 均稳定在 0.9999934649) ...
PASSED
============================== 1 passed in 15.24s ==============================
```

---

### 测试 7.3：6U 子网格分区测试（test_mesh_partition_rm）

- **源码**：`tests/nightly/tg/ccl/test_mesh_partition_6U.py`

**功能说明**：
验证 32 卡物理集群被动态切割划分为不同逻辑形状的子网格后，进行集合通信时的数据一致性。

```bash
pytest tests/nightly/tg/ccl/test_mesh_partition_6U.py::test_mesh_partition_rm -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (6 passed in 11.92s)
- **功能验证**：验证了 32 卡物理集群（8x4 Grid）在不同子网格切割方式（mesh_axes=0/1/None）和不同通信使能方案（True/False）下，共 6 种参数组合的数据一致性全部通过，所有输出 Tensor 的 PCC = 1.0（完全精确）。
- **输出日志片段**：
```
2026-06-11 16:13:01.370 | INFO | run_mesh_partition_test:222 - tt_output per-device shape Shape([1, 1, 8, 7168])
2026-06-11 16:13:01.370 | INFO | run_mesh_partition_test:223 - golden shape torch.Size([8, 4, 8, 7168])
2026-06-11 16:13:01.372 | INFO | run_mesh_partition_test:226 - Output tensor 0 has result Max ATOL Delta: 0.0, Max RTOL Delta: 0.0, PCC: 1.0
2026-06-11 16:13:01.411 | INFO | run_mesh_partition_test:226 - Output tensor 1 has result Max ATOL Delta: 0.0, Max RTOL Delta: 0.0, PCC: 1.0
PASSED tests/nightly/tg/ccl/test_mesh_partition_6U.py::test_mesh_partition_rm[...mesh_axes0-0...True...]
PASSED tests/nightly/tg/ccl/test_mesh_partition_6U.py::test_mesh_partition_rm[...mesh_axes0-0...False...]
PASSED tests/nightly/tg/ccl/test_mesh_partition_6U.py::test_mesh_partition_rm[...mesh_axes0-1...True...]
PASSED tests/nightly/tg/ccl/test_mesh_partition_6U.py::test_mesh_partition_rm[...mesh_axes0-1...False...]
PASSED tests/nightly/tg/ccl/test_mesh_partition_6U.py::test_mesh_partition_rm[...mesh_axes0-None...True...]
PASSED tests/nightly/tg/ccl/test_mesh_partition_6U.py::test_mesh_partition_rm[...mesh_axes0-None...False...]
============================== 6 passed in 11.92s ==============================
```

---

### 测试 7.4：Galaxy 8x4 All-To-All Dispatch/Combine 调度测试

- **源码**：`tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py`
- **源码**：`tests/nightly/tg/ccl/test_all_to_all_combine_6U.py`

```bash
# 测试 8x4 上的 All-to-All 调度分发
pytest tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x4 -v

# 测试 8x4 上的 All-to-All 组合接收
pytest tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x4 -v
```

**运行结果**： [已通过 ✅]
- **测试状态**：PASSED (18 passed in 210.57s)
- **功能验证**：Galaxy 8x4 上的 All-to-All Dispatch 和 Combine 全面通过，覆盖了以下所有参数组合：
  - **内存布局**：`dram_in_l1_out`、`l1_in_dram_out`（Dispatch）；`dram_in_l1_out_axis0`、`l1_in_dram_out_axis1`（Combine）
  - **分布模式**：`dense`、`sparse`（Combine）
  - **Fabric 拓扑**：`fabric_1d_line`、`fabric_1d_ring`、`fabric_2d`
  - Dispatch 测试共 6 个（3 种 Fabric × 2 种内存布局），Combine 测试共 12 个（3 种 Fabric × 2 种内存布局 × 2 种模式），全部通过。
- **输出日志片段**：
```
PASSED tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x4[...l1_in_dram_out...fabric_1d_line]
PASSED tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x4[...l1_in_dram_out...fabric_1d_ring]
PASSED tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x4[...l1_in_dram_out...fabric_2d]
PASSED tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x4[...dram_in_l1_out...fabric_1d_line]
PASSED tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x4[...dram_in_l1_out...fabric_1d_ring]
PASSED tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x4[...dram_in_l1_out...fabric_2d]
PASSED tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x4[...dense...fabric_2d]
PASSED tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x4[...dense...fabric_1d_line]
PASSED tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x4[...dense...fabric_1d_ring]
PASSED tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x4[...sparse...fabric_2d]
PASSED tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x4[...sparse...fabric_1d_line]
PASSED tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x4[...sparse...fabric_1d_ring]
... (共 18 个测试全部通过) ...
======================== 18 passed in 210.57s (0:03:30) ========================
```

---

## 快速健康检查流程

以下是推荐的**最小测试集**，用于快速验证系统基本健康状态（约 20 分钟）：

```bash
# ==============================
# 步骤 1：物理拓扑与链路自检（~1 分钟）
# ==============================
./build/test/tt_metal/tt_fabric/test_system_health
./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.PrintHostTopology"

# ==============================
# 步骤 1.5：Metal CQ / Program / Buffer 基础 API 验证（~3 分钟）
# ==============================
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="UnitMeshCQSingleCardFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="UnitMeshCQSingleCardProgramFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="UnitMeshCQSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"

# ==============================
# 步骤 2：Fabric 数据通路基本检查（~3 分钟）
# ==============================
./build/test/tt_metal/tt_fabric/fabric_smoke_tests
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

# ==============================
# 步骤 3：MeshDevice 打开与关闭（~2 分钟）
# ==============================
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_open_close_galaxy_mesh -v

# ==============================
# 步骤 4：基本计算验证（~3 分钟）
# ==============================
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py -k "single_op_unary or single_op_binary or data_parallel_matmul" -v

# ==============================
# 步骤 5：6U 专属 CCL 融合算子快速检查（~5 分钟）
# ==============================
pytest tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py::test_all_reduce_qkv_heads_fuse_perf_6U -v
```

---

## 完整测试流程

建议用于全面验收硬件状态（约 1-2 小时）：

```bash
echo "============ 层级 1：系统健康与拓扑发现 ============"
./build/test/tt_metal/tt_fabric/test_system_health
./build/test/tt_metal/tt_fabric/test_physical_discovery
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardProgramFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"
# 以太网微基准（耗时较长，按需执行）：
# pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py
# pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py

echo "============ 层级 2：Fabric 通路与基准 ============"
./build/test/tt_metal/tt_fabric/fabric_smoke_tests
./build/test/tt_metal/tt_fabric/test_addrgen_write
./build/test/tt_metal/tt_fabric/bench_unicast
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

echo "============ 层级 3：MeshDevice 初始化 ============"
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py::test_multi_device_open_close_galaxy_mesh -v
./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="MeshDeviceInitTest.*:MeshDeviceTest.CheckFabricNodeIds"

echo "============ 层级 4：多设备基本计算 ============"
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device.py -k "not all_gather" -v
pytest tests/ttnn/distributed/test_multidevice_TG.py -k "matmul_2d_fracture or eltwise or attn_matmul or rotary or nlp_create_heads" -v

echo "============ 层级 5：Trace 回放 ============"
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_trace_TG.py -v
./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="MeshTraceSweepTest4x8*"

echo "============ 层级 6：CCL 与 6U 大模型融合通信 ============"
pytest tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py::test_all_reduce -k "bfloat8_b" -v --timeout 1500
pytest tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_async_6U.py -v
pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py -k "6u_trace_rms or concat_fuse_6u" -v
pytest tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py::test_all_reduce_qkv_heads_fuse_perf_6U -v
pytest tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_create_heads_async_TG.py::test_rs_create_heads_6u_trace -v
pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py -k "6u" -v

echo "============ 层级 7：Nightly 性能压测与并发调度 ============"
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_rs_matmul_1d_gather_in0.py::test_6U_matmul_1d_ring_llama_with_rs_perf -v
pytest tests/nightly/tg/ccl/test_minimal_all_gather_matmul_async_nightly.py::test_all_gather_async -v
pytest tests/nightly/tg/ccl/test_mesh_partition_6U.py::test_mesh_partition_rm -v
pytest tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x4 -v
pytest tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x4 -v
```

---

## 故障排查

### 问题 1：链路 DOWN
**排查步骤**：
1. 运行 `./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.PrintHostTopology"` 查看是否由于布线或 Host 间互联故障引起。
2. 宿主机上进行 warm reset:
   ```bash
   source ~/.tenstorrent-venv/bin/activate
   tt-smi -r
   ```
3. 若报错依然存在，运行 `tt-smi -glx_reset` 彻底重置 UBB 状态。

### 问题 2：Fabric 数据通路异常或 Hang
**排查步骤**：
1. 先运行基础的单播 smoke: `fabric_smoke_tests`。
2. 运行 `test_addrgen_write` 排除多播寻址寄存器配置故障。
3. 检查遥测数据：`test_bandwidth_telemetry_validation` 是否能正确监控流量状态。

### 问题 3：CCL 大模型融合算子超时/Hang
**排查步骤**：
1. 观察是否所有普通 CCL 测试（如 Level 6.1）通过，但融合算子（Level 6.9/6.10）超时。如果是，这通常由于 NOC 拥塞或者 subdevice 边界 semaphore 耗尽。
2. 检查当前是否使能了正确的 `fabric_config`。6U 系统在大模型并行下必须开启 `ttnn.FabricConfig.FABRIC_1D_RING`（环形拓扑），使用默认的 Linear 拓扑会造成严重的端口阻塞。
3. 尝试降低 iterations 参数以定位是在前几次循环还是长时间压力后出现泄露或超时。
