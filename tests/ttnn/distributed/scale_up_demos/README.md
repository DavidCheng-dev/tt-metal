# MeshDevice Scale-Up Demos

A small, self-contained set of TT-NN demos that show how to scale one program from a
**single chip up to 8 chips** on a Wormhole Galaxy, going from the simplest op to a full
attention layer. Every demo:

1. runs the op across a `(1, N)` mesh for `N = 1, 2, 4, 8`,
2. checks the result against a **CPU/torch golden** with a PCC comparison, and
3. prints a **speedup table** so you can see the compute getting faster as chips are added.

These complement the prose guides
[`Programming_Mesh_of_Devices_with_TT-NN.md`](../../../../tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md)
and [`LLMs/llms.md`](../../../../tech_reports/LLMs/llms.md).

## The two ways we scale

| Strategy | What is split | What is copied | When to use |
|---|---|---|---|
| **Data parallel (DP)** | the input data (batch) | the weights | the model fits on one chip; you want more throughput |
| **Tensor parallel (TP)** | the weights / heads | the input data | the model (a weight or the heads) is too big for one chip |

All demos use **strong scaling**: the *total* problem size is fixed and sharded across the
mesh, so each chip does `1/N` of the work and the per-iteration wall time should drop
roughly `N x`. The mesh dispatches to all chips in lock-step, so this parallelism is
"free" once the data is distributed.

## The demos

| File | Op | Strategy | Distribution |
|---|---|---|---|
| `demo_01_sin.py` | `ttnn.sin` | DP | shard input on batch dim, gather back |
| `demo_02_add.py` | `ttnn.add` | DP | shard both inputs on batch dim |
| `demo_03_matmul.py` | `ttnn.matmul` | DP **and** TP | DP: shard activation, replicate weight. TP: replicate activation, shard weight on width |
| `demo_04_attention.py` | `ttnn.transformer.scaled_dot_product_attention` | DP **and** TP | DP: shard batch. TP: shard heads (heads are independent → no communication) |

`sin` and `add` are elementwise: they have no weights and no cross-device reduction, so
data parallel is the natural (and only sensible) fit. `matmul` and `attention` add the
tensor-parallel view on top.

## How to run

Everything runs inside the `ct_metal` container (see the repo `CLAUDE.md`).

Standalone (prints the full 1→8 speedup table — the best way to *see* scaling):

```sh
docker exec -it -w /root/tt-metal ct_metal bash -c "python tests/ttnn/distributed/scale_up_demos/demo_01_sin.py"
docker exec -it -w /root/tt-metal ct_metal bash -c "python tests/ttnn/distributed/scale_up_demos/demo_03_matmul.py"
```

As pytest (per-mesh-size correctness checks, good for CI):

```sh
docker exec -it -w /root/tt-metal ct_metal bash -c "pytest tests/ttnn/distributed/scale_up_demos/demo_04_attention.py -s"
```

The demos automatically use only the mesh sizes your system supports (`available_mesh_sizes`),
so they also run on a smaller box (e.g. an N300 only does `N = 1, 2`).

## 2D mesh shapes

The pytest path also covers **2D meshes** `(2, 4)` and `(4, 2)` via the `*_2d` tests, using
**genuine 2D parallelism**: one mesh axis does data-parallel sharding and the other does
tensor-parallel sharding (`ttnn.ShardTensor2dMesh` / `ttnn.ConcatMesh2dToTensor`).

| Demo | Mesh axis 0 (rows) | Mesh axis 1 (cols) |
|---|---|---|
| `sin` / `add` | shard batch | shard height (data parallel both ways — no weights) |
| `matmul` | data parallel (shard batch of activation, replicate weight) | tensor parallel (replicate activation, shard weight width) |
| `attention` | data parallel (shard batch) | tensor parallel (shard heads) |

Because the two axes carry different strategies, `(2, 4)` and `(4, 2)` are **not** equivalent:
e.g. for `matmul`/`attention`, `(2, 4)` is 2-way DP x 4-way TP while `(4, 2)` is 4-way DP x
2-way TP. These need an 8-chip machine; on fewer chips the cases auto-skip. Run them with:

```sh
docker exec -it -w /root/tt-metal ct_metal bash -c "pytest tests/ttnn/distributed/scale_up_demos/ -s -k 2d"
```

## Reading the speedup table

```
Demo 1: ttnn.sin (data parallel)
--------------------------------------------------------------------
 chips   time/iter (ms)   speedup  efficiency   throughput (Gelem/s)
--------------------------------------------------------------------
     1            0.200     1.00x    100.00%                  41.93
     2            0.097     2.06x    102.78%                  86.19
     4            0.053     3.78x     94.51%                 158.50
     8            0.026     7.82x     97.75%                 327.87
--------------------------------------------------------------------
```

- **time/iter** — mean device time for one op, measured with trace capture (host dispatch
  overhead excluded) so the comparison is fair.
- **speedup** — `time(1 chip) / time(N chips)`.
- **efficiency** — `speedup / N`. `100%` is perfect linear scaling; it falls as
  communication / dispatch overhead grows relative to the shrinking per-chip compute.
- **throughput** — total work per second (elements/s or FLOP/s); rises as time falls.

## Notes

- **Accuracy.** Results are bf16, compared to an fp32 CPU golden via PCC. Elementwise ops
  reach ~0.999. `matmul` accumulates a deep (K=1024–2048) reduction, so it uses a
  `WormholeComputeKernelConfig` with `fp32_dest_acc_en=True` to stay ≥0.99; attention uses
  exact-exp + fp32 accumulation and is checked at ≥0.98. These compute-kernel configs are
  the main precision knobs on TT hardware.
- **Why TP efficiency can taper.** As a sharded dimension gets small (e.g. the per-chip
  matmul width at 8 chips), fixed overhead becomes a larger fraction of each iteration, so
  efficiency drops. The demos size the TP problems large enough that this stays mild.
- **Production tensor parallel keeps results on-device.** Here the TP demos gather
  per-chip slices on the host for a simple correctness check. Real models instead stitch
  the slices back on-device with a collective (`ttnn.experimental.all_gather_async`); see
  `tests/ttnn/unit_tests/operations/ccl/test_ag_rs_llama_prefill_TG.py` for the full
  fabric + global-semaphore setup that requires.

## Test results

Measured on a Wormhole Galaxy (32 chips, 8 used here), 2026-06-03.
All 32 pytest cases passed (`32 passed in 49.53s`).

### Demo 1 — `ttnn.sin` (data parallel)

```
Demo 1: ttnn.sin (data parallel)
--------------------------------------------------------------------
 chips   time/iter (ms)   speedup  efficiency   throughput (Gelem/s)
--------------------------------------------------------------------
     1            0.206     1.00x    100.00%                  40.74
     2            0.091     2.26x    113.05%                  92.12
     4            0.053     3.91x     97.76%                 159.32
     8            0.025     8.15x    101.89%                 332.11
--------------------------------------------------------------------

Demo 1b: ttnn.sin (2D mesh, data parallel on both axes: batch x height)
--------------------------------------------------------------------
  mesh   time/iter (ms)   speedup  efficiency   throughput (Gelem/s)
--------------------------------------------------------------------
   1x1            0.197     1.00x    100.00%                  42.59
   4x2            0.025     7.76x     96.98%                 330.44
   2x4            0.026     7.69x     96.16%                 327.66
--------------------------------------------------------------------
```

### Demo 2 — `ttnn.add` (data parallel)

```
Demo 2: ttnn.add (data parallel)
--------------------------------------------------------------------
 chips   time/iter (ms)   speedup  efficiency   throughput (Gelem/s)
--------------------------------------------------------------------
     1            0.219     1.00x    100.00%                  38.27
     2            0.110     1.99x     99.74%                  76.34
     4            0.060     3.67x     91.82%                 140.56
     8            0.030     7.40x     92.51%                 283.23
--------------------------------------------------------------------

Demo 2b: ttnn.add (2D mesh, data parallel on both axes: batch x height)
--------------------------------------------------------------------
  mesh   time/iter (ms)   speedup  efficiency   throughput (Gelem/s)
--------------------------------------------------------------------
   1x1            0.220     1.00x    100.00%                  38.11
   4x2            0.030     7.45x     93.10%                 283.84
   2x4            0.030     7.43x     92.88%                 283.18
--------------------------------------------------------------------
```

### Demo 3 — `ttnn.matmul` (data parallel + tensor parallel)

```
Demo 3a: ttnn.matmul (data parallel)
--------------------------------------------------------------------
 chips   time/iter (ms)   speedup  efficiency   throughput (GFLOP/s)
--------------------------------------------------------------------
     1            0.320     1.00x    100.00%               26828.91
     2            0.160     1.99x     99.74%               53520.46
     4            0.081     3.95x     98.79%              106022.07
     8            0.041     7.78x     97.19%              208604.20
--------------------------------------------------------------------

Demo 3b: ttnn.matmul (tensor parallel)
--------------------------------------------------------------------
 chips   time/iter (ms)   speedup  efficiency   throughput (GFLOP/s)
--------------------------------------------------------------------
     1            0.665     1.00x    100.00%               51692.81
     2            0.324     2.05x    102.48%              105945.30
     4            0.168     3.96x     98.98%              204666.25
     8            0.106     6.29x     78.63%              325188.76
--------------------------------------------------------------------

Demo 3c: ttnn.matmul (2D mesh, data parallel x tensor parallel)
--------------------------------------------------------------------
  mesh   time/iter (ms)   speedup  efficiency   throughput (GFLOP/s)
--------------------------------------------------------------------
   1x1            5.307     1.00x    100.00%               51799.63
   4x2            0.668     7.95x     99.37%              411774.32
   2x4            0.677     7.84x     97.95%              405890.85
--------------------------------------------------------------------
```

### Demo 4 — `ttnn.transformer.scaled_dot_product_attention` (data parallel + tensor parallel)

```
Demo 4a: attention (data parallel)
--------------------------------------------------------------------
 chips   time/iter (ms)   speedup  efficiency   throughput (GFLOP/s)
--------------------------------------------------------------------
     1            3.050     1.00x    100.00%               45057.33
     2            1.576     1.94x     96.77%               87206.90
     4            0.771     3.95x     98.85%              178149.32
     8            0.388     7.86x     98.30%              354319.45
--------------------------------------------------------------------

Demo 4b: attention (tensor parallel / head-parallel)
--------------------------------------------------------------------
 chips   time/iter (ms)   speedup  efficiency   throughput (GFLOP/s)
--------------------------------------------------------------------
     1            3.043     1.00x    100.00%               45162.42
     2            1.571     1.94x     96.86%               87484.46
     4            0.771     3.95x     98.64%              178198.71
     8            0.390     7.81x     97.57%              352515.00
--------------------------------------------------------------------

Demo 4c: attention (2D mesh, data parallel x tensor parallel / head-parallel)
--------------------------------------------------------------------
  mesh   time/iter (ms)   speedup  efficiency   throughput (GFLOP/s)
--------------------------------------------------------------------
   1x1            3.044     1.00x    100.00%               45154.36
   4x2            0.387     7.86x     98.28%              355022.90
   2x4            0.388     7.85x     98.18%              354665.06
--------------------------------------------------------------------
```

## Files

- `scale_up_common.py` — shared helpers: open a `(1, N)` mesh, trace-based timing,
  the PCC check, and the `SpeedupTable`. The demo files stay short and readable.
- `demo_01_sin.py`, `demo_02_add.py`, `demo_03_matmul.py`, `demo_04_attention.py` — the four demos.
