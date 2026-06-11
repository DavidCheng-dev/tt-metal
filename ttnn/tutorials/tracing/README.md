# TT-NN Trace Capture / Replay — Tutorial

This folder is a hands-on introduction to **trace capture & replay** in TT-NN, the
feature that lets you record a sequence of device operations once and then replay it
many times with almost no host involvement. It is one of the most important tools for
getting high throughput out of Tenstorrent hardware.

> **Two different "tracing" features — don't confuse them.**
> This tutorial is about the **execution trace** API (`ttnn.begin_trace_capture`,
> `ttnn.execute_trace`, …), a *performance* feature. It is **not** the same as
> `ttnn.tracer` (in `ttnn/ttnn/tracer.py`), which records an operation *graph* for
> visualization/debugging. If you came here looking for a picture of your op graph,
> you want `ttnn.tracer`, not this.

---

## 1. The problem tracing solves: host dispatch overhead

When you call a TT-NN op like `ttnn.matmul(a, b)`, the host (your Python process) does a
surprising amount of work *every single call*: it looks up the program, builds the
command stream, and pushes commands over the PCIe command queue to the device. For a
single big op this host time is hidden behind the device compute. But for workloads made
of **many small ops** — think a transformer decode step that emits one token at a time —
the device finishes each op so fast that it sits **idle waiting for the host** to dispatch
the next one. The host, not the math, becomes the bottleneck. This is *dispatch-bound*
execution.

```
Eager execution (host re-dispatches every op):

 host:  [dispatch op0][dispatch op1][dispatch op2] ...   <- the bottleneck
 dev:        [op0]        [op1]        [op2]              <- idle gaps between ops


Traced execution (host issues ONE "replay" command):

 host:  [execute_trace] ......................            <- almost nothing to do
 dev:   [op0][op1][op2][op0][op1][op2] ...                <- back-to-back, no gaps
```

A **trace** is a recording of the device-side commands for a whole op sequence, stored in
a dedicated DRAM region *on the device*. Replaying it with `ttnn.execute_trace` tells the
device "run that recorded command stream again" — the host issues one command instead of
hundreds, and the device runs the ops back-to-back.

---

## 2. The trace lifecycle (5 steps)

Every use of tracing follows the same shape:

1. **Open the device with a `trace_region_size`.** This reserves the on-device DRAM that
   will hold the recorded commands. It must be non-zero to trace anything.
   ```python
   device = ttnn.open_device(device_id=0, trace_region_size=16 << 20)  # 16 MB
   # or, for a multi-device mesh:
   mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 << 20)
   ```
2. **Warm-up / compile run.** Run the op chain once *before* capturing. Kernels are
   compiled on first use; capture records *dispatch*, not *compilation*, so the kernels
   must already exist.
3. **Capture.** Everything between `begin_trace_capture` and `end_trace_capture` is
   recorded instead of executed.
   ```python
   tid = ttnn.begin_trace_capture(device, cq_id=0)
   output = run_op_chain(input_dev)          # records, does not run
   ttnn.end_trace_capture(device, tid, cq_id=0)
   ```
4. **Execute (replay), as many times as you like.**
   ```python
   ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
   ```
5. **Release** the trace when done, then close the device.
   ```python
   ttnn.release_trace(device, tid)
   ```

The same five function calls work unchanged on both `Device` and `MeshDevice`.

---

## 3. The most important constraint: fixed inputs/outputs

A trace records the *exact* operations on the *exact* tensor buffers (addresses) it saw
during capture. On replay it re-runs those commands against **the same buffers**. That
has two consequences you must design around:

- **Pre-allocate your input/output tensors once, and feed new data by updating their
  contents in place** — not by creating new tensors. Use
  `ttnn.allocate_tensor_on_device(...)` for the buffers and
  `ttnn.copy_host_to_device_tensor(host_tensor, dev_tensor)` to load each new input.
- **Shapes, dtypes, layouts and addresses are baked in.** A trace captured for one shape
  cannot run a different shape; you capture a separate trace per shape you need.
- **Allocate all buffers before the first `begin_trace_capture`.** Allocating inside a
  live trace region is unsafe — those allocations will be overwritten on replay.

If you allocate a fresh tensor inside the captured region, the trace ties itself to
*that* allocation — replaying after it's been freed/reused is undefined. Keep allocations
outside the loop.

### When does tracing help?

| Situation | Tracing benefit |
|---|---|
| Many small ops, fixed shapes, run repeatedly (e.g. LLM decode) | **Large** — removes dispatch overhead |
| A few large, compute-bound ops | Small — host time was already hidden |
| Shapes change every call | Limited — you'd re-capture constantly |
| You run the sequence only once | None — capture cost isn't amortized |

### Sizing `trace_region_size`

It's the DRAM budget (in bytes) for recorded commands. Too small and capture fails; too
large and you waste DRAM that could hold tensors. Start generous (e.g. `16 << 20`) for a
tutorial and tune down for production. Real models keep per-model/per-device values — see
`models/tt_transformers/demo/trace_region_config.py` for examples.

---

## 4. The scripts

Run **inside the `ct_metal` container** (see the repo `CLAUDE.md`). From the repo root:

```sh
docker exec -it -w /root/tt-metal ct_metal bash -c "python ttnn/tutorials/tracing/01_trace_basics.py"
docker exec -it -w /root/tt-metal ct_metal bash -c "python ttnn/tutorials/tracing/02_trace_performance.py"
docker exec -it -w /root/tt-metal ct_metal bash -c "python ttnn/tutorials/tracing/03_trace_two_cq.py"
docker exec -it -w /root/tt-metal ct_metal bash -c "python ttnn/tutorials/tracing/04_trace_mesh_device.py"
docker exec -it -w /root/tt-metal ct_metal bash -c "python ttnn/tutorials/tracing/05_trace_determinism.py"
```

| Script | What it teaches |
|---|---|
| [`01_trace_basics.py`](01_trace_basics.py) | The minimal capture → execute → release lifecycle on a single device, with each step labelled. Start here. |
| [`02_trace_performance.py`](02_trace_performance.py) | Times the **same** op chain run eagerly vs. traced and prints the speedup, so you can *see* the dispatch overhead disappear. |
| [`03_trace_two_cq.py`](03_trace_two_cq.py) | **Advanced.** Uses two command queues to overlap host↔device data movement (CQ1) with traced compute (CQ0) — the pattern real model demos use. Also benchmarks single-CQ vs two-CQ pipelined latency so you can see when pipelining helps. |
| [`04_trace_mesh_device.py`](04_trace_mesh_device.py) | **Multi-device.** Shows that the same trace API works on a `MeshDevice` with `ShardTensorToMesh` / `ReplicateTensorToMesh` / `ConcatMeshToTensor` for distributed data-parallel inference. |
| [`05_trace_determinism.py`](05_trace_determinism.py) | **Determinism.** Verifies that replaying a trace twice with identical inputs produces bit-for-bit identical output on a single device — TT hardware is fully deterministic. Uses `torch.equal` (not PCC) to confirm. |

All scripts verify their traced output against an eager (golden) result so you can trust
that replay produces the same numbers as direct execution.

---

## 5. Measured results (Wormhole N150, single card)

Results measured on 2026-06-04 against the `main` branch on a Wormhole B0 system.

### Script 01 — basics
All 3 trace replays produce output matching the torch golden (PCC ≥ 0.99). ✓

### Script 02 — performance comparison

Chain of **40 small eltwise ops** (`gelu → relu → neg → add`, repeated 10×), 100 timed
iterations each.

| Mode   | Latency (ms/iter) | Throughput (iter/s) |
|--------|------------------:|--------------------:|
| Eager  |             2.075 |                 482 |
| Traced |             0.451 |               2,216 |
| **Speedup** | — | **4.60×** |

The traced output matches the eager golden (PCC ≥ 0.99). The speedup grows with more,
smaller ops: increasing `CHAIN_REPEATS` in the script will widen the gap further.

### Script 03 — 2-CQ pipelining

Chain of **32 small eltwise ops** (`gelu → relu → neg → add`, repeated 8×) on a 512×512
bfloat16 tensor. 8 warm-up + 32 timed iterations, each including a host↔device copy on both
sides.

| Mode | Latency (ms/iter) | Throughput (iter/s) |
|------|------------------:|--------------------:|
| Single CQ (serial) | 0.864 | 1,157 |
| Two CQ (pipelined) | 0.846 | 1,182 |
| **Speedup** | — | **1.02×** |

All outputs match golden (PCC ≥ 0.99). ✓

The modest speedup here is **expected**: for a 512×512 tensor (~512 KB per direction), the
PCIe transfer takes only ~0.1 ms while the compute chain takes ~0.75 ms. When compute
dominates, there is little idle time for the DMA to overlap with. The two-CQ advantage
grows with **larger tensors** or **shorter compute chains**, where data movement becomes a
bigger fraction of each iteration. In real LLM decode loops that move multi-MB KV-cache
slices, the pipelining speedup is typically 1.3–2×.

### Script 04 — MeshDevice (4-device mesh, 1×4)
3 replays of the main trace (ShardTensorToMesh) + 1 replay of the replicated-weight
trace, across 4 Wormhole devices. All outputs match golden (PCC ≥ 0.99). ✓

### Script 05 — bit-exact determinism (single device)
Replaying the same trace twice with identical inputs produces bit-for-bit identical
output across all tested shapes. ✓

---

## 6. Where the API lives

- Python entry points are exported in `ttnn/ttnn/__init__.py`.
- Bindings: `ttnn/cpp/ttnn-nanobind/operations/trace.cpp`; implementation:
  `ttnn/cpp/ttnn/operations/trace.cpp`.
- Canonical unit-test examples:
  - Single-device: `tests/ttnn/unit_tests/base_functionality/test_single_device_trace.py`
  - Multi-device: `tests/ttnn/unit_tests/base_functionality/test_multi_device_trace.py`
  - Bit-exact determinism: [`ttnn/tutorials/tracing/05_trace_determinism.py`](05_trace_determinism.py)
- 2-CQ production example: `models/experimental/functional_unet/tests/test_unet_trace.py`.

### Higher-level helpers
- `models/tt_dit/utils/tracing.py` wraps capture/replay in a reusable `Tracer` class and
  `@traced_function` decorator.
- `models/tt_transformers/demo/trace_region_config.py` has per-model/per-hardware
  `trace_region_size` values for reference.
