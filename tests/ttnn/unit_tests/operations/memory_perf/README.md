# Memory-Layout Performance Benchmark Suite

This suite answers a single question: **for a fixed operator and tensor shape, how much does
the choice of memory location (DRAM vs L1) and memory layout (interleaved vs sharded) change
throughput?**

It is the companion to the shape-efficiency demos in
[`tests/ttnn/unit_tests/operations/eltwise/test_sin_shape_efficiency.py`](../eltwise/test_sin_shape_efficiency.py)
and
[`tests/ttnn/unit_tests/operations/matmul/test_matmul_shape_efficiency.py`](../matmul/test_matmul_shape_efficiency.py).
Those demos hold memory config fixed and sweep shape; this suite does the opposite.

---

## Memory configurations swept

| Config name | Location | Layout |
|---|---|---|
| `dram_interleaved` | DRAM | Interleaved (default) |
| `l1_interleaved` | L1 | Interleaved |
| `l1_height_sharded` | L1 | Height-sharded across 8×8 core grid |
| `l1_width_sharded` | L1 | Width-sharded across 8×8 core grid |
| `l1_block_sharded` | L1 | Block-sharded across 8×8 core grid |

All five are applied to each operator where the operator supports them.
Two gates prevent a config from running:

1. **Geometry gate** (`build_memcfgs`): the per-core shard dimension must be
   ≥ 32 elements (one tile) and tile-aligned. For an 8×8 grid (64 cores),
   HEIGHT and WIDTH sharding require rows/cols ≥ 2048; BLOCK sharding requires
   rows and cols ≥ 256 each.
2. **Op gate**: some ops impose additional layout constraints (e.g., `ttnn.matmul`
   requires tensor B to be INTERLEAVED; SDPA requires all operands to be
   INTERLEAVED). Configs rejected by the op are logged and shown as `(skipped)`.

---

## Operators and shapes

| Test file | Operator | Shape | Mode | Metric |
|---|---|---|---|---|
| `test_sin_memory_perf.py` | `ttnn.sin` (SFPU unary) | 2048×2048 bf16 | — | GOps/s |
| `test_mul_memory_perf.py` | `ttnn.mul` (SFPU binary) | 2048×2048 bf16 | — | GOps/s |
| `test_softmax_memory_perf.py` | `ttnn.softmax` (row-wise, dim=-1) | 2048×2048 bf16 | — | GOps/s |
| `test_matmul_memory_perf.py` | `ttnn.matmul` (LoFi bf16), A & B | 2048×2048×2048 | prefill | TFLOPS |
| `test_matmul_memory_perf.py` | `ttnn.matmul` (LoFi bf16), A & B | 32×4096×4096 | decode | TFLOPS |
| `test_attention_memory_perf.py` | SDPA causal (Q/K/V) | B=1,H=8,S=512,D=128 | prefill | TFLOPS |
| `test_attention_memory_perf.py` | SDPA non-causal (Q/K/V) | B=1,H=8,S_q=32,S_kv=2048,D=128 | decode | TFLOPS |

**Why 2048×2048 for sin and mul?**
With an 8×8 = 64-core grid, HEIGHT/WIDTH sharding gives `2048 / 64 = 32` elements
per core — exactly one tile. Smaller shapes (e.g., 1024×1024) yield 16 elements,
which is below the tile minimum and are automatically skipped.

**Softmax (dim=-1, row-wise):** Softmax computes a per-row max, exp, and sum — the
reduction axis is the last (width) dimension. HEIGHT sharding assigns a contiguous block
of rows to each core: the entire reduction stays local, so the operation is fully
data-parallel. WIDTH sharding splits each row across all 64 cores, forcing a global
all-reduce for every row; this cross-core traffic makes width-sharded softmax *slower*
than DRAM-interleaved. BLOCK sharding splits rows across only the column dimension of
the core grid (8 cores per row), so the cross-core overhead is smaller but still
measurable.

**Matmul (A & B):** Both activation tensor A and weight tensor B share the same memcfg
in the sweep. `ttnn.matmul` requires B to be INTERLEAVED (op constraint), so all three
sharded configs are rejected regardless of geometry — only `dram_interleaved` and
`l1_interleaved` run. This tests whether moving both activations and weights into L1
together changes throughput.

**Matmul decode shape (M=32):** M=32 represents one tile row — a single padded
decode token or a batch of 32. HEIGHT sharding of A requires 32/64 < 32 elements
per core (geometry invalid); BLOCK sharding requires 32/8 = 4 rows per core (geometry
invalid); WIDTH-sharded B is rejected by the op. Only interleaved configs survive.

**Attention (Q/K/V):** Q, K, and V all share the same memcfg in the sweep.
SDPA (`ttnn.transformer.scaled_dot_product_attention`) requires all operands to be
INTERLEAVED, so all sharded configs are skipped unconditionally.

**Attention decode shape (S_q=32, S_kv=2048):** S_q=32 is the minimum tile-aligned
query length (representing one padded decode token). The Q shard shape (256, 128) fails
all three sharding geometry gates: HEIGHT → 4 rows/core, WIDTH → 2 cols/core, BLOCK →
32 rows but only 16 cols/core (must be ≥ 32 each). Only interleaved configs run, and
SDPA additionally requires INTERLEAVED operands in any case.

---

## Measured results (Wormhole N300, 2026-06-03)

### sin — 2048×2048 bf16

```
sin  shape=2048x2048  dtype=bfloat16
Takeaway: sin is bandwidth-bound — L1 sharded variants should outperform DRAM-interleaved
because L1 offers ~10x higher bandwidth per core.
------------------------------------------------------------
           memory config    ms/iter       GOps/s   % of best
------------------------------------------------------------
        dram_interleaved      0.094        44.55      62.4%
          l1_interleaved      0.059        71.41     100.0%
       l1_height_sharded      0.064        65.21      91.3%
        l1_width_sharded      0.064        65.30      91.4%
        l1_block_sharded      0.064        65.26      91.4%
------------------------------------------------------------
```

**Takeaway:** L1-interleaved wins at 71.4 GOps/s, 1.6× faster than DRAM-interleaved.
Sharded variants reach 91–92% of L1-interleaved — slightly slower because the shard
boundary handling adds a small overhead relative to fully-contiguous L1 placement.
The bandwidth gain from L1 is the dominant effect.

---

### mul — 2048×2048 bf16

```
mul  shape=2048x2048  dtype=bfloat16
Takeaway: mul reads 2 inputs + writes 1 output (3x bandwidth vs sin). L1 sharded layouts benefit more than DRAM-interleaved.
------------------------------------------------------------
           memory config    ms/iter       GOps/s   % of best
------------------------------------------------------------
        dram_interleaved      0.111        37.83      50.9%
          l1_interleaved      0.060        70.07      94.4%
       l1_height_sharded      0.056        74.26     100.0%
        l1_width_sharded      0.057        74.05      99.7%
        l1_block_sharded      0.057        74.13      99.8%
------------------------------------------------------------
```

**Takeaway:** DRAM-interleaved falls to only 51% of the best config — the 3× memory
traffic (read A, read B, write C) amplifies the DRAM bandwidth bottleneck. L1
height-sharded reaches 74.3 GOps/s (2× faster than DRAM). The sharded variants
edge out L1-interleaved because each core owns a private shard and avoids contention
on the shared L1 banking.

---

### softmax — 2048×2048 bf16 dim=-1 (row-wise)

```
softmax  shape=2048x2048  dim=-1  dtype=bfloat16
Takeaway: softmax is a row-wise reduction — height-sharding keeps each row on one core (local reduction). Width/block sharding may be rejected because the reduction crosses shard boundaries.
------------------------------------------------------------
           memory config    ms/iter       GOps/s   % of best
------------------------------------------------------------
        dram_interleaved      0.112        37.45      33.4%
          l1_interleaved      0.095        44.00      39.2%
       l1_height_sharded      0.037       112.23     100.0%
        l1_width_sharded      0.134        31.34      27.9%
        l1_block_sharded      0.069        60.82      54.2%
------------------------------------------------------------
```

**Takeaway:** `l1_height_sharded` is the clear winner at 112.2 GOps/s — **3× faster**
than DRAM-interleaved. Because softmax reduces along the last dimension (dim=-1),
height-sharding assigns a private set of complete rows to each core, so the exp,
max-subtract, and sum operations are entirely local with no cross-core communication.

`l1_width_sharded` is the *slowest* config (31.3 GOps/s), even slower than
DRAM-interleaved. Splitting each row across all 64 cores forces a global all-reduce
for every row's sum and max — the NoC traffic for this cross-core synchronization
outweighs the L1 bandwidth benefit.

`l1_block_sharded` lands between the two: rows are split across only 8 cores (the
x-dimension of the 8×8 grid), so the reduction overhead is smaller but still present.

The lesson: **for reduction-style ops, layout determines which configs are efficient
and which are catastrophically slow — not just bandwidth, but communication pattern.**

---

### matmul — 2048×2048×2048 bf16 LoFi (A and B same memcfg)

```
matmul  shape=2048x2048x2048  dtype=bfloat16  fidelity=LoFi  (A and B same memcfg)
Takeaway: both A and B share the same memcfg. Prefill is compute-bound so DRAM vs L1 matters less; sharded B is rejected by the op.
------------------------------------------------------------
           memory config    ms/iter       TFLOPS   % of best
------------------------------------------------------------
        dram_interleaved      0.279        61.66     100.0%
          l1_interleaved      0.476        36.10      58.5%
       l1_height_sharded  (skipped)
        l1_width_sharded  (skipped)
        l1_block_sharded  (skipped)
------------------------------------------------------------
```

**Takeaway:** Since B must be INTERLEAVED (op constraint), all three sharded configs are
rejected. DRAM-interleaved A+B is 1.7× faster than L1-interleaved A+B — the
auto-selected DRAM→DRAM program config uses a highly-optimized multicast tile schedule.
Placing both A and B in L1-interleaved forces a different program-config path that is
less optimized for this large prefill shape. The lesson: for matmul, program-config
compatibility matters as much as memory location.

---

### attention — B=1 H=8 S=512 D=128 causal SDPA (Q/K/V same memcfg)

```
SDPA prefill  B=1 H=8 S=512 D=128  is_causal=True  dtype=bfloat16  (Q/K/V same memcfg)
Takeaway: Q, K, V all share the same memcfg (all-DRAM vs all-L1). SDPA requires INTERLEAVED operands; sharded configs are skipped.
------------------------------------------------------------
           memory config    ms/iter       TFLOPS   % of best
------------------------------------------------------------
        dram_interleaved      0.079        13.64     100.0%
          l1_interleaved      0.079        13.63      99.9%
       l1_height_sharded  (skipped)
        l1_width_sharded  (skipped)
        l1_block_sharded  (skipped)
------------------------------------------------------------
```

**Takeaway:** SDPA prefill is compute-bound at S=512, D=128 — the QK^T and softmax
dominate. Moving all of Q, K, and V from DRAM to L1-interleaved makes no measurable
difference (within noise). Sharded configs are not supported by this SDPA variant;
those entries show as skipped.

---

### matmul decode — 32×4096×4096 bf16 LoFi (A and B same memcfg)

```
matmul decode  shape=32x4096x4096  dtype=bfloat16  fidelity=LoFi  (A and B same memcfg)
Takeaway: B=32 MB dominates bandwidth. Moving both A and B from DRAM to L1 shows the true memory benefit — only interleaved configs run for this tiny M.
------------------------------------------------------------
           memory config    ms/iter       TFLOPS   % of best
------------------------------------------------------------
        dram_interleaved      0.164         6.54      77.9%
          l1_interleaved      0.128         8.39     100.0%
       l1_height_sharded  (skipped)
        l1_width_sharded  (skipped)
        l1_block_sharded  (skipped)
------------------------------------------------------------
```

**Takeaway:** With both A and B moving to L1-interleaved, the 32 MB weight load is
served from L1 rather than DRAM, and throughput improves by 28% (8.39 vs 6.54 TFLOPS).
All sharded variants remain invalid: HEIGHT/BLOCK fail the geometry gate for M=32
(< 1 tile per core), and WIDTH-sharded B is rejected by the op. The key lesson:
**for decode-shaped matmuls, the bandwidth benefit of L1 is real — but only when the
weight tensor B also resides in L1, which requires the full weight to fit.**

---

### attention decode — B=1 H=8 S_q=32 S_kv=2048 D=128 SDPA (Q/K/V same memcfg)

```
SDPA decode  B=1 H=8 S_q=32 S_kv=2048 D=128  is_causal=False  dtype=bfloat16  (Q/K/V same memcfg)
Takeaway: KV = 8 MB dominates bandwidth. Moving Q/K/V from DRAM to L1 together eliminates the KV load cost — only interleaved configs run for this tiny S_q.
------------------------------------------------------------
           memory config    ms/iter       TFLOPS   % of best
------------------------------------------------------------
        dram_interleaved      0.104         2.57      88.1%
          l1_interleaved      0.092         2.92     100.0%
       l1_height_sharded  (skipped)
        l1_width_sharded  (skipped)
        l1_block_sharded  (skipped)
------------------------------------------------------------
```

**Takeaway:** With Q, K, and V all moving to L1-interleaved, the ~8 MB KV load is
served from L1 rather than DRAM, and throughput improves by 14% (2.92 vs 2.57 TFLOPS).
All sharding strategies remain invalid for the tiny Q shard shape (S_q=32 on an 8×8
grid), and SDPA requires INTERLEAVED operands in any case. The lesson: **in decode
mode, moving the KV cache into L1 yields a measurable speedup — but only if it fits.**

---

## How to run

All commands must be run inside the `ct_metal` Docker container (see the project
[CLAUDE.md](../../../../../CLAUDE.md)):

```sh
# Enter the container
./start_container.sh

# Or run a single command without entering
docker exec -it -w /root/tt-metal ct_metal bash -c "<command>"
```

### Run the full pytest suite

```sh
pytest -s tests/ttnn/unit_tests/operations/memory_perf/
```

### Run only the correctness gates (fast, no timing)

```sh
pytest -s tests/ttnn/unit_tests/operations/memory_perf/ -k correctness
```

### Run one operator's sweep

```sh
pytest -s tests/ttnn/unit_tests/operations/memory_perf/test_sin_memory_perf.py
```

### Run standalone (prints formatted table to stdout)

Each standalone script prints both prefill **and** decode benchmark tables.

```sh
python tests/ttnn/unit_tests/operations/memory_perf/test_sin_memory_perf.py
python tests/ttnn/unit_tests/operations/memory_perf/test_mul_memory_perf.py
python tests/ttnn/unit_tests/operations/memory_perf/test_softmax_memory_perf.py
python tests/ttnn/unit_tests/operations/memory_perf/test_matmul_memory_perf.py
python tests/ttnn/unit_tests/operations/memory_perf/test_attention_memory_perf.py
```

### Run only decode tests

```sh
pytest -s tests/ttnn/unit_tests/operations/memory_perf/ -k "decode"
```

### Select a single config

```sh
pytest -s tests/ttnn/unit_tests/operations/memory_perf/test_sin_memory_perf.py \
    -k "l1_height_sharded"
```

---

## File map

| File | Role |
|---|---|
| [`_memcfg_utils.py`](_memcfg_utils.py) | Shared timing, `build_memcfgs`, `MemcfgResultTable` |
| [`test_sin_memory_perf.py`](test_sin_memory_perf.py) | sin benchmark |
| [`test_mul_memory_perf.py`](test_mul_memory_perf.py) | mul benchmark |
| [`test_softmax_memory_perf.py`](test_softmax_memory_perf.py) | softmax benchmark |
| [`test_matmul_memory_perf.py`](test_matmul_memory_perf.py) | matmul benchmark |
| [`test_attention_memory_perf.py`](test_attention_memory_perf.py) | SDPA prefill benchmark |

---

## Methodology

Each benchmark:

1. **Constructs all five memory configs** via `build_memcfgs` in `_memcfg_utils.py`. Two
   layers of filtering apply:
   - *Geometry gate*: shard dimensions must be tile-aligned (≥ 32 elements and divisible
     by 32). Configs that fail this pre-check are returned as `None` without calling the
     TTNN API, and the test reports `pytest.skip` rather than failing.
   - *Op gate*: configs that pass geometry but are rejected by the op itself (e.g., matmul
     requires B to be INTERLEAVED, SDPA requires all operands to be INTERLEAVED) are caught
     and shown as `(skipped)` in the table.

2. **Times using trace capture** — the op is first compiled, then captured into two traces
   (warmup and main). Wall-time difference between the two execution calls cancels the
   fixed per-trace launch overhead, leaving only pure op cycles. Falls back to eager timing
   if trace capture is unavailable.

3. **Reports "% of best"** normalized to the fastest config in the table, so the numbers are
   meaningful on any Wormhole or Blackhole board without hard-coding a peak constant.

4. **All operands share the same memcfg** in the sweep (A and B for matmul; Q, K, and V
   for attention). This is the most direct test of the memory subsystem's effect: if an op
   supports a layout for all its inputs, does using it end-to-end help or hurt?
