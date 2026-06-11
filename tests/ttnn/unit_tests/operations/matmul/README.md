# Matmul Shape-Efficiency Demo

`test_matmul_shape_efficiency.py` answers: **for a fixed operator and fidelity, how much does
tensor shape change throughput?**

It is the companion to the memory-layout benchmark in
[`tests/ttnn/unit_tests/operations/memory_perf/`](../memory_perf/) — that suite holds shape fixed
and sweeps memory config; this demo holds memory config fixed and sweeps shape.

---

## What is measured

Fidelity is fixed at **LoFi** (the highest bf16 matrix-engine rate on Wormhole) so the only
variable is shape.  Three case sets are run:

| Set | What changes | What is fixed | Question answered |
|---|---|---|---|
| A | M (activation rows) | K=N=512 | How badly do skinny / non-aligned matmuls waste engine cycles? |
| B | M=K=N (square) | — | How fast does throughput scale with total problem size? |
| C | M = seq\_len | K=N=2048 (d\_model) | What is the TFLOPS vs. sequence-length curve for prefill? |

TFLOPS is computed on the **real** (un-padded) shape, so tile-padding waste appears as a drop.

---

## Measured results (Wormhole N300, 2026-06-03)

### Set A — shape / tile utilization (K=N=512, LoFi bf16)

```
Set A: shape / tile utilization (K=N=512, LoFi bf16)
Takeaway: skinny (small-M) and non-32-aligned shapes waste matrix-engine cycles -> lower TFLOPS than a fat square matmul.
-----------------------------------------------------------------------
       shape (MxKxN)    tiles (MxKxN)    ms/iter     TFLOPS   % of best
-----------------------------------------------------------------------
          32x512x512          1x16x16      0.009       1.77      11.7%
          33x512x512          2x16x16      0.010       1.78      11.8%
          40x512x512          2x16x16      0.010       2.12      14.0%
          64x512x512          2x16x16      0.010       3.51      23.2%
          65x512x512          3x16x16      0.010       3.33      22.0%
          72x512x512          3x16x16      0.010       3.71      24.5%
         128x512x512          4x16x16      0.011       6.24      41.2%
         129x512x512          5x16x16      0.011       6.07      40.1%
         136x512x512          5x16x16      0.011       6.36      42.0%
         256x512x512          8x16x16      0.013      10.57      69.8%
         257x512x512          9x16x16      0.013      10.05      66.4%
         264x512x512          9x16x16      0.014      10.24      67.7%
         512x512x512         16x16x16      0.018      14.63      96.7%
         513x512x512         17x16x16      0.054       4.96      32.8%
         520x512x512         17x16x16      0.055       4.92      32.5%
        2048x512x512         64x16x16      0.091      11.81      78.0%
        2049x512x512         65x16x16      0.093      11.60      76.7%
        4096x512x512        128x16x16      0.142      15.13     100.0%
-----------------------------------------------------------------------
```

**Takeaway:** A perfectly-aligned M=32 (one tile) achieves only 11.7% of best — the engine
is mostly idle. Adding a single row (M=33) forces a second tile row but delivers essentially
the same throughput (1.78 TFLOPS) because the padding is wasted compute.
The sharpest cliff is at M=513: adding one row over a 512 boundary causes the runtime to spill
into a new tile row, but the core grid cannot accommodate the extra row efficiently, dropping
from 14.63 to 4.96 TFLOPS (−66%).

---

### Set B — core-grid / total size (square M=K=N, LoFi bf16)

```
Set B: core-grid / total size (square M=K=N, LoFi bf16)
Takeaway: tiny matmuls are dispatch/overhead-bound; only large ones saturate the core grid and approach peak TFLOPS.
-----------------------------------------------------------------------
       shape (MxKxN)    tiles (MxKxN)    ms/iter     TFLOPS   % of best
-----------------------------------------------------------------------
         128x128x128            4x4x4      0.005       0.92       1.3%
         256x256x256            8x8x8      0.008       4.07       5.8%
         512x512x512         16x16x16      0.018      14.85      21.3%
      1024x1024x1024         32x32x32      0.066      32.76      47.1%
      2048x2048x2048         64x64x64      0.277      62.01      89.1%
      4096x4096x4096      128x128x128      1.975      69.60     100.0%
-----------------------------------------------------------------------
```

**Takeaway:** 128³ is almost entirely dispatch overhead (0.92 TFLOPS, 1.3% of best). The
engine only begins to saturate above 2048³ (89.1%), and the full 4096³ case reaches 69.60 TFLOPS.
Doubling the linear dimension multiplies FLOPs by 8 but wall time by ~7×, so efficiency
improves steadily — a clear sign of an overhead-bound → compute-bound transition.

---

### Set C — prefill seq-len sweep (K=N=2048, LoFi bf16)

```
Set C: prefill seq-len sweep (K=N=2048, LoFi bf16)
Takeaway: short sequences (small M) starve the matrix engine; as seq_len grows the engine saturates and TFLOPS rises toward peak.
-----------------------------------------------------------------------
       shape (MxKxN)    tiles (MxKxN)    ms/iter     TFLOPS   % of best
-----------------------------------------------------------------------
        32x2048x2048          1x64x64      0.049       5.48       8.9%
        64x2048x2048          2x64x64      0.060       8.90      14.5%
       128x2048x2048          4x64x64      0.083      12.90      21.0%
       256x2048x2048          8x64x64      0.135      15.92      25.9%
       512x2048x2048         16x64x64      0.234      18.35      29.8%
      1024x2048x2048         32x64x64      0.155      55.59      90.4%
      2048x2048x2048         64x64x64      0.279      61.51     100.0%
      4096x2048x2048        128x64x64      0.600      57.27      93.1%
-----------------------------------------------------------------------
```

**Takeaway:** At seq\_len=32 (one tile row against a 2048-wide weight), TFLOPS is only 8.9%
of best — the core grid is massively under-utilised. There is a large jump between 512 and 1024
tokens (29.8% → 90.4%) as the problem finally fills enough of the 8×8 grid to amortise
dispatch. Beyond 2048 tokens, throughput plateaus (61–62 TFLOPS), reflecting a transition to
compute saturation.

---

## How to run

All commands must be run inside the `ct_metal` Docker container (see the project
[CLAUDE.md](../../../../CLAUDE.md)):

```sh
# Run standalone — prints all three tables to stdout
python tests/ttnn/unit_tests/operations/matmul/test_matmul_shape_efficiency.py

# Run via pytest
pytest -s tests/ttnn/unit_tests/operations/matmul/test_matmul_shape_efficiency.py

# Correctness check only (fast, no timing)
pytest -s tests/ttnn/unit_tests/operations/matmul/test_matmul_shape_efficiency.py -k correctness

# Single case set
pytest -s tests/ttnn/unit_tests/operations/matmul/test_matmul_shape_efficiency.py -k test_prefill_seq_len
```

---

## Methodology

1. **Fidelity fixed at LoFi** — `WormholeComputeKernelConfig(math_fidelity=LoFi)` keeps the
   matrix engine at its top bf16 rate so shape is the only variable.
2. **Trace-capture timing** — the op is compiled once, then captured into two traces
   (warmup: 10 iters, main: 30 iters). Wall-time subtraction cancels the fixed per-trace
   launch overhead, leaving pure op cycles.
3. **TFLOPS on real shape** — `2*M*K*N / t / 1e12` uses the original (un-padded) dimensions,
   so tile-padding waste directly depresses the reported number.
4. **"% of best"** is normalised to the highest TFLOPS in each table, making results
   meaningful on any Wormhole board without hard-coding a peak constant.
