# SDPA Prefill Shape-Efficiency Demo

`test_sdpa_shape_efficiency.py` answers: **for causal scaled dot-product attention during
prefill, how much does tensor shape (sequence length, head dimension) change throughput?**

It is the companion to `test_matmul_shape_efficiency.py` — same two-trace timing methodology
applied to the Tensix attention engine rather than the matrix engine.

---

## What is measured

All runs use causal SDPA prefill (B=1) with HiFi2 fidelity and `exp_approx_mode=True`.
Two case sets are run:

| Set | What changes | What is fixed | Question answered |
|---|---|---|---|
| A | S (sequence length) | H=8, D=128, chunk=64 | How does TFLOPS scale with sequence length? |
| B | D (head dimension) | H=8, S=1024, chunk=128 | Does wider D improve compute/memory balance? |

FLOPs are counted as `4·B·H·S²·D` (two S×D @ D×S matmuls: QKᵀ and PV, each `2·S·D·S` ops).

---

## Measured results (Wormhole N300, 2026-06-03)

### Set A — seq-len scaling (H=8, D=128, causal bf16)

```
Set A: seq-len scaling  H=8 D=128  causal bf16
Takeaway: short sequences are overhead-bound (few chunks); TFLOPS rises as S grows because S²·D FLOPs dominate S·D memory traffic.
------------------------------------------
       S    ms/iter     TFLOPS   % of best
------------------------------------------
      64      0.015       1.15       4.4%
     128      0.027       2.53       9.8%
     256      0.039       6.90      26.6%
     512      0.065      16.42      63.3%
    1024      0.194      22.11      85.2%
    2048      0.695      24.71      95.3%
    4096      2.650      25.93     100.0%
------------------------------------------
```

**Takeaway:** At S=64 (one chunk), only 4.4% of peak TFLOPS is reached — the attention engine
is almost entirely idle. The curve rises steeply as S grows: S=512 is already 63.3% of best.
The asymptotic climb from S=1024 to S=4096 (85% → 95% → 100%) reflects a transition to
compute saturation as S²·D FLOPs come to dominate S·D memory traffic.

---

### Set B — head-dim scaling (H=8, S=1024, causal bf16)

```
Set B: head-dim scaling  H=8 S=1024  causal bf16
Takeaway: wider heads pack more FLOPs per byte — D=128 should be more compute-bound than D=64 at the same S, reaching higher TFLOPS.
------------------------------------------
       D    ms/iter     TFLOPS   % of best
------------------------------------------
      64      0.112      19.17      61.9%
      96      0.123      26.12      84.3%
     128      0.139      30.99     100.0%
------------------------------------------
```

**Takeaway:** Wider heads raise TFLOPS by packing more FLOPs per byte of K/V bandwidth
(FLOPs ∝ S²·D, bandwidth ∝ S·D). D=128 achieves 30.99 TFLOPS — 1.6× D=64 — confirming that
the attention engine is more compute-bound at wider head dimensions.

---

## How to run

All commands must be run inside the `ct_metal` Docker container (see the project
[CLAUDE.md](../../../../CLAUDE.md)):

```sh
# Run standalone — prints both tables to stdout
python tests/ttnn/unit_tests/operations/sdpa/test_sdpa_shape_efficiency.py

# Run via pytest
pytest -s tests/ttnn/unit_tests/operations/sdpa/test_sdpa_shape_efficiency.py

# Correctness check only (fast, no timing)
pytest -s tests/ttnn/unit_tests/operations/sdpa/test_sdpa_shape_efficiency.py -k correctness

# Single case set
pytest -s tests/ttnn/unit_tests/operations/sdpa/test_sdpa_shape_efficiency.py -k test_seq_len_scaling
pytest -s tests/ttnn/unit_tests/operations/sdpa/test_sdpa_shape_efficiency.py -k test_head_dim_scaling
```

---

## Methodology

1. **Fidelity fixed at HiFi2 + exp\_approx\_mode** — mirrors realistic inference settings
   (the same config used in production prefill kernels) so shape is the only variable.
2. **Trace-capture timing** — same two-trace subtraction trick as `test_matmul_shape_efficiency.py`:
   warmup (10 iters) and main (30 iters) traces; wall-time difference cancels per-trace launch
   overhead, leaving pure op cycles.
3. **FLOPs = 4·B·H·S²·D** — counts both the QKᵀ matmul and the PV matmul (each 2·S·D·S ops),
   so the reported TFLOPS reflects full attention arithmetic intensity.
4. **"% of best"** is normalised to the highest TFLOPS in each table, making results
   meaningful on any Wormhole board without hard-coding a peak constant.
