# Precision / Data-Format Efficiency Suite

Companion benchmark suite to the shape-efficiency demos in `operations/eltwise/` and `operations/matmul/`. Where those demos fix dtype and fidelity and sweep **shape**, this suite fixes a representative shape and sweeps the **precision configuration** axes.

The goal is to quantify the throughput cost — or absence of cost — of each precision choice across four hardware-representative ops:

| Op | Engine | Fixed shape |
|---|---|---|
| `sin` | SFPU (unary) | 2048×2048 |
| `mul` | SFPU (binary) | 2048×2048 |
| `matmul` | Matrix engine | 2048×2048×2048 |
| `attention` (SDPA) | Matrix engine + SFPU | batch=1, heads=8, seq=2048, head\_dim=128 |

---

## Sweep axes

Each test file runs **one axis at a time**, holding the other three at the baseline. This avoids a 64-cell cross-product and keeps each table's takeaway unambiguous.

**Baseline**: `dtype=bfloat16`, `math_fidelity=HiFi2`, `fp32_dest_acc_en=False`, `packer_l1_acc=False`.

| Axis | Values | What it controls |
|---|---|---|
| `dtype` | `bfloat16` (2B), `bfloat8_b` (1B), `bfloat4_b` (0.5B), `float32` (4B) | DRAM bytes per element; storage fidelity of weights/activations |
| `math_fidelity` | `LoFi`, `HiFi2`, `HiFi3`, `HiFi4` | Matrix-engine cycles per tile (no effect on SFPU ops) |
| `fp32_dest_acc_en` | `False`, `True` | Destination register format; `True` halves dest tile capacity → potential TFLOPS drop |
| `packer_l1_acc` | `False`, `True` | L1 accumulation in the output packer; reduces DRAM writes for large-K matmuls |

---

## Expected findings

| Op | dtype | math\_fidelity | fp32\_dest\_acc\_en | packer\_l1\_acc |
|---|---|---|---|---|
| sin | bfloat8\_b ≈ 2× GB/s of bfloat16 | **flat** (SFPU ignores fidelity) | **flat** | **flat** |
| mul | bfloat8\_b ≈ 2× GB/s of bfloat16 | **flat** | **flat** | **flat** |
| matmul | smaller dtype → higher TFLOPS | LoFi > HiFi2 > HiFi3 > HiFi4 | True slightly slower | True can improve large-K |
| attention | bfloat8\_b faster (less DRAM) | LoFi > … > HiFi4 | True slightly slower | small effect |

The "flat" SFPU results are deliberate **negative results** — they confirm that the fidelity axis is correctly isolated and does not bleed into SFPU ops.

---

## How to read the tables

Each table has four columns:

```
       value    ms/iter       GOps/s   % of best
```

- **value** — the axis value being tested (e.g., `bfloat8_b`, `HiFi2`).
- **ms/iter** — wall time per op iteration, measured by trace-capture timing (or eager fallback).
- **GOps/s / TFLOPS** — throughput using padded element counts for SFPU ops (to reflect actual hardware work) and real FLOP counts for matrix ops.
- **% of best** — normalized to the fastest row in that table, so the relative cost is immediately visible without knowing peak hardware numbers.

Rows that fail (e.g., `bfloat4_b` on an unsupported op) are shown as `[skip] <reason>` rather than silently omitted.

---

## How to run

All commands must be executed **inside the `ct_metal` container**. From the repo root:

```sh
# Enter the container shell
./start_container.sh
```

Or run a single script without entering the shell:

```sh
docker exec -it -w /root/tt-metal ct_metal bash -c \
  "python tests/ttnn/unit_tests/operations/precision_efficiency/test_sin_precision.py"

docker exec -it -w /root/tt-metal ct_metal bash -c \
  "python tests/ttnn/unit_tests/operations/precision_efficiency/test_mul_precision.py"

docker exec -it -w /root/tt-metal ct_metal bash -c \
  "python tests/ttnn/unit_tests/operations/precision_efficiency/test_matmul_precision.py"

docker exec -it -w /root/tt-metal ct_metal bash -c \
  "python tests/ttnn/unit_tests/operations/precision_efficiency/test_attention_precision.py"
```

Run all tests via pytest:

```sh
docker exec -it -w /root/tt-metal ct_metal bash -c \
  "pytest -s tests/ttnn/unit_tests/operations/precision_efficiency/"
```

Run a specific test file:

```sh
docker exec -it -w /root/tt-metal ct_metal bash -c \
  "pytest -s tests/ttnn/unit_tests/operations/precision_efficiency/test_matmul_precision.py"
```

---

## Card reset

If a run hangs or the card enters a bad state, reset from the **host** (not inside the container) with the `tenstorrent-venv` activated:

```sh
source ~/.tenstorrent-venv/bin/activate
tt-smi -r
```

On Galaxy systems use `tt-smi -glx_reset` if the above fails (requires CPLD FW ≥ v1.16).

---

## Benchmark results

Hardware: Wormhole B0. Measured 2026-06-03. Each number is the median of 50 warm iterations captured via program trace.

### `sin` — SFPU unary, 2048×2048

**A. dtype sweep**

| dtype | ms/iter | GB/s | % of best |
|---|---|---|---|
| bfloat16 | 0.090 | 187.40 | 100.0% |
| bfloat8\_b | 0.059 | 142.26 | 75.9% |
| bfloat4\_b | 0.058 | 72.21 | 38.5% |

> bfloat8\_b is faster in ms but lower GB/s because the tensor is half the bytes — the SFPU finishes in fewer cycles but the bandwidth denominator is also halved. bfloat4\_b shows the same pattern amplified.

**B. math\_fidelity sweep** (baseline dtype = bfloat16)

| fidelity | ms/iter | GOps/s | % of best |
|---|---|---|---|
| LoFi | 0.092 | 45.36 | 96.5% |
| HiFi2 | 0.095 | 44.10 | 93.8% |
| HiFi3 | 0.094 | 44.75 | 95.2% |
| HiFi4 | 0.089 | 47.01 | 100.0% |

> Variation < 7% — effectively flat, confirming SFPU bypasses the matrix engine's fidelity path.

**C. fp32\_dest\_acc\_en sweep**

| value | ms/iter | GOps/s | % of best |
|---|---|---|---|
| False | 0.090 | 46.54 | 100.0% |
| True | 0.091 | 46.32 | 99.5% |

> No measurable impact, as expected for SFPU ops.

**D. packer\_l1\_acc sweep**

| value | ms/iter | GOps/s | % of best |
|---|---|---|---|
| False | 0.089 | 47.08 | 99.8% |
| True | 0.089 | 47.16 | 100.0% |

> No measurable impact, as expected for SFPU ops.

---

### `mul` — SFPU binary, 2048×2048

**A. dtype sweep**

| dtype | ms/iter | GB/s | % of best |
|---|---|---|---|
| bfloat16 | 0.110 | 228.68 | 97.5% |
| bfloat8\_b | 0.061 | 205.49 | 87.6% |
| bfloat4\_b | 0.042 | 151.00 | 64.4% |

> mul is 3-tensor (2 reads + 1 write), so peak GB/s is higher than sin. Smaller dtypes are faster in absolute time but move fewer bytes, pulling GB/s down.

**B. math\_fidelity sweep**

| fidelity | ms/iter | GOps/s | % of best |
|---|---|---|---|
| LoFi | 0.111 | 37.87 | 99.5% |
| HiFi2 | 0.110 | 38.08 | 100.0% |
| HiFi3 | 0.110 | 38.05 | 99.9% |
| HiFi4 | 0.111 | 37.96 | 99.7% |

> Perfectly flat — fidelity has zero effect on SFPU binary ops.

**C. fp32\_dest\_acc\_en sweep**

| value | ms/iter | GOps/s | % of best |
|---|---|---|---|
| False | 0.111 | 37.94 | 99.6% |
| True | 0.110 | 38.10 | 100.0% |

> No measurable impact.

**D. packer\_l1\_acc sweep**

| value | ms/iter | GOps/s | % of best |
|---|---|---|---|
| False | 0.110 | 38.01 | 100.0% |
| True | 0.110 | 38.00 | 100.0% |

> No measurable impact.

---

### `matmul` — Matrix engine, 2048×2048×2048

**A. dtype sweep**

| dtype | ms/iter | TFLOPS | % of best |
|---|---|---|---|
| bfloat16 | 0.307 | 55.89 | 61.7% |
| bfloat8\_b | 0.224 | 76.57 | 84.5% |
| bfloat4\_b | 0.190 | 90.64 | 100.0% |

> Smaller dtype → fewer DRAM bytes per tile → higher achieved TFLOPS. bfloat4\_b is 1.62× faster than bfloat16 at this shape.

**B. math\_fidelity sweep**

| fidelity | ms/iter | TFLOPS | % of best |
|---|---|---|---|
| LoFi | 0.279 | 61.57 | 100.0% |
| HiFi2 | 0.310 | 55.35 | 89.9% |
| HiFi3 | 0.355 | 48.35 | 78.5% |
| HiFi4 | 0.413 | 41.64 | 67.6% |

> Strong monotonic effect: each fidelity step costs ~10–11% throughput. LoFi is 1.48× faster than HiFi4.

**C. fp32\_dest\_acc\_en sweep**

| value | ms/iter | TFLOPS | % of best |
|---|---|---|---|
| False | 0.309 | 55.64 | 100.0% |
| True | 0.344 | 49.96 | 89.8% |

> fp32 dest halves tile capacity in the destination registers, costing ~10% throughput.

**D. packer\_l1\_acc sweep**

| value | ms/iter | TFLOPS | % of best |
|---|---|---|---|
| False | 0.309 | 55.55 | 95.1% |
| True | 0.294 | 58.43 | 100.0% |

> L1 accumulation saves ~5% by reducing DRAM writes for this large-K shape. Enable for production large-K matmuls.

---

## File layout

```
precision_efficiency/
├── __init__.py
├── README.md
├── precision_utils.py          # shared timing, constants, table formatter, sweep driver
├── test_sin_precision.py       # sin — 4 axis sweeps + correctness
├── test_mul_precision.py       # mul — 4 axis sweeps + correctness
├── test_matmul_precision.py    # matmul — 4 axis sweeps + correctness
└── test_attention_precision.py # SDPA — 4 axis sweeps + correctness
```
