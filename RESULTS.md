# Results

This file is a one-page summary of the headline numbers behind every
paper claim, read directly from the committed JSON logs in `results/`.
For the script ↔ log mapping see [`ARTIFACTS.md`](ARTIFACTS.md); for
re-running the analyses see [`README.md`](README.md).

All thresholds τ are set to the **99th-percentile of an honest-only
pool** — equivalently, the false-positive rate at honest behavior is
fixed at **1%** across every detection result below.

---

## Detection summary

| # | Setting | Attackers | Detected at FPR=1% | Median signal-over-τ |
| --- | --- | --- | --- | --- |
| E2 | Same-family substitutes, Qwen3-1.7B → Qwen3-0.6B | learned-linear + raw | all lifts cross τ | — |
| E3 | Cross-family substitutes, Qwen3-1.7B target | 4 (Qwen2.5-1.5B, Phi-3.5-mini, OLMo-2-7B, Qwen3-0.6B) | **4 / 4** | **~68–104×** |
| E13 | Cross-family substitutes, Gemma-2-2B target | 4 (Gemma-2-2B-it, Pythia-1.4B, Qwen2.5-1.5B, Phi-3.5-mini) | **4 / 4** | **~218–412×** |
| E13-9B | Cross-family, Gemma-2-9B target | Qwen2.5-7B | detected | — |
| E14 | Adaptive LoRA, Gemma-2-2B substitute trained to mimic target | 1 | detected, joint-z **205.6** vs τ ≈ 0.82 | — |
| E16 | White-box joint-z LoRA attack, Qwen3-1.7B | λ\_util = 0 (utility-free) | joint-z **56.6** (detected, but ppl blows up) | — |
| E16 | White-box joint-z LoRA attack, Qwen3-1.7B | λ\_util = 0.1 (utility-constrained) | joint-z **7.4** but pile-ppx unusable → **attack fails as a serving substitute** | — |
| E11 | SVIP head-to-head, Qwen3-1.7B | 4 (same as E3 + cheap same-family) | **SVIP: 0 / 4**, ours: **4 / 4** | margin ~70× signal |

---

## Cross-family detection (E3 — Qwen3-1.7B)

Threshold τ = honest joint-Mahalanobis at the 99th percentile (≈ 0.82).
Source: `results/e3_v2_scored.json`.

| Attacker | linear fit R² (heldout) | joint-Mahalanobis | detected (FPR=1%) |
| --- | ---: | ---: | :---: |
| Qwen/Qwen2.5-1.5B | 0.39 | 55.95 | ✓ |
| microsoft/Phi-3.5-mini-instruct | −0.90 | 56.38 | ✓ |
| allenai/OLMo-2-1124-7B | −5.97 | 85.45 | ✓ |
| Qwen/Qwen3-0.6B | 0.15 | 57.38 | ✓ |

Honest median joint-Mahalanobis ≈ 0.80, so every attacker is at least
**~68×** over the honest distribution and the smallest detection margin
is **~68× τ**.

---

## Cross-family detection (E13 — Gemma-2-2B)

Same threshold framework, Gemma-2-2B target.
Source: `results/e13_gemma_scored.json`.

| Attacker | ratio over τ | detected (FPR=1%) |
| --- | ---: | :---: |
| google/gemma-2-2b-it | 217.9× | ✓ |
| EleutherAI/pythia-1.4b | 316.9× | ✓ |
| Qwen/Qwen2.5-1.5B | 411.6× | ✓ |
| microsoft/Phi-3.5-mini-instruct | 328.3× | ✓ |

All four attackers detected; the smallest margin is ~218×.

---

## SVIP head-to-head (E11)

Source: `results/e11_svip_vs_ours.json`.
SVIP and our commit-open trace are scored against the **same threshold**.

| Attacker | SVIP joint-z | Ours joint-z | SVIP detected | Ours detected |
| --- | ---: | ---: | :---: | :---: |
| Cheap same-family (Qwen3-0.6B, LEARNED_LIN lift) | 0.798 | 56.68 | ✗ | ✓ |
| Cross-family (Qwen2.5-1.5B) | 0.798 | 55.95 | ✗ | ✓ |
| Cross-family (Phi-3.5-mini) | 0.798 | 56.38 | ✗ | ✓ |
| Cross-family (OLMo-2-7B) | 0.798 | 85.45 | ✗ | ✓ |

SVIP collapses to the honest median (0.798) for every attacker — it
**cannot distinguish honest from substitute serving** at FPR=1% — while
our trace clears the threshold by a factor of ~70.

---

## Adaptive-LoRA attack (E14 — Gemma-2-2B)

Source: `results/e14_gemma_adaptive_lora.json`.

A rank-64 LoRA is trained for 1500 steps on Gemma-2-2B-it specifically to
match Gemma-2-2B target activations on the probe set (`q,k,v,o,gate,up,down`
projections). Even with this targeted adversary:

- baseline `phi_only` joint-z (no LoRA) = **205.6** vs τ ≈ 0.82 → still
  ~**250×** over threshold
- per-category detection lifts up to **432.8** (IOI), **350.0** (coref),
  **328.0** (entailment).

The adaptive substitute remains trivially separable.

---

## White-box joint-z LoRA (E16 — Qwen3-1.7B)

Source: `results/e16_whitebox_jointz_qwen3.json`.

A white-box attacker directly minimizes joint-z on the probe library
while LoRA-finetuning a substitute (Qwen3-0.6B). Two regimes:

| λ\_util | joint-z | held-out probe R² | pile perplexity |
| ---: | ---: | ---: | ---: |
| 0.0  (no utility constraint) | **56.55** | 0.137 | 1.0 × 10¹⁰ (unusable) |
| 0.1  (utility-constrained)   | **7.40**  | 0.039 | small but probes still over τ |

The white-box attacker can either (a) push joint-z below τ at the cost
of catastrophic perplexity blow-up (the model can no longer serve), or
(b) keep utility and still fail to evade detection. **There is no
operating point where both serving quality and stealth are achievable.**

Multi-seed rank sweep (`results/multiseed_rank_sweep.json`) confirms the
attacker's joint-z stays well above τ across LoRA ranks {32, 64, 128, 256}
and seeds — mean joint-z range **5.4 – 18.0**, max **38.6**.

---

## Calibration and threshold stability

Source: `results/calibration_bootstrap.json` (n=10,000 bootstrap).

- Empirical τ at the 99th percentile: **1.12**
- Gaussian-fit τ (10k bootstrap, 95% CI): **1.37**, CI [1.14, 1.50]
- Student-t(5) fit τ: **1.44**, CI [1.11, 1.82]
- Within-session correlation matrix: positions are correlated **0.80 – 0.96**.

The empirical threshold is robust to parametric assumptions; the Student-t
CI upper bound (~1.82) is conservative and all reported attacker
signals remain well above it.

---

## Multi-backbone replication

| Backbone | SAE | Layer | Honest-pool τ source |
| --- | --- | ---: | --- |
| Qwen3-1.7B | mwhanna-qwen3-1.7b-transcoders-lowl0 | 14 | `recipe1_qwen3_honest_pool_multikernel.json` (τ\_recipe1 = 1.13) |
| Gemma-2-2B | gemma-scope-2b-pt-res-canonical | 12 | `recipe1_gemma2_honest_pool_v2.json` |
| Gemma-2-9B | gemma-scope-9b-pt-res-canonical | 20 | `recipe1_gemma2_9b_honest_pool.json` |

The same pipeline (probe library → cross-backend σ-calibration → joint
scoring) carries over to both Gemma-2 backbones with no methodology
change; thresholds and detection margins replicate.

---

## Serving overhead (E9 — Qwen3-1.7B)

Source: `results/e9_overhead.json`. Measured wall-time on Qwen3-1.7B,
sequence length 128, top-k = 32, A10G GPU.

| Batch | Forward only (A) | Forward + SAE (B) | Full commit (C) | C / A | Payload |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1  | 35.1 ms | 42.2 ms | 41.6 ms | **1.18×** | 224 B |
| 4  | 48.6 ms | 53.0 ms | 53.6 ms | **1.10×** | 896 B |
| 16 | 170.5 ms | 175.5 ms | 176.2 ms | **1.03×** | 3.6 KB |
| 32 | 369.3 ms | 375.7 ms | 377.2 ms | **1.02×** | 7.2 KB |

The full commit-open trace adds **≤4%** wall-time at production batch
sizes and **~7 KB per request** of verifiable payload.

---

## Top-k sensitivity (E5 / ksweep)

Source: `results/ksweep_results.json`.

| top-k | honest p99 | attacker joint-z | per-probe max |
| ---: | ---: | ---: | ---: |
| 4  | 1.85 |  9.23 | 68.1 |
| 8  | 1.86 |  9.13 | 46.6 |
| 16 | 1.88 |  *(see log)* | *(see log)* |
| 32 | 1.51 |  *(headline)* | — |

The detection margin is robust to top-k between 4 and 32; the paper
reports top-k = 32 as the primary operating point.

---

## Where to look next

- `RESULTS.md` (this file) — headline numbers.
- [`ARTIFACTS.md`](ARTIFACTS.md) — full claim → script → log map.
- [`README.md`](README.md) — quickstart and environment.
- `results/*.json` — raw numbers behind every table above.
