# Results

This page summarizes the numbers a reviewer is most likely to check first.
For the exact claim -> script -> log mapping, see [`ARTIFACTS.md`](ARTIFACTS.md).
For raw measurements, inspect `results/*.json`.

Unless stated otherwise, thresholds are honest-pool 99th percentiles, so the
nominal false-positive rate on honest behavior is 1%.

---

## Main Claims

| Paper claim | Reviewer artifact |
| --- | --- |
| The SAE trace uses a 96-probe named-circuit library over 11 categories. | `results/probe_library_qwen3_1.7b_L14_k96.json`, `results/e23_perhead_ablation.json` |
| Probe-aware LoRA attackers show the paper's capacity-dependent per-class crossover, but the full-library joint score remains above threshold. | `results/e14_gemma_adaptive_lora.json`, `results/multiseed_rank_sweep.json`, `results/per_class_rank_matrix.json` |
| Cross-family substitutes are separated on Qwen3-1.7B and Gemma-2-2B. | `results/e3_v2_scored.json`, `results/e13_gemma_scored.json` |
| White-box and attention-only variants remain outside the accepted serving envelope. | `results/e16_whitebox_jointz_qwen3.json`, `results/e24_attn_only_qwen3_r64_attn_only_mse.json` |
| Commit-open closes the parallel-serve gap left by probe-after-return baselines. | `results/e11_svip_vs_ours.json`, `results/recipe3_svip_two_backbone.json`, `paper_figures/svip_vs_commit_open.png` |
| Serving overhead is small at production batch sizes. | `results/e9_overhead.json`, `paper_figures/serving_overhead.png` |

---

## Capacity-Dependent Crossover

Source: `results/multiseed_rank_sweep.json`.

| LoRA rank | Mean joint-z | Min joint-z | Max joint-z |
| ---: | ---: | ---: | ---: |
| 32 | 5.61 | 3.74 | 8.38 |
| 64 | 5.37 | 3.87 | 8.03 |
| 128 | 6.09 | 4.54 | 8.72 |
| 256 | 18.02 | 5.42 | 38.60 |

The manuscript's per-class figures show the qualitative inversion: at lower
rank, attention-pattern classes such as induction, IOI, and coreference are
most attackable; at higher rank, surface classes such as factual recall,
syntax, and language become easier to imitate while induction recovers.

For visual inspection, see:

- `paper_figures/adaptive_lora_profile.png`
- `paper_figures/per_class_attackability.png`
- `paper_figures/joint_probe_sweep.png`

---

## Cross-Family Detection

### Qwen3-1.7B Target

Source: `results/e3_v2_scored.json`. Threshold τ is the honest
joint-Mahalanobis 99th percentile, approximately 0.82 in this scoring run.

| Attacker | Held-out linear-fit R2 | Joint score | Detected |
| --- | ---: | ---: | :---: |
| Qwen/Qwen2.5-1.5B | 0.39 | 55.95 | yes |
| microsoft/Phi-3.5-mini-instruct | -0.90 | 56.38 | yes |
| allenai/OLMo-2-1124-7B | -5.97 | 85.45 | yes |
| Qwen/Qwen3-0.6B | 0.15 | 57.38 | yes |

All four tested substitutes clear threshold by a large margin.

### Gemma-2-2B Target

Source: `results/e13_gemma_scored.json`.

| Attacker | Ratio over τ | Detected |
| --- | ---: | :---: |
| google/gemma-2-2b-it | 217.9x | yes |
| EleutherAI/pythia-1.4b | 316.9x | yes |
| Qwen/Qwen2.5-1.5B | 411.6x | yes |
| microsoft/Phi-3.5-mini-instruct | 328.3x | yes |

The Gemma-2-2B replication uses the same probe-library -> calibration ->
joint-scoring pipeline rather than a backbone-specific detector.

---

## White-Box and Attention-Only Stress Tests

Source: `results/e16_whitebox_jointz_qwen3.json`.

| Setting | Joint-z | Utility signal |
| --- | ---: | --- |
| White-box LoRA, lambda_util = 0.0 | 56.55 | pile perplexity approximately 1.0e10, unusable as a serving substitute |
| White-box LoRA, lambda_util = 0.1 | 7.40 | pile perplexity 243.24, still above threshold |

Source: `results/e24_attn_only_qwen3_r64_attn_only_mse.json`.

The attention-only rank-64 LoRA stress test remains above the Qwen3 threshold
while exposing which circuit categories are most vulnerable under a restricted
attacker parameterization.

---

## Commit-Open vs Probe-After-Return

The paper's application result is about timing: the same trace score is useful
only if the provider commits before the verifier asks what to open. A
probe-after-return baseline can be bypassed by parallel serving: route normal
users to a substitute and route verifier probes to the advertised model.

Committed reviewer artifacts:

| Artifact | Scope |
| --- | --- |
| `results/e11_svip_vs_ours.json` | Qwen3 core comparison; 4 matched attackers |
| `results/recipe3_svip_two_backbone.json` | Qwen3 + Gemma-2-2B comparison; 11 matched attackers |
| `paper_figures/svip_vs_commit_open.png` | Manuscript figure for the full matched comparison |

Across the committed JSON comparisons, SVIP-style probe-after-return misses
every matched parallel-serve attacker and commit-open detects every matched
attacker. The manuscript figure extends this matched comparison to the full
15-attacker set reported in the paper.

---

## Serving Overhead

Source: `results/e9_overhead.json`. Measured on Qwen3-1.7B, sequence length
128, top-k = 32, A10G GPU.

| Batch | Forward only | Forward + SAE | Full commit | Full / forward | Payload / request |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.1 ms | 42.2 ms | 41.6 ms | 1.185x | 224 B |
| 4 | 48.6 ms | 53.0 ms | 53.6 ms | 1.102x | 224 B |
| 16 | 170.5 ms | 175.5 ms | 176.2 ms | 1.034x | 224 B |
| 32 | 369.3 ms | 375.7 ms | 377.2 ms | 1.021x | 224 B |

The production-batch headline is batch 32: full commit-open tracing adds about
2.1% wall-clock overhead with a 224-byte per-request payload.

---

## Threshold and Calibration Checks

Source: `results/calibration_bootstrap.json`.

- Empirical τ at the 99th percentile: 1.12.
- Gaussian-fit τ, 10k bootstrap: 1.37 with 95% CI [1.14, 1.50].
- Student-t(5) τ: 1.44 with 95% CI [1.11, 1.82].
- Within-session position correlations are high, so the paper uses
  honest-pool calibration rather than assuming independent positions.

These thresholds remain far below the cross-family and most adversarial
substitute scores reported above.

---

## Where to Look Next

- [`ARTIFACTS.md`](ARTIFACTS.md) is the authoritative map from paper claims to
  scripts and logs.
- [`README.md`](README.md) gives the reviewer-oriented quickstart.
- `results/*.json` contains the raw numbers behind the summaries.
- `paper_figures/` mirrors selected manuscript figures for visual inspection.
