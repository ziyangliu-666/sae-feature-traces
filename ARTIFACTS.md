# Artifact Map

This file maps the paper's main empirical claims to the checked-in scripts and
result logs. The JSON logs are the exact artifacts used for the submission
tables; the Modal scripts are included for GPU-backed reruns.

## Core Qwen3 Pipeline

| Paper item | Scripts | Result logs |
| --- | --- | --- |
| Probe library | `scripts/01_extract_probes.py` | `results/probe_library_qwen3_1.7b_L14_k96.json` |
| Cross-backend calibration and Qwen3 threshold | `scripts/07_e7_cross_backend_calibration.py`, `modal/e17_qwen3_honest_pool_multikernel.py` | `results/sigma_calibration_qwen3_1.7b_L14.json`, `results/recipe1_qwen3_honest_pool_multikernel.json` |
| Same-family lifts (E2) | `scripts/08_e2_same_family_separability.py`, `modal/e2_same_family.py` | `results/e2_separability.json` |
| Cross-family substitutes (E3) | `modal/e3_cross_family.py`, `scripts/09_e3_score.py`, `scripts/11_e3_v2_score.py` | `results/e3_cross_family_results.json`, `results/e3_cross_family_results_v2.json`, `results/e3_cross_family_scored.json`, `results/e3_v2_scored.json` |
| Adaptive LoRA (E4) | `modal/e4_v2_library_aware.py`, `modal/e4_adaptive_lora.py` | `results/e4_v2_library_aware.json`, `results/e4_adaptive_lora.json` |
| Joint-probe count sweep (E5) | `scripts/13_e5_joint_n_sweep.py` | `results/e5_v2_joint_n_sweep.json`, `results/e5_e6_post_hoc.json` |
| Calibration bootstrap | `scripts/14_calibration_bootstrap.py` | `results/calibration_bootstrap.json` |
| Top-k sensitivity | `scripts/18_ksweep_posthoc.py` | `results/ksweep_results.json` |
| Mask-flip sensitivity | `scripts/19_mask_flip.py` | `results/mask_flip_sensitivity.json` |
| Intrinsic-dimension analysis | `scripts/20_intrinsic_dim_analysis.py` | `results/intrinsic_dim.json` |
| Per-class covariate analysis | `scripts/22_covariate_analysis.py` | `results/22_covariate_analysis.json` |

## Hosted-LLM Verification

| Paper item | Scripts | Result logs |
| --- | --- | --- |
| Commit-open binding gate (B1) | `scripts/05_b1_binding.py` | `results/b1_results.json`, `results/b1_summary.txt` |
| Serving overhead (E9) | `modal/e9_overhead.py` | `results/e9_overhead.json` |
| SVIP comparison (E11 / Recipe 3) | `scripts/15_e11_svip_analysis.py`, `scripts/18_svip_two_backbone.py` | `results/e11_svip_vs_ours.json`, `results/recipe3_svip_two_backbone.json` |

## README Figure

| Figure asset | Paper role | Backing artifacts |
| --- | --- | --- |
| `paper_figures/f1.png` | Main reviewer-facing overview figure | `README.md`, manuscript figure build |

## Multi-Backbone Replication

| Paper item | Scripts | Result logs |
| --- | --- | --- |
| Gemma-2-2B probe library and threshold (E12/E15) | `modal/e12_gemma_pilot.py`, `modal/e15_gemma_honest_pool_v2.py` | `results/e12_gemma_pilot.json`, `results/sigma_calibration_gemma2_2b_L12.json`, `results/recipe1_gemma2_honest_pool_v2.json` |
| Gemma-2-2B cross-family substitutes (E13) | `modal/e13_gemma_e3.py`, `scripts/16_e13_gemma_score.py` | `results/e13_gemma_cross_family.json`, `results/e13_gemma_scored.json` |
| Gemma-2-2B adaptive LoRA (E14) | `modal/e14_gemma_adaptive_lora.py` | `results/e14_gemma_adaptive_lora.json` |
| Gemma-2-9B scale-up (E12/E13/E14-9B) | `modal/e12_gemma_9b_pilot.py`, `modal/e13_gemma_9b_cross_family.py`, `modal/e14_gemma_9b_adaptive_lora.py`, `modal/e15_gemma_9b_honest_pool.py` | `results/e12_gemma_9b_pilot.json`, `results/e13_gemma_9b_cross_family.json`, `results/e14_gemma_9b_adaptive_lora.json`, `results/sigma_calibration_gemma2_9b_L20.json`, `results/recipe1_gemma2_9b_honest_pool.json` |

## White-Box and Ablations

| Paper item | Scripts | Result logs |
| --- | --- | --- |
| White-box joint-z LoRA (E16) | `modal/e16_whitebox_jointz.py` | `results/e16_whitebox_jointz_qwen3*.json`, `results/multiseed_rank_sweep.json`, `results/per_class_rank_matrix.json` |
| 8-class circuit ablation (E10b) | `modal/e10b_circuit_ablation_8cats.py` | `results/e10b_circuit_ablation_8cats.json` |
| Per-head ablation (E23) | `modal/e23_perhead_ablation.py` | `results/e23_perhead_ablation.json` |
| Attention-only LoRA (E24) | `modal/e24_attn_only_lora.py` | `results/e24_attn_only_qwen3_r64_attn_only_mse.json` |

## Included for Completeness

The repository also includes earlier local pilots and post-hoc stress tests
whose outputs support appendix discussion or negative-scope checks:

- `02_p2_substitute_attacks.py`, `03_p3_adversarial_distill.py`,
  `04_p3b_distribution_matched_attack.py`
- `e18_forgery_local.py`, `e18_forgery_f3_local.py`,
  `e18_forgery_f3_gemma.py`, `e19_library_split_generalization.py`
- corresponding `p2_*`, `p3_*`, `p3b_*`, `e18_*`, and `e19_*` logs
