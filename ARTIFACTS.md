# Artifact Map

This file maps the paper's main empirical claims to the checked-in scripts and
result logs. The JSON logs are the exact artifacts used for the submission
tables; the Modal scripts are included for GPU-backed reruns.

## Core Qwen3 Pipeline

| Paper item | Scripts | Result logs |
| --- | --- | --- |
| Probe library | `pilots/p1_probe_lib/01_extract_probes.py` | `logs/probe_library_qwen3_1.7b_L14_k96.json` |
| Cross-backend calibration and Qwen3 threshold | `pilots/p1_probe_lib/07_e7_cross_backend_calibration.py`, `modal_runs/e17_qwen3_honest_pool_multikernel.py` | `logs/sigma_calibration_qwen3_1.7b_L14.json`, `logs/recipe1_qwen3_honest_pool_multikernel.json` |
| Same-family lifts (E2) | `pilots/p1_probe_lib/08_e2_same_family_separability.py`, `modal_runs/e2_same_family.py` | `logs/e2_separability.json` |
| Cross-family substitutes (E3) | `modal_runs/e3_cross_family.py`, `pilots/p1_probe_lib/09_e3_score.py`, `pilots/p1_probe_lib/11_e3_v2_score.py` | `logs/e3_cross_family_results.json`, `logs/e3_cross_family_results_v2.json`, `logs/e3_cross_family_scored.json`, `logs/e3_v2_scored.json` |
| Adaptive LoRA (E4) | `modal_runs/e4_v2_library_aware.py`, `modal_runs/e4_adaptive_lora.py` | `logs/e4_v2_library_aware.json`, `logs/e4_adaptive_lora.json` |
| Joint-probe count sweep (E5) | `pilots/p1_probe_lib/13_e5_joint_n_sweep.py` | `logs/e5_v2_joint_n_sweep.json`, `logs/e5_e6_post_hoc.json` |
| Calibration bootstrap | `pilots/p1_probe_lib/14_calibration_bootstrap.py` | `logs/calibration_bootstrap.json` |
| Top-k sensitivity | `pilots/p1_probe_lib/18_ksweep_posthoc.py` | `logs/ksweep_results.json` |
| Mask-flip sensitivity | `pilots/p1_probe_lib/19_mask_flip.py` | `logs/mask_flip_sensitivity.json` |
| Intrinsic-dimension analysis | `pilots/p1_probe_lib/20_intrinsic_dim_analysis.py` | `logs/intrinsic_dim.json` |
| Per-class covariate analysis | `pilots/p1_probe_lib/22_covariate_analysis.py` | `logs/22_covariate_analysis.json` |

## Hosted-LLM Verification

| Paper item | Scripts | Result logs |
| --- | --- | --- |
| Commit-open binding gate (B1) | `pilots/p1_probe_lib/05_b1_binding.py` | `logs/b1_results.json`, `logs/b1_summary.txt` |
| Serving overhead (E9) | `modal_runs/e9_overhead.py` | `logs/e9_overhead.json` |
| SVIP comparison (E11 / Recipe 3) | `pilots/p1_probe_lib/15_e11_svip_analysis.py`, `pilots/p1_probe_lib/18_svip_two_backbone.py` | `logs/e11_svip_vs_ours.json`, `logs/recipe3_svip_two_backbone.json` |

## Multi-Backbone Replication

| Paper item | Scripts | Result logs |
| --- | --- | --- |
| Gemma-2-2B probe library and threshold (E12/E15) | `modal_runs/e12_gemma_pilot.py`, `modal_runs/e15_gemma_honest_pool_v2.py` | `logs/e12_gemma_pilot.json`, `logs/sigma_calibration_gemma2_2b_L12.json`, `logs/recipe1_gemma2_honest_pool_v2.json` |
| Gemma-2-2B cross-family substitutes (E13) | `modal_runs/e13_gemma_e3.py`, `pilots/p1_probe_lib/16_e13_gemma_score.py` | `logs/e13_gemma_cross_family.json`, `logs/e13_gemma_scored.json` |
| Gemma-2-2B adaptive LoRA (E14) | `modal_runs/e14_gemma_adaptive_lora.py` | `logs/e14_gemma_adaptive_lora.json` |
| Gemma-2-9B scale-up (E12/E13/E14-9B) | `modal_runs/e12_gemma_9b_pilot.py`, `modal_runs/e13_gemma_9b_cross_family.py`, `modal_runs/e14_gemma_9b_adaptive_lora.py`, `modal_runs/e15_gemma_9b_honest_pool.py` | `logs/e12_gemma_9b_pilot.json`, `logs/e13_gemma_9b_cross_family.json`, `logs/e14_gemma_9b_adaptive_lora.json`, `logs/sigma_calibration_gemma2_9b_L20.json`, `logs/recipe1_gemma2_9b_honest_pool.json` |

## White-Box and Ablations

| Paper item | Scripts | Result logs |
| --- | --- | --- |
| White-box joint-z LoRA (E16) | `modal_runs/e16_whitebox_jointz.py` | `logs/e16_whitebox_jointz_qwen3*.json`, `logs/multiseed_rank_sweep.json`, `logs/per_class_rank_matrix.json` |
| 8-class circuit ablation (E10b) | `modal_runs/e10b_circuit_ablation_8cats.py` | `logs/e10b_circuit_ablation_8cats.json` |
| Per-head ablation (E23) | `modal_runs/e23_perhead_ablation.py` | `logs/e23_perhead_ablation.json` |
| Attention-only LoRA (E24) | `modal_runs/e24_attn_only_lora.py` | `logs/e24_attn_only_qwen3_r64_attn_only_mse.json` |

## Included for Completeness

The repository also includes earlier local pilots and post-hoc stress tests
whose outputs support appendix discussion or negative-scope checks:

- `02_p2_substitute_attacks.py`, `03_p3_adversarial_distill.py`,
  `04_p3b_distribution_matched_attack.py`
- `e18_forgery_local.py`, `e18_forgery_f3_local.py`,
  `e18_forgery_f3_gemma.py`, `e19_library_split_generalization.py`
- corresponding `p2_*`, `p3_*`, `p3b_*`, `e18_*`, and `e19_*` logs
