# Committed SAE Feature Traces

This repository contains the anonymized artifact for the EMNLP 2026 submission
on sparse-autoencoder feature traces for hosted language-model verification.

The artifact includes:

- probe-library extraction and calibration scripts;
- same-family, cross-family, adaptive LoRA, white-box, and SVIP comparison
  experiment scripts;
- Modal entrypoints used for GPU-backed experiments;
- JSON result logs used to produce the paper tables.

## Layout

```text
pilots/p1_probe_lib/
  00_smoke_test.py                       small local smoke test
  01_extract_probes.py                   Qwen3 probe-library extraction
  07_e7_cross_backend_calibration.py     cross-backend sigma calibration
  08_e2_same_family_separability.py      same-family separability
  09_e3_score.py                         cross-family post-hoc scoring
  13_e5_joint_n_sweep.py                 joint-probe count sweep
  14_calibration_bootstrap.py            threshold bootstrap analysis
  15_e11_svip_analysis.py                SVIP comparison
  16_e13_gemma_score.py                  Gemma scoring
  17_honest_pool_expansion.py            honest pool expansion
  18_ksweep_posthoc.py                   top-k sensitivity post-hoc
  19_mask_flip.py                        mask-flip analysis
  20_intrinsic_dim_analysis.py           intrinsic dimension analysis
  22_covariate_analysis.py               per-class covariate analysis
  modal_runs/                            Modal GPU entrypoints
  logs/                                  paper result JSON files

```

See `ARTIFACTS.md` for the mapping from paper claims/tables to scripts and
checked-in result logs.

## Environment

The CPU-only post-hoc analysis scripts use:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The full model experiments require GPU execution and access to the public
Hugging Face model/SAE releases listed in `pilots/p1_probe_lib/config.py`.
Modal scripts expect a Modal secret named `huggingface-secret` containing
`HF_TOKEN` for gated model access where required.

## Quick Checks

Run post-hoc analyses from the repository root:

```bash
python pilots/p1_probe_lib/09_e3_score.py
python pilots/p1_probe_lib/14_calibration_bootstrap.py
python pilots/p1_probe_lib/15_e11_svip_analysis.py
python pilots/p1_probe_lib/18_ksweep_posthoc.py
python pilots/p1_probe_lib/22_covariate_analysis.py
```

## Notes

The checked-in JSON logs are the result artifacts used for the submission.
Large model weights, Hugging Face caches, Modal volumes, and local virtual
environments are intentionally excluded.
