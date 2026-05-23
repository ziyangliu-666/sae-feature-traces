<div align="center">

# Committed SAE Feature Traces

**Sparse-autoencoder feature traces for verifying hosted language models.**
*Anonymized artifact — EMNLP 2026 submission.*

[![EMNLP 2026](https://img.shields.io/badge/EMNLP_2026-Anonymous_Submission-1f6feb)](https://2026.emnlp.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Modal](https://img.shields.io/badge/Modal-GPU_backend-7b3fe4)](https://modal.com)
[![SAE Lens](https://img.shields.io/badge/SAE_Lens-6.39-2ea44f)]()
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-models-yellow)](https://huggingface.co)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-reproducible-success)]()
[![Review](https://img.shields.io/badge/double--blind-anonymized-lightgrey)]()

[![Scripts](https://img.shields.io/badge/scripts-27-blue)]()
[![Modal entrypoints](https://img.shields.io/badge/modal-24-blueviolet)]()
[![Result logs](https://img.shields.io/badge/results-78_JSON-informational)]()
[![Backbones](https://img.shields.io/badge/backbones-Qwen3_%C2%B7_Gemma--2--2B_%C2%B7_Gemma--2--9B-orange)]()

</div>

---

## TL;DR

- A **probe-library** of SAE features, plus calibration and scoring code, lets
  a verifier check that a hosted LLM is actually serving the model it claims.
- All paper experiments — same-family, cross-family, adaptive LoRA, white-box,
  SVIP comparison, multi-backbone replication — are reproducible from the
  checked-in JSON logs without re-running GPUs.
- The full claim → script → log map is in **[`ARTIFACTS.md`](ARTIFACTS.md)**;
  headline numbers from each experiment are summarized in **[`RESULTS.md`](RESULTS.md)**.

---

## Results at a glance

All numbers are read directly from the committed JSON logs.

| Experiment | Backbone | Result | Source |
| --- | --- | --- | --- |
| Cross-family detection (E3) | Qwen3-1.7B | **4 / 4** attackers detected at FPR=1%; joint-Mahalanobis ratio **68–104× τ** | `results/e3_v2_scored.json` |
| Cross-family detection (E13) | Gemma-2-2B | **4 / 4** detected; ratio **218–412× τ** | `results/e13_gemma_scored.json` |
| Adaptive-LoRA attack (E14) | Gemma-2-2B | substitute trained to mimic target → still detected, joint-z = **205.6** vs τ ≈ 0.82 | `results/e14_gemma_adaptive_lora.json` |
| White-box joint-z LoRA (E16) | Qwen3-1.7B | unconstrained attacker (λ\_util=0) → joint-z **56.6**; with utility constraint (λ\_util=0.1) → ppl-blowup; **constrained attack fails** | `results/e16_whitebox_jointz_qwen3.json` |
| SVIP head-to-head (E11) | Qwen3-1.7B | SVIP **does not detect** any of 4 attackers at FPR=1%; ours detects all (margin **~70× signal**) | `results/e11_svip_vs_ours.json` |
| Serving overhead (E9) | Qwen3-1.7B | batch=16 → **+2.9%** wall-time, **3.6 KB** commit payload; batch=32 → **+1.8%**, 7.2 KB | `results/e9_overhead.json` |

τ here is the honest-pool 99th-percentile threshold (FPR=1%) used uniformly
across attackers; see RESULTS.md for the per-experiment definitions.

---

## Highlights

- **End-to-end pipeline** — probe extraction → cross-backend calibration →
  scoring → bootstrap and sensitivity analyses.
- **Multi-backbone replication** — same pipeline, three open-weight
  backbones (Qwen3-1.7B, Gemma-2-2B, Gemma-2-9B).
- **Adversarial stress tests** — adaptive LoRA, white-box joint-z LoRA,
  attention-only LoRA, per-head ablation, circuit ablation.
- **Head-to-head baseline** — SVIP comparison included as a first-class
  experiment, not an afterthought.
- **Offline-verifiable** — every paper table is reproducible from the
  committed JSON logs without re-running GPUs.

---

## Quickstart

### CPU-only — recompute paper tables from committed logs

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/09_e3_score.py
python scripts/14_calibration_bootstrap.py
python scripts/15_e11_svip_analysis.py
python scripts/18_ksweep_posthoc.py
python scripts/22_covariate_analysis.py
```

### GPU — re-run experiments on Modal

```bash
modal secret create huggingface-secret HF_TOKEN=...
modal run modal/e2_same_family.py
modal run modal/e3_cross_family.py
modal run modal/e12_gemma_pilot.py
```

Hugging Face access to the public Qwen3, Gemma-2, and corresponding SAE
releases is required for the GPU paths.

---

## Repository layout

```text
sae-feature-traces-anon/
├── README.md
├── ARTIFACTS.md                 paper claims → scripts → logs
├── RESULTS.md                   headline numbers from every experiment
├── LICENSE                      MIT (anonymized)
├── requirements.txt
├── config.py                    backbone / SAE / kernel configuration
├── scripts/                     27 pilot + post-hoc scripts (numbered 00–22 + e18/e19)
├── modal/                       24 Modal GPU entrypoints
└── results/                     78 result JSONs + 4 summary .txt files
```

Naming convention: numbered `0X_*` scripts follow the order of the paper;
`eXX_*` scripts are appendix / stress tests. `ARTIFACTS.md` is the
authoritative claim-to-script mapping.

---

## Reproducing the paper

| Paper item | Script | Result log |
| --- | --- | --- |
| Probe library | `scripts/01_extract_probes.py` | `results/probe_library_qwen3_1.7b_L14_k96.json` |
| Cross-backend calibration | `scripts/07_e7_cross_backend_calibration.py` | `results/sigma_calibration_qwen3_1.7b_L14.json` |
| Same-family lifts (E2) | `scripts/08_e2_same_family_separability.py` | `results/e2_separability.json` |
| Cross-family substitutes (E3) | `scripts/09_e3_score.py`, `scripts/11_e3_v2_score.py` | `results/e3_cross_family_scored.json` |
| SVIP comparison (E11) | `scripts/15_e11_svip_analysis.py` | `results/e11_svip_vs_ours.json` |

The full claim → script → log map (Qwen3 core pipeline, hosted-LLM
verification, Gemma-2 / Gemma-2-9B replication, white-box and ablations)
lives in **[`ARTIFACTS.md`](ARTIFACTS.md)**.

---

## Environment

- **Python**: 3.10+
- **CPU path**: only the requirements in `requirements.txt`.
- **GPU path**: Modal with a secret named `huggingface-secret`
  exposing `HF_TOKEN` for gated model access. Pinned versions: PyTorch
  2.6, `transformers` 4.56.2, `sae_lens` 6.39.0, `datasets` 3.1.0,
  `peft`, `scikit-learn`, `scipy`, `zstandard`.
- Backbone, SAE release, and kernel choices are centralized in
  `config.py` at the repo root.

---

## Citation

```bibtex
@inproceedings{anonymous2026saetraces,
  title     = {Committed SAE Feature Traces for Hosted LLM Verification},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026},
  note      = {Under double-blind review}
}
```

---

## License and anonymity

Released under the [MIT License](LICENSE) for double-blind review. Author
identities, institution affiliations, and identifying repository metadata
have been removed. Large model weights, Hugging Face caches, Modal
volumes, and local virtual environments are intentionally excluded —
every checked-in file is either source code, configuration, or a result
log used to produce the paper tables.
