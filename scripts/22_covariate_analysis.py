"""E-Cov (W4/Q2): per-class covariate analysis.

Reviewer asks whether the distillation asymmetry between attention-pattern
circuits (induction/IOI/coref) and surface circuits (syntax/factual/lang)
is a property of the *circuits* or of the *probes*. Attention-pattern probes
may simply be harder for an attacker to match because they elicit more
distinctive SAE features.

We test three confounds per class:
  (a) top-32 feature-set entropy  — how "spread" the top-k support is
      across the SAE width (high entropy = many distinct features used
      across the class's 12 probes; low entropy = same few features
      keep showing up).
  (b) honest-pool sigma magnitude  — per-class median of sigma_{i,j}
      averaged across the class's 32-feature support. Larger sigma
      means honest variation is wider, so a fixed deviation produces
      smaller z. This is the most direct "is this class easy to fake"
      proxy purely from calibration.
  (c) feature-ID uniqueness  — number of distinct SAE feature IDs
      appearing in the union of top-32 across the 12 probes of the
      class, divided by 32*12. Low → repeating features (one circuit
      activated repeatedly); high → diffuse features.

Then we look at the rank correlation (Spearman) between these covariates
and the per-class attackability from the master results (lower joint-z
under StageA = more attackable).

Pure CPU. ~30s. Output: results/22_covariate_analysis.json
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

LOG = Path(__file__).parent.parent / "results"
lib = json.loads((LOG / "probe_library_qwen3_1.7b_L14_k96.json").read_text())
sig = json.loads((LOG / "sigma_calibration_qwen3_1.7b_L14.json").read_text())

# Group probes by class
by_cat: dict[str, list[dict]] = {}
for p in lib["probes"]:
    by_cat.setdefault(p["category"], []).append(p)

sig_by_pid = {r["probe_id"]: r for r in sig["calibration"]}


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    return -sum((c / total) * math.log(c / total)
                for c in counter.values() if c > 0)


per_class: dict[str, dict[str, float]] = {}
for cat, probes in sorted(by_cat.items()):
    # (a) top-32 feature-set entropy across class
    union = Counter()
    for p in probes:
        for f in p["top_k_feature_ids"]:
            union[f] += 1
    n_feat_slots = sum(union.values())            # 32 * |class|
    n_unique = len(union)
    feature_entropy = shannon_entropy(union)
    feature_uniqueness = n_unique / max(n_feat_slots, 1)

    # (b) honest-pool sigma magnitude — median across the class's per-feature sigmas
    sigmas: list[float] = []
    for p in probes:
        cal = sig_by_pid.get(p["probe_id"])
        if cal is None:
            continue
        sigmas.extend(cal["sigma_cross_backend"])
    sigma_med = float(np.median(sigmas)) if sigmas else float("nan")
    sigma_mean = float(np.mean(sigmas)) if sigmas else float("nan")

    # honest within-class joint-z variance — taken from the
    # class's specificity field (top_k_specificity is a per-probe robustness score)
    spec_vals = [p.get("top_k_specificity", float("nan")) for p in probes]
    spec_med = float(np.nanmedian(spec_vals)) if spec_vals else float("nan")

    per_class[cat] = {
        "n_probes": len(probes),
        "feature_entropy": feature_entropy,
        "feature_uniqueness": feature_uniqueness,
        "n_unique_features": n_unique,
        "sigma_median": sigma_med,
        "sigma_mean": sigma_mean,
        "specificity_median": spec_med,
    }

# Attackability ranking from §4.3 (StageA per-cat min joint-z, Qwen3, r=64)
# Lower = more attackable.
# Values pulled from e16/e4_v2 results in the paper.
# These come from results/e4_v2_library_aware.json -> stageA_pure_probe / per_cat_mean_z
v2_path = LOG / "e4_v2_library_aware.json"
attackability: dict[str, float] = {}
if v2_path.exists():
    v2 = json.loads(v2_path.read_text())
    pc = v2["results"]["stageA_pure_probe"]["per_cat_mean_z"]
    attackability = {c: float(v) for c, v in pc.items()}
else:
    # fallback: use e16 r=128 if v2 absent
    e16 = json.loads((LOG / "e16_whitebox_jointz_qwen3_p02_r128_mse.json").read_text())
    attackability = {c: float(v) for c, v in
                     e16["results"]["alpha_jz_0.0_lambda_util_0.0"]["per_cat_mean_z"].items()}

print(f"[load] {len(per_class)} classes, attackability source has {len(attackability)} classes")

# Align: only classes that appear in BOTH
common = sorted(set(per_class) & set(attackability))
print(f"[align] {len(common)} classes in common")

x_entropy = np.array([per_class[c]["feature_entropy"] for c in common])
x_unique = np.array([per_class[c]["feature_uniqueness"] for c in common])
x_sigma = np.array([per_class[c]["sigma_median"] for c in common])
y_attack = np.array([attackability[c] for c in common])

# Spearman: positive corr means high covariate → high joint-z (less attackable)
sp_ent = spearmanr(x_entropy, y_attack)
sp_uni = spearmanr(x_unique, y_attack)
sp_sig = spearmanr(x_sigma, y_attack)

# Also partial: regress y on (entropy, uniqueness, sigma) and report residual ranking
# vs the labelled "attention-pattern" tag
ATTN_CATS = {"ioi", "induction", "coref"}
attn_mask = np.array([c in ATTN_CATS for c in common])
y_resid = y_attack.copy()
# Simple linear partial-out:
X = np.stack([x_entropy, x_unique, x_sigma], axis=1)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
beta, *_ = np.linalg.lstsq(np.hstack([X, np.ones((len(common), 1))]), y_attack, rcond=None)
y_pred_from_cov = X @ beta[:3] + beta[3]
y_residual = y_attack - y_pred_from_cov

attn_residual_mean = float(y_residual[attn_mask].mean())
surf_residual_mean = float(y_residual[~attn_mask].mean())
attn_raw_mean = float(y_attack[attn_mask].mean())
surf_raw_mean = float(y_attack[~attn_mask].mean())

print(f"\n=== per-class covariates ===")
for c in common:
    pc = per_class[c]
    tag = "[attn]" if c in ATTN_CATS else "[surf]"
    print(f"  {tag} {c:>11}  entropy={pc['feature_entropy']:.3f}  "
          f"uniq={pc['feature_uniqueness']:.3f}  "
          f"sigma={pc['sigma_median']:.3f}  "
          f"attack_z={attackability[c]:.2f}")

print(f"\n=== Spearman (covariate vs joint-z under attack) ===")
print(f"  feature_entropy   rho={sp_ent.statistic:+.3f}  p={sp_ent.pvalue:.3f}")
print(f"  feature_uniqueness rho={sp_uni.statistic:+.3f}  p={sp_uni.pvalue:.3f}")
print(f"  sigma_median      rho={sp_sig.statistic:+.3f}  p={sp_sig.pvalue:.3f}")

print(f"\n=== attn vs surface gap (raw vs covariate-partialled residual) ===")
print(f"  raw mean joint-z      attn={attn_raw_mean:.2f}  surf={surf_raw_mean:.2f}  gap={attn_raw_mean - surf_raw_mean:+.2f}")
print(f"  residual mean         attn={attn_residual_mean:+.2f}  surf={surf_residual_mean:+.2f}  gap={attn_residual_mean - surf_residual_mean:+.2f}")
print(f"  -> if residual gap >> 0, asymmetry survives partialling out covariates.")

out = {
    "per_class": per_class,
    "attackability_used": attackability,
    "common_classes": common,
    "attn_cats": sorted(ATTN_CATS),
    "spearman": {
        "feature_entropy": {"rho": float(sp_ent.statistic), "p": float(sp_ent.pvalue)},
        "feature_uniqueness": {"rho": float(sp_uni.statistic), "p": float(sp_uni.pvalue)},
        "sigma_median": {"rho": float(sp_sig.statistic), "p": float(sp_sig.pvalue)},
    },
    "partial_regression": {
        "beta_entropy": float(beta[0]),
        "beta_uniqueness": float(beta[1]),
        "beta_sigma": float(beta[2]),
        "intercept": float(beta[3]),
        "attn_mean_raw": attn_raw_mean,
        "surf_mean_raw": surf_raw_mean,
        "raw_gap": float(attn_raw_mean - surf_raw_mean),
        "attn_mean_residual": attn_residual_mean,
        "surf_mean_residual": surf_residual_mean,
        "residual_gap": float(attn_residual_mean - surf_residual_mean),
    },
}

out_path = LOG / "22_covariate_analysis.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n[save] {out_path}")
