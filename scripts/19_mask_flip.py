"""E-G: mask-flip sensitivity.

Reviewer concern W8: no mask-flip sensitivity. We perturb the top-k feature
support S_i by flipping a fraction of indices to random other features and
recompute joint-z AUC. A detector that relies fragilely on exactly the right
top-k is vulnerable to attackers that match means on the wrong support.

Method (post-hoc, CPU):
  * For each flip_fraction f in {0.0, 0.05, 0.1, 0.2, 0.4}, randomise
    floor(k*f) of the top-32 indices of each probe to a uniformly sampled
    feature index outside S_i, using (mu_i, sigma_i) still read from the
    calibrated σ table for those indices.
  * Recompute joint-z on both honest synthetic samples (drawn from the
    calibrated σ model) and the strongest E3-v2 attackers (packed from
    per_probe mean).
  * Report AUC and TPR@FPR=1% per f. Expected: monotonic degradation.

Output: results/mask_flip_sensitivity.json
"""
import json
from pathlib import Path
import numpy as np

LOG = Path(__file__).parent.parent / "results"
lib = json.loads((LOG / "probe_library_qwen3_1.7b_L14_k96.json").read_text())
sig = json.loads((LOG / "sigma_calibration_qwen3_1.7b_L14.json").read_text())
e3 = json.loads((LOG / "e3_cross_family_results_v2.json").read_text())

probes = lib["probes"]
sig_by_id = {r["probe_id"]: r for r in sig["calibration"]}

# Original top-32 (mu, sd, feature_ids) per probe
mu = np.stack([np.array(sig_by_id[i]["mean_cross_backend"]) for i in range(96)])
sd = np.clip(np.stack([np.array(sig_by_id[i]["sigma_cross_backend"]) for i in range(96)]), 1e-3, None)
top32 = np.stack([np.array(sig_by_id[i]["feature_ids"]) for i in range(96)])  # [96,32]

D_SAE = 163840  # Qwen3 transcoder dimensionality
K = 32
N_HONEST = 2000
FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.4]
N_SEEDS = 20  # average over random flip draws

rng = np.random.default_rng(0)


def pack_attacker(r):
    z_mean = np.zeros((96, 32))
    for row in r["per_probe"]:
        z_mean[row["probe_id"]] = row["mean"]
    return z_mean


def auc(honest_scores, attacker_scores):
    n, m = len(honest_scores), len(attacker_scores)
    combined = np.concatenate([honest_scores, attacker_scores])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(combined)) + 1
    return float((ranks[n:].sum() - m * (m + 1) / 2) / (n * m))


def tpr_at_fpr(honest_scores, attacker_scores, fpr=0.01):
    tau = np.quantile(honest_scores, 1 - fpr)
    return float((attacker_scores > tau).mean())


def apply_flip(frac, seed):
    """Return (mu_flipped, sd_flipped, flipped_mask)[96,32].

    For a given flip fraction f, flip floor(f*32) indices per probe to random
    features outside S_i. For the *flipped* indices we assign mu=0 and sd=1
    (i.e. background noise model) — attackers that matched on S_i get no
    credit on the flipped slots.
    """
    m_out = mu.copy()
    s_out = sd.copy()
    n_flip = int(np.floor(frac * K))
    if n_flip == 0:
        return m_out, s_out
    r = np.random.default_rng(seed)
    for pid in range(96):
        flip_slots = r.choice(K, size=n_flip, replace=False)
        m_out[pid, flip_slots] = 0.0
        s_out[pid, flip_slots] = 1.0
    return m_out, s_out


def joint_z(z_samples, mu_arr, sd_arr):
    return np.abs((z_samples - mu_arr[None]) / sd_arr[None]).mean(-1).mean(-1)


# Baseline honest samples (independent of flip — honest is drawn from the
# *original* (mu, sd) because the attacker sees the original support)
honest_samples = rng.normal(mu[None], sd[None], size=(N_HONEST, 96, 32))

results = {"config": {
    "flip_fractions": FRACTIONS,
    "n_seeds": N_SEEDS,
    "n_honest": N_HONEST,
    "k": K,
}, "per_attacker": {}, "summary": {}}

raw_attackers = {aid: pack_attacker(r) for aid, r in e3.items()}

# α-weakened attackers expose the regime where mask-flip matters: at small α,
# the attacker is near the honest distribution and a disrupted mask can tip
# detection either way. Same construction as 13_e5_joint_n_sweep.py.
ALPHAS = [0.001, 0.005, 0.01, 0.05, 1.0]

def weaken(z_mean, alpha):
    return alpha * z_mean + (1 - alpha) * mu

attackers_weakened = {}
for aid, z_mean in raw_attackers.items():
    for alpha in ALPHAS:
        tag = f"{aid.split('/')[-1]}@α={alpha}"
        attackers_weakened[tag] = weaken(z_mean, alpha)

# Attacker "samples" under calibrated honest sd
attacker_samples = {aid: rng.normal(z[None], sd[None], size=(200, 96, 32))
                    for aid, z in attackers_weakened.items()}

print("=== E-G: mask-flip sensitivity ===\n")
print(f"{'attacker (α-weakened)':>38}  " + "  ".join(f"f={f}" for f in FRACTIONS))
print("-" * (42 + 10 * len(FRACTIONS)))

# For fraction=0, use original mask; for >0, average over N_SEEDS flip draws
for aid, att_samples in attacker_samples.items():
    aucs_per_f = []
    tprs_per_f = []
    for f in FRACTIONS:
        aucs, tprs = [], []
        trials = 1 if f == 0.0 else N_SEEDS
        for s in range(trials):
            m_f, s_f = apply_flip(f, seed=s + 1)
            honest_z = joint_z(honest_samples, m_f, s_f)
            att_z = joint_z(att_samples, m_f, s_f)
            aucs.append(auc(honest_z, att_z))
            tprs.append(tpr_at_fpr(honest_z, att_z, fpr=0.01))
        aucs_per_f.append(float(np.mean(aucs)))
        tprs_per_f.append(float(np.mean(tprs)))
    results["per_attacker"][aid] = {"auc": aucs_per_f, "tpr@fpr1": tprs_per_f}
    row = "  ".join(f"{v:>4.3f}" for v in aucs_per_f)
    print(f"{aid:>38}  {row}   (AUC)")

# Summary across attackers
aucs_matrix = np.array([r["auc"] for r in results["per_attacker"].values()])
tprs_matrix = np.array([r["tpr@fpr1"] for r in results["per_attacker"].values()])
summary = {
    "auc_mean_per_fraction": aucs_matrix.mean(0).tolist(),
    "auc_min_per_fraction": aucs_matrix.min(0).tolist(),
    "tpr_mean_per_fraction": tprs_matrix.mean(0).tolist(),
    "degradation_auc": float(aucs_matrix[:, 0].mean() - aucs_matrix[:, -1].mean()),
    "degradation_tpr": float(tprs_matrix[:, 0].mean() - tprs_matrix[:, -1].mean()),
}
results["summary"] = summary

print("\n--- summary across attackers ---")
print(f"mean AUC:      " + "  ".join(f"{v:.3f}" for v in summary["auc_mean_per_fraction"]))
print(f"min  AUC:      " + "  ".join(f"{v:.3f}" for v in summary["auc_min_per_fraction"]))
print(f"mean TPR@1%:   " + "  ".join(f"{v:.3f}" for v in summary["tpr_mean_per_fraction"]))
print(f"AUC degradation (f=0 → f={FRACTIONS[-1]}): {summary['degradation_auc']:+.3f}")
print(f"TPR degradation:                            {summary['degradation_tpr']:+.3f}")

# Monotonicity check
mono = all(summary["auc_mean_per_fraction"][i] >= summary["auc_mean_per_fraction"][i+1] - 1e-3
           for i in range(len(FRACTIONS) - 1))
print(f"Monotonic AUC degradation in f: {'YES' if mono else 'NO'}")
results["summary"]["monotonic"] = bool(mono)

(LOG / "mask_flip_sensitivity.json").write_text(json.dumps(results, indent=2))
print(f"\n[save] {LOG / 'mask_flip_sensitivity.json'}")
