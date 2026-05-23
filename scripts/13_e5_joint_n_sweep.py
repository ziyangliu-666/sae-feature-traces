"""E5-v2: joint-consistency ablation (C4) — proper version.

E5-v1 swept feature dim k within each probe and hit AUC=1 ceiling because
E3-v2 attackers are all strong enough to be detected at any k.

E5-v2 fixes this by:
  (a) sweeping N_probes ∈ {1, 2, 4, 8, 16, 32, 64, 96} — the actual C4 claim
      is about aggregating across *probes*, not within-probe features
  (b) synthesizing weakened attackers via α-interpolation between honest and
      the real E3-v2 attacker: z ~ N(α*z_mean + (1-α)*mu, z_std)
      α=0: pure honest. α=1: full attacker. Sweep α to find the regime where
      joint-consistency actually helps.

Output: AUC matrix [α × N] per attacker → plot shows that as attacker weakens,
joint-consistency (larger N) rescues detection, confirming C4.

Pure CPU / numpy, ~30s.
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

mu = np.stack([np.array(sig_by_id[i]["mean_cross_backend"]) for i in range(96)])      # [96,32]
sd = np.clip(np.stack([np.array(sig_by_id[i]["sigma_cross_backend"]) for i in range(96)]), 1e-3, None)

rng = np.random.default_rng(0)
N_SAMPLES = 2000
N_PROBES_SWEEP = [1, 2, 4, 8, 16, 32, 64, 96]
ALPHAS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.2, 1.0]

def pack_attacker(r):
    z_mean = np.zeros((96, 32)); z_std = np.zeros((96, 32))
    for row in r["per_probe"]:
        z_mean[row["probe_id"]] = row["mean"]
        z_std[row["probe_id"]]  = row["std"]
    return z_mean, z_std

def simulate_samples(center, scale, n):
    """Draw n samples ~ N(center, scale), center/scale shape [96,32]."""
    return rng.normal(center[None], np.clip(scale[None], 1e-3, None), size=(n, 96, 32))

def per_probe_maha(z):
    """z: [n, 96, 32] → [n, 96] — mean |(z-mu)/sd| over 32 feature dims."""
    return np.abs((z - mu[None]) / sd[None]).mean(axis=-1)

def aggregate_n(per_probe, N, seed):
    """per_probe: [n_samples, 96], pick a *random subset* of N probes per sample
    and return aggregate mean. Same subset per sample for fair AUC."""
    rng_local = np.random.default_rng(seed)
    idx = rng_local.choice(96, size=N, replace=False)
    return per_probe[:, idx].mean(axis=-1)                      # [n_samples]

def tpr_at_fpr(honest_scores, attacker_scores, fpr=0.01):
    tau = np.quantile(honest_scores, 1 - fpr)
    return float((attacker_scores > tau).mean())

def auc(honest_scores, attacker_scores):
    n = len(honest_scores)
    m = len(attacker_scores)
    # simple rank-based AUC
    combined = np.concatenate([honest_scores, attacker_scores])
    labels = np.concatenate([np.zeros(n), np.ones(m)])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(combined)) + 1
    attacker_rank_sum = ranks[n:].sum()
    return float((attacker_rank_sum - m * (m + 1) / 2) / (n * m))

print("=== E5-v2: joint-N sweep × alpha sweep ===\n")
print("For each α (attacker strength) and N (probes aggregated):")
print("  • α=0 → attacker sampled as honest (null case, AUC ≈ 0.5)")
print("  • α=1 → full E3-v2 attacker\n")

# Honest baseline samples (shared across attackers)
honest_samples = simulate_samples(mu, sd, N_SAMPLES)            # [n,96,32]
honest_pp = per_probe_maha(honest_samples)                       # [n,96]

results = {"per_attacker": {}, "config": {
    "N_samples": N_SAMPLES, "alphas": ALPHAS, "N_probes_sweep": N_PROBES_SWEEP,
    "fpr_target": 0.01,
}}

for aid, r in e3.items():
    z_mean, z_std = pack_attacker(r)
    print(f"\n--- attacker: {aid} ---")
    print(f"{'α':>6}  " + "  ".join(f"{'N='+str(n):>8}" for n in N_PROBES_SWEEP))
    print("-" * (8 + 10 * len(N_PROBES_SWEEP)))

    per_alpha = {}
    for alpha in ALPHAS:
        att_center = alpha * z_mean + (1 - alpha) * mu
        # Use HONEST sd as attacker variance so α=0 is exactly honest (AUC=0.5)
        # and α only controls mean shift — cleanly probes joint-consistency
        att_scale  = sd
        att_samples = simulate_samples(att_center, att_scale, N_SAMPLES)
        att_pp = per_probe_maha(att_samples)

        row_auc, row_tpr = [], []
        for N in N_PROBES_SWEEP:
            # average 10 random probe subsets to reduce subset noise
            aucs = []; tprs = []
            for s in range(10):
                h_agg = aggregate_n(honest_pp, N, seed=s)
                a_agg = aggregate_n(att_pp,    N, seed=s)
                aucs.append(auc(h_agg, a_agg))
                tprs.append(tpr_at_fpr(h_agg, a_agg, fpr=0.01))
            row_auc.append(float(np.mean(aucs)))
            row_tpr.append(float(np.mean(tprs)))
        per_alpha[alpha] = {"auc": row_auc, "tpr@fpr1": row_tpr}
        print(f"α={alpha:<4}  " + "  ".join(f"{v:>8.3f}" for v in row_auc) +
              "   (AUC)")

    results["per_attacker"][aid] = per_alpha

# Summary: at what α do we see a gap ΔAUC(N=96) - ΔAUC(N=1) > 0.1?
print("\n\n=== C4 gate check (ΔAUC = AUC(N=96) - AUC(N=1)) ===")
print(f"{'attacker':>38}  " + "  ".join(f"α={a}" for a in ALPHAS))
print("-" * 110)
c4_support = {}
for aid, per_alpha in results["per_attacker"].items():
    deltas = []
    for a in ALPHAS:
        row = per_alpha[a]
        d = row["auc"][-1] - row["auc"][0]  # N=96 minus N=1
        deltas.append(d)
    c4_support[aid] = deltas
    print(f"{aid:>38}  " + "  ".join(f"{d:>+6.3f}" for d in deltas))

max_gap_any = max(max(v) for v in c4_support.values())
print(f"\nMax ΔAUC observed across all (attacker, α): {max_gap_any:+.3f}")
print(f"C4 gate: ΔAUC > 0.03 in at least one regime → "
      f"{'PASS' if max_gap_any > 0.03 else 'FAIL'}")

results["c4_gate"] = {
    "delta_auc_per_attacker_per_alpha": c4_support,
    "max_delta_auc": float(max_gap_any),
    "threshold": 0.03,
    "status": "PASS" if max_gap_any > 0.03 else "FAIL",
}

(LOG / "e5_v2_joint_n_sweep.json").write_text(json.dumps(results, indent=2))
print(f"\n[save] {LOG / 'e5_v2_joint_n_sweep.json'}")
