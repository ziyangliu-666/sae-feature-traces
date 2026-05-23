"""E5 (joint-consistency ablation, C4) + E6 (attackability ranking).

Both are post-hoc on E3 per-probe magnitudes + E7 σ calibration. No GPU.

E5: sweep k ∈ {1,2,4,8,16,32} on the top-k feature magnitudes and measure
how well an attacker is distinguished from honest at each k. Kill gate
(EXPERIMENT_PLAN): AUC(k=32) - AUC(k=1) < 0.03 triggers C4 kill.

E6: per-category attack success rate aggregated across the 4 attackers
from E3 + the LEARNED_LIN run from E2. Extends the P3b asymmetry to k=96.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

LOG = Path(__file__).parent / "logs"
lib = json.loads((LOG / "probe_library_qwen3_1.7b_L14_k96.json").read_text())
sig = json.loads((LOG / "sigma_calibration_qwen3_1.7b_L14.json").read_text())
e3 = json.loads((LOG / "e3_cross_family_results_v2.json").read_text())

probes = lib["probes"]
sig_by_id = {r["probe_id"]: r for r in sig["calibration"]}
cat_by_id = {p["probe_id"]: p["category"] for p in probes}

# --- pack per-attacker per-probe magnitudes ---
def pack_attacker(per_probe_list):
    z = np.zeros((96, 32), dtype=np.float64)
    for row in per_probe_list:
        z[row["probe_id"]] = row["mean"]
    return z

attackers = {aid: pack_attacker(r["per_probe"]) for aid, r in e3.items()}

# honest "center" (from E7) is stored per-probe as mean_cross_backend (k=32)
mu_honest = np.stack([np.array(sig_by_id[i]["mean_cross_backend"]) for i in range(96)])
sd_honest = np.clip(np.stack([np.array(sig_by_id[i]["sigma_cross_backend"]) for i in range(96)]), 1e-3, None)

# Approximate honest sample distribution: assume honest is Gaussian(μ, σ).
# We draw N_HONEST synthetic samples per probe to compute AUC properly.
# This is conservative — honest in-distribution maha ~1 by construction.
rng = np.random.default_rng(0)
N_HONEST = 100
honest_samples = rng.normal(mu_honest[None, ...], sd_honest[None, ...], size=(N_HONEST, 96, 32))

def maha_per_probe(z_samples, k):
    """z_samples: [..., 96, 32] → maha score: [..., 96], using first-k features."""
    zk = z_samples[..., :k]
    muk = mu_honest[:, :k]
    sdk = sd_honest[:, :k]
    return np.abs((zk - muk) / sdk).mean(axis=-1)

def detection_auc(attacker_z, k):
    """Single-threshold AUC comparing attacker vs honest synthetic samples.
    Uses aggregate-over-probes maha as the score per sample (joint-consistency
    is *mean over 96 probes*, i.e., single-sample → single scalar)."""
    honest_scores = maha_per_probe(honest_samples, k).mean(axis=-1)  # [N_HONEST]
    attacker_score = maha_per_probe(attacker_z, k).mean(axis=-1)      # scalar
    rank_attacker = (honest_scores >= attacker_score).sum() / len(honest_scores)
    return float(1.0 - rank_attacker)  # AUC-like: if attacker >> honest, →1

def tpr_at_fpr(attacker_z, k, fpr=0.01):
    honest_scores = maha_per_probe(honest_samples, k).mean(axis=-1)
    tau = np.quantile(honest_scores, 1 - fpr)
    attacker_score = maha_per_probe(attacker_z, k).mean(axis=-1)
    return float(attacker_score > tau)

# ===== E5: joint-consistency k-ablation =====
print("=== E5: joint-consistency k-sweep ===")
print(f"{'k':>3}  " + "  ".join(f"{a[:20]:>20}" for a in attackers) + "   gate")
print("-" * 120)
K_VALUES = [1, 2, 4, 8, 16, 32]
e5_results = {}
for k in K_VALUES:
    aucs = {aid: detection_auc(z, k) for aid, z in attackers.items()}
    e5_results[k] = aucs
    row = f"{k:>3}  " + "  ".join(f"{aucs[a]:>20.3f}" for a in attackers)
    mean_auc = np.mean(list(aucs.values()))
    print(f"{row}   mean={mean_auc:.3f}")

auc32 = np.mean(list(e5_results[32].values()))
auc1 = np.mean(list(e5_results[1].values()))
gate = auc32 - auc1
print(f"\nAUC(k=32) - AUC(k=1) = {auc32:.3f} - {auc1:.3f} = {gate:.3f}")
print(f"C4 kill gate: <0.03 → {'KILL C4 (benchmark-only paper)' if gate < 0.03 else 'PASS'}")

# ===== E6: per-category attackability =====
print("\n=== E6: per-category attackability ===")
# For each category × attacker, compute mean maha at k=32
per_cat = defaultdict(list)
for aid, z in attackers.items():
    maha = maha_per_probe(z, k=32)  # [96]
    for pid in range(96):
        per_cat[cat_by_id[pid]].append((aid, maha[pid]))

print(f"\n{'category':>14}  n  {'med_maha':>8}  {'min_maha':>8}  {'max_maha':>8}  status")
print("-" * 70)
e6_results = {}
for cat, entries in sorted(per_cat.items()):
    mahas = [m for _, m in entries]
    med, mn, mx = np.median(mahas), np.min(mahas), np.max(mahas)
    # Attackable: median maha < 5 means attacker's magnitudes are <5σ from honest (survives)
    status = "LEAK" if med < 5 else "ROBUST"
    e6_results[cat] = {"n_entries": len(mahas), "median_maha": float(med),
                       "min_maha": float(mn), "max_maha": float(mx), "status": status}
    print(f"{cat:>14}  {len(mahas):>2}  {med:>8.1f}  {mn:>8.1f}  {mx:>8.1f}  {status}")

# Rank categories by robustness
ranked = sorted(e6_results.items(), key=lambda x: x[1]["median_maha"], reverse=True)
print(f"\nRanked by robustness (highest maha = most robust):")
for cat, r in ranked:
    print(f"  {cat:>14}: med_maha={r['median_maha']:.1f}  {r['status']}")

# ===== save =====
out = {
    "e5_joint_consistency_k_sweep": {
        str(k): {aid: float(v) for aid, v in aucs.items()}
        for k, aucs in e5_results.items()
    },
    "e5_kill_gate": {
        "auc_k32_mean": float(auc32),
        "auc_k1_mean": float(auc1),
        "delta": float(gate),
        "threshold": 0.03,
        "status": "PASS" if gate >= 0.03 else "KILL",
    },
    "e6_per_category": e6_results,
    "e6_ranked_by_robustness": [
        {"category": c, **r} for c, r in ranked
    ],
}
(LOG / "e5_e6_post_hoc.json").write_text(json.dumps(out, indent=2))
print(f"\n[save] {LOG / 'e5_e6_post_hoc.json'}")
