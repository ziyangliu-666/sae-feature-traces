"""Score E3-v2 with the same joint-consistency scorer."""
import json
from pathlib import Path
import numpy as np

LOG = Path(__file__).parent.parent / "results"
lib = json.loads((LOG / "probe_library_qwen3_1.7b_L14_k96.json").read_text())
sig = json.loads((LOG / "sigma_calibration_qwen3_1.7b_L14.json").read_text())
e3 = json.loads((LOG / "e3_cross_family_results_v2.json").read_text())

probes = lib["probes"]
sig_by_id = {r["probe_id"]: r for r in sig["calibration"]}

rng = np.random.default_rng(0)
N_HONEST = 500
mu = np.stack([np.array(sig_by_id[i]["mean_cross_backend"]) for i in range(96)])
sd = np.clip(np.stack([np.array(sig_by_id[i]["sigma_cross_backend"]) for i in range(96)]), 1e-3, None)
honest = rng.normal(mu[None], sd[None], size=(N_HONEST, 96, 32))

def maha_per_probe(z, k=32):
    return np.abs((z[..., :k] - mu[:, :k]) / sd[:, :k]).mean(-1)

honest_joint = maha_per_probe(honest).mean(-1)          # [N_HONEST]
tau_1pct = np.quantile(honest_joint, 0.99)

print("=== E3-v2 scoring (k=32 joint maha) ===")
print(f"Honest synthetic: median={np.median(honest_joint):.3f} 99th-pct(τ_FPR=1%)={tau_1pct:.3f}")
print()
print(f"{'attacker':>30} {'R²_tr':>7} {'R²_ho':>7} {'joint_maha':>10} {'detected':>9}")
print("-" * 75)

out_summary = []
for aid, r in e3.items():
    z = np.zeros((96, 32))
    for row in r["per_probe"]:
        z[row["probe_id"]] = row["mean"]
    joint = maha_per_probe(z).mean()
    detected = joint > tau_1pct
    out_summary.append({
        "attacker": aid,
        "r2_train": r["linear_fit_r2_train"],
        "r2_heldout": r["linear_fit_r2_heldout"],
        "joint_maha": float(joint),
        "detected_at_fpr1pct": bool(detected),
    })
    print(f"{aid:>30} {r['linear_fit_r2_train']:>7.3f} {r['linear_fit_r2_heldout']:>7.3f} {joint:>10.2f} {'YES' if detected else 'NO':>9}")

out = {
    "tau_honest_joint_99th_pct": float(tau_1pct),
    "honest_joint_median": float(np.median(honest_joint)),
    "per_attacker": out_summary,
    "n_detected": sum(1 for s in out_summary if s["detected_at_fpr1pct"]),
    "n_total": len(out_summary),
}
(LOG / "e3_v2_scored.json").write_text(json.dumps(out, indent=2))
print(f"\n[save] {LOG / 'e3_v2_scored.json'}")
print(f"Overall: {out['n_detected']}/{out['n_total']} detected at FPR=1%")
