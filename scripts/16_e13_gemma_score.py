"""Score E13 (Gemma cross-family) joint-Mahalanobis against the Gemma pilot library."""
import json
from pathlib import Path
import numpy as np

LOG = Path(__file__).parent.parent / "results"
pilot = json.loads((LOG / "e12_gemma_pilot.json").read_text())
e13 = json.loads((LOG / "e13_gemma_cross_family.json").read_text())

probes = pilot["probes"]
sig_by_id = {int(k): v for k, v in pilot["sigma_calibration"].items()}

rng = np.random.default_rng(0)
N_HONEST = 500
mu = np.stack([np.array(sig_by_id[i]["mean_cross_backend"]) for i in range(96)])
sd = np.clip(np.stack([np.array(sig_by_id[i]["sigma_cross_backend"]) for i in range(96)]), 1e-3, None)

honest = rng.normal(mu[None], sd[None], size=(N_HONEST, 96, 32))

def maha_per_probe(z, k=32):
    return np.abs((z[..., :k] - mu[:, :k]) / sd[:, :k]).mean(-1)

honest_joint = maha_per_probe(honest).mean(-1)
tau = float(np.quantile(honest_joint, 0.99))
print(f"Gemma honest synthetic joint: median={np.median(honest_joint):.3f} 99-pct={tau:.3f}")
print()
print(f"{'attacker':>32} {'R²_tr':>7} {'R²_ho':>8} {'joint_maha':>10} {'×τ':>6} {'detected':>9}")
print("-"*82)

results = []
for aid, r in e13.items():
    z = np.zeros((96, 32))
    for row in r["per_probe"]:
        z[row["probe_id"]] = row["mean"]
    joint = float(maha_per_probe(z).mean())
    ratio = joint / tau
    detected = joint > tau
    results.append({
        "attacker": aid, "r2_train": r["linear_fit_r2_train"],
        "r2_heldout": r["linear_fit_r2_heldout"],
        "joint_maha": joint, "ratio_tau": ratio,
        "detected_at_fpr1pct": detected,
    })
    print(f"{aid:>32} {r['linear_fit_r2_train']:>7.3f} {r['linear_fit_r2_heldout']:>8.3f} {joint:>10.2f} {ratio:>5.2f}x {('YES' if detected else 'NO'):>9}")

out = {
    "tau_honest_joint_99th_pct": tau,
    "honest_joint_median": float(np.median(honest_joint)),
    "per_attacker": results,
    "n_detected": sum(1 for s in results if s["detected_at_fpr1pct"]),
    "n_total": len(results),
}
(LOG / "e13_gemma_scored.json").write_text(json.dumps(out, indent=2))
print(f"\n[save] {LOG/'e13_gemma_scored.json'}")
print(f"Detection: {out['n_detected']}/{out['n_total']} at FPR=1%")
