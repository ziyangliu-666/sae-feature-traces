"""E-F post-hoc k-sweep.

Consumes:
  logs/recipe1_qwen3_honest_pool_multikernel.json  (per_probe_z: [96, 32] per config)
  logs/e16_whitebox_jointz_qwen3_ksweep_strongest.json  (z_top32: [96, 32])
  logs/sigma_calibration_qwen3_1.7b_L14.json  (mu, sd at top-32)

For each k' in {4, 8, 16, 32}:
  honest joint-z = mean_{probes}[mean_{j<k'}(|z[p,j] - mu[p,j]| / sd[p,j])]
  tau_real(k')  = max honest joint-z
  attacker joint-z at same k'
  margin ratio = attacker_jz / tau_real

Writes logs/ksweep_results.json.
"""
import json
import numpy as np
from pathlib import Path

LOG = Path(__file__).parent / "logs"
HONEST = LOG / "recipe1_qwen3_honest_pool_multikernel.json"
ATTACKER = LOG / "e16_whitebox_jointz_qwen3_ksweep_strongest.json"
SIG = LOG / "sigma_calibration_qwen3_1.7b_L14.json"
OUT = LOG / "ksweep_results.json"

K_VALUES = [4, 8, 16, 32]

def load():
    hon = json.loads(HONEST.read_text())
    att = json.loads(ATTACKER.read_text())
    sig = json.loads(SIG.read_text())
    return hon, att, sig

def build_musd(sig):
    n_probes = max(c["probe_id"] for c in sig["calibration"]) + 1
    mu = np.zeros((n_probes, 32))
    sd = np.zeros((n_probes, 32))
    for c in sig["calibration"]:
        pid = c["probe_id"]
        mu[pid] = np.array(c["mean_cross_backend"])
        sd[pid] = np.clip(np.array(c["sigma_cross_backend"]), 1e-3, None)
    return mu, sd

def sweep_honest(hon, mu, sd):
    results = {}
    configs = hon.get("all_configs", [])
    if not configs:
        print("WARNING: no all_configs in honest JSON")
        return results
    z_arr = np.array([c["per_probe_z"] for c in configs])  # [n_cfg, n_probes, 32]
    for k in K_VALUES:
        maha = np.abs((z_arr[:, :, :k] - mu[None, :, :k]) / sd[None, :, :k]).mean(-1)
        joints = maha.mean(-1)
        results[k] = {
            "n_configs": int(z_arr.shape[0]),
            "min": float(joints.min()),
            "median": float(np.median(joints)),
            "mean": float(joints.mean()),
            "max": float(joints.max()),
            "p99": float(np.quantile(joints, 0.99)),
        }
    return results

def sweep_attacker(att, mu, sd):
    results = {}
    res = att["results"]
    key = list(res.keys())[0]
    z = np.array(res[key]["z_top32"])  # [n_probes, 32]
    for k in K_VALUES:
        maha = np.abs((z[:, :k] - mu[:, :k]) / sd[:, :k]).mean(-1)
        joint = float(maha.mean())
        results[k] = {
            "joint_z": joint,
            "per_probe_max": float(maha.max()),
            "per_probe_median": float(np.median(maha)),
        }
    return results, key

def main():
    hon, att, sig = load()
    mu, sd = build_musd(sig)
    print(f"[config] n_probes={mu.shape[0]}, top_k_stored=32")

    honest_k = sweep_honest(hon, mu, sd)
    attacker_k, attacker_key = sweep_attacker(att, mu, sd)

    print(f"\n[attacker config] {attacker_key}")
    print(f"{'k':>4}  {'tau_real':>10}  {'attacker_jz':>12}  {'margin':>8}")
    out = {"k_values": K_VALUES, "honest": {}, "attacker": {}, "margin": {}}
    for k in K_VALUES:
        tau = honest_k[k]["max"]
        a_jz = attacker_k[k]["joint_z"]
        margin = a_jz / tau if tau > 0 else float("inf")
        out["honest"][k] = honest_k[k]
        out["attacker"][k] = attacker_k[k]
        out["margin"][k] = {"tau_real": tau, "attacker_jz": a_jz, "ratio": margin}
        print(f"{k:>4}  {tau:>10.4f}  {a_jz:>12.4f}  {margin:>7.2f}x")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {OUT}")

if __name__ == "__main__":
    main()
