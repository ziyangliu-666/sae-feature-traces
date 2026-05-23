"""F3 local: probe-aware exact-optimum feature-forgery attacker.

Since (a) the attacker commits one top-32 feature selection per position,
(b) the probe library is position-invariant, and (c) joint-z is a
separable sum over probe-slots, the optimal attacker strategy has a
closed form. Gradient descent / Gumbel-softmax (the "F3" strategy in
the original plan) cannot exceed this bound — this script computes it
exactly.

Attacker picks T = {f1, ..., f32} (distinct feature ids) with values
v(fi) in bf16. For each probe i and slot s with feature id f = S[i,s]:
  penalty(i, s) = |v(f) - mu[i,s]| / sigma[i,s]  if f in T
                = |mu[i,s]| / sigma[i,s]         otherwise
joint-z = mean over 96 probes of mean over 32 slots of penalty(i,s)
       = (1/3072) * sum over (i,s) of penalty(i,s).

Per feature f, define the "gain" of including it with optimal value:
  G*(f) = max_v sum over (i,s) with S[i,s]=f of
          [|mu[i,s]|/sigma[i,s] - |v - mu[i,s]|/sigma[i,s]]
The unconstrained best v* is the weighted-median of {mu[i,s]} weighted
by {1/sigma[i,s]} — solved by sort + cumulative-weight lookup.

Given G*(f) for every feature, optimal T = top-32 by G*.

Usage: python scripts/e18_forgery_f3_local.py
Output: results/e18_forgery_f3_local.json
"""
from pathlib import Path
import json
import numpy as np
from collections import defaultdict


ROOT = Path(__file__).resolve().parent
LIB_PATH = ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_PATH = ROOT / "logs" / "sigma_calibration_qwen3_1.7b_L14.json"
OUT_PATH = ROOT / "logs" / "e18_forgery_f3_local.json"

TOP_K = 32
TAU_POOL = 1.509


def load_library():
    lib = json.loads(LIB_PATH.read_text())
    sig = json.loads(SIG_PATH.read_text())
    probes = lib["probes"]
    sig_by_id = {r["probe_id"]: r for r in sig["calibration"]}
    S = np.array([p["top_k_feature_ids"][:TOP_K] for p in probes])
    mu = np.array([sig_by_id[p["probe_id"]]["mean_cross_backend"] for p in probes])
    sd = np.clip(
        np.array([sig_by_id[p["probe_id"]]["sigma_cross_backend"] for p in probes]),
        1e-3, None,
    )
    return S, mu, sd


def weighted_median(values, weights):
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order]
    cw = np.cumsum(sw)
    half = cw[-1] / 2.0
    idx = np.searchsorted(cw, half)
    idx = min(idx, len(sv) - 1)
    return float(sv[idx])


def bf16_round(x):
    """Emulate bf16 rounding: keep 8-bit exponent + 7-bit mantissa."""
    f32 = np.float32(x).view(np.uint32)
    # round-to-nearest-even: add 0x8000 then mask
    rounded = (f32 + 0x8000) & 0xFFFF0000
    return float(np.uint32(rounded).view(np.float32))


def solve_f3(S, mu, sd):
    """Returns optimal (picked_features, values, joint_z, per_feature_gain)."""
    n_probes, k = S.shape
    total_slots = n_probes * k

    ratio = np.abs(mu) / sd  # (96, 32) — baseline penalty if attacker zeroes
    total_baseline = float(ratio.sum() / total_slots)  # = F0 expected joint-z

    # For each feature id, collect its (mu, sigma) occurrences across probes
    feat_occur = defaultdict(list)
    for i in range(n_probes):
        for s in range(k):
            fid = int(S[i, s])
            feat_occur[fid].append((mu[i, s], sd[i, s]))

    # For each feature, compute G*(f) and optimal v*
    feat_gain = {}
    feat_value = {}
    for fid, occs in feat_occur.items():
        mus = np.array([m for m, _ in occs])
        sds = np.array([s for _, s in occs])
        weights = 1.0 / sds
        baseline_sum = float((np.abs(mus) / sds).sum())  # penalty if attacker=0

        v_star_cont = weighted_median(mus, weights)
        v_star = bf16_round(v_star_cont)  # bf16 quantize
        reduced_sum = float((np.abs(v_star - mus) / sds).sum())

        feat_gain[fid] = baseline_sum - reduced_sum
        feat_value[fid] = v_star

    # Pick top-32 features by gain
    sorted_feats = sorted(feat_gain.items(), key=lambda x: -x[1])
    picked = [f for f, _ in sorted_feats[:TOP_K]]
    total_gain = sum(feat_gain[f] for f in picked)

    # Now compute actual joint-z with picked set
    joint_z_optimal = float((total_baseline * total_slots - total_gain) / total_slots)

    # Per-probe breakdown under this attack
    picked_set = set(picked)
    per_probe_z = np.zeros(n_probes)
    coverage = np.zeros(n_probes, dtype=int)
    for i in range(n_probes):
        row_sum = 0.0
        for s in range(k):
            fid = int(S[i, s])
            if fid in picked_set:
                penalty = abs(feat_value[fid] - mu[i, s]) / sd[i, s]
                coverage[i] += 1
            else:
                penalty = abs(mu[i, s]) / sd[i, s]
            row_sum += penalty
        per_probe_z[i] = row_sum / k

    return {
        "picked_features": picked,
        "picked_values": [feat_value[f] for f in picked],
        "per_feature_gain_sorted": [feat_gain[f] for f in picked],
        "top_gain_feature_gains": [feat_gain[f] for f, _ in sorted_feats[:20]],
        "joint_z_optimal": joint_z_optimal,
        "joint_z_baseline_zero": total_baseline,
        "total_gain": total_gain,
        "coverage_mean": float(coverage.mean()),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "per_probe_z_min": float(per_probe_z.min()),
        "per_probe_z_median": float(np.median(per_probe_z)),
        "per_probe_z_max": float(per_probe_z.max()),
        "margin_over_tau": joint_z_optimal / TAU_POOL,
    }


def main():
    S, mu, sd = load_library()
    print(f"Loaded library: {S.shape[0]} probes, top-{S.shape[1]} each.")
    print(f"|union S_i| = {len(set(S.flatten().tolist()))}")
    print(f"tau_pool = {TAU_POOL}")
    print()

    res = solve_f3(S, mu, sd)

    report = {
        "metadata": {
            "library": str(LIB_PATH.relative_to(ROOT)),
            "sigma_calibration": str(SIG_PATH.relative_to(ROOT)),
            "tau_pool": TAU_POOL,
            "attacker_model": "F3-exact-optimum-discrete-selection",
            "notes": (
                "Closed-form optimum for attacker picking top-32 feature ids "
                "with bf16 values, one selection applied at every position. "
                "No gradient descent required — this is the joint-z infimum "
                "over the entire discrete feature-fabrication strategy space."
            ),
        },
        "result": res,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x)))

    print(f"=== F3 Exact-Optimum Attacker (discrete, bf16 values) ===")
    print(f"Baseline (zeroed): joint-z = {res['joint_z_baseline_zero']:.2f}")
    print(f"Optimal attacker:  joint-z = {res['joint_z_optimal']:.2f}")
    print(f"Total gain / slots: {res['total_gain']:.2f} / 3072 = {res['total_gain']/3072:.3f}")
    print(f"Probe coverage: mean={res['coverage_mean']:.2f} min={res['coverage_min']} max={res['coverage_max']}")
    print(f"Per-probe z: min={res['per_probe_z_min']:.2f} med={res['per_probe_z_median']:.2f} max={res['per_probe_z_max']:.2f}")
    print(f"Margin over tau: {res['margin_over_tau']:.2f}x")
    print(f"Top-5 feature gains: {res['top_gain_feature_gains'][:5]}")
    print(f"\nOutput: {OUT_PATH}")


if __name__ == "__main__":
    main()
