"""F3 exact-optimum forgery for Gemma-2-2B spot-check.

Same closed-form solver as e18_forgery_f3_local.py, but loads the
Gemma-2-2B probe library from e12_gemma_pilot.json + cross-backend
calibration from sigma_calibration_gemma2_2b_L12.json. Operating
threshold: tau_Gemma = 1.09 (cf. e16_whitebox_jointz_gemma.json).

Usage: python pilots/p1_probe_lib/e18_forgery_f3_gemma.py
Output: logs/e18_forgery_f3_gemma.json
"""
from pathlib import Path
import json
import numpy as np
from collections import defaultdict


ROOT = Path(__file__).resolve().parent
E12_PATH = ROOT / "logs" / "e12_gemma_pilot.json"
SIG_PATH = ROOT / "logs" / "sigma_calibration_gemma2_2b_L12.json"
OUT_PATH = ROOT / "logs" / "e18_forgery_f3_gemma.json"

TOP_K = 32
TAU_GEMMA = 1.09


def load_library():
    e12 = json.loads(E12_PATH.read_text())
    sig = json.loads(SIG_PATH.read_text())
    probes = e12["probes"]
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
    f32 = np.float32(x).view(np.uint32)
    rounded = (f32 + 0x8000) & 0xFFFF0000
    return float(np.uint32(rounded).view(np.float32))


def solve_f3(S, mu, sd):
    n_probes, k = S.shape
    total_slots = n_probes * k
    ratio = np.abs(mu) / sd
    total_baseline = float(ratio.sum() / total_slots)

    feat_occur = defaultdict(list)
    for i in range(n_probes):
        for s in range(k):
            feat_occur[int(S[i, s])].append((mu[i, s], sd[i, s]))

    feat_gain, feat_value = {}, {}
    for fid, occs in feat_occur.items():
        mus = np.array([m for m, _ in occs])
        sds = np.array([s for _, s in occs])
        weights = 1.0 / sds
        baseline_sum = float((np.abs(mus) / sds).sum())
        v_star = bf16_round(weighted_median(mus, weights))
        reduced_sum = float((np.abs(v_star - mus) / sds).sum())
        feat_gain[fid] = baseline_sum - reduced_sum
        feat_value[fid] = v_star

    sorted_feats = sorted(feat_gain.items(), key=lambda x: -x[1])
    picked = [f for f, _ in sorted_feats[:TOP_K]]
    total_gain = sum(feat_gain[f] for f in picked)
    joint_z_optimal = float((total_baseline * total_slots - total_gain) / total_slots)

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

    union_feats = len(feat_occur)
    return {
        "n_probes": n_probes,
        "union_feats": union_feats,
        "joint_z_optimal": joint_z_optimal,
        "joint_z_baseline_zero": total_baseline,
        "total_gain": total_gain,
        "coverage_mean": float(coverage.mean()),
        "per_probe_z_min": float(per_probe_z.min()),
        "per_probe_z_median": float(np.median(per_probe_z)),
        "per_probe_z_max": float(per_probe_z.max()),
        "margin_over_tau": joint_z_optimal / TAU_GEMMA,
    }


def main():
    S, mu, sd = load_library()
    print(f"Loaded Gemma library: {S.shape[0]} probes, top-{S.shape[1]} each.")
    print(f"tau_Gemma = {TAU_GEMMA}")
    print()
    res = solve_f3(S, mu, sd)

    report = {
        "metadata": {
            "probe_library": str(E12_PATH.relative_to(ROOT)),
            "sigma_calibration": str(SIG_PATH.relative_to(ROOT)),
            "backbone": "google/gemma-2-2b",
            "layer": 12,
            "tau_gemma": TAU_GEMMA,
            "attacker_model": "F3-exact-optimum-discrete-selection",
        },
        "result": res,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2))

    print(f"=== F3 Exact-Optimum Attacker (Gemma-2-2B, L12) ===")
    print(f"|union S_i|       = {res['union_feats']}")
    print(f"Baseline (zero):    joint-z = {res['joint_z_baseline_zero']:.2f}")
    print(f"Optimal attacker:   joint-z = {res['joint_z_optimal']:.2f}")
    print(f"Probe coverage mean = {res['coverage_mean']:.2f} / 32")
    print(f"Per-probe z: min={res['per_probe_z_min']:.2f} med={res['per_probe_z_median']:.2f} max={res['per_probe_z_max']:.2f}")
    print(f"Margin over tau_Gemma({TAU_GEMMA}): {res['margin_over_tau']:.2f}x")
    print(f"\nOutput: {OUT_PATH}")


if __name__ == "__main__":
    main()
