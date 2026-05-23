"""Intrinsic-dimension analysis for the commit-open probe library.

Computes the feasibility lower bound on joint-z for any feature-fabrication
attacker that does not run M_hon. The bound is combinatorial: per position,
the attacker picks ONE top-32 feature selection. The 96 probes collectively
require |union S_i| features to be "on", far more than 32. The overlap
structure between S_i determines how many probes can be satisfied by a
single top-32 pick.

Usage: python pilots/p1_probe_lib/20_intrinsic_dim_analysis.py
Output: pilots/p1_probe_lib/logs/intrinsic_dim.json
"""
from pathlib import Path
import json
import numpy as np
from collections import Counter


ROOT = Path(__file__).resolve().parent
LIB_PATH = ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_PATH = ROOT / "logs" / "sigma_calibration_qwen3_1.7b_L14.json"
OUT_PATH = ROOT / "logs" / "intrinsic_dim.json"

TOP_K = 32
TAU_POOL = 1.509


def load_library():
    lib = json.loads(LIB_PATH.read_text())
    sig = json.loads(SIG_PATH.read_text())
    probes = lib["probes"]
    sig_by_id = {r["probe_id"]: r for r in sig["calibration"]}
    # S_i, mu_i, sigma_i aligned by probe_id
    S = np.array([p["top_k_feature_ids"][:TOP_K] for p in probes])   # (96, 32)
    mu = np.array([sig_by_id[p["probe_id"]]["mean_cross_backend"] for p in probes])  # (96, 32)
    sd = np.clip(
        np.array([sig_by_id[p["probe_id"]]["sigma_cross_backend"] for p in probes]),
        1e-3, None,
    )  # (96, 32)
    return probes, S, mu, sd


def coverage_stats(S):
    n_probes, k = S.shape
    all_feats = S.flatten()
    union = set(all_feats.tolist())
    memberships = Counter(all_feats.tolist())
    sizes = np.array(list(memberships.values()))
    return {
        "n_probes": int(n_probes),
        "k": int(k),
        "total_memberships": int(n_probes * k),  # 96*32 = 3072
        "union_size": int(len(union)),           # |union S_i|
        "compression_ratio": float(len(union) / (n_probes * k)),
        "membership_mean": float(sizes.mean()),   # avg probes per feature
        "membership_median": int(np.median(sizes)),
        "membership_max": int(sizes.max()),
        "features_in_1_probe": int((sizes == 1).sum()),
        "features_in_ge2_probes": int((sizes >= 2).sum()),
    }


def pairwise_overlap(S):
    n = S.shape[0]
    overlap = np.zeros((n, n), dtype=int)
    sets = [set(row.tolist()) for row in S]
    for i in range(n):
        for j in range(i + 1, n):
            overlap[i, j] = len(sets[i] & sets[j])
            overlap[j, i] = overlap[i, j]
    off = overlap[np.triu_indices(n, k=1)]
    return {
        "pairwise_overlap_mean": float(off.mean()),
        "pairwise_overlap_max": int(off.max()),
        "pairs_zero_overlap": int((off == 0).sum()),
        "pairs_total": int(len(off)),
    }


def c_min_stats(mu, sd):
    # per-feature "penalty floor": |mu_{i,j}| / sigma_{i,j}
    # attacker that outputs 0 for feature j of probe i incurs exactly this
    ratio = np.abs(mu) / sd
    return {
        "mu_over_sigma_min": float(ratio.min()),
        "mu_over_sigma_p05": float(np.percentile(ratio, 5)),
        "mu_over_sigma_median": float(np.median(ratio)),
        "mu_over_sigma_mean": float(ratio.mean()),
        "mu_over_sigma_max": float(ratio.max()),
    }


def feasibility_bound(S, mu, sd):
    """For any attacker with one top-32 feature selection per position,
    compute a lower bound on joint-z attainable assuming attacker zeroes
    all probe features NOT in its selection and optimally matches probe
    features that ARE in its selection.

    Optimal attacker picks top-32 features that maximize the SUM over
    probes of (#features covered * avg |mu/sigma|). We compute:

    (a) Greedy coverage: pick the 32 features that appear in the most
        probe supports. This is a greedy upper bound on attacker coverage.
    (b) Per-probe coverage given greedy pick -> per-probe z_i lower bound
        -> joint-z lower bound.
    """
    n_probes, k = S.shape
    ratio = np.abs(mu) / sd  # (96, 32)

    # Build feature -> list of (probe_id, slot, penalty) lookup
    from collections import defaultdict
    feat_to_probes = defaultdict(list)
    for i in range(n_probes):
        for slot in range(k):
            fid = int(S[i, slot])
            feat_to_probes[fid].append((i, slot, ratio[i, slot]))

    # Greedy: score each feature by sum of its penalty across appearances
    # (attacker removes that penalty by matching mu exactly)
    feat_score = {f: sum(p for _, _, p in lst) for f, lst in feat_to_probes.items()}
    top_feats = sorted(feat_score.items(), key=lambda x: -x[1])[:k]
    picked = set(f for f, _ in top_feats)

    # Per-probe: features covered by picked vs forced-zero
    per_probe_z_lb = np.zeros(n_probes)
    covered_counts = np.zeros(n_probes, dtype=int)
    for i in range(n_probes):
        covered_penalties = []
        forced_penalties = []
        for slot in range(k):
            fid = int(S[i, slot])
            if fid in picked:
                covered_penalties.append(0.0)      # attacker matches exactly
                covered_counts[i] += 1
            else:
                forced_penalties.append(ratio[i, slot])  # attacker forced to 0
        all_penalties = covered_penalties + forced_penalties
        per_probe_z_lb[i] = np.mean(all_penalties)

    joint_z_lb = float(per_probe_z_lb.mean())
    return {
        "greedy_picked": k,
        "probe_coverage_mean": float(covered_counts.mean()),
        "probe_coverage_max": int(covered_counts.max()),
        "probe_coverage_min": int(covered_counts.min()),
        "per_probe_z_lb_min": float(per_probe_z_lb.min()),
        "per_probe_z_lb_median": float(np.median(per_probe_z_lb)),
        "per_probe_z_lb_max": float(per_probe_z_lb.max()),
        "joint_z_lb_greedy": joint_z_lb,
        "joint_z_lb_margin_over_tau": float(joint_z_lb / TAU_POOL),
    }


def main():
    probes, S, mu, sd = load_library()
    cov = coverage_stats(S)
    pw = pairwise_overlap(S)
    cm = c_min_stats(mu, sd)
    fb = feasibility_bound(S, mu, sd)

    report = {
        "metadata": {
            "library": str(LIB_PATH.relative_to(ROOT)),
            "sigma_calibration": str(SIG_PATH.relative_to(ROOT)),
            "tau_pool": TAU_POOL,
        },
        "coverage": cov,
        "pairwise_overlap": pw,
        "penalty_floor": cm,
        "feasibility_bound": fb,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n=== Intrinsic-Dim Analysis (Qwen3-1.7B, L14, k=32, N=96) ===")
    print(f"|union S_i|              = {cov['union_size']}  (of {cov['total_memberships']} memberships)")
    print(f"probes per feature mean  = {cov['membership_mean']:.2f}")
    print(f"features in >=2 probes   = {cov['features_in_ge2_probes']} / {cov['union_size']}")
    print(f"pairwise overlap mean    = {pw['pairwise_overlap_mean']:.2f}")
    print(f"pairs with zero overlap  = {pw['pairs_zero_overlap']} / {pw['pairs_total']}")
    print(f"")
    print(f"|mu/sigma| median        = {cm['mu_over_sigma_median']:.2f}")
    print(f"|mu/sigma| mean          = {cm['mu_over_sigma_mean']:.2f}")
    print(f"|mu/sigma| p05           = {cm['mu_over_sigma_p05']:.2f}")
    print(f"")
    print(f"Greedy attacker (32 picks covering maximum penalty):")
    print(f"  probe coverage mean    = {fb['probe_coverage_mean']:.2f} / 32 features")
    print(f"  per-probe z_i mean LB  = {fb['per_probe_z_lb_median']:.2f}")
    print(f"  JOINT-Z LOWER BOUND    = {fb['joint_z_lb_greedy']:.2f}")
    print(f"  tau_pool               = {TAU_POOL}")
    print(f"  margin over tau        = {fb['joint_z_lb_margin_over_tau']:.2f}x")
    print(f"")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
