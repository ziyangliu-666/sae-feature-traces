"""F0-F1 local (no GPU) feature-forgery attacks against the probe library.

F0 = uniform random top-32 per position: baseline/null attacker.
F1 = sample from honest marginal (per-probe Gaussian at stored mu_i, sigma_i).

F2 (proxy) and F3/F4 (gradient descent) require GPU -> Modal (see e18_feature_forgery.py).

Usage:
  python scripts/e18_forgery_local.py --n-positions 500 --n-seeds 5
Output: results/e18_forgery_local.json
"""
from pathlib import Path
import argparse
import json
import numpy as np


ROOT = Path(__file__).resolve().parent
LIB_PATH = ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_PATH = ROOT / "logs" / "sigma_calibration_qwen3_1.7b_L14.json"
OUT_PATH = ROOT / "logs" / "e18_forgery_local.json"

TOP_K = 32
D_SAE = 163_840
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


def joint_z(attacker_top32_indices, attacker_top32_values, S, mu, sd):
    """Evaluate joint-z for a single position.

    attacker_top32_indices: (32,) np.int array of feature indices the attacker chose
    attacker_top32_values:  (32,) np.float bf16-equivalent values
    S, mu, sd: (96, 32) per-probe supports, cross-backend means, cross-backend stds

    For each probe i and each slot j in S_i:
      - if S[i,j] in attacker indices -> use attacker's value at that index
      - else -> 0 (feature not in attacker's top-32, SAE encode would read 0)
    z_i = mean_j |f_hat_{i,j} - mu_{i,j}| / sd_{i,j}
    joint-z = mean_i z_i
    """
    n_probes, k = S.shape
    atk_lookup = dict(zip(attacker_top32_indices.tolist(), attacker_top32_values.tolist()))
    f_hat = np.zeros_like(mu)  # (96, 32) — attacker reads at probe's expected slots
    for i in range(n_probes):
        for slot in range(k):
            fid = int(S[i, slot])
            if fid in atk_lookup:
                f_hat[i, slot] = atk_lookup[fid]
    z_per_probe = np.abs(f_hat - mu) / sd
    z_per_probe_mean = z_per_probe.mean(axis=1)  # (96,)
    return float(z_per_probe_mean.mean()), z_per_probe_mean


def attack_f0(rng, n_positions):
    """F0: random top-32 indices, uniform values in [0, 1]."""
    idx = np.stack([rng.choice(D_SAE, size=TOP_K, replace=False) for _ in range(n_positions)])
    vals = rng.uniform(0, 1, size=(n_positions, TOP_K)).astype(np.float32)
    return idx, vals


def attack_f1(rng, n_positions, S, mu, sd):
    """F1: sample from per-probe marginal. Attacker picks top-32 features that
    appear in the most probe supports (to maximize coverage) and samples values
    from the pooled honest distribution (mu, sd are per-slot; we use flattened
    pooled marginal).
    """
    # Pooled marginal: collect all (feature_id, mu, sd) triples, flatten
    from collections import defaultdict
    feat_stats = defaultdict(list)
    n_probes, k = S.shape
    for i in range(n_probes):
        for slot in range(k):
            fid = int(S[i, slot])
            feat_stats[fid].append((mu[i, slot], sd[i, slot]))

    # Rank features by how many probes they appear in
    by_count = sorted(feat_stats.items(), key=lambda x: -len(x[1]))
    # Attacker picks the top-32 most-repeated features
    picked_feats = [f for f, _ in by_count[:TOP_K]]
    # Average honest mu, sd per picked feature
    picked_mu = np.array([np.mean([m for m, _ in feat_stats[f]]) for f in picked_feats])
    picked_sd = np.array([np.mean([s for _, s in feat_stats[f]]) for f in picked_feats])

    idx = np.tile(np.array(picked_feats, dtype=np.int64)[None, :], (n_positions, 1))
    # Sample around mu with sd (will slightly help the probes that contain these features)
    vals = rng.normal(
        loc=picked_mu[None, :],
        scale=picked_sd[None, :],
        size=(n_positions, TOP_K),
    ).astype(np.float32)
    vals = np.clip(vals, 0, None)  # SAE activations non-negative
    return idx, vals


def run(attack_fn, attack_name, rng, n_positions, S, mu, sd, extra_args=None):
    if extra_args:
        idx, vals = attack_fn(rng, n_positions, *extra_args)
    else:
        idx, vals = attack_fn(rng, n_positions)
    z_list = []
    for t in range(n_positions):
        jz, _ = joint_z(idx[t], vals[t], S, mu, sd)
        z_list.append(jz)
    z = np.array(z_list)
    return {
        "tier": attack_name,
        "n_positions": n_positions,
        "joint_z_min": float(z.min()),
        "joint_z_p05": float(np.percentile(z, 5)),
        "joint_z_median": float(np.median(z)),
        "joint_z_mean": float(z.mean()),
        "joint_z_p95": float(np.percentile(z, 95)),
        "joint_z_max": float(z.max()),
        "tau_pool": TAU_POOL,
        "frac_below_tau": float((z < TAU_POOL).mean()),
        "margin_min_over_tau": float(z.min() / TAU_POOL),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-positions", type=int, default=500)
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    S, mu, sd = load_library()
    print(f"Loaded library: {S.shape[0]} probes, top-{S.shape[1]} each.")
    print(f"|union S_i| = {len(set(S.flatten().tolist()))}")
    print(f"tau_pool = {TAU_POOL}")
    print()

    results = {
        "metadata": {
            "n_positions": args.n_positions,
            "n_seeds": args.n_seeds,
            "library": str(LIB_PATH.relative_to(ROOT)),
            "sigma_calibration": str(SIG_PATH.relative_to(ROOT)),
            "d_sae": D_SAE,
            "top_k": TOP_K,
            "tau_pool": TAU_POOL,
        },
        "results": [],
    }

    for seed in range(args.n_seeds):
        rng = np.random.default_rng(42 + seed)

        print(f"[seed {seed}] F0 (random top-32)...")
        r0 = run(attack_f0, "F0", rng, args.n_positions, S, mu, sd)
        r0["seed"] = seed
        results["results"].append(r0)
        print(
            f"  min={r0['joint_z_min']:.2f} med={r0['joint_z_median']:.2f} "
            f"max={r0['joint_z_max']:.2f} frac<tau={r0['frac_below_tau']:.3f}"
        )

        print(f"[seed {seed}] F1 (marginal, top-32 most-covered features)...")
        r1 = run(attack_f1, "F1", rng, args.n_positions, S, mu, sd, extra_args=(S, mu, sd))
        r1["seed"] = seed
        results["results"].append(r1)
        print(
            f"  min={r1['joint_z_min']:.2f} med={r1['joint_z_median']:.2f} "
            f"max={r1['joint_z_max']:.2f} frac<tau={r1['frac_below_tau']:.3f}"
        )

    # Aggregate across seeds
    agg = {}
    for tier in ("F0", "F1"):
        tier_mins = [r["joint_z_min"] for r in results["results"] if r["tier"] == tier]
        tier_meds = [r["joint_z_median"] for r in results["results"] if r["tier"] == tier]
        agg[tier] = {
            "min_of_mins": float(np.min(tier_mins)),
            "mean_of_medians": float(np.mean(tier_meds)),
            "seeds": len(tier_mins),
        }
    results["aggregate"] = agg

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nAggregate:")
    for tier, a in agg.items():
        print(
            f"  {tier}: min-of-mins={a['min_of_mins']:.2f}  "
            f"mean-of-medians={a['mean_of_medians']:.2f}  "
            f"margin={a['min_of_mins']/TAU_POOL:.1f}x tau"
        )
    print(f"\nOutput: {OUT_PATH}")


if __name__ == "__main__":
    main()
