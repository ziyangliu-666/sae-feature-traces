"""Library-rotation generalization test via within-library split.

Instead of building a larger (256) probe pool and drawing random 96-subsets
per session, we run a cheaper within-library split:
  - Train: F3 attacker computes optimal top-32 against TRAIN probes only
  - Test: evaluate the SAME top-32 against held-out TEST probes
  - Repeat over K random 48/48 splits, bootstrap joint-z distribution

This captures the essence of rotation: a probe-aware attacker who trained
against one draw of probes does not transfer to a different draw. Rotation
(drawing a 96-subset from 256 at session time) raises this transfer gap.

Usage: python scripts/e19_library_split_generalization.py
Output: results/e19_library_split.json
"""
from pathlib import Path
import json
import numpy as np
from collections import defaultdict


ROOT = Path(__file__).resolve().parent
LIB_PATH = ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_PATH = ROOT / "logs" / "sigma_calibration_qwen3_1.7b_L14.json"
OUT_PATH = ROOT / "logs" / "e19_library_split.json"

TOP_K = 32
TAU_POOL = 1.509
N_SPLITS = 50
SPLIT_TRAIN = 48
SPLIT_TEST = 48


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
    idx = np.searchsorted(cw, cw[-1] / 2.0)
    return float(sv[min(idx, len(sv) - 1)])


def bf16_round(x):
    f32 = np.float32(x).view(np.uint32)
    rounded = (f32 + 0x8000) & 0xFFFF0000
    return float(np.uint32(rounded).view(np.float32))


def solve_f3_on_subset(S_sub, mu_sub, sd_sub):
    """Return optimal (picked_features, feat_value_map)."""
    n_probes, k = S_sub.shape
    feat_occur = defaultdict(list)
    for i in range(n_probes):
        for s in range(k):
            feat_occur[int(S_sub[i, s])].append((mu_sub[i, s], sd_sub[i, s]))

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
    return picked, feat_value


def eval_joint_z(picked, feat_value, S_sub, mu_sub, sd_sub):
    n_probes, k = S_sub.shape
    picked_set = set(picked)
    per_probe_z = np.zeros(n_probes)
    for i in range(n_probes):
        row_sum = 0.0
        for s in range(k):
            fid = int(S_sub[i, s])
            if fid in picked_set:
                v = feat_value.get(fid, 0.0)
                row_sum += abs(v - mu_sub[i, s]) / sd_sub[i, s]
            else:
                row_sum += abs(mu_sub[i, s]) / sd_sub[i, s]
        per_probe_z[i] = row_sum / k
    return float(per_probe_z.mean()), per_probe_z


def main():
    S, mu, sd = load_library()
    n_probes = S.shape[0]
    print(f"Loaded library: {n_probes} probes; running {N_SPLITS} random 48/48 splits.")
    print()

    train_zs, test_zs, diffs = [], [], []
    for seed in range(N_SPLITS):
        rng = np.random.default_rng(1000 + seed)
        perm = rng.permutation(n_probes)
        train_idx = perm[:SPLIT_TRAIN]
        test_idx = perm[SPLIT_TRAIN:SPLIT_TRAIN + SPLIT_TEST]

        picked, feat_value = solve_f3_on_subset(S[train_idx], mu[train_idx], sd[train_idx])

        z_train, _ = eval_joint_z(picked, feat_value, S[train_idx], mu[train_idx], sd[train_idx])
        z_test, _ = eval_joint_z(picked, feat_value, S[test_idx], mu[test_idx], sd[test_idx])

        train_zs.append(z_train)
        test_zs.append(z_test)
        diffs.append(z_test - z_train)

    train_zs = np.array(train_zs)
    test_zs = np.array(test_zs)
    diffs = np.array(diffs)

    report = {
        "metadata": {
            "library": str(LIB_PATH.relative_to(ROOT)),
            "sigma_calibration": str(SIG_PATH.relative_to(ROOT)),
            "tau_pool": TAU_POOL,
            "n_splits": N_SPLITS,
            "split_train": SPLIT_TRAIN,
            "split_test": SPLIT_TEST,
            "experiment": "F3 probe-aware attacker trained on 48 probes, tested on held-out 48",
        },
        "train_z": {
            "min": float(train_zs.min()),
            "p05": float(np.percentile(train_zs, 5)),
            "median": float(np.median(train_zs)),
            "mean": float(train_zs.mean()),
            "p95": float(np.percentile(train_zs, 95)),
            "max": float(train_zs.max()),
            "frac_below_tau": float((train_zs < TAU_POOL).mean()),
        },
        "test_z": {
            "min": float(test_zs.min()),
            "p05": float(np.percentile(test_zs, 5)),
            "median": float(np.median(test_zs)),
            "mean": float(test_zs.mean()),
            "p95": float(np.percentile(test_zs, 95)),
            "max": float(test_zs.max()),
            "frac_below_tau": float((test_zs < TAU_POOL).mean()),
        },
        "test_minus_train": {
            "min": float(diffs.min()),
            "median": float(np.median(diffs)),
            "mean": float(diffs.mean()),
            "max": float(diffs.max()),
        },
    }

    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"=== F3 train-vs-test (N={N_SPLITS} splits of 48/48) ===")
    print(f"Train joint-z: min={train_zs.min():.2f} med={np.median(train_zs):.2f} max={train_zs.max():.2f}")
    print(f"Test  joint-z: min={test_zs.min():.2f} med={np.median(test_zs):.2f} max={test_zs.max():.2f}")
    print(f"Test - Train : mean={diffs.mean():+.2f} (rotation transfer gap)")
    print(f"All runs: frac joint-z < tau_pool ({TAU_POOL}) = {(test_zs < TAU_POOL).mean():.3f} (test), "
          f"{(train_zs < TAU_POOL).mean():.3f} (train)")
    print(f"\nOutput: {OUT_PATH}")


if __name__ == "__main__":
    main()
