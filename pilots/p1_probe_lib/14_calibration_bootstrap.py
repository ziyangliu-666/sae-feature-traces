"""E-C: Empirical calibration with dependence-aware bootstrap.

Reviewer W3/W4: session-level FPR relies on independence across openings;
joint-z aggregator assumes per-feature Gaussian-like deviations. We replace
i.i.d. synthetic bootstrap with:
  (a) empirical bootstrap directly on the n=64 real honest joint-z pool,
      no distributional assumption;
  (b) parametric Gaussian and Student-t(df=5) fits for tail-robustness check;
  (c) within-session position correlation matrix estimated from the
      16 (dtype, kernel, companion_seed) tuples × 4 positions structure in
      recipe1_qwen3_honest_pool.json;
  (d) empirical session-FPR for k openings under the estimated correlation,
      compared to the naive kα independence bound.

Pure CPU / numpy + scipy, ~5s.

Output: logs/calibration_bootstrap.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import norm, beta

LOG = Path(__file__).parent / "logs"
POOL = json.loads((LOG / "recipe1_qwen3_honest_pool.json").read_text())
SIG = json.loads((LOG / "sigma_calibration_qwen3_1.7b_L14.json").read_text())

joints = np.asarray(POOL["joint_z_values"], dtype=np.float64)
configs = POOL["configs"]
n = len(joints)
print(f"[load] n={n} honest joint-z values, min={joints.min():.3f} "
      f"med={np.median(joints):.3f} max={joints.max():.3f}")


# -------------------------------------------------------------------------
# (a) Empirical bootstrap of tau_99 (no distributional assumption)
# -------------------------------------------------------------------------
N_BOOT = 10_000
rng = np.random.default_rng(0)
tau99_boot = np.empty(N_BOOT)
for b in range(N_BOOT):
    draw = rng.choice(joints, size=n, replace=True)
    tau99_boot[b] = np.quantile(draw, 0.99)
tau99_empirical = float(np.quantile(joints, 0.99))
print(f"\n(a) Empirical bootstrap of tau_99:")
print(f"    empirical p99 on n=64: {tau99_empirical:.4f}")
print(f"    bootstrap median:      {np.median(tau99_boot):.4f}")
print(f"    bootstrap 95% CI:      [{np.quantile(tau99_boot, 0.025):.4f}, "
      f"{np.quantile(tau99_boot, 0.975):.4f}]")


# -------------------------------------------------------------------------
# (b) Parametric tail fits — Gaussian vs Student-t(df=5)
# -------------------------------------------------------------------------
mu_hat = joints.mean()
sd_hat = joints.std(ddof=1)
tau99_gaussian = float(mu_hat + norm.ppf(0.99) * sd_hat)

# MLE Student-t with df=5 (fixed df, fit location & scale)
# For df=5, the scale is sd_hat / sqrt(df/(df-2)) = sd_hat / sqrt(5/3).
df_t = 5
t_scale = sd_hat / np.sqrt(df_t / (df_t - 2))
tau99_student = float(mu_hat + student_t.ppf(0.99, df=df_t) * t_scale)

# Parametric bootstrap CIs
gauss_boot = np.empty(N_BOOT)
stud_boot = np.empty(N_BOOT)
for b in range(N_BOOT):
    g = rng.normal(mu_hat, sd_hat, size=n)
    s = mu_hat + t_scale * rng.standard_t(df_t, size=n)
    gauss_boot[b] = np.quantile(g, 0.99)
    stud_boot[b] = np.quantile(s, 0.99)

print(f"\n(b) Parametric tail fits:")
print(f"    Gaussian tau99:       {tau99_gaussian:.4f} "
      f"(boot 95% CI [{np.quantile(gauss_boot, 0.025):.4f}, "
      f"{np.quantile(gauss_boot, 0.975):.4f}])")
print(f"    Student-t(df=5) tau99: {tau99_student:.4f} "
      f"(boot 95% CI [{np.quantile(stud_boot, 0.025):.4f}, "
      f"{np.quantile(stud_boot, 0.975):.4f}])")


# -------------------------------------------------------------------------
# (c) Within-session position correlation matrix
#     Group by (dtype, kernel, companion_seed) — each tuple has 4 positions.
# -------------------------------------------------------------------------
tuples = defaultdict(dict)
for c in configs:
    key = (c["dtype"], c["kernel"], c["companion_seed"])
    tuples[key][c["pos"]] = c["joint_z"]

positions = sorted({p for v in tuples.values() for p in v.keys()})
n_tuples = len(tuples)
mat = np.array([[v[p] for p in positions] for v in tuples.values()])  # [n_tuples, 4]
assert mat.shape == (n_tuples, len(positions)), f"got {mat.shape}"

cov_pos = np.cov(mat, rowvar=False, ddof=1)
corr_pos = np.corrcoef(mat, rowvar=False)
mean_rho = float((corr_pos.sum() - np.trace(corr_pos)) / (corr_pos.size - corr_pos.shape[0]))
print(f"\n(c) Within-session position correlation ({n_tuples} sessions × 4 positions):")
print(f"    correlation matrix:\n{np.round(corr_pos, 3)}")
print(f"    mean off-diagonal ρ: {mean_rho:.3f}")
print(f"    session marginal σ:  {np.sqrt(np.diag(cov_pos)).round(4).tolist()}")


# -------------------------------------------------------------------------
# (d) Empirical session-FPR under dependence vs naive kα independence bound
#     Simulate honest sessions with the empirical (mu, Σ) for k=4 openings,
#     decide session-positive if ANY opening exceeds tau_99_empirical.
# -------------------------------------------------------------------------
def sample_session_mvn(mu_vec, cov, n_draws, rng):
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(len(mu_vec)))
    z = rng.standard_normal(size=(n_draws, len(mu_vec)))
    return mu_vec[None] + z @ L.T

def sample_session_iid(mu_scalar, sd_scalar, n_draws, k, rng):
    return rng.normal(mu_scalar, sd_scalar, size=(n_draws, k))

def session_fpr(samples, tau, k_use):
    k_actual = min(k_use, samples.shape[1])
    return float((samples[:, :k_actual] > tau).any(axis=1).mean())

K_OPENINGS_SWEEP = [1, 2, 3, 4]
N_SIM = 50_000
mu_vec = mat.mean(axis=0)
session_results = []

for k_use in K_OPENINGS_SWEEP:
    # MVN (dependent) — use the empirical Σ truncated to k positions
    k_effective = min(k_use, len(mu_vec))
    mu_k = mu_vec[:k_effective]
    cov_k = cov_pos[:k_effective, :k_effective]
    samp_mvn = sample_session_mvn(mu_k, cov_k, N_SIM, rng)
    fpr_mvn = session_fpr(samp_mvn, tau99_empirical, k_use)

    # IID baseline — same marginal mean/std but independent
    sd_marg = float(np.sqrt(np.mean(np.diag(cov_pos))))
    mu_marg = float(mu_vec.mean())
    samp_iid = sample_session_iid(mu_marg, sd_marg, N_SIM, k_effective, rng)
    fpr_iid = session_fpr(samp_iid, tau99_empirical, k_use)

    # Naive kα bound (paper's current claim)
    alpha = 0.01
    bound_ka = min(k_use * alpha, 1.0)

    session_results.append({
        "k": k_use,
        "fpr_mvn_dependent": fpr_mvn,
        "fpr_iid": fpr_iid,
        "naive_bound_k_alpha": bound_ka,
    })

print(f"\n(d) Session-FPR at tau={tau99_empirical:.3f} under dependence vs IID:")
print(f"    {'k':>3}  {'kα bound':>10}  {'IID sim':>10}  {'MVN (dep)':>10}")
for r in session_results:
    print(f"    {r['k']:>3}  {r['naive_bound_k_alpha']:>10.4f}  "
          f"{r['fpr_iid']:>10.4f}  {r['fpr_mvn_dependent']:>10.4f}")


# -------------------------------------------------------------------------
# (e) Sanity: realized FPR at tau_empirical across 1000 reseeds
# -------------------------------------------------------------------------
N_RESEED = 1000
fprs = np.empty(N_RESEED)
for s in range(N_RESEED):
    rng_s = np.random.default_rng(s + 100_000)
    draw = rng_s.choice(joints, size=n, replace=True)
    fprs[s] = (draw > tau99_empirical).mean()
print(f"\n(e) Realized FPR at tau={tau99_empirical:.3f} under empirical bootstrap:")
print(f"    median={np.median(fprs):.4f}, CI95=["
      f"{np.quantile(fprs, 0.025):.4f}, {np.quantile(fprs, 0.975):.4f}]")


# -------------------------------------------------------------------------
# Save
# -------------------------------------------------------------------------
out = {
    "config": {
        "n_honest": n,
        "n_bootstrap": N_BOOT,
        "n_tuples": n_tuples,
        "positions": positions,
        "student_t_df": df_t,
        "k_openings_sweep": K_OPENINGS_SWEEP,
        "n_session_sim": N_SIM,
    },
    "tau_99": {
        "empirical": tau99_empirical,
        "empirical_bootstrap_median": float(np.median(tau99_boot)),
        "empirical_bootstrap_ci95": [
            float(np.quantile(tau99_boot, 0.025)),
            float(np.quantile(tau99_boot, 0.975)),
        ],
        "gaussian_fit": tau99_gaussian,
        "gaussian_boot_ci95": [
            float(np.quantile(gauss_boot, 0.025)),
            float(np.quantile(gauss_boot, 0.975)),
        ],
        "student_t5_fit": tau99_student,
        "student_t5_boot_ci95": [
            float(np.quantile(stud_boot, 0.025)),
            float(np.quantile(stud_boot, 0.975)),
        ],
    },
    "within_session_correlation": {
        "matrix": corr_pos.tolist(),
        "mean_off_diagonal_rho": mean_rho,
        "session_marginal_sd": np.sqrt(np.diag(cov_pos)).tolist(),
    },
    "session_fpr_by_k": session_results,
    "realized_fpr_at_empirical_tau": {
        "median": float(np.median(fprs)),
        "ci95": [float(np.quantile(fprs, 0.025)), float(np.quantile(fprs, 0.975))],
    },
}

(LOG / "calibration_bootstrap.json").write_text(json.dumps(out, indent=2))
print(f"\n[save] {LOG / 'calibration_bootstrap.json'}")
