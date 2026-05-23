"""E-D: SPRT decision latency under empirical dependence, Holm-corrected.

Reviewer W3 (session-FPR independence) and W9 (SPRT deferred). We replace
the i.i.d. Wald SPRT with:
  (1) Per-probe LLR accumulation walked across a random permutation of 96
      probes WITHIN a single opening, with Holm step-down thresholds
      controlling probe-level family-wise error.
  (2) Session-level decision across k openings, using the empirical
      position-correlation (ρ≈0.88 from 14_calibration_bootstrap.py) —
      we simulate dependent per-opening joint-z via MVN and report realized
      session-FPR under null vs the naive kα union bound.

Protocol:
  H0 honest:  s_i ~ N(mu0_i, sd0_i)
  H1 attacker: s_i ~ N(mu1_i, sd1_i)   (per-probe from E3-v2)

  Holm step-down: at step n (having observed n probe scores), require LLR
  above log((1-beta_n)/alpha_n) with alpha_n = alpha / (N_PROBES - n + 1)
  for the step-down variant controlling FWER. We use the composite bound
  log((1-beta)/alpha) * (1 + log(N_PROBES)/N_PROBES) as a Holm-like
  conservative upper threshold; lower bound is symmetric.

  Max openings: MAX_OPENINGS=96 (the probe budget).

Output: logs/e8_sprt_latency.json
"""
import json
from pathlib import Path
import numpy as np

LOG = Path(__file__).parent / "logs"
lib = json.loads((LOG / "probe_library_qwen3_1.7b_L14_k96.json").read_text())
sig = json.loads((LOG / "sigma_calibration_qwen3_1.7b_L14.json").read_text())
e3 = json.loads((LOG / "e3_cross_family_results_v2.json").read_text())
calib = json.loads((LOG / "calibration_bootstrap.json").read_text())

probes = lib["probes"]
sig_by_id = {r["probe_id"]: r for r in sig["calibration"]}
MAX_OPENINGS = 96
ALPHA = 0.01
BETA = 0.01

rng = np.random.default_rng(0)

# ---- per-probe honest (mu0, sd0) from the existing synthetic noise model ----
N_HONEST_SIM = 2000
mu = np.stack([np.array(sig_by_id[i]["mean_cross_backend"]) for i in range(96)])
sd = np.clip(np.stack([np.array(sig_by_id[i]["sigma_cross_backend"]) for i in range(96)]), 1e-3, None)
honest_draws = rng.normal(mu[None], sd[None], size=(N_HONEST_SIM, 96, 32))
honest_s = np.abs((honest_draws - mu) / sd).mean(-1)   # [N_SIM, 96]
mu0 = honest_s.mean(0)
sd0 = honest_s.std(0)
print(f"[calibrate] per-probe s_i: mu0 med={np.median(mu0):.3f} sd0 med={np.median(sd0):.3f}")


def pack_attacker(r):
    z_mean = np.zeros((96, 32))
    z_std = np.zeros((96, 32))
    for row in r["per_probe"]:
        z_mean[row["probe_id"]] = row["mean"]
        z_std[row["probe_id"]]  = row["std"]
    s_mean = np.abs((z_mean - mu) / sd).mean(-1)
    s_std  = (z_std / sd).mean(-1)
    return s_mean, s_std


def log_gaussian(x, m, s):
    return -0.5 * ((x - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)


def holm_thresholds(N_PROBES, alpha=ALPHA, beta=BETA):
    """Holm step-down: at step n, use alpha_n = alpha/(N - n + 1).

    Returns arrays of per-step UB/LB for the LLR.
    """
    steps = np.arange(1, N_PROBES + 1)
    alpha_n = alpha / (N_PROBES - steps + 1)
    beta_n = beta / (N_PROBES - steps + 1)
    UB = np.log((1 - beta_n) / alpha_n)
    LB = np.log(beta_n / (1 - alpha_n))
    return UB, LB


UB_HOLM, LB_HOLM = holm_thresholds(MAX_OPENINGS)
UB_WALD = np.full(MAX_OPENINGS, np.log((1 - BETA) / ALPHA))
LB_WALD = np.full(MAX_OPENINGS, np.log(BETA / (1 - ALPHA)))


def simulate_probe_level_sprt(mu1, sd1, thresholds_ub, thresholds_lb,
                              n_trials=1000, attacker_true=True):
    stops, decisions, llr_final = [], [], []
    for _ in range(n_trials):
        order = rng.permutation(96)
        llr = 0.0
        stop, decision = -1, -1
        for step, pid in enumerate(order, 1):
            if attacker_true:
                s = rng.normal(mu1[pid], max(sd1[pid], 1e-4))
            else:
                s = rng.normal(mu0[pid], sd0[pid])
            llr += (log_gaussian(s, mu1[pid], max(sd1[pid], 1e-4))
                    - log_gaussian(s, mu0[pid], sd0[pid]))
            if llr >= thresholds_ub[step - 1]:
                stop, decision = step, 1
                break
            if llr <= thresholds_lb[step - 1]:
                stop, decision = step, 0
                break
        if stop == -1:
            stop = 96
            decision = 1 if llr > 0 else 0
        stops.append(stop)
        decisions.append(decision)
        llr_final.append(llr)
    return np.array(stops), np.array(decisions), np.array(llr_final)


print("\n=== E-D SPRT latency, Wald vs Holm-corrected (α=β=0.01) ===")
print(f"{'attacker':>38}  {'rule':>7} {'med_n*':>8} {'p95_n*':>8} {'TPR':>7} {'FNR':>7}")
print("-" * 90)

out = {
    "config": {
        "alpha": ALPHA, "beta": BETA, "max_openings": MAX_OPENINGS,
        "holm_thresholds_first5": UB_HOLM[:5].tolist(),
        "wald_threshold": float(UB_WALD[0]),
    },
    "per_attacker": [], "honest_baseline": {},
    "session_fpr_by_k": [],
}

# honest baseline — both rules
first_aid = next(iter(e3.keys()))
mu1_0, sd1_0 = pack_attacker(e3[first_aid])
for rule, UB, LB in [("wald", UB_WALD, LB_WALD), ("holm", UB_HOLM, LB_HOLM)]:
    h_stops, h_dec, _ = simulate_probe_level_sprt(mu1_0, sd1_0, UB, LB, 2000, attacker_true=False)
    fpr = float((h_dec == 1).mean())
    med_n = int(np.median(h_stops)); p95_n = int(np.percentile(h_stops, 95))
    print(f"{'HONEST (null)':>38}  {rule:>7} {med_n:>8} {p95_n:>8} {'':>7} FPR={fpr:.3f}")
    out["honest_baseline"][rule] = {"med_n": med_n, "p95_n": p95_n, "FPR": fpr}

# per-attacker
for aid, r in e3.items():
    mu1, sd1 = pack_attacker(r)
    attack_out = {"attacker": aid}
    for rule, UB, LB in [("wald", UB_WALD, LB_WALD), ("holm", UB_HOLM, LB_HOLM)]:
        stops, dec, _ = simulate_probe_level_sprt(mu1, sd1, UB, LB, 1000, attacker_true=True)
        tpr = float((dec == 1).mean())
        fnr = float((dec == 0).mean())
        med_n = int(np.median(stops)); p95_n = int(np.percentile(stops, 95))
        print(f"{aid:>38}  {rule:>7} {med_n:>8} {p95_n:>8} {tpr:>7.3f} {fnr:>7.3f}")
        attack_out[rule] = {
            "med_n_stop": med_n,
            "p95_n_stop": p95_n,
            "mean_n_stop": float(stops.mean()),
            "TPR": tpr, "FNR": fnr,
        }
    out["per_attacker"].append(attack_out)

# -------------------------------------------------------------------------
# Session-level FPR under empirical position correlation — Gaussian-copula
# framing. Assume marginal per-opening FPR = alpha (target); apply the
# measured within-session correlation ρ to the latent normal score; report
# the resulting session-FPR = P(any of k openings exceeds tau). This
# isolates the DEPENDENCE effect from marginal calibration error.
# -------------------------------------------------------------------------
from scipy.stats import norm
print("\n=== Session-FPR via Gaussian copula (marginal α fixed) ===")
corr_pos = np.array(calib["within_session_correlation"]["matrix"])
z_tau = norm.ppf(1 - ALPHA)  # per-opening critical value under standard normal

def simulate_session_copula(corr, k, z_tau, n_sim, rng_):
    L = np.linalg.cholesky(corr[:k, :k] + 1e-8 * np.eye(k))
    z = rng_.standard_normal(size=(n_sim, k))
    latent = z @ L.T
    return float((latent > z_tau).any(axis=1).mean())

print(f"  marginal per-opening α = {ALPHA}, z_tau = {z_tau:.3f}")
print(f"  {'k':>3}  {'kα bound':>10}  {'indep sim':>10}  {'copula (ρ≈0.88)':>18}")
for k in [1, 2, 3, 4]:
    # Independent baseline under the same marginal
    indep = 1 - (1 - ALPHA) ** k
    # Copula under empirical ρ
    fpr_cop = simulate_session_copula(corr_pos, k, z_tau, 100_000, rng)
    bound = min(k * ALPHA, 1.0)
    out["session_fpr_by_k"].append({
        "k": k,
        "naive_k_alpha_union": bound,
        "independent_exact": float(indep),
        "empirical_copula": fpr_cop,
        "mean_rho": calib["within_session_correlation"]["mean_off_diagonal_rho"],
    })
    print(f"  {k:>3}  {bound:>10.4f}  {indep:>10.4f}  {fpr_cop:>18.4f}")

(LOG / "e8_sprt_latency.json").write_text(json.dumps(out, indent=2))
print(f"\n[save] {LOG / 'e8_sprt_latency.json'}")
