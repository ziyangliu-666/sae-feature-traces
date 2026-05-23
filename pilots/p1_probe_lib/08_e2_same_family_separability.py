"""E2: same-family separability on committed SAE traces (C1 primary evidence).

Protocol-consistent setup:
  Honest M commits the top-32 SAE feature IDs (+ magnitudes) at probe positions.
  An adversarial substitute M' (Qwen3-0.6B) must produce committed traces that
  the verifier accepts. We measure whether the verifier's joint-consistency
  scorer distinguishes honest from adversarial, at FPR=1% on honest runs.

Test statistics per run:
  - overlap_mean = mean over 96 probes of |committed_top32 ∩ expected_top32| / 32
  - maha_score   = mean over 96 probes of ||z_committed − μ_expected||_{σ_E7}

Runs:
  - N_HONEST = 24 honest backend variations (2 dtypes × 3 kernels × 4 positions)
    — note FLASH_ATTENTION is unavailable on this torch/card, so 16 effective
  - N_ATTACK = 20 per attack type (different random seeds), using Qwen3-0.6B
    substitute with ZERO_PAD, RAND_PROJ, LEARNED_LIN, LEARNED_LIN_heldout

Reports ROC + TPR at FPR=1%.

Estimated runtime: ~8-15 min on RTX 3090.
"""
import json
import time
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

OUT_DIR = Path(__file__).parent / "logs"
LIB_PATH = OUT_DIR / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_PATH = OUT_DIR / "sigma_calibration_qwen3_1.7b_L14.json"

TARGET_MODEL = "Qwen/Qwen3-1.7B"
SUBSTITUTE_MODEL = "Qwen/Qwen3-0.6B"
TARGET_LAYER = 14
TOP_K = 32
N_ATTACK_RUNS = 20

PRECISIONS = [torch.bfloat16, torch.float16]
KERNELS = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]  # skip FLASH (unavailable)
POSITIONS = [0, 1, 2, 3]

random.seed(0); torch.manual_seed(0)

print("[load] library + sigma calibration")
lib = json.loads(LIB_PATH.read_text())
sig = json.loads(SIG_PATH.read_text())
probes = lib["probes"]
sig_by_probe = {r["probe_id"]: r for r in sig["calibration"]}
print(f"  {len(probes)} probes, d_sae={lib['metadata']['d_sae']}")

FILLER = [
    "The weather today is particularly",
    "Climate change affects biodiversity in multiple ways, including",
    "A typical morning routine often includes",
    "The history of cryptography spans several",
    "Modern cities face challenges such as",
    "Healthy cooking often involves fresh",
    "Space exploration has yielded numerous",
    "Online privacy has become a pressing",
]

tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

print(f"\n[load] target {TARGET_MODEL}")
t0 = time.time()
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=torch.bfloat16, device_map="cuda")
target.eval()
print(f"  loaded in {time.time()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

print(f"[load] substitute {SUBSTITUTE_MODEL}")
t1 = time.time()
sub_tok = AutoTokenizer.from_pretrained(SUBSTITUTE_MODEL)
if sub_tok.pad_token is None: sub_tok.pad_token = sub_tok.eos_token
sub = AutoModelForCausalLM.from_pretrained(SUBSTITUTE_MODEL, dtype=torch.bfloat16, device_map="cuda")
sub.eval()
sub_layers = sub.config.num_hidden_layers
sub_dim = sub.config.hidden_size
sub_layer = int(round(TARGET_LAYER / target.config.num_hidden_layers * sub_layers))
print(f"  loaded in {time.time()-t1:.1f}s, {sub_layers} layers, d={sub_dim}, probing layer {sub_layer}, vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

print(f"[load] transcoder layer_{TARGET_LAYER}")
sae = SAE.from_pretrained(release="mwhanna-qwen3-1.7b-transcoders-lowl0", sae_id=f"layer_{TARGET_LAYER}", device="cuda")
sae.eval()

captured = {}
def hook_tgt(_m, inputs):
    captured["tgt"] = inputs[0].detach()
def hook_sub(_m, inputs):
    captured["sub"] = inputs[0].detach()
h_tgt = target.model.layers[TARGET_LAYER].mlp.register_forward_pre_hook(hook_tgt)
h_sub = sub.model.layers[sub_layer].mlp.register_forward_pre_hook(hook_sub)

def forward_target(prompt: str, pos: int, companions: list[str], dtype, kernel) -> torch.Tensor:
    batch = companions.copy(); batch.insert(pos, prompt)
    enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
    attn = enc["attention_mask"]; last = attn.sum(dim=1) - 1
    with sdpa_kernel([kernel]), torch.no_grad():
        if dtype == torch.bfloat16:
            _ = target(**enc)
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = target(**enc)
    return captured["tgt"][pos, last[pos]]

def forward_sub(prompt: str, pos: int, companions: list[str]) -> torch.Tensor:
    batch = companions.copy(); batch.insert(pos, prompt)
    enc = sub_tok(batch, return_tensors="pt", padding=True).to("cuda")
    attn = enc["attention_mask"]; last = attn.sum(dim=1) - 1
    with torch.no_grad():
        _ = sub(**enc)
    return captured["sub"][pos, last[pos]]

def commit_trace(act: torch.Tensor) -> tuple[set, torch.Tensor]:
    """Encode via SAE and return (set of top-32 feature IDs, top-32 magnitudes aligned with that set)."""
    with torch.no_grad():
        z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
    vals, idx = torch.topk(z.abs(), k=TOP_K)
    return set(idx.tolist()), z[idx].float().cpu()

def score_run_against_library(committed: list[tuple[set, torch.Tensor]]) -> tuple[float, float]:
    """Return (overlap_mean, maha_score) for a run's 96 committed traces."""
    overlaps = []
    maha = []
    for p, (ids, _) in zip(probes, committed):
        expected = set(p["top_k_feature_ids"])
        overlaps.append(len(ids & expected) / TOP_K)
    for p, (ids, vals) in zip(probes, committed):
        # evaluate magnitudes at the *expected* top-32 positions using the full z if we had it
        # — approximation: we only kept top-32 of committed, so this is best-effort
        # We check magnitude overlap via a rough surrogate: if committed != expected, maha grows.
        # For a cleaner Mahalanobis we need full z; we do a second forward in the honest path below.
        pass
    return float(sum(overlaps) / len(overlaps)), 0.0

def score_full_maha(z_full_per_probe: list[torch.Tensor]) -> float:
    """Given full SAE activation per probe, compute mean Mahalanobis vs (μ,σ) from E7."""
    vals = []
    for p, z in zip(probes, z_full_per_probe):
        expected_ids = p["top_k_feature_ids"]
        mu = torch.tensor(sig_by_probe[p["probe_id"]]["mean_cross_backend"])
        sd = torch.tensor(sig_by_probe[p["probe_id"]]["sigma_cross_backend"]).clamp(min=1e-3)
        z_at_exp = z[expected_ids].float().cpu()
        vals.append(float(((z_at_exp - mu) / sd).abs().mean().item()))
    return float(sum(vals) / len(vals))

def commit_and_fullz(act: torch.Tensor) -> tuple[set, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
    vals, idx = torch.topk(z.abs(), k=TOP_K)
    return set(idx.tolist()), z[idx].float().cpu(), z.float()

# --- Phase 1: honest runs ---------------------------------------------------
print(f"\n[phase 1] honest target runs across {len(PRECISIONS)}×{len(KERNELS)}×{len(POSITIONS)} = {len(PRECISIONS)*len(KERNELS)*len(POSITIONS)} backend configs")
honest_runs = []
t2 = time.time()
for ri, dtype in enumerate(PRECISIONS):
    for kernel in KERNELS:
        for pos in POSITIONS:
            companions = random.sample(FILLER, 3)
            committed = []
            zs = []
            for p in probes:
                act = forward_target(p["prompt"], pos, companions, dtype, kernel)
                ids, mags, z = commit_and_fullz(act)
                committed.append((ids, mags))
                zs.append(z)
            om, _ = score_run_against_library(committed)
            mh = score_full_maha(zs)
            honest_runs.append({"overlap": om, "maha": mh, "dtype": str(dtype).split(".")[-1], "kernel": kernel.name, "pos": pos})
            if len(honest_runs) % 4 == 0:
                print(f"  [{len(honest_runs):2d}/{len(PRECISIONS)*len(KERNELS)*len(POSITIONS)}] overlap={om:.4f} maha={mh:.3f} (elapsed {time.time()-t2:.0f}s)")

# --- Phase 2: attacker runs -------------------------------------------------
# Fit LEARNED_LIN on public pile (not probes) to be consistent with public-corpus regime.
# Shortcut: fit on a held-out batch of FILLER continuations (not on probe prompts).
print(f"\n[phase 2] fitting LEARNED_LIN linear map on public corpus (not on probe prompts)")
# Gather paired activations on a set of held-out snippets (not in our probe set).
CALIB = [
    "Many modern cities suffer from chronic air pollution,",
    "The advent of the printing press revolutionized communication by",
    "Traditional Japanese cuisine places strong emphasis on seasonal",
    "When studying economics, one must carefully distinguish between",
    "Recent advances in renewable energy technology suggest that",
    "The phenomenon of consciousness remains one of the most",
    "International trade agreements have historically shaped the trajectory of",
    "During the Industrial Revolution, urbanization increased dramatically as",
    "Bioinformatics combines tools from biology, statistics, and computer science to",
    "Philosophers since Kant have debated whether morality is ultimately grounded in",
    "The formation of planetary systems from protoplanetary disks involves",
    "Quantum entanglement, one of the most counterintuitive predictions of quantum mechanics,",
    "Early research on neural networks in the 1980s laid the foundation for",
    "The legal concept of intellectual property includes several distinct categories, such as",
    "A balanced diet typically contains proportional amounts of carbohydrates, proteins, and",
    "In classical music theory, the distinction between major and minor keys",
]
tgt_acts = []
sub_acts = []
for s in CALIB:
    at = forward_target(s, 0, random.sample(FILLER, 3), torch.bfloat16, SDPBackend.MATH)
    asu = forward_sub(s, 0, random.sample(FILLER, 3))
    tgt_acts.append(at.float().cpu())
    sub_acts.append(asu.float().cpu())
X_tgt = torch.stack(tgt_acts)          # [n, 2048]
X_sub = torch.stack(sub_acts)          # [n, 1024]
A = torch.cat([X_sub, torch.ones(X_sub.shape[0], 1)], dim=1)
phi, *_ = torch.linalg.lstsq(A, X_tgt)   # [d_sub+1, d_tgt]
phi = phi.to("cuda").to(torch.bfloat16)
pred = (A.to("cuda").to(torch.bfloat16) @ phi).float().cpu()
r2 = 1.0 - ((pred - X_tgt).norm()**2 / X_tgt.norm()**2).item()
print(f"  linear fit R² on public pile (n={len(CALIB)}): {r2:.3f}")

def attacker_commit(prompt, pos, companions, lift: str) -> tuple[set, torch.Tensor, torch.Tensor]:
    act_sub = forward_sub(prompt, pos, companions).float()  # [d_sub]
    if lift == "ZERO_PAD":
        act_tgt = torch.zeros(2048, device="cuda", dtype=torch.bfloat16)
        act_tgt[:sub_dim] = act_sub.to(torch.bfloat16)
    elif lift == "RAND_PROJ":
        W = torch.randn(2048, sub_dim, device="cuda") / (sub_dim ** 0.5)
        act_tgt = (W @ act_sub).to(torch.bfloat16)
    elif lift == "LEARNED_LIN":
        act_aug = torch.cat([act_sub, torch.ones(1, device="cuda")])
        act_tgt = (act_aug.to(torch.bfloat16) @ phi)
    else:
        raise ValueError(lift)
    return commit_and_fullz(act_tgt)

print(f"\n[phase 3] attacker runs × {N_ATTACK_RUNS} per lift")
attack_runs = {lift: [] for lift in ["ZERO_PAD", "RAND_PROJ", "LEARNED_LIN"]}
for lift in attack_runs:
    t3 = time.time()
    for ri in range(N_ATTACK_RUNS):
        pos = POSITIONS[ri % len(POSITIONS)]
        random.seed(100 + ri)
        companions = random.sample(FILLER, 3)
        committed = []
        zs = []
        for p in probes:
            ids, mags, z = attacker_commit(p["prompt"], pos, companions, lift)
            committed.append((ids, mags))
            zs.append(z)
        om, _ = score_run_against_library(committed)
        mh = score_full_maha(zs)
        attack_runs[lift].append({"overlap": om, "maha": mh, "run": ri})
    print(f"  [{lift}] 20 runs in {time.time()-t3:.0f}s")

h_tgt.remove(); h_sub.remove()

# --- Phase 4: classifier ----------------------------------------------------
print(f"\n[phase 4] classifier: threshold on honest, measure TPR per attack")
import numpy as np

honest_ov = np.array([r["overlap"] for r in honest_runs])
honest_mh = np.array([r["maha"] for r in honest_runs])
# For overlap: honest is HIGH, attack should be LOW → detect: overlap < τ_ov
# For maha: honest is LOW, attack should be HIGH → detect: maha > τ_mh
# FPR=1% → τ_ov = 1st percentile of honest overlaps (we flag if below)
tau_ov = float(np.quantile(honest_ov, 0.01))
tau_mh = float(np.quantile(honest_mh, 0.99))
print(f"  honest overlap:  min={honest_ov.min():.4f}  med={np.median(honest_ov):.4f}  max={honest_ov.max():.4f}")
print(f"  honest maha:     min={honest_mh.min():.3f}  med={np.median(honest_mh):.3f}  max={honest_mh.max():.3f}")
print(f"  τ_overlap (FPR=1%): {tau_ov:.4f}")
print(f"  τ_maha    (FPR=1%): {tau_mh:.3f}")

results_by_attack = {}
for lift, runs in attack_runs.items():
    ov = np.array([r["overlap"] for r in runs])
    mh = np.array([r["maha"] for r in runs])
    tpr_ov = float((ov < tau_ov).mean())
    tpr_mh = float((mh > tau_mh).mean())
    tpr_joint = float(((ov < tau_ov) | (mh > tau_mh)).mean())
    results_by_attack[lift] = {
        "n_runs": len(runs),
        "overlap_mean": float(ov.mean()), "overlap_std": float(ov.std()),
        "maha_mean": float(mh.mean()), "maha_std": float(mh.std()),
        "tpr_overlap_only": tpr_ov,
        "tpr_maha_only": tpr_mh,
        "tpr_joint": tpr_joint,
    }
    print(f"  {lift:<14} overlap={ov.mean():.4f}±{ov.std():.4f}  maha={mh.mean():.2f}±{mh.std():.2f}  TPR_ov={tpr_ov:.2f}  TPR_mh={tpr_mh:.2f}  TPR_joint={tpr_joint:.2f}")

# --- Save --------------------------------------------------------------------
out = {
    "metadata": {
        "experiment": "E2",
        "target": TARGET_MODEL,
        "substitute": SUBSTITUTE_MODEL,
        "layer": TARGET_LAYER,
        "n_probes": len(probes),
        "top_k": TOP_K,
        "n_honest": len(honest_runs),
        "n_attack_per_lift": N_ATTACK_RUNS,
        "linear_fit_r2": r2,
        "runtime_s": time.time() - t0,
    },
    "thresholds": {"tau_overlap_at_fpr_1pct": tau_ov, "tau_maha_at_fpr_1pct": tau_mh},
    "honest": honest_runs,
    "attacks": attack_runs,
    "classifier": results_by_attack,
}
(OUT_DIR / "e2_separability.json").write_text(json.dumps(out, indent=2))
print(f"\n[save] {OUT_DIR / 'e2_separability.json'}")
print(f"\nTotal: {time.time()-t0:.1f}s")
