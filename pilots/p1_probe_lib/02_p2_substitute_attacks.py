"""P2: substitute-model attacks against the Qwen3-1.7B probe library.

Threat model: adversary runs Qwen3-0.6B (d_model=1024) but must return
d_model=2048 activations that match M's latent-pattern expectations under
M's SAE basis. We evaluate three attacker lift strategies:

  1. ZERO_PAD — naive: h' (1024) || zeros (1024) → (2048). Lowest effort.
  2. RAND_PROJ — random linear W: (2048 × 1024), fixed, then W @ h'.
  3. LEARNED_LIN — W fitted by least squares on a held-out calibration set
     of (h' , h) pairs. Strongest linear attacker.

For each attack, encode through M's SAE and compare to M's top-32 pattern:
  • Jaccard (ID overlap)
  • Pearson corr on top-32 magnitudes
  • "Joint score" = mean Mahalanobis distance over probe library

Also: Qwen3-4B as alternative substitute (d_model=2560 → 2048 via linear lift).
Output: logs/p2_results.json, logs/p2_summary.txt
Estimated runtime: ~10-15 min on RTX 3090.
"""
import json
import time
import random
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

OUT_DIR = Path(__file__).parent / "logs"
LIB_PATH = OUT_DIR / "probe_library_qwen3_1.7b_L14.json"

TARGET_MODEL = "Qwen/Qwen3-1.7B"
TARGET_LAYER = 14
SUBSTITUTE_MODELS = [
    {"id": "Qwen/Qwen3-0.6B",  "layer_frac": 14/28},   # 0.6B has 28 layers too actually; same fraction
    {"id": "Qwen/Qwen3-4B",    "layer_frac": 14/28},
]

random.seed(0); torch.manual_seed(0)

lib = json.loads(LIB_PATH.read_text())
probes = lib["probes"]
d_sae = lib["metadata"]["d_sae"]
d_in = lib["metadata"]["d_in"]
assert d_in == 2048
print(f"[load lib] {len(probes)} probes, d_sae={d_sae}, d_in={d_in}")

# Load target SAE once
print(f"[load sae] mwhanna-qwen3-1.7b-transcoders-lowl0 / layer_{TARGET_LAYER}")
sae = SAE.from_pretrained(
    release="mwhanna-qwen3-1.7b-transcoders-lowl0",
    sae_id=f"layer_{TARGET_LAYER}",
    device="cuda",
)

# Also load target model once to:
#   (a) recompute target-model activations (for sanity check of our pipeline)
#   (b) collect calibration pairs for LEARNED_LIN attack
print(f"[load target] {TARGET_MODEL}")
tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=torch.bfloat16, device_map="cuda")
target.eval()
print(f"  vram={torch.cuda.memory_allocated()/1e9:.2f} GB  layers={target.config.num_hidden_layers}")

# --- helpers ----------------------------------------------------------------
def make_hook(container, key):
    def hook(_mod, inputs):
        container[key] = inputs[0].detach()
    return hook

def batch_mlp_in(model, layer_idx, prompts, token_pos="last"):
    """Run prompts, capture mlp_in at layer_idx, return [B, d_model] at last non-pad pos."""
    captured = {}
    h = model.model.layers[layer_idx].mlp.register_forward_pre_hook(make_hook(captured, "x"))
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    try:
        with torch.no_grad():
            _ = model(**enc)
        mlp_in = captured["x"]  # [B, T, d]
        last = enc["attention_mask"].sum(dim=1) - 1
        return torch.stack([mlp_in[i, last[i]] for i in range(mlp_in.shape[0])]).float()
    finally:
        h.remove()

def encode_top_pattern(h, top_ids):
    """h: [d_in] on GPU/CPU, top_ids: list of int → returns magnitudes at those ids."""
    with torch.no_grad():
        z = sae.encode(h.unsqueeze(0).to(sae.dtype).to(sae.W_enc.device))
    return z[0, top_ids].float().cpu()

def encode_all(h):
    with torch.no_grad():
        z = sae.encode(h.unsqueeze(0).to(sae.dtype).to(sae.W_enc.device))
    return z[0].float().cpu()

# --- Step 1: recompute M's activations so we have h_M reference for calibration --
print("\n[step 1] Recomputing target-model activations for all probes (for calibration + sanity check)")
prompt_list = [p["prompt"] for p in probes]
# filler to match P1 batching of 4
FILLER_PROMPTS = [
    "The weather today is sunny.",
    "Quantum mechanics describes the behavior of",
    "In the beginning, there was light.",
]
h_M_per_probe = []  # [n_probes, d_in]
t0 = time.time()
for i, prompt in enumerate(prompt_list):
    batch = [prompt] + FILLER_PROMPTS[:3]
    feats = batch_mlp_in(target, TARGET_LAYER, batch)  # [4, 2048]
    h_M_per_probe.append(feats[0])
H_M = torch.stack(h_M_per_probe)  # [n, 2048]
print(f"  done in {time.time()-t0:.1f}s. shape={tuple(H_M.shape)}")

# Sanity: encode H_M through sae and check top-1 matches library
print("  sanity: match top-1 feature vs library ...")
match = 0
for i, p in enumerate(probes):
    z = encode_all(H_M[i])
    top1 = int(z.abs().argmax().item())
    if top1 == p["top_k_feature_ids"][0]:
        match += 1
print(f"  top-1 match: {match}/{len(probes)} (should be ~all)")

# --- Step 2: run substitute models, collect their activations -----------------
SUB_ACTIVATIONS = {}  # model_id → tensor [n_probes, d_model_sub]

for sub in SUBSTITUTE_MODELS:
    print(f"\n[step 2] Loading substitute {sub['id']}")
    # free target model memory if tight: we keep it, 3090 has 24G
    m = AutoModelForCausalLM.from_pretrained(sub["id"], dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    nlay = m.config.num_hidden_layers
    dmod = m.config.hidden_size
    lay_idx = int(round(sub["layer_frac"] * nlay))
    # clip to valid range
    lay_idx = min(max(lay_idx, 0), nlay - 1)
    print(f"  {sub['id']}: {nlay} layers, d={dmod}, probing layer {lay_idx}")
    t0 = time.time()
    H_sub = []
    for i, prompt in enumerate(prompt_list):
        batch = [prompt] + FILLER_PROMPTS[:3]
        feats = batch_mlp_in(m, lay_idx, batch)  # [4, dmod]
        H_sub.append(feats[0])
    H_sub = torch.stack(H_sub)  # [n, dmod]
    SUB_ACTIVATIONS[sub["id"]] = {"H": H_sub.cpu(), "layer": lay_idx, "d": dmod}
    print(f"  captured {tuple(H_sub.shape)} in {time.time()-t0:.1f}s")
    del m
    torch.cuda.empty_cache()
    print(f"  vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

# keep H_M on cpu too
H_M_cpu = H_M.cpu()

# --- Step 3: for each substitute, run three attacks and score ----------------
def score_attack(H_fake_ongpu, H_M_ref, probes):
    """H_fake: [n, 2048]. Returns per-probe metrics + joint score."""
    results = []
    d = sae.cfg.d_in
    for i, p in enumerate(probes):
        top_ids = p["top_k_feature_ids"]
        top_means = torch.tensor(p["top_k_means"])
        top_stds = torch.tensor(p["top_k_stds"])
        z_fake = encode_top_pattern(H_fake_ongpu[i], top_ids)
        z_M = encode_top_pattern(H_M_ref[i], top_ids)

        # M-reported should match expected
        z_M_arr = z_M.numpy()
        z_fake_arr = z_fake.numpy()

        # Jaccard: what fraction of top-32 features are among the top-32 of the fake's full latent
        z_all_fake = encode_all(H_fake_ongpu[i])
        _, fake_top32_idx = torch.topk(z_all_fake.abs(), k=32)
        fake_top32 = set(fake_top32_idx.tolist())
        M_top32 = set(top_ids)
        jacc = len(M_top32 & fake_top32) / len(M_top32 | fake_top32)

        # Pearson on top-K magnitudes
        import numpy as np
        if z_fake.std() > 1e-6 and z_M.std() > 1e-6:
            pear = float(np.corrcoef(z_fake_arr, z_M_arr)[0, 1])
        else:
            pear = 0.0

        # Mahalanobis-like: how many sigmas away is fake from expected
        resid = (z_fake - top_means).abs()
        sig = (top_stds.clamp(min=1e-3))
        maha = float((resid / sig).mean().item())

        # L2 on magnitudes
        l2 = float((z_fake - top_means).norm().item())

        results.append({
            "probe_id": p["probe_id"], "category": p["category"],
            "jaccard_top32": jacc, "pearson": pear, "mahalanobis_mean": maha, "l2": l2,
        })
    return results

def summarize(results, label):
    import statistics
    js = [r["jaccard_top32"] for r in results]
    pr = [r["pearson"] for r in results]
    ma = [r["mahalanobis_mean"] for r in results]
    l2 = [r["l2"] for r in results]
    return {
        "label": label,
        "jaccard_mean": statistics.mean(js), "jaccard_median": statistics.median(js),
        "pearson_mean": statistics.mean(pr), "pearson_median": statistics.median(pr),
        "mahalanobis_mean": statistics.mean(ma), "mahalanobis_median": statistics.median(ma),
        "l2_mean": statistics.mean(l2), "l2_median": statistics.median(l2),
    }

print("\n[step 3] Running attacks and scoring")
all_results = {}

# --- Reference: M itself (should match near-perfectly) -----
print("  [ref] M on M (should match library)")
ref_r = score_attack(H_M, H_M_cpu, probes)
all_results["M_on_M"] = {"per_probe": ref_r, "summary": summarize(ref_r, "M_on_M")}
print(f"    summary: {all_results['M_on_M']['summary']}")

for model_id, sub in SUB_ACTIVATIONS.items():
    H_sub = sub["H"]  # [n, d_sub]
    d_sub = sub["d"]
    print(f"\n  === substitute: {model_id} (d={d_sub}) ===")

    # Attack A: ZERO_PAD or ZERO_TRUNC
    if d_sub <= d_in:
        H_pad = torch.zeros(H_sub.shape[0], d_in)
        H_pad[:, :d_sub] = H_sub
    else:  # d_sub > d_in, truncate
        H_pad = H_sub[:, :d_in].clone()
    r = score_attack(H_pad.cuda(), H_M_cpu, probes)
    all_results[f"{model_id}::ZERO_PAD"] = {"per_probe": r, "summary": summarize(r, f"{model_id}/ZERO_PAD")}
    print(f"    ZERO_PAD: {all_results[f'{model_id}::ZERO_PAD']['summary']}")

    # Attack B: random projection
    torch.manual_seed(42)
    W_rand = torch.randn(d_in, d_sub) * (1.0 / (d_sub ** 0.5))  # preserves magnitude
    H_rand = (H_sub @ W_rand.T)
    r = score_attack(H_rand.cuda(), H_M_cpu, probes)
    all_results[f"{model_id}::RAND_PROJ"] = {"per_probe": r, "summary": summarize(r, f"{model_id}/RAND_PROJ")}
    print(f"    RAND_PROJ: {all_results[f'{model_id}::RAND_PROJ']['summary']}")

    # Attack C: learned linear map via least squares.
    # Use 20/31 probes as calibration, evaluate on the remaining 11.
    # W minimizes ||W H_sub_calib.T - H_M_calib.T||_F
    # => W = H_M_calib.T @ pinv(H_sub_calib.T)
    n = H_sub.shape[0]
    idx_perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_cal = int(0.65 * n)
    idx_cal = idx_perm[:n_cal]
    idx_eval = idx_perm[n_cal:]
    H_sub_cal = H_sub[idx_cal]   # [n_cal, d_sub]
    H_M_cal = H_M_cpu[idx_cal]   # [n_cal, d_in]
    # Solve W_fit: d_in × d_sub
    # H_M_cal.T  = W_fit @ H_sub_cal.T
    # W_fit = H_M_cal.T @ pinv(H_sub_cal.T)
    W_fit = H_M_cal.float().T @ torch.linalg.pinv(H_sub_cal.float().T)
    # Apply to all probes (train+eval) but we'll report held-out separately
    H_learned = (H_sub.float() @ W_fit.T)
    r_all = score_attack(H_learned.cuda(), H_M_cpu, probes)
    all_results[f"{model_id}::LEARNED_LIN_all"] = {"per_probe": r_all, "summary": summarize(r_all, f"{model_id}/LEARNED_LIN_all")}
    eval_r = [r_all[i] for i in idx_eval.tolist()]
    all_results[f"{model_id}::LEARNED_LIN_heldout"] = {"per_probe": eval_r, "summary": summarize(eval_r, f"{model_id}/LEARNED_LIN_heldout")}
    print(f"    LEARNED_LIN (all): {all_results[f'{model_id}::LEARNED_LIN_all']['summary']}")
    print(f"    LEARNED_LIN (held-out {len(idx_eval)}): {all_results[f'{model_id}::LEARNED_LIN_heldout']['summary']}")

# --- Save -------------------------------------------------------------------
out = {
    "metadata": {
        "target": TARGET_MODEL,
        "target_layer": TARGET_LAYER,
        "substitutes": [s["id"] for s in SUBSTITUTE_MODELS],
        "n_probes": len(probes),
        "d_sae": d_sae,
    },
    "results": all_results,
}
(OUT_DIR / "p2_results.json").write_text(json.dumps(out, indent=2))
print(f"\n[save] {OUT_DIR / 'p2_results.json'}")

# --- Text summary -----------------------------------------------------------
summary_lines = []
summary_lines.append("P2: substitute attack results — fake M-compatible activations vs M's probe library")
summary_lines.append("=" * 85)
summary_lines.append(f"Target model: {TARGET_MODEL} layer {TARGET_LAYER}, d_sae={d_sae}, {len(probes)} probes")
summary_lines.append("")
summary_lines.append(f"{'attack':<45} {'jacc':>8} {'pear':>8} {'maha':>10} {'l2':>10}")
summary_lines.append("-" * 85)
for label, d in all_results.items():
    s = d["summary"]
    summary_lines.append(
        f"{label:<45} {s['jaccard_median']:>8.3f} {s['pearson_median']:>8.3f} "
        f"{s['mahalanobis_median']:>10.3f} {s['l2_median']:>10.3f}"
    )
summary_lines.append("")
summary_lines.append("Interpretation:")
summary_lines.append("  • jaccard  = top-32 feature-id overlap. M_on_M should be ~1.0.")
summary_lines.append("  • pearson  = correlation of magnitudes on M's top-32. Higher = closer.")
summary_lines.append("  • maha     = |residual|/sigma_M averaged. Smaller = closer to M.")
summary_lines.append("  • l2       = euclidean distance on top-32 magnitude vector.")
summary_lines.append("")
summary_lines.append("If LEARNED_LIN_heldout jacc < 0.3 AND pear < 0.3 → strong separability; attack is hard.")
summary_lines.append("If LEARNED_LIN_heldout jacc > 0.7 AND pear > 0.8 → attack succeeds; direction in danger.")
txt = "\n".join(summary_lines)
(OUT_DIR / "p2_summary.txt").write_text(txt)
print(txt)
print(f"\n[save] {OUT_DIR / 'p2_summary.txt'}")
