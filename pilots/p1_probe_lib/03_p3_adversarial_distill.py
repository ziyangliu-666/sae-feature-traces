"""P3: adversarial distillation attack against the probe library.

Threat model (strong):
  - Attacker runs Qwen3-0.6B (d=1024) but must return d=2048 activations that
    pattern-match Qwen3-1.7B's SAE latents on unseen probe prompts.
  - Attacker has white-box access to both models.
  - Attacker can collect N (prompt, h'_{0.6B}, h_{1.7B}) triplets on ANY prompts
    (simulating the realistic case: probe library is secret; attacker calibrates
    on random text and hopes to generalize).
  - Attacker trains a non-linear lift g_θ : R^1024 → R^2048 to minimize
    ||g_θ(h') - h||^2 on calibration.
  - The 31 library probes are the held-out test set — attacker has never seen them.

We sweep N ∈ {100, 500, 2000, 5000} and report:
  - calibration MSE
  - held-out MSE (same statistic, but probe positions only)
  - latent-pattern metrics (Jaccard, Pearson, Mahalanobis) after g_θ-lift + M-SAE encode

Kill criterion for the project:
  If at N=500 the attacker achieves Jaccard > 0.7 AND Mahalanobis < 5 on held-out,
  the protocol is too weak and we pivot.

Estimated runtime: ~15-20 min on RTX 3090.
"""
import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

OUT_DIR = Path(__file__).parent / "logs"
LIB_PATH = OUT_DIR / "probe_library_qwen3_1.7b_L14.json"

TARGET_MODEL = "Qwen/Qwen3-1.7B"
SUBSTITUTE_MODEL = "Qwen/Qwen3-0.6B"
LAYER = 14
SWEEP_N = [100, 500, 2000, 5000]

random.seed(0); torch.manual_seed(0)

# --- Load probe library ------------------------------------------------------
lib = json.loads(LIB_PATH.read_text())
probes = lib["probes"]
probe_prompts = [p["prompt"] for p in probes]
n_probes = len(probes)
print(f"[load] library: {n_probes} probes")

# --- Load SAE for target model (our verifier's tool) -------------------------
print("[load] target SAE")
sae = SAE.from_pretrained(
    release="mwhanna-qwen3-1.7b-transcoders-lowl0",
    sae_id=f"layer_{LAYER}",
    device="cuda",
)

# --- Load both models --------------------------------------------------------
print(f"[load] {TARGET_MODEL}")
tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=torch.bfloat16, device_map="cuda")
target.eval()
print(f"  vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

print(f"[load] {SUBSTITUTE_MODEL}")
substitute = AutoModelForCausalLM.from_pretrained(SUBSTITUTE_MODEL, dtype=torch.bfloat16, device_map="cuda")
substitute.eval()
print(f"  vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

# --- Calibration corpus ------------------------------------------------------
# Generic prompts drawn from common English patterns; NOT overlapping with the probes.
# We keep them short (<30 tokens) for speed.
CORPUS = [
    # Narrative
    "Once upon a time in a small village,",
    "The scientist spent years studying",
    "It was a dark and stormy night when",
    "The young prince discovered a hidden",
    "After the battle ended, the soldiers",
    # Technical
    "The algorithm runs in polynomial time because",
    "Neural networks can approximate any function, but",
    "In distributed systems, consistency is",
    "The memory hierarchy consists of",
    "Compilers optimize code by",
    # Question-answer
    "What is the meaning of life?",
    "How does photosynthesis work?",
    "Why is the sky blue?",
    "When was the Roman Empire founded?",
    "Where does the Amazon river begin?",
    # Instructions
    "To bake a cake you will need",
    "The first step in learning Python is",
    "Before starting your car, check",
    "When writing an email, remember to",
    "In negotiations, it helps to",
    # Lists
    "The planets of our solar system include",
    "Common programming languages are",
    "Three kinds of renewable energy are",
    "The primary colors are",
    "Famous Greek philosophers include",
    # Poetic / abstract
    "Love is like",
    "Time moves forward but",
    "The ocean whispers secrets about",
    "Every morning brings a new",
    "Dreams often contain",
    # News / factual
    "The stock market closed today with",
    "Scientists recently announced a breakthrough in",
    "The conference will be held in",
    "According to the latest report,",
    "Economists predict that",
    # Dialogue
    "She said, \"I can't believe you",
    "He replied, \"That's not what I meant when",
    "They shouted, \"Run!\" and then",
    "\"Excuse me,\" the stranger asked,",
    "The child whispered, \"Look,",
    # Code / math fragments
    "def fibonacci(n):",
    "for i in range(10):",
    "The integral of x squared is",
    "Differentiating sin(x) gives",
    "A matrix is invertible if and only if",
    # Everyday
    "On a sunny afternoon, we decided to",
    "The traffic was terrible because",
    "My grandmother always said",
    "The new restaurant downtown serves",
    "Walking through the park, I saw",
]

# We expand by taking prefixes up to 5 positions per prompt — gives us more samples
def make_calibration_prompts(n_total):
    # cycle through corpus with slight variations
    base = list(CORPUS)
    random.shuffle(base)
    out = []
    while len(out) < n_total:
        for s in base:
            if len(out) >= n_total: break
            out.append(s)
    return out

# --- Hook infrastructure -----------------------------------------------------
def make_hook(container, key):
    def hook(_mod, inputs):
        container[key] = inputs[0].detach()
    return hook

def last_token_mlp_in(model, layer_idx, prompts, batch_size=32):
    """Return stack of last-token hidden-states at mlp_in of layer_idx."""
    outs = []
    captured = {}
    h = model.model.layers[layer_idx].mlp.register_forward_pre_hook(make_hook(captured, "x"))
    try:
        for i in range(0, len(prompts), batch_size):
            b = prompts[i:i+batch_size]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
            with torch.no_grad():
                _ = model(**enc)
            x = captured["x"]  # [B, T, d]
            last = enc["attention_mask"].sum(dim=1) - 1
            for j in range(x.shape[0]):
                outs.append(x[j, last[j]].float().cpu())
        return torch.stack(outs)
    finally:
        h.remove()

# --- Gather max N calibration activations once -------------------------------
N_MAX = max(SWEEP_N)
cal_prompts = make_calibration_prompts(N_MAX)
print(f"\n[calib] collecting {N_MAX} paired activations")
t0 = time.time()
H_cal_target = last_token_mlp_in(target, LAYER, cal_prompts)      # [N, 2048]
print(f"  target: {H_cal_target.shape} in {time.time()-t0:.1f}s")
t0 = time.time()
H_cal_sub = last_token_mlp_in(substitute, LAYER, cal_prompts)    # [N, 1024]
print(f"  sub:    {H_cal_sub.shape} in {time.time()-t0:.1f}s")

# --- Gather held-out (probe) activations ------------------------------------
print("\n[eval] collecting probe-set activations")
H_eval_target = last_token_mlp_in(target, LAYER, probe_prompts)  # [n_probes, 2048]
H_eval_sub = last_token_mlp_in(substitute, LAYER, probe_prompts) # [n_probes, 1024]
print(f"  target probe shape: {H_eval_target.shape}")
print(f"  sub probe shape:    {H_eval_sub.shape}")

# Free substitute model; we don't need it during training
del substitute
torch.cuda.empty_cache()

# --- Attack: train MLP lift on N calibration samples ------------------------
class LiftMLP(nn.Module):
    def __init__(self, d_in, d_out, d_hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, x): return self.net(x)

def train_lift(H_in, H_out, n_train, seed=0):
    """Returns trained model. Splits H_in/H_out into train/val (90/10)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(H_in.shape[0], generator=g)
    H_in_s = H_in[perm][:n_train].cuda()
    H_out_s = H_out[perm][:n_train].cuda()
    # 90/10 split
    sp = max(int(0.9 * n_train), 1)
    X_tr, X_val = H_in_s[:sp], H_in_s[sp:]
    Y_tr, Y_val = H_out_s[:sp], H_out_s[sp:]

    model = LiftMLP(H_in.shape[1], H_out.shape[1], d_hidden=4096).cuda()
    # for small N we need to avoid overfitting the MLP; use more epochs but LR schedule
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    n_epochs = 60 if n_train < 500 else 40 if n_train < 2000 else 20
    batch = min(64, sp)
    best_val = float("inf")
    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(sp)
        for i in range(0, sp, batch):
            b = idx[i:i+batch]
            pred = model(X_tr[b])
            loss = F.mse_loss(pred, Y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            if X_val.shape[0] > 0:
                val_mse = F.mse_loss(model(X_val), Y_val).item()
            else:
                val_mse = loss.item()
        if val_mse < best_val: best_val = val_mse
    model.eval()
    return model, best_val

def score_fake(H_fake):
    """H_fake: [n_probes, 2048] on GPU. Return metrics vs M's probe library."""
    import numpy as np
    per_probe = []
    for i, p in enumerate(probes):
        top_ids = p["top_k_feature_ids"]
        top_means = torch.tensor(p["top_k_means"])
        top_stds = torch.tensor(p["top_k_stds"]).clamp(min=1e-3)
        with torch.no_grad():
            z_all = sae.encode(H_fake[i:i+1].to(sae.dtype).cuda()).squeeze(0).float().cpu()
        z_on_top = z_all[top_ids]
        # top-32 of fake
        _, fake_top32 = torch.topk(z_all.abs(), k=32)
        fake_set = set(fake_top32.tolist())
        M_set = set(top_ids)
        jacc = len(M_set & fake_set) / len(M_set | fake_set)
        if z_on_top.std() > 1e-6:
            pear = float(np.corrcoef(z_on_top.numpy(), top_means.numpy())[0, 1])
        else:
            pear = 0.0
        maha = float(((z_on_top - top_means).abs() / top_stds).mean().item())
        per_probe.append({"probe_id": p["probe_id"], "category": p["category"],
                          "jaccard": jacc, "pearson": pear, "mahalanobis": maha})
    import statistics
    js = [r["jaccard"] for r in per_probe]
    pr = [r["pearson"] for r in per_probe]
    ma = [r["mahalanobis"] for r in per_probe]
    return per_probe, {
        "jaccard_mean": statistics.mean(js), "jaccard_median": statistics.median(js),
        "pearson_mean": statistics.mean(pr), "pearson_median": statistics.median(pr),
        "mahalanobis_mean": statistics.mean(ma), "mahalanobis_median": statistics.median(ma),
    }

# --- Run sweep ---------------------------------------------------------------
print("\n" + "=" * 70)
print("P3 SWEEP: attacker sample size → held-out probe detection")
print("=" * 70)

results = {}
for N in SWEEP_N:
    print(f"\n[N={N}] training lift MLP on {N} pairs")
    t_start = time.time()
    g, val_mse = train_lift(H_cal_sub, H_cal_target, n_train=N, seed=0)
    with torch.no_grad():
        H_fake_eval = g(H_eval_sub.cuda())  # [n_probes, 2048]
    # MSE on held-out probes (as raw activations)
    with torch.no_grad():
        hold_mse = F.mse_loss(H_fake_eval, H_eval_target.cuda()).item()
        ref_mse = F.mse_loss(torch.zeros_like(H_eval_target).cuda(), H_eval_target.cuda()).item()
    print(f"  train val_mse={val_mse:.4f}  eval_mse={hold_mse:.4f}  ref_mse(zero)={ref_mse:.4f}")
    per_probe, summary = score_fake(H_fake_eval)
    summary["N"] = N
    summary["cal_val_mse"] = val_mse
    summary["eval_mse"] = hold_mse
    summary["ref_mse_zero"] = ref_mse
    summary["runtime_sec"] = time.time() - t_start
    results[f"N={N}"] = {"per_probe": per_probe, "summary": summary}
    print(f"  summary: jacc_med={summary['jaccard_median']:.3f}  pear_med={summary['pearson_median']:.3f}  maha_med={summary['mahalanobis_median']:.2f}")

# --- Save -------------------------------------------------------------------
out = {
    "metadata": {
        "target": TARGET_MODEL,
        "substitute": SUBSTITUTE_MODEL,
        "layer": LAYER,
        "sweep_N": SWEEP_N,
        "n_probes_heldout": n_probes,
        "attack_arch": "3-layer MLP, hidden 4096, GELU",
    },
    "results": results,
}
(OUT_DIR / "p3_results.json").write_text(json.dumps(out, indent=2))
print(f"\n[save] {OUT_DIR / 'p3_results.json'}")

# --- Summary ----------------------------------------------------------------
lines = []
lines.append("P3: adversarial distillation attack — held-out probe detection vs attacker sample size")
lines.append("=" * 92)
lines.append(f"{'N_calib':>10} {'val_mse':>10} {'eval_mse':>10} {'jacc_med':>10} {'pear_med':>10} {'maha_med':>10} {'detect?':>10}")
lines.append("-" * 92)
for k, d in results.items():
    s = d["summary"]
    # simple detector: maha_median > 5 AND jacc_median < 0.5 → detected
    detected = "YES" if (s["mahalanobis_median"] > 5 and s["jaccard_median"] < 0.5) else "no"
    lines.append(f"{s['N']:>10} {s['cal_val_mse']:>10.4f} {s['eval_mse']:>10.4f} "
                 f"{s['jaccard_median']:>10.3f} {s['pearson_median']:>10.3f} {s['mahalanobis_median']:>10.2f} {detected:>10}")

lines.append("")
lines.append("Kill criterion: if at N=500 attacker achieves jacc_med > 0.7 AND maha_med < 5,")
lines.append("the protocol is too weak. Otherwise, attacker's sample efficiency is our forcing function.")
txt = "\n".join(lines)
(OUT_DIR / "p3_summary.txt").write_text(txt)
print("\n" + txt)
print(f"\n[save] {OUT_DIR / 'p3_summary.txt'}")
