"""E10: circuit-ablation auditability (C5).

For each probe category, ablate the probe's top-k SAE features and measure
how it distorts the downstream logits. The key question is whether ablations
are *class-specific* — i.e., ablating `induction` features harms induction
prompts more than it harms `factual` prompts, and vice versa.

Baselines compared:
  - clean(x)               : unmodified forward
  - null(x)                : replace MLP(x) by transcoder_decode(transcoder_encode(x))
                             (reconstruction noise only; no ablation)
  - ablated(x, FIDS)       : same as null, but zero out features FIDS in z before decode

Per-pair metric:  KL(clean || ablated) − KL(clean || null)
                  = pure ablation effect with reconstruction noise subtracted.

For each pair (category A → probe B), we test:
  M[A→B] = mean_{probe p in A, probe q in B} effect(p's FIDS applied to q's prompt)

If M[A→A] >> M[A→B] for A≠B, the fingerprint is class-specific (C5).

Categories tested: ioi, induction, syntax, factual — 4 categories × 8 probes.

Estimated ≈ 12 min on Modal L4, ≈ $0.16.
"""
import json
import modal
from pathlib import Path

app = modal.App("e10-circuit-ablation")
GPU = "L4"
VOL = modal.Volume.from_name("e3-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers==4.56.2", "sae_lens==6.39.0",
        "datasets==3.1.0", "numpy<2", "zstandard",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

TARGET_MODEL = "Qwen/Qwen3-1.7B"
TARGET_LAYER = 14
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"
TOP_K = 32
CATS = ["ioi", "induction", "syntax", "factual"]
N_PROBES_PER_CAT = 8

LOCAL_ROOT = Path(__file__).parent.parent
LIB_LOCAL = LOCAL_ROOT / "results" / "probe_library_qwen3_1.7b_L14_k96.json"


@app.function(gpu=GPU, image=image, timeout=1800, volumes={"/cache": VOL},
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])])
def run(lib: dict) -> dict:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    # ---- pick N probes per category ----
    by_cat = {c: [] for c in CATS}
    for p in lib["probes"]:
        c = p["category"]
        if c in by_cat and len(by_cat[c]) < N_PROBES_PER_CAT:
            by_cat[c].append(p)
    for c, ps in by_cat.items():
        print(f"  category {c}: {len(ps)} probes")

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{TARGET_LAYER}", device="cuda")
    sae.eval()

    captured = {}
    # Transcoder-based substitution: replace mlp(x) by sae.decode(maybe_ablate(sae.encode(x)))
    # Install both pre-hook (to capture input) and forward-hook (to override output).

    target_layer = model.model.layers[TARGET_LAYER]
    mlp = target_layer.mlp

    ablation_fids = {"active": None}   # None = null (no ablation); list = ablate

    def pre_hook(_m, inputs):
        captured["x"] = inputs[0]

    def override(_m, inputs, outputs):
        x = captured["x"]
        z = sae.encode(x.to(sae.dtype))
        if ablation_fids["active"] is not None:
            for fid in ablation_fids["active"]:
                z[..., fid] = 0
        reconstructed = sae.decode(z)
        return reconstructed.to(outputs.dtype)

    def clean_forward(prompt):
        enc = tok([prompt], return_tensors="pt", truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            out = model(**enc)
        return F.log_softmax(out.logits[0, -1].float(), dim=-1)

    def swap_forward(prompt, fids):
        """fids=None → null reconstruct; fids=list → ablate those SAE features."""
        enc = tok([prompt], return_tensors="pt", truncation=True, max_length=64).to("cuda")
        ablation_fids["active"] = fids
        h1 = mlp.register_forward_pre_hook(pre_hook)
        h2 = mlp.register_forward_hook(override)
        with torch.no_grad():
            out = model(**enc)
        h1.remove(); h2.remove()
        ablation_fids["active"] = None
        return F.log_softmax(out.logits[0, -1].float(), dim=-1)

    def kl(logp, logq):
        p = logp.exp()
        return float((p * (logp - logq)).sum().item())

    # ---- gather clean / null logits per probe ----
    clean_logits = {}  # probe_id -> logp tensor
    null_logits  = {}
    print("[phase 1] clean + null forward for each probe")
    for c in CATS:
        for p in by_cat[c]:
            pid = p["probe_id"]
            clean_logits[pid] = clean_forward(p["prompt"])
            null_logits[pid]  = swap_forward(p["prompt"], None)

    # ---- cross-ablation: features from cat A applied to prompts in cat B ----
    print("[phase 2] cross-ablation K matrix")
    effect = np.zeros((len(CATS), len(CATS)))          # A × B: mean effect
    recon_noise = np.zeros(len(CATS))                  # per-cat null-vs-clean KL
    per_pair = {a: {b: [] for b in CATS} for a in CATS}

    for ia, A in enumerate(CATS):
        probes_A = by_cat[A]
        for p_a in probes_A:
            fids_a = p_a["top_k_feature_ids"][:TOP_K]
            for ib, B in enumerate(CATS):
                for p_b in by_cat[B]:
                    pid_b = p_b["probe_id"]
                    ab_logits = swap_forward(p_b["prompt"], fids_a)
                    eff = kl(clean_logits[pid_b], ab_logits) - kl(clean_logits[pid_b], null_logits[pid_b])
                    per_pair[A][B].append(eff)
        print(f"  done cat={A}  on-diag avg eff={np.mean(per_pair[A][A]):.4f}")

    # aggregate
    for ia, A in enumerate(CATS):
        for ib, B in enumerate(CATS):
            effect[ia, ib] = float(np.mean(per_pair[A][B]))

    # recon noise per probe (clean vs null)
    recon_per_cat = {}
    for c in CATS:
        vals = [kl(clean_logits[p["probe_id"]], null_logits[p["probe_id"]]) for p in by_cat[c]]
        recon_per_cat[c] = float(np.mean(vals))

    # class-specificity score per source cat = diag / mean(off-diag)
    spec = {}
    for ia, A in enumerate(CATS):
        diag = effect[ia, ia]
        off = np.delete(effect[ia], ia).mean()
        spec[A] = float(diag / (off + 1e-9))

    print("\n=== effect matrix (rows=src cat, cols=target prompt) ===")
    print("       " + "  ".join(f"{b:>10}" for b in CATS))
    for ia, A in enumerate(CATS):
        print(f"{A:>7} " + "  ".join(f"{effect[ia,ib]:>10.4f}" for ib in range(len(CATS))))

    print(f"\nreconstruction KL per cat: {recon_per_cat}")
    print(f"class-specificity (diag/off): {spec}")

    return {
        "cats": CATS,
        "effect_matrix": effect.tolist(),
        "recon_kl_per_cat": recon_per_cat,
        "specificity": spec,
        "n_probes_per_cat": N_PROBES_PER_CAT,
        "top_k": TOP_K,
    }


@app.local_entrypoint()
def main():
    lib = json.loads(LIB_LOCAL.read_text())
    r = run.remote(lib)
    out = LOCAL_ROOT / "results" / "e10_circuit_ablation.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\n[save] {out}")
    print(f"specificity: {r['specificity']}")
