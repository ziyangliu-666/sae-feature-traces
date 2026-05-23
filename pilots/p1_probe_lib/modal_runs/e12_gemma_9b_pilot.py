"""E12-9B: Gemma-2-9B probe library + sigma baseline (P0-1 scale-up).

Mirrors e12_gemma_pilot.py (Gemma-2-2B) but:
  - Target: google/gemma-2-9b @ layer 20
  - SAE: gemma-scope-9b-pt-res-canonical / layer_20/width_131k/canonical
  - GPU: A100-40GB (9B bf16 ~18.5GB + 131k SAE ~1.9GB bf16 + activations)
  - SKIPS Phase 3 attackers (redundant; E13/E14 handle cross-family + LoRA).

Output: /cache/e12_9b_out/e12_gemma_9b_pilot.json (volume) +
  logs/e12_gemma_9b_pilot.json (local). Format matches 2B pilot so
  e15/e13/e14 9B clones can consume it directly.

Est. wall: ~45-60 min on A100-40GB -> ~$1.6-2.1.
"""
import json
import modal
from pathlib import Path

app = modal.App("e12-gemma-9b-pilot")
GPU = "A100-40GB"
VOL = modal.Volume.from_name("e3-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.56.2",
        "sae_lens==6.39.0",
        "datasets==3.1.0",
        "numpy<2",
        "accelerate>=0.33",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

# Backbone
TARGET_MODEL   = "google/gemma-2-9b"
SAE_RELEASE    = "gemma-scope-9b-pt-res-canonical"
SAE_ID         = "layer_20/width_131k/canonical"
LAYER          = 20
D_MODEL_TARGET = 3584

TOP_K = 32
N_REPEAT_LIB = 30
BATCH_COMPANIONS = 3

LOCAL_ROOT = Path(__file__).parent.parent
LIB_K96_QWEN = LOCAL_ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
OUT_JSON = LOCAL_ROOT / "logs" / "e12_gemma_9b_pilot.json"


@app.function(
    gpu=GPU,
    image=image,
    timeout=7200,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(shared_prompts: list) -> dict:
    import os, time, random
    import torch
    import numpy as np
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

    random.seed(0); torch.manual_seed(0)

    FILLER = [
        "The weather today is particularly",
        "Climate change affects biodiversity in multiple ways, including",
        "A typical morning routine often includes",
        "The history of cryptography spans several",
        "Modern cities face challenges such as",
        "Healthy cooking often involves fresh",
        "Space exploration has yielded numerous",
        "Online privacy has become a pressing",
        "Machine learning research has advanced rapidly and",
        "Economic policy debates often center on",
    ]

    # ------------------------------------------------------------------
    # Load target + SAE
    # ------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    target.eval()
    target_layers = target.config.num_hidden_layers
    print(f"[load] target Gemma-2-9B in {time.time()-t0:.1f}s, layers={target_layers}, d={target.config.hidden_size}, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
    assert target.config.hidden_size == D_MODEL_TARGET

    t1 = time.time()
    sae, sae_cfg, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    sae.eval()
    print(f"[load] Gemma-Scope SAE layer_{LAYER} width_131k in {time.time()-t1:.1f}s, d_sae={sae.cfg.d_sae}, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    captured = {}

    def hook_target(_m, _inputs, outputs):
        # Gemma-2 residual post: outputs is a tuple (hidden_states, ...)
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        captured["tgt"] = h.detach()

    h_tgt = target.model.layers[LAYER].register_forward_hook(hook_target)

    def forward_target(prompt, pos, companions, dtype, kernel):
        batch = companions.copy(); batch.insert(pos, prompt)
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        last = enc["attention_mask"].sum(dim=1) - 1
        with sdpa_kernel([kernel, SDPBackend.MATH]), torch.no_grad():
            if dtype == torch.bfloat16:
                _ = target(**enc)
            else:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = target(**enc)
        return captured["tgt"][pos, last[pos]]

    # ------------------------------------------------------------------
    # Phase 1: build probe library on Gemma-2-9B (reuse 96 prompts from Qwen3 E1)
    # ------------------------------------------------------------------
    print(f"\n[phase 1] build probe library on Gemma-2-9B, {len(shared_prompts)} probes x {N_REPEAT_LIB} repeats")
    t2 = time.time()
    probes = []
    for pidx, (cat, prompt) in enumerate(shared_prompts):
        latents = []
        for rep in range(N_REPEAT_LIB):
            companions = random.sample(FILLER, BATCH_COMPANIONS)
            batch = [prompt] + companions
            random.shuffle(batch)
            tp = batch.index(prompt)
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
            last = enc["attention_mask"].sum(dim=1) - 1
            with torch.no_grad():
                _ = target(**enc)
            act = captured["tgt"][tp, last[tp]]
            with torch.no_grad():
                z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
            latents.append(z.float().cpu())
        L = torch.stack(latents)
        mean = L.mean(dim=0); std = L.std(dim=0)
        topk_vals, topk_idx = torch.topk(mean.abs(), k=TOP_K)
        probes.append({
            "probe_id": pidx, "category": cat, "prompt": prompt,
            "top_k_feature_ids": topk_idx.tolist(),
            "top_k_means": mean[topk_idx].tolist(),
            "top_k_stds": std[topk_idx].tolist(),
        })
        if (pidx+1) % 16 == 0:
            el = time.time() - t2
            eta = el / (pidx+1) * (len(shared_prompts) - pidx - 1)
            print(f"  [{pidx+1:2d}/{len(shared_prompts)}] top1={topk_idx[0].item()} mag={mean[topk_idx[0]]:.2f} (elapsed {el:.0f}s eta {eta:.0f}s)")

    print(f"[phase 1] library built in {time.time()-t2:.0f}s")

    # ------------------------------------------------------------------
    # Phase 2: sigma calibration across honest backends (8 configs/probe)
    # ------------------------------------------------------------------
    PRECISIONS = [torch.bfloat16, torch.float16]
    KERNELS = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    POSITIONS = [0, 2]
    print(f"\n[phase 2] sigma calibration, {len(probes)} x {len(PRECISIONS)*len(KERNELS)*len(POSITIONS)} cfgs")
    t3 = time.time()
    sig_by_probe = {}
    for p in probes:
        snaps = []
        feat_ids = p["top_k_feature_ids"]
        for dtype in PRECISIONS:
            for kernel in KERNELS:
                for pos in POSITIONS:
                    companions = random.sample(FILLER, 3)
                    act = forward_target(p["prompt"], pos, companions, dtype, kernel)
                    with torch.no_grad():
                        z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
                    snaps.append(z[feat_ids].float().cpu())
        S = torch.stack(snaps)
        sig_by_probe[p["probe_id"]] = {
            "mean_cross_backend": S.mean(dim=0).tolist(),
            "sigma_cross_backend": S.std(dim=0).tolist(),
        }
    print(f"[phase 2] sigma done in {time.time()-t3:.0f}s")

    h_tgt.remove()

    # Quick diagnostic on sigma magnitudes (flag numerical-floor features early)
    all_sds = np.concatenate([
        np.array(sig_by_probe[p["probe_id"]]["sigma_cross_backend"]) for p in probes
    ])
    print(f"[diag] sigma p01={np.quantile(all_sds,0.01):.4f} p50={np.quantile(all_sds,0.5):.4f} p95={np.quantile(all_sds,0.95):.4f} max={all_sds.max():.4f}")
    print(f"[diag] frac sigma < 1e-3: {(all_sds<1e-3).mean():.4f}; < 1e-2: {(all_sds<1e-2).mean():.4f}")

    out = {
        "metadata": {
            "experiment": "E12-9B (Gemma-2-9B pilot, P0-1 scale-up)",
            "target": TARGET_MODEL,
            "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
            "layer": LAYER, "d_model": D_MODEL_TARGET,
            "d_sae": int(sae.cfg.d_sae), "top_k": TOP_K,
            "n_probes": len(probes),
            "wall_s": time.time() - t0,
            "gpu": GPU,
            "note": "Phase 3 attackers skipped; dedicated E13-9B/E14-9B runs handle cross-family + LoRA.",
        },
        "probes": probes,
        "sigma_calibration": sig_by_probe,
        "sigma_diag": {
            "p01": float(np.quantile(all_sds, 0.01)),
            "p50": float(np.quantile(all_sds, 0.5)),
            "p95": float(np.quantile(all_sds, 0.95)),
            "max": float(all_sds.max()),
            "frac_below_1e3": float((all_sds < 1e-3).mean()),
            "frac_below_1e2": float((all_sds < 1e-2).mean()),
        },
    }

    os.makedirs("/cache/e12_9b_out", exist_ok=True)
    with open("/cache/e12_9b_out/e12_gemma_9b_pilot.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[persist] /cache/e12_9b_out/e12_gemma_9b_pilot.json ({os.path.getsize('/cache/e12_9b_out/e12_gemma_9b_pilot.json')} bytes)")
    return out


@app.local_entrypoint()
def main():
    if not LIB_K96_QWEN.exists():
        raise SystemExit(f"Missing Qwen3 library for prompts: {LIB_K96_QWEN}")
    lib = json.loads(LIB_K96_QWEN.read_text())
    shared = [(p["category"], p["prompt"]) for p in lib["probes"]]
    print(f"reusing {len(shared)} probe prompts from Qwen3 E1 library")
    out = run.remote(shared)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {OUT_JSON}")
    print(f"wall: {out['metadata']['wall_s']:.0f}s, n_probes={out['metadata']['n_probes']}")
