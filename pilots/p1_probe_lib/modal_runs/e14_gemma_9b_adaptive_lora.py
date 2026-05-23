"""E14-9B: Gemma-2-9B adaptive LoRA (P0-1 scale-up).

Mirrors e14_gemma_adaptive_lora.py (2B target) but:
  - Target: google/gemma-2-9b @ layer 20 + width_131k canonical SAE
  - Substitute: google/gemma-2-2b-it (same-family, LoRA-lightweight)
  - Joint-phi: (d_sub+1, d_tgt) = (2305, 3584) bf16 => ~32MB param
  - StageA only: lambda_util=0, r=64, 1500 steps, 7 proj modules
  - Target unloaded after phi-fit (target_acts pre-cached on CPU)

Reads:
  /cache/e12_9b_out/e12_gemma_9b_pilot.json
Writes:
  logs/e14_gemma_9b_adaptive_lora.json
  /cache/e14_9b_out/e14_gemma_9b_adaptive_lora.json (volume backup)

Est. wall on A100-40GB: ~1.5-2.0h -> ~$3.2-4.2.
"""
import json
from pathlib import Path
import modal

app = modal.App("e14-gemma-9b-adaptive-lora")
GPU = "A100-40GB"
VOL = modal.Volume.from_name("e3-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.56.2",
        "sae_lens==6.39.0",
        "datasets==3.1.0",
        "peft==0.14.0",
        "numpy<2",
        "zstandard",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

TARGET_MODEL = "google/gemma-2-9b"
SUBSTITUTE_MODEL = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_20/width_131k/canonical"
LAYER_TGT = 20        # layer index in 9B (42 layers)
LAYER_SUB = 12        # layer index in 2B (26 layers) — depth-matched
TOP_K = 32
D_MODEL_TGT = 3584
D_MODEL_SUB = 2304

LOCAL_ROOT = Path(__file__).parent.parent
PILOT_JSON = LOCAL_ROOT / "logs" / "e12_gemma_9b_pilot.json"
OUT_LOCAL  = LOCAL_ROOT / "logs" / "e14_gemma_9b_adaptive_lora.json"


@app.function(
    gpu=GPU,
    image=image,
    timeout=9000,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(pilot_blob: dict) -> dict:
    import os
    import time
    import random
    import gc
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset

    torch.manual_seed(0)
    random.seed(0)
    rng = random.Random(0)
    t0 = time.time()

    probes = pilot_blob["probes"]
    sig_by_id = {int(k): v for k, v in pilot_blob["sigma_calibration"].items()}
    n_probes = len(probes)

    mu_full = np.stack([np.array(sig_by_id[i]["mean_cross_backend"]) for i in range(n_probes)])
    sd_full = np.clip(
        np.stack([np.array(sig_by_id[i]["sigma_cross_backend"]) for i in range(n_probes)]),
        1e-3, None,
    )
    print(f"[load] {n_probes} probes, mu shape={mu_full.shape}")

    # ---- target (9B) + SAE (frozen) ----
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tgt.eval()
    for p in tgt.parameters():
        p.requires_grad = False
    print(f"[load] target Gemma-2-9B in {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    sae_out = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    sae = sae_out[0] if isinstance(sae_out, tuple) else sae_out
    sae.eval()
    print(f"[load] SAE 131k in vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    cap = {}

    def hook_tgt(_m, _inputs, outputs):
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        cap["tgt"] = h.detach()

    h_tgt = tgt.model.layers[LAYER_TGT].register_forward_hook(hook_tgt)

    sub_tok = AutoTokenizer.from_pretrained(SUBSTITUTE_MODEL)
    if sub_tok.pad_token is None:
        sub_tok.pad_token = sub_tok.eos_token

    # ---- Pre-cache target activations for all probes ----
    target_acts = {}
    for pid in range(n_probes):
        enc = tok([probes[pid]["prompt"]], return_tensors="pt", truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            _ = tgt(**enc)
            target_acts[pid] = cap["tgt"][0, -1].float().cpu()
    print(f"[label] {n_probes} target acts cached")

    # ---- Pile corpus ----
    print("[pile] streaming 2000 samples")
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    pile_texts = []
    for ex in ds:
        if len(ex["text"]) > 50:
            pile_texts.append(ex["text"][:512])
        if len(pile_texts) >= 2000:
            break
    rng.shuffle(pile_texts)
    pile_train = pile_texts[:1600]
    pile_eval = pile_texts[1600:]  # 400 held-out

    # ===========================================================
    # Builders
    # ===========================================================
    def build_substitute(rank=64, seed=0):
        torch.manual_seed(seed)
        sub = AutoModelForCausalLM.from_pretrained(
            SUBSTITUTE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        cfg = LoraConfig(
            r=rank, lora_alpha=2 * rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        sub = get_peft_model(sub, cfg)

        def hook_sub(_m, _inputs, outputs):
            h = outputs[0] if isinstance(outputs, tuple) else outputs
            cap["sub"] = h  # NO detach — gradient flows in train
        h_handle = sub.base_model.model.model.layers[LAYER_SUB].register_forward_hook(hook_sub)
        return sub, h_handle

    def fit_initial_phi(sub_model):
        """Pre-fit phi (linear + bias) via LSQ on ~1000 Pile pairs. BOTH models must be loaded."""
        sub_model.eval()
        X_s, X_t = [], []
        with torch.no_grad():
            for i in range(0, 1000, 16):
                b = pile_train[i:i + 16]
                enc_t = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
                _ = tgt(**enc_t)
                idx_t = enc_t["attention_mask"].sum(dim=1) - 1
                Ht = torch.stack([cap["tgt"][k, idx_t[k]] for k in range(cap["tgt"].shape[0])]).float().cpu()
                enc_s = sub_tok(b, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
                _ = sub_model(**enc_s)
                idx_s = enc_s["attention_mask"].sum(dim=1) - 1
                Hs = torch.stack([cap["sub"][k, idx_s[k]] for k in range(cap["sub"].shape[0])]).float().cpu()
                X_s.append(Hs); X_t.append(Ht)
        X_s = torch.cat(X_s); X_t = torch.cat(X_t)
        n_tr = int(0.8 * X_s.shape[0])
        A_tr = torch.cat([X_s[:n_tr], torch.ones(n_tr, 1)], dim=1)
        phi_init, *_ = torch.linalg.lstsq(A_tr, X_t[:n_tr])
        A_ho = torch.cat([X_s[n_tr:], torch.ones(X_s.shape[0] - n_tr, 1)], dim=1)
        pred = A_ho @ phi_init
        r2_ho = float(1.0 - ((pred - X_t[n_tr:]).norm() ** 2 / X_t[n_tr:].norm() ** 2).item())
        phi = nn.Parameter(phi_init.to("cuda").to(torch.bfloat16))
        return phi, r2_ho

    # ===========================================================
    # Eval helpers
    # ===========================================================
    def joint_z_from_top32(z_per_probe_top32):
        per_probe = np.abs((z_per_probe_top32 - mu_full[:, :TOP_K]) / sd_full[:, :TOP_K]).mean(-1)
        return float(per_probe.mean()), per_probe

    def eval_substitute(sub_model, phi):
        sub_model.eval()
        z_top32 = np.zeros((n_probes, TOP_K))
        per_cat = {}
        with torch.no_grad():
            for pid in range(n_probes):
                enc = sub_tok([probes[pid]["prompt"]], return_tensors="pt", truncation=True, max_length=64).to("cuda")
                _ = sub_model(**enc)
                h_sub = cap["sub"][0, -1]
                h_aug = torch.cat([h_sub, torch.ones(1, device="cuda", dtype=h_sub.dtype)])
                proj = (h_aug @ phi).float()
                z_all = sae.encode(proj.unsqueeze(0).to(sae.dtype))[0].float().cpu().numpy()
                fids = probes[pid]["top_k_feature_ids"][:TOP_K]
                z_top32[pid] = z_all[fids]
                z_pid = np.abs((z_top32[pid] - mu_full[pid, :TOP_K]) / sd_full[pid, :TOP_K]).mean()
                per_cat.setdefault(probes[pid]["category"], []).append(float(z_pid))
        joint, per_probe = joint_z_from_top32(z_top32)
        return {
            "joint_z": joint,
            "per_cat_mean_z": {c: float(np.mean(v)) for c, v in per_cat.items()},
            "per_probe_z_median": float(np.median(per_probe)),
            "per_probe_z_max": float(np.max(per_probe)),
            "per_probe_z_min": float(np.min(per_probe)),
        }

    def eval_perplexity(sub_model, texts, batch=8):
        sub_model.eval()
        total_loss, total_tok = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                b = texts[i:i + batch]
                enc = sub_tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
                lbls = enc["input_ids"].clone()
                lbls[enc["attention_mask"] == 0] = -100
                out = sub_model(**enc, labels=lbls)
                n_tok = int((lbls != -100).sum().item()) - len(b)
                if n_tok <= 0:
                    continue
                total_loss += out.loss.item() * n_tok
                total_tok += n_tok
        return float(np.exp(total_loss / max(total_tok, 1)))

    def train_attack(sub, phi, lambda_probe, lambda_util, n_steps,
                     probe_batch=8, util_batch=4, lr=3e-4, log_every=200):
        opt = torch.optim.AdamW(
            [{"params": [p for p in sub.parameters() if p.requires_grad], "lr": lr},
             {"params": [phi], "lr": lr}],
            weight_decay=0.0,
        )
        pl_hist, ul_hist = [], []
        sub.train()
        for step in range(n_steps):
            pids = rng.sample(range(n_probes), probe_batch)
            prompts = [probes[p]["prompt"] for p in pids]
            enc_s = sub_tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
            _ = sub(**enc_s)
            h = cap["sub"]
            idx = enc_s["attention_mask"].sum(dim=1) - 1
            h_last = torch.stack([h[k, idx[k]] for k in range(h.shape[0])])
            ones = torch.ones(h_last.shape[0], 1, device="cuda", dtype=h_last.dtype)
            pred_act = (torch.cat([h_last, ones], dim=1) @ phi).float()
            tgt_acts = torch.stack([target_acts[p] for p in pids]).to("cuda")
            L_probe = F.mse_loss(pred_act, tgt_acts)

            L_util = torch.tensor(0.0, device="cuda")
            if lambda_util > 0:
                ub = rng.sample(pile_train, util_batch)
                enc_u = sub_tok(ub, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
                lbls = enc_u["input_ids"].clone()
                lbls[enc_u["attention_mask"] == 0] = -100
                out = sub(**enc_u, labels=lbls)
                L_util = out.loss

            L = lambda_probe * L_probe + lambda_util * L_util
            opt.zero_grad()
            L.backward()
            opt.step()
            pl_hist.append(float(L_probe.item()))
            ul_hist.append(float(L_util.item()) if lambda_util > 0 else 0.0)
            if (step + 1) % log_every == 0:
                print(f"  [{step+1}/{n_steps}] L_probe={np.mean(pl_hist[-log_every:]):.4f} "
                      f"L_util={np.mean(ul_hist[-log_every:]):.4f} "
                      f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB "
                      f"elapsed={time.time()-t0:.0f}s")
        return pl_hist, ul_hist

    # ===========================================================
    # Run: Gemma Stage A only
    # ===========================================================
    results = {}

    print("\n=== Baseline-clean: sub-it no-LoRA Pile ppx ===")
    sub_clean = AutoModelForCausalLM.from_pretrained(
        SUBSTITUTE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    sub_clean.eval()
    base_ppx_clean = eval_perplexity(sub_clean, pile_eval[:200])
    print(f"  sub-it-clean Pile ppx = {base_ppx_clean:.2f}")
    del sub_clean
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n=== Build Gemma substitute (rank=64) ===  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
    sub, hh = build_substitute(rank=64, seed=0)
    sub.print_trainable_parameters()
    print(f"  after sub-build vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
    phi, r2_phi = fit_initial_phi(sub)
    print(f"[phi] heldout R^2 = {r2_phi:.3f}  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ---- unload target to free ~18GB for training backward pass ----
    h_tgt.remove()
    del tgt, tok
    gc.collect(); torch.cuda.empty_cache()
    print(f"[unload] target freed, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    print("\n=== Baseline_phi: sub-it + phi-init only (no LoRA training) ===")
    base_eval = eval_substitute(sub, phi)
    base_eval["pile_perplexity"] = eval_perplexity(sub, pile_eval[:200])
    base_eval["phi_heldout_r2_init"] = r2_phi
    print(f"  joint_z={base_eval['joint_z']:.2f}  ppx={base_eval['pile_perplexity']:.2f}")
    results["baseline_phi_only"] = base_eval

    print("\n=== Stage A (Gemma): lambda_util=0, lambda_probe=1, 1500 steps ===")
    train_attack(sub, phi, lambda_probe=1.0, lambda_util=0.0, n_steps=1500)
    sa_eval = eval_substitute(sub, phi)
    sa_eval["pile_perplexity"] = eval_perplexity(sub, pile_eval[:200])
    sa_eval["lambda_probe"] = 1.0; sa_eval["lambda_util"] = 0.0
    print(f"  joint_z={sa_eval['joint_z']:.2f}  ppx={sa_eval['pile_perplexity']:.2f}")
    print(f"  per_cat={sa_eval['per_cat_mean_z']}")
    results["stageA_pure_probe"] = sa_eval

    hh.remove(); del sub, phi
    gc.collect(); torch.cuda.empty_cache()

    # Honest tau (synthetic bootstrap against the pilot sigma)
    rng_np = np.random.default_rng(0)
    honest = rng_np.normal(mu_full[None], sd_full[None], size=(500, n_probes, TOP_K))
    honest_joint = np.abs((honest - mu_full[:, :TOP_K]) / sd_full[:, :TOP_K]).mean(-1).mean(-1)
    tau_99_synth = float(np.quantile(honest_joint, 0.99))

    out = {
        "metadata": {
            "target": TARGET_MODEL,
            "substitute": SUBSTITUTE_MODEL,
            "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
            "layer_target": LAYER_TGT, "layer_substitute": LAYER_SUB,
            "d_model_target": D_MODEL_TGT, "d_model_substitute": D_MODEL_SUB,
            "lora_rank": 64,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
            "n_probes": n_probes,
            "top_k": TOP_K,
            "n_train_steps": 1500,
            "honest_tau_99_synthetic": tau_99_synth,
            "sub_baseline_pile_ppx_no_lora": base_ppx_clean,
            "wall_s": time.time() - t0,
            "gpu": GPU,
        },
        "results": results,
    }

    os.makedirs("/cache/e14_9b_out", exist_ok=True)
    with open("/cache/e14_9b_out/e14_gemma_9b_adaptive_lora.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[persist] /cache/e14_9b_out/e14_gemma_9b_adaptive_lora.json")
    return out


@app.local_entrypoint()
def main():
    pilot = json.loads(PILOT_JSON.read_text())
    result = run.remote(pilot)
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOCAL.write_text(json.dumps(result, indent=2))
    md = result["metadata"]
    print(f"\n[save] {OUT_LOCAL}")
    print(f"\n=== Gemma-9B Stage A summary (tau_99_synth = {md['honest_tau_99_synthetic']:.2f}) ===")
    print(f"sub-it-clean Pile ppx (no LoRA, no phi): {md['sub_baseline_pile_ppx_no_lora']:.2f}")
    for stage, r in result["results"].items():
        det = "DETECTED" if r["joint_z"] > md["honest_tau_99_synthetic"] else "EVADED"
        ratio = r["joint_z"] / md["honest_tau_99_synthetic"]
        ppx_ratio = r.get("pile_perplexity", 0) / md["sub_baseline_pile_ppx_no_lora"]
        print(f"  {stage:<25} joint_z={r['joint_z']:7.2f} ({ratio:5.2f}x tau)  ppx={r.get('pile_perplexity', 0):7.2f} ({ppx_ratio:.2f}x clean)  -> {det}")
    print(f"\nwall_s = {md['wall_s']:.0f}s")
