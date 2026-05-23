"""E4-v2: library-aware adaptive LoRA attack (Round 4 — combined objective).

Round-4 reviewer recommendation: rank-64 LoRA on Qwen3-0.6B substitute,
trained against a combined objective on the FULL released 96-probe library:
    L = lambda_probe * MSE(phi(sub_act_probe), tgt_act_probe)
        + lambda_util  * NLL(sub on Pile sequences)

The lift phi: R^{1024} -> R^{2048} is a learned linear+bias map. It is
pre-fit by least-squares on 1000 Pile pairs (sub baseline vs target),
then fine-tuned jointly with the LoRA.

We sweep four operating points to draw the attacker trade-off curve:
  - baseline_phi:        sub + phi-init only (no LoRA training)
  - stageA_pure_probe:   lambda_util = 0     (max effort on probe-matching)
  - stageB_balanced:     lambda_util = 1.0   (matched scales)
  - stageC_low_util:     lambda_util = 0.1   (probe-prioritised)

Reports per stage:
  - joint-z (vs honest tau_99 = 0.82)
  - per-category mean z
  - Pile held-out perplexity (utility)

Modal L4 (24 GB), ~1-1.5 GPU-h total.
Output: pilots/p1_probe_lib/logs/e4_v2_library_aware.json
"""
import json
from pathlib import Path
import modal

app = modal.App("e4-v2-library-aware")
GPU = "L4"
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

TARGET_MODEL = "Qwen/Qwen3-1.7B"
SUBSTITUTE_MODEL = "Qwen/Qwen3-0.6B"
TARGET_LAYER = 14
SUB_LAYER = 14
TOP_K = 32
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"

LOCAL_ROOT = Path(__file__).parent.parent
LIB_LOCAL = LOCAL_ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_LOCAL = LOCAL_ROOT / "logs" / "sigma_calibration_qwen3_1.7b_L14.json"


@app.function(
    gpu=GPU,
    image=image,
    timeout=10800,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(lib: dict, sig: dict) -> dict:
    import time
    import random
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

    probes = lib["probes"]
    sig_by_id = {r["probe_id"]: r for r in sig["calibration"]}
    n_probes = len(probes)

    mu_full = np.stack([np.array(sig_by_id[i]["mean_cross_backend"]) for i in range(n_probes)])
    sd_full = np.clip(
        np.stack([np.array(sig_by_id[i]["sigma_cross_backend"]) for i in range(n_probes)]),
        1e-3, None,
    )
    print(f"[load] {n_probes} probes, mu shape={mu_full.shape}")

    # ---- target + SAE (frozen) ----
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tgt.eval()
    for p in tgt.parameters():
        p.requires_grad = False
    print(f"[load] target in {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{TARGET_LAYER}", device="cuda")
    sae.eval()

    cap = {}
    def hook_tgt(_m, inputs):
        cap["tgt"] = inputs[0].detach()
    tgt.model.layers[TARGET_LAYER].mlp.register_forward_pre_hook(hook_tgt)

    sub_tok = AutoTokenizer.from_pretrained(SUBSTITUTE_MODEL)
    if sub_tok.pad_token is None:
        sub_tok.pad_token = sub_tok.eos_token
    # left-pad for next-token loss (Qwen3 default is right; we keep right and mask)

    d_tgt = tgt.config.hidden_size  # 2048

    # ---- Pre-cache target activations for all 96 probes ----
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
    pile_eval = pile_texts[1600:]  # 400 held-out for utility ppx

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

        def hook_sub(_m, inputs):
            cap["sub"] = inputs[0]   # NO detach (gradient required during train)
        h_handle = sub.base_model.model.model.layers[SUB_LAYER].mlp.register_forward_pre_hook(hook_sub)
        return sub, h_handle

    def fit_initial_phi(sub_model):
        """Pre-fit phi (linear + bias) via least-squares on 1000 Pile pairs.
        At init, LoRA-B is zero so sub forward is the base substitute."""
        sub_model.eval()
        X_s, X_t = [], []
        with torch.no_grad():
            for i in range(0, 1000, 16):
                b = pile_train[i:i + 16]
                # target
                enc_t = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
                _ = tgt(**enc_t)
                idx_t = enc_t["attention_mask"].sum(dim=1) - 1
                Ht = torch.stack([cap["tgt"][k, idx_t[k]] for k in range(cap["tgt"].shape[0])]).float().cpu()
                # sub
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
                n_tok = int((lbls != -100).sum().item()) - len(b)  # CE shifts labels by 1
                if n_tok <= 0:
                    continue
                total_loss += out.loss.item() * n_tok
                total_tok += n_tok
        return float(np.exp(total_loss / max(total_tok, 1)))

    # ===========================================================
    # Trainer
    # ===========================================================
    def train_attack(sub, phi, lambda_probe, lambda_util, n_steps,
                     probe_batch=8, util_batch=4, lr=3e-4, log_every=200):
        opt = torch.optim.AdamW(
            [{"params": [p for p in sub.parameters() if p.requires_grad], "lr": lr},
             {"params": [phi], "lr": lr}],
            weight_decay=0.0,
        )
        probe_loss_hist, util_loss_hist = [], []
        sub.train()
        for step in range(n_steps):
            # ---- probe batch ----
            pids = rng.sample(range(n_probes), probe_batch)
            prompts = [probes[p]["prompt"] for p in pids]
            enc_s = sub_tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
            _ = sub(**enc_s)
            h = cap["sub"]                       # [B, T, d_sub]
            idx = enc_s["attention_mask"].sum(dim=1) - 1
            h_last = torch.stack([h[k, idx[k]] for k in range(h.shape[0])])  # [B, d_sub]
            ones = torch.ones(h_last.shape[0], 1, device="cuda", dtype=h_last.dtype)
            pred_act = (torch.cat([h_last, ones], dim=1) @ phi).float()       # [B, d_tgt]
            tgt_acts = torch.stack([target_acts[p] for p in pids]).to("cuda")  # [B, d_tgt]
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
            probe_loss_hist.append(float(L_probe.item()))
            util_loss_hist.append(float(L_util.item()) if lambda_util > 0 else 0.0)
            if (step + 1) % log_every == 0:
                print(f"  [{step+1}/{n_steps}] L_probe={np.mean(probe_loss_hist[-log_every:]):.4f} "
                      f"L_util={np.mean(util_loss_hist[-log_every:]):.4f} "
                      f"elapsed={time.time()-t0:.0f}s")
        return probe_loss_hist, util_loss_hist

    # ===========================================================
    # Run
    # ===========================================================
    results = {}

    # ---- Baseline: clean sub (no LoRA, no phi) perplexity reference ----
    print("\n=== Baseline-clean: sub no-LoRA Pile ppx ===")
    sub_clean = AutoModelForCausalLM.from_pretrained(
        SUBSTITUTE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    sub_clean.eval()
    base_ppx_clean = eval_perplexity(sub_clean, pile_eval[:200])
    print(f"  sub-clean Pile ppx = {base_ppx_clean:.2f}")
    del sub_clean
    torch.cuda.empty_cache()

    # ---- Build substitute and pre-fit phi ----
    print("\n=== Build substitute #1 (rank=64) ===")
    sub, hh = build_substitute(rank=64, seed=0)
    sub.print_trainable_parameters()
    phi, r2_phi = fit_initial_phi(sub)
    print(f"[phi] heldout R^2 = {r2_phi:.3f}")

    print("\n=== Baseline_phi: sub + phi-init only (no LoRA training) ===")
    base_eval = eval_substitute(sub, phi)
    base_eval["pile_perplexity"] = eval_perplexity(sub, pile_eval[:200])
    base_eval["phi_heldout_r2_init"] = r2_phi
    print(f"  joint_z={base_eval['joint_z']:.2f}  ppx={base_eval['pile_perplexity']:.2f}")
    print(f"  per_cat={base_eval['per_cat_mean_z']}")
    results["baseline_phi_only"] = base_eval

    print("\n=== Stage A: lambda_util=0, lambda_probe=1, 1500 steps ===")
    train_attack(sub, phi, lambda_probe=1.0, lambda_util=0.0, n_steps=1500)
    sa_eval = eval_substitute(sub, phi)
    sa_eval["pile_perplexity"] = eval_perplexity(sub, pile_eval[:200])
    sa_eval["lambda_probe"] = 1.0; sa_eval["lambda_util"] = 0.0
    print(f"  joint_z={sa_eval['joint_z']:.2f}  ppx={sa_eval['pile_perplexity']:.2f}")
    print(f"  per_cat={sa_eval['per_cat_mean_z']}")
    results["stageA_pure_probe"] = sa_eval

    hh.remove(); del sub, phi
    torch.cuda.empty_cache()

    print("\n=== Build substitute #2 (rank=64) ===")
    sub, hh = build_substitute(rank=64, seed=1)
    phi, r2_phi = fit_initial_phi(sub)
    print(f"[phi] heldout R^2 = {r2_phi:.3f}")
    print("\n=== Stage B: lambda_util=1.0, lambda_probe=1.0, 1500 steps ===")
    train_attack(sub, phi, lambda_probe=1.0, lambda_util=1.0, n_steps=1500)
    sb_eval = eval_substitute(sub, phi)
    sb_eval["pile_perplexity"] = eval_perplexity(sub, pile_eval[:200])
    sb_eval["lambda_probe"] = 1.0; sb_eval["lambda_util"] = 1.0
    print(f"  joint_z={sb_eval['joint_z']:.2f}  ppx={sb_eval['pile_perplexity']:.2f}")
    print(f"  per_cat={sb_eval['per_cat_mean_z']}")
    results["stageB_balanced"] = sb_eval

    hh.remove(); del sub, phi
    torch.cuda.empty_cache()

    print("\n=== Build substitute #3 (rank=64) ===")
    sub, hh = build_substitute(rank=64, seed=2)
    phi, r2_phi = fit_initial_phi(sub)
    print(f"[phi] heldout R^2 = {r2_phi:.3f}")
    print("\n=== Stage C: lambda_util=0.1, lambda_probe=1.0, 1500 steps ===")
    train_attack(sub, phi, lambda_probe=1.0, lambda_util=0.1, n_steps=1500)
    sc_eval = eval_substitute(sub, phi)
    sc_eval["pile_perplexity"] = eval_perplexity(sub, pile_eval[:200])
    sc_eval["lambda_probe"] = 1.0; sc_eval["lambda_util"] = 0.1
    print(f"  joint_z={sc_eval['joint_z']:.2f}  ppx={sc_eval['pile_perplexity']:.2f}")
    print(f"  per_cat={sc_eval['per_cat_mean_z']}")
    results["stageC_low_util"] = sc_eval

    # ---- Honest tau (recompute for sanity) ----
    rng_np = np.random.default_rng(0)
    honest = rng_np.normal(mu_full[None], sd_full[None], size=(500, n_probes, TOP_K))
    honest_joint = np.abs((honest - mu_full[:, :TOP_K]) / sd_full[:, :TOP_K]).mean(-1).mean(-1)
    tau_99 = float(np.quantile(honest_joint, 0.99))

    return {
        "metadata": {
            "target": TARGET_MODEL,
            "substitute": SUBSTITUTE_MODEL,
            "lora_rank": 64,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
            "n_probes": n_probes,
            "top_k": TOP_K,
            "n_train_steps_per_stage": 1500,
            "honest_tau_99": tau_99,
            "sub_baseline_pile_ppx_no_lora": base_ppx_clean,
            "wall_s": time.time() - t0,
        },
        "results": results,
    }


@app.local_entrypoint()
def main():
    lib = json.loads(LIB_LOCAL.read_text())
    sig = json.loads(SIG_LOCAL.read_text())
    result = run.remote(lib, sig)
    out = LOCAL_ROOT / "logs" / "e4_v2_library_aware.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[save] {out}")
    md = result["metadata"]
    print(f"\n=== Summary (tau_99 = {md['honest_tau_99']:.2f}) ===")
    print(f"sub_clean_ppx (no LoRA, no phi)   : {md['sub_baseline_pile_ppx_no_lora']:.2f}")
    for stage, r in result["results"].items():
        det = "DETECTED" if r["joint_z"] > md["honest_tau_99"] else "EVADED"
        print(f"  {stage:<25} joint_z={r['joint_z']:7.2f}  ppx={r.get('pile_perplexity', 0):7.2f}  -> {det}")
    print(f"\nwall_s = {md['wall_s']:.0f}s")
