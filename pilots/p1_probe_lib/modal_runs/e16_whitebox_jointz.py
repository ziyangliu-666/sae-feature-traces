"""E16 (E-E): White-box LoRA adaptive attack with joint-z objective.

v1 (e4_v2_library_aware) trained to match honest-side activations via
MSE(phi(h_sub), h_tgt). That's an INDIRECT surrogate for joint-z. For
the reviewer's W6 complaint we close the circuit: the training loss
IS the joint-z aggregator, with gradient flowing through the SAE
encoder (SAE params frozen).

  L_probe = mean_{probe, top-k feature} | (z_hat - mu) / sigma |
  L_util  = NLL on Pile
  L       = L_probe + lambda_util * L_util

SAE.requires_grad_(False) — frozen. No torch.no_grad() around the SAE
encode during training. phi is a learnable linear lift
R^{d_sub} -> R^{d_tgt}.

Two operating points per backbone (small sweep — expand after first
run validates the gradient path):
  Qwen3 r=64, lambda_util in {0.0, 0.1}, 3000 steps
  Gemma r=64, lambda_util in {0.0, 0.1}, 3000 steps

Modal A10G (~1.8 GPU-h) -> ~$2 for the Qwen3 pair, similar for Gemma.
Output: pilots/p1_probe_lib/logs/e16_whitebox_jointz_{backbone}.json
"""
import json
import argparse
from pathlib import Path
import modal

app = modal.App("e16-whitebox-jointz")
GPU = "A10G"
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

BACKBONES = {
    "qwen3": {
        "target": "Qwen/Qwen3-1.7B",
        "substitute": "Qwen/Qwen3-0.6B",
        "layer_tgt": 14,
        "layer_sub": 14,
        "sae_release": "mwhanna-qwen3-1.7b-transcoders-lowl0",
        "sae_id": "layer_14",
        "hook_kind": "mlp_in",
        "lib": "probe_library_qwen3_1.7b_L14_k96.json",
        "sigma": "sigma_calibration_qwen3_1.7b_L14.json",
        "tau_real": 1.13,
    },
    "gemma": {
        "target": "google/gemma-2-2b",
        "substitute": "google/gemma-2-2b-it",  # same d_model, different weights
        "layer_tgt": 12,
        "layer_sub": 12,
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_12/width_16k/canonical",
        "hook_kind": "residual_post",
        "lib": "e12_gemma_pilot.json",          # probes nested inside this file
        "sigma": "sigma_calibration_gemma2_2b_L12.json",
        "tau_real": 1.09,
    },
}

LOCAL_ROOT = Path(__file__).parent.parent


@app.function(
    gpu=GPU,
    image=image,
    timeout=14400,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(backbone: str, lib_data: dict, sig_data: dict,
        lambda_utils: list, alpha_jzs: list, n_steps: int, lora_rank: int,
        public_pids: list = None, seed_offset: int = 0) -> dict:
    import time, random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset

    cfg = BACKBONES[backbone]
    torch.manual_seed(0); random.seed(0)
    rng = random.Random(0)
    t0 = time.time()

    # Unpack library + sigma (handle both formats)
    if backbone == "gemma":
        probes = lib_data["probes"]
        sig_by_id = {int(k): v for k, v in sig_data.items()} \
            if isinstance(sig_data, dict) and "calibration" not in sig_data \
            else {r["probe_id"]: r for r in sig_data["calibration"]}
    else:
        probes = lib_data["probes"]
        sig_by_id = {r["probe_id"]: r for r in sig_data["calibration"]}
    n_probes = len(probes)
    TOP_K = 32

    # E-H: public/secret probe split. Training sees only public ids; eval reports both.
    if public_pids is None:
        train_pids = list(range(n_probes))
        secret_pids = []
    else:
        train_pids = list(public_pids)
        secret_pids = [i for i in range(n_probes) if i not in set(public_pids)]
    print(f"[split] train pids={len(train_pids)}, secret={len(secret_pids)}")

    mu_full = np.stack([np.array(sig_by_id[i]["mean_cross_backend"]) for i in range(n_probes)])
    sd_full = np.clip(
        np.stack([np.array(sig_by_id[i]["sigma_cross_backend"]) for i in range(n_probes)]),
        1e-3, None,
    )
    mu_t = torch.tensor(mu_full, device="cuda", dtype=torch.float32)  # [96, 32]
    sd_t = torch.tensor(sd_full, device="cuda", dtype=torch.float32)  # [96, 32]
    feat_ids = torch.tensor(
        np.stack([p["top_k_feature_ids"][:TOP_K] for p in probes]),
        device="cuda", dtype=torch.long,
    )  # [96, 32]
    print(f"[load] {n_probes} probes, mu {mu_full.shape}")

    # ---- Target + SAE ----
    tok = AutoTokenizer.from_pretrained(cfg["target"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(
        cfg["target"], torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tgt.eval()
    for p in tgt.parameters():
        p.requires_grad_(False)

    sae_res = SAE.from_pretrained(
        release=cfg["sae_release"], sae_id=cfg["sae_id"], device="cuda"
    )
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    print(f"[load] target+SAE in {time.time()-t0:.1f}s, d_sae={sae.cfg.d_sae}")

    # Target hook (for labelling)
    cap = {}
    if cfg["hook_kind"] == "mlp_in":
        def hook_tgt(_m, inputs):
            cap["tgt"] = inputs[0].detach()
        tgt.model.layers[cfg["layer_tgt"]].mlp.register_forward_pre_hook(hook_tgt)
    else:
        def hook_tgt(_m, _inputs, outputs):
            h = outputs[0] if isinstance(outputs, tuple) else outputs
            cap["tgt"] = h.detach()
        tgt.model.layers[cfg["layer_tgt"]].register_forward_hook(hook_tgt)

    # ---- Cache target activations for each probe (needed for dense MSE) ----
    target_acts = {}
    for pid in range(n_probes):
        enc = tok([probes[pid]["prompt"]], return_tensors="pt",
                  truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            _ = tgt(**enc)
            target_acts[pid] = cap["tgt"][0, -1].float().cpu()
    print(f"[label] {n_probes} target acts cached")

    # ---- Pile ----
    print("[pile] streaming 2000 samples")
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    pile_texts = []
    for ex in ds:
        if len(ex["text"]) > 50:
            pile_texts.append(ex["text"][:512])
        if len(pile_texts) >= 2000: break
    rng.shuffle(pile_texts)
    pile_train = pile_texts[:1600]
    pile_eval = pile_texts[1600:]

    # ---- Substitute builder ----
    sub_tok = AutoTokenizer.from_pretrained(cfg["substitute"])
    if sub_tok.pad_token is None: sub_tok.pad_token = sub_tok.eos_token
    sub_proto = AutoModelForCausalLM.from_pretrained(
        cfg["substitute"], torch_dtype=torch.bfloat16, device_map="cuda"
    )
    d_sub = sub_proto.config.hidden_size
    d_tgt = tgt.config.hidden_size
    del sub_proto; torch.cuda.empty_cache()
    print(f"d_sub={d_sub}, d_tgt={d_tgt}")

    def build_sub(rank, seed):
        torch.manual_seed(seed)
        sub = AutoModelForCausalLM.from_pretrained(
            cfg["substitute"], torch_dtype=torch.bfloat16, device_map="cuda"
        )
        lora_cfg = LoraConfig(
            r=rank, lora_alpha=2 * rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        sub = get_peft_model(sub, lora_cfg)

        if cfg["hook_kind"] == "mlp_in":
            def hook_sub(_m, inputs):
                cap["sub"] = inputs[0]  # no detach, gradient flows
            h = sub.base_model.model.model.layers[cfg["layer_sub"]].mlp.register_forward_pre_hook(hook_sub)
        else:
            def hook_sub(_m, _inputs, outputs):
                cap["sub"] = outputs[0] if isinstance(outputs, tuple) else outputs
            h = sub.base_model.model.model.layers[cfg["layer_sub"]].register_forward_hook(hook_sub)
        return sub, h

    # ---- Phi init (least-squares on Pile) ----
    def fit_phi(sub):
        sub.eval()
        X_s, X_t = [], []
        with torch.no_grad():
            for i in range(0, 1000, 16):
                b = pile_train[i:i+16]
                enc_t = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
                _ = tgt(**enc_t)
                idx_t = enc_t["attention_mask"].sum(dim=1) - 1
                Ht = torch.stack([cap["tgt"][k, idx_t[k]] for k in range(cap["tgt"].shape[0])]).float().cpu()
                enc_s = sub_tok(b, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
                _ = sub(**enc_s)
                idx_s = enc_s["attention_mask"].sum(dim=1) - 1
                Hs = torch.stack([cap["sub"][k, idx_s[k]].detach() for k in range(cap["sub"].shape[0])]).float().cpu()
                X_s.append(Hs); X_t.append(Ht)
        X_s = torch.cat(X_s); X_t = torch.cat(X_t)
        n_tr = int(0.8 * X_s.shape[0])
        A_tr = torch.cat([X_s[:n_tr], torch.ones(n_tr, 1)], dim=1)
        phi_init, *_ = torch.linalg.lstsq(A_tr, X_t[:n_tr])
        A_ho = torch.cat([X_s[n_tr:], torch.ones(X_s.shape[0] - n_tr, 1)], dim=1)
        pred = A_ho @ phi_init
        r2 = float(1.0 - ((pred - X_t[n_tr:]).norm()**2 / X_t[n_tr:].norm()**2).item())
        phi = nn.Parameter(phi_init.to("cuda").to(torch.bfloat16))
        return phi, r2

    # ---- Joint-z loss (gradient flows through SAE.encode) ----
    def jointz_loss(pred_act_batch, pids_tensor):
        """
        pred_act_batch: [B, d_tgt]  (float; output of h_sub @ phi)
        pids_tensor:    [B]         (probe ids for batch)
        Returns scalar joint-z loss.
        """
        # SAE encode with gradient
        z_full = sae.encode(pred_act_batch.to(sae.dtype))  # [B, d_sae]
        z_full = z_full.float()
        # Gather top-k features per probe in batch: [B, 32]
        fids_b = feat_ids[pids_tensor]  # [B, 32]
        z_pick = torch.gather(z_full, 1, fids_b)  # [B, 32]
        mu_b = mu_t[pids_tensor]
        sd_b = sd_t[pids_tensor]
        z_score = ((z_pick - mu_b) / sd_b).abs()  # [B, 32]
        return z_score.mean()

    # ---- Eval (no grad) ----
    def eval_substitute(sub, phi):
        sub.eval()
        z_top32 = np.zeros((n_probes, TOP_K))
        per_cat = {}
        with torch.no_grad():
            for pid in range(n_probes):
                enc = sub_tok([probes[pid]["prompt"]], return_tensors="pt",
                              truncation=True, max_length=64).to("cuda")
                _ = sub(**enc)
                h_sub = cap["sub"][0, -1]
                h_aug = torch.cat([h_sub, torch.ones(1, device="cuda", dtype=h_sub.dtype)])
                proj = (h_aug @ phi).float()
                z_all = sae.encode(proj.unsqueeze(0).to(sae.dtype))[0].float().cpu().numpy()
                fids = probes[pid]["top_k_feature_ids"][:TOP_K]
                z_top32[pid] = z_all[fids]
                z_pid = np.abs((z_top32[pid] - mu_full[pid, :TOP_K]) / sd_full[pid, :TOP_K]).mean()
                per_cat.setdefault(probes[pid]["category"], []).append(float(z_pid))
        per_probe = np.abs((z_top32 - mu_full[:, :TOP_K]) / sd_full[:, :TOP_K]).mean(-1)
        split_scores = {}
        if secret_pids:
            split_scores["joint_z_public"] = float(per_probe[train_pids].mean())
            split_scores["joint_z_secret"] = float(per_probe[secret_pids].mean())
            split_scores["per_probe_max_public"] = float(per_probe[train_pids].max())
            split_scores["per_probe_max_secret"] = float(per_probe[secret_pids].max())
        return {
            "joint_z": float(per_probe.mean()),
            **split_scores,
            "per_probe_max": float(per_probe.max()),
            "per_probe_median": float(np.median(per_probe)),
            "per_cat_mean_z": {c: float(np.mean(v)) for c, v in per_cat.items()},
            "z_top32": z_top32.tolist(),
        }

    def eval_ppx(sub, texts, batch=8):
        sub.eval()
        tot_loss, tot_tok = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                b = texts[i:i+batch]
                enc = sub_tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
                lbls = enc["input_ids"].clone()
                lbls[enc["attention_mask"] == 0] = -100
                out = sub(**enc, labels=lbls)
                n_tok = int((lbls != -100).sum().item()) - len(b)
                if n_tok <= 0: continue
                tot_loss += out.loss.item() * n_tok
                tot_tok += n_tok
        return float(np.exp(tot_loss / max(tot_tok, 1)))

    # ---- White-box trainer (dense MSE + joint-z combined) ----
    def train(sub, phi, lambda_util, alpha_jointz, n_steps,
              probe_batch=8, util_batch=4, lr=3e-4):
        """
        Combined loss to close the SAE-sparsity gradient gap:
          L = MSE(phi(h_sub), h_tgt)            # dense, always has gradient
              + alpha_jointz * |(z-mu)/sigma|   # sparse, targets joint-z directly
              + lambda_util * NLL(Pile)
        The MSE term gets the substitute into the right hidden-state ballpark;
        the joint-z term then fine-tunes for the specific top-k mask.
        """
        opt = torch.optim.AdamW(
            [{"params": [p for p in sub.parameters() if p.requires_grad], "lr": lr},
             {"params": [phi], "lr": lr}],
            weight_decay=0.0,
        )
        hist_mse, hist_jz, hist_util = [], [], []
        sub.train()
        for step in range(n_steps):
            pids = rng.sample(train_pids, probe_batch)
            prompts = [probes[p]["prompt"] for p in pids]
            enc_s = sub_tok(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=64).to("cuda")
            _ = sub(**enc_s)
            h = cap["sub"]
            idx = enc_s["attention_mask"].sum(dim=1) - 1
            h_last = torch.stack([h[k, idx[k]] for k in range(h.shape[0])])
            ones = torch.ones(h_last.shape[0], 1, device="cuda", dtype=h_last.dtype)
            pred_act = (torch.cat([h_last, ones], dim=1) @ phi).float()  # [B, d_tgt]

            # Dense MSE term
            tgt_act_batch = torch.stack([target_acts[p] for p in pids]).to("cuda")  # [B, d_tgt]
            L_mse = F.mse_loss(pred_act, tgt_act_batch)

            # White-box joint-z term (gradient through SAE.encode)
            pids_t = torch.tensor(pids, device="cuda", dtype=torch.long)
            L_jz = jointz_loss(pred_act, pids_t)

            # Utility
            L_util = torch.tensor(0.0, device="cuda")
            if lambda_util > 0:
                ub = rng.sample(pile_train, util_batch)
                enc_u = sub_tok(ub, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to("cuda")
                lbls = enc_u["input_ids"].clone()
                lbls[enc_u["attention_mask"] == 0] = -100
                out = sub(**enc_u, labels=lbls)
                L_util = out.loss

            L = L_mse + alpha_jointz * L_jz + lambda_util * L_util
            opt.zero_grad()
            L.backward()
            opt.step()
            hist_mse.append(float(L_mse.item()))
            hist_jz.append(float(L_jz.item()))
            hist_util.append(float(L_util.item()) if lambda_util > 0 else 0.0)
            if (step+1) % 250 == 0:
                print(f"  [{step+1}/{n_steps}] L_mse={np.mean(hist_mse[-250:]):.4f} "
                      f"L_jz={np.mean(hist_jz[-250:]):.3f} "
                      f"L_util={np.mean(hist_util[-250:]):.3f} "
                      f"elapsed={time.time()-t0:.0f}s")
        return hist_mse, hist_jz, hist_util

    # ---- Baseline (clean substitute, no LoRA, no phi) ----
    print("\n=== Baseline clean sub ppx ===")
    sub_clean = AutoModelForCausalLM.from_pretrained(
        cfg["substitute"], torch_dtype=torch.bfloat16, device_map="cuda"
    )
    sub_clean.eval()
    base_ppx = eval_ppx(sub_clean, pile_eval[:200])
    print(f"  clean ppx = {base_ppx:.2f}")
    del sub_clean; torch.cuda.empty_cache()

    # ---- Sweep over (alpha_jz, lambda_util) ----
    results = {}
    configs = [(a, lu) for a in alpha_jzs for lu in lambda_utils]
    for (ALPHA_JZ, lu) in configs:
        tag = f"alpha_jz_{ALPHA_JZ}_lambda_util_{lu}"
        print(f"\n=== backbone={backbone} alpha_jz={ALPHA_JZ} lambda_util={lu} r={lora_rank} steps={n_steps} ===")
        sub, hh = build_sub(lora_rank, seed=int(lu * 10 + ALPHA_JZ * 3) + seed_offset * 131)
        phi, r2 = fit_phi(sub)
        print(f"  phi heldout R^2 = {r2:.3f}")

        # Pre-train eval
        pre = eval_substitute(sub, phi)
        print(f"  pre-train joint_z={pre['joint_z']:.3f}")

        # Train
        hist_mse, hist_jz, hist_u = train(sub, phi, lambda_util=lu,
                                          alpha_jointz=ALPHA_JZ, n_steps=n_steps)

        # Post-train eval
        post = eval_substitute(sub, phi)
        post["pile_perplexity"] = eval_ppx(sub, pile_eval[:200])
        post["lambda_util"] = lu
        post["alpha_jointz"] = ALPHA_JZ
        post["lora_rank"] = lora_rank
        post["n_steps"] = n_steps
        post["phi_heldout_r2"] = r2
        post["pre_train_joint_z"] = pre["joint_z"]
        post["train_loss_last100_mse"] = float(np.mean(hist_mse[-100:]))
        post["train_loss_last100_jz"] = float(np.mean(hist_jz[-100:]))
        post["train_loss_last100_util"] = float(np.mean(hist_u[-100:]))

        det = "DETECTED" if post["joint_z"] > cfg["tau_real"] else "EVADED"
        print(f"  POST joint_z={post['joint_z']:.3f} ppx={post['pile_perplexity']:.2f} "
              f"({det} vs tau={cfg['tau_real']})")

        results[tag] = post
        hh.remove(); del sub, phi
        torch.cuda.empty_cache()

    return {
        "metadata": {
            "experiment": "E16: white-box joint-z LoRA attack (E-E)",
            "backbone": backbone,
            "target": cfg["target"],
            "substitute": cfg["substitute"],
            "lora_rank": lora_rank,
            "n_steps": n_steps,
            "lambda_utils": lambda_utils,
            "alpha_jzs": alpha_jzs,
            "tau_real": cfg["tau_real"],
            "base_ppx_no_lora_no_phi": base_ppx,
            "gpu": GPU,
            "wall_s": time.time() - t0,
        },
        "results": results,
    }


@app.local_entrypoint()
def main(backbone: str = "qwen3", n_steps: int = 3000, lora_rank: int = 64,
         lambda_utils: str = "0.0,0.1", alpha_jzs: str = "1.0",
         out_tag: str = "", public_split_seed: int = -1, n_public: int = 64,
         seed_offset: int = 0):
    cfg = BACKBONES[backbone]
    lib_path = LOCAL_ROOT / "logs" / cfg["lib"]
    sig_path = LOCAL_ROOT / "logs" / cfg["sigma"]
    if not lib_path.exists():
        raise SystemExit(f"Missing: {lib_path}")
    if not sig_path.exists():
        raise SystemExit(f"Missing: {sig_path}")
    lib = json.loads(lib_path.read_text())
    sig = json.loads(sig_path.read_text())
    lus = [float(x) for x in lambda_utils.split(",")]
    ajs = [float(x) for x in alpha_jzs.split(",")]
    public_pids = None
    if public_split_seed >= 0:
        import random as _r
        probes = lib["probes"]
        ids = list(range(len(probes)))
        _r.Random(public_split_seed).shuffle(ids)
        public_pids = sorted(ids[:n_public])
        print(f"[e-h] public split seed={public_split_seed}: {len(public_pids)}/{len(ids)} pids public")
    result = run.remote(backbone, lib, sig, lus, ajs, n_steps, lora_rank, public_pids, seed_offset)
    suffix = f"_{out_tag}" if out_tag else ""
    out = LOCAL_ROOT / "logs" / f"e16_whitebox_jointz_{backbone}{suffix}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[save] {out}")
    print(f"\n=== Summary (backbone={backbone}) ===")
    md = result["metadata"]
    print(f"  tau_real = {md['tau_real']}")
    print(f"  base ppx (clean, no LoRA) = {md['base_ppx_no_lora_no_phi']:.2f}")
    for k, r in result["results"].items():
        det = "DETECTED" if r["joint_z"] > md["tau_real"] else "EVADED"
        print(f"  {k:<35} joint_z={r['joint_z']:7.3f}  ppx={r['pile_perplexity']:7.2f}  -> {det}")
