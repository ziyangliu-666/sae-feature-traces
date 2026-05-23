"""E4 diagnostic v2: reproduce the full pipeline but track LoRA B-norm
across stages + re-score at identical points to isolate root cause of
byte-identical stage A/B outputs.

Runs 200-step stage A + 200-step stage B on only 8 train / 8 held probes.
Prints LoRA B norm before stage A, after stage A, after stage B. Also
re-scores using same weights twice to confirm eval-determinism.

~90s on L4, ~$0.02.
"""
import json
from pathlib import Path
import modal

app = modal.App("e4-diag2")
GPU = "L4"
VOL = modal.Volume.from_name("e3-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers==4.56.2", "sae_lens==6.39.0",
        "datasets==3.1.0", "peft==0.14.0", "numpy<2", "zstandard",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

TARGET_MODEL = "Qwen/Qwen3-1.7B"
SUBSTITUTE_MODEL = "Qwen/Qwen3-0.6B"
TARGET_LAYER = 14
SUB_LAYER = 14
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"

LOCAL_ROOT = Path(__file__).parent.parent
LIB_LOCAL = LOCAL_ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_LOCAL = LOCAL_ROOT / "logs" / "sigma_calibration_qwen3_1.7b_L14.json"


@app.function(gpu=GPU, image=image, timeout=900, volumes={"/cache": VOL},
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])])
def run(lib: dict, sig: dict) -> dict:
    import time, random
    import torch, torch.nn as nn
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset

    torch.manual_seed(0); random.seed(0)
    t0 = time.time()
    sig_by_id = {r["probe_id"]: r for r in sig["calibration"]}

    # split: take first 16 probes, 8/8
    probes = lib["probes"][:16]
    train_ids = list(range(8))
    held_ids  = list(range(8, 16))

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    tgt.eval()
    for p in tgt.parameters(): p.requires_grad = False

    sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{TARGET_LAYER}", device="cuda")
    sae.eval()

    sub_tok = AutoTokenizer.from_pretrained(SUBSTITUTE_MODEL)
    if sub_tok.pad_token is None: sub_tok.pad_token = sub_tok.eos_token
    sub = AutoModelForCausalLM.from_pretrained(SUBSTITUTE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    sub = get_peft_model(sub, lora_cfg)

    cap = {}
    def hook_tgt(_m, inputs): cap["tgt"] = inputs[0].detach()
    def hook_sub(_m, inputs): cap["sub"] = inputs[0]
    tgt.model.layers[TARGET_LAYER].mlp.register_forward_pre_hook(hook_tgt)
    sub.base_model.model.model.layers[SUB_LAYER].mlp.register_forward_pre_hook(hook_sub)

    # ---- proper phi from Pile (exactly like real E4) ----
    print(f"[calib] loading 500 pile samples (smaller for diag)")
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    pile_texts = []
    for ex in ds:
        if len(ex["text"]) > 50: pile_texts.append(ex["text"])
        if len(pile_texts) >= 500: break

    def get_last_act(model, tokenizer, prompts, key):
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
        idx = enc["attention_mask"].sum(1) - 1
        with torch.no_grad():
            _ = model(**enc)
        h = cap[key]
        return torch.stack([h[i, idx[i]] for i in range(h.shape[0])])

    X_s, X_t = [], []
    sub.eval()
    for i in range(0, len(pile_texts), 16):
        b = pile_texts[i:i+16]
        X_s.append(get_last_act(sub, sub_tok, b, "sub").float().cpu())
        X_t.append(get_last_act(tgt, tok, b, "tgt").float().cpu())
    X_s = torch.cat(X_s); X_t = torch.cat(X_t)
    n_tr = int(0.8 * X_s.shape[0])
    A_tr = torch.cat([X_s[:n_tr], torch.ones(n_tr, 1)], dim=1)
    phi, *_ = torch.linalg.lstsq(A_tr, X_t[:n_tr])
    phi = phi.to("cuda").to(torch.bfloat16)
    # check phi magnitude
    phi_max = float(phi.abs().max())
    phi_norm = float(phi.float().norm())
    print(f"[calib] phi max={phi_max:.3f}  norm={phi_norm:.1f}  shape={tuple(phi.shape)}")
    # quick test: does phi produce finite values?
    test_proj = torch.cat([X_s[0].to("cuda").to(torch.bfloat16), torch.ones(1, device="cuda", dtype=torch.bfloat16)]) @ phi
    print(f"[calib] test proj finite={bool(test_proj.isfinite().all())} max={float(test_proj.abs().max()):.3f}")

    def target_act_for(prompt):
        enc = tok([prompt], return_tensors="pt", truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            _ = tgt(**enc)
            h = cap["tgt"][0, -1].float()
        return h.detach()

    def sub_projected_act(prompt):
        enc = sub_tok([prompt], return_tensors="pt", truncation=True, max_length=64).to("cuda")
        _ = sub(**enc)
        h = cap["sub"][0, -1]
        h_aug = torch.cat([h, torch.ones(1, device="cuda", dtype=h.dtype)])
        proj = h_aug @ phi
        return proj.float()

    def sub_projected_z(prompt):
        proj = sub_projected_act(prompt)
        z = sae.encode(proj.unsqueeze(0).to(sae.dtype))[0].float()
        return z

    labels = {pid: target_act_for(probes[pid]["prompt"]) for pid in train_ids}

    opt = torch.optim.AdamW([p for p in sub.parameters() if p.requires_grad], lr=5e-4)

    # LoRA B tracking
    B_params = [(n, p) for n, p in sub.named_parameters() if p.requires_grad and "lora_B" in n]
    def lora_B_total_norm():
        return float(sum(p.detach().float().pow(2).sum() for _, p in B_params) ** 0.5)

    def score_held():
        scores = {}
        sub.eval()
        with torch.no_grad():
            for pid in held_ids:
                z_proj = sub_projected_z(probes[pid]["prompt"])
                fids = probes[pid]["top_k_feature_ids"]
                z_top = z_proj[fids].detach().cpu().numpy()
                sigrec = sig_by_id[pid]
                mu = np.array(sigrec["mean_cross_backend"])
                sd = np.clip(np.array(sigrec["sigma_cross_backend"]), 1e-3, None)
                maha = float(np.abs((z_top - mu) / sd).mean())
                scores[pid] = maha
        return scores

    B0 = lora_B_total_norm()
    scores_init = score_held()
    print(f"\n[stage 0] LoRA B norm={B0:.4f}  init scores: {scores_init}")

    # ---- stage A: 200 steps ----
    sub.train()
    for step in range(200):
        pid = random.choice(train_ids)
        pred = sub_projected_act(probes[pid]["prompt"])
        loss = nn.functional.mse_loss(pred, labels[pid])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 50 == 0:
            print(f"  [A {step+1}/200] loss={loss.item():.4f} B_norm={lora_B_total_norm():.4f}")
    B1 = lora_B_total_norm()
    scores_A = score_held()
    print(f"\n[stage A] LoRA B norm={B1:.4f} (Δ={B1-B0:+.4f})")
    print(f"  stage A scores: {scores_A}")

    # ---- RE-SCORE: same weights, same prompts. Should be identical to scores_A. ----
    scores_A2 = score_held()
    all_same_A = all(abs(scores_A[k] - scores_A2[k]) < 1e-9 for k in scores_A)
    print(f"[determinism check] re-score with same weights matches: {all_same_A}")

    # ---- stage B: 200 more steps ----
    sub.train()
    for step in range(200):
        pid = random.choice(train_ids)
        pred = sub_projected_act(probes[pid]["prompt"])
        loss = nn.functional.mse_loss(pred, labels[pid])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 50 == 0:
            print(f"  [B {step+1}/200] loss={loss.item():.4f} B_norm={lora_B_total_norm():.4f}")
    B2 = lora_B_total_norm()
    scores_B = score_held()
    print(f"\n[stage B] LoRA B norm={B2:.4f} (Δ from A={B2-B1:+.4f})")
    print(f"  stage B scores: {scores_B}")

    # ---- compare scores stage A vs B ----
    all_same_AB = all(abs(scores_A[k] - scores_B[k]) < 1e-9 for k in scores_A)
    diff_scores = {k: scores_B[k] - scores_A[k] for k in scores_A}
    print(f"\n[A vs B] stage A and stage B scores byte-identical: {all_same_AB}")
    print(f"[A vs B] score differences: {diff_scores}")

    return {
        "phi_max": phi_max, "phi_norm": phi_norm,
        "B_norm_init": B0, "B_norm_stageA": B1, "B_norm_stageB": B2,
        "scores_init": scores_init,
        "scores_A": scores_A, "scores_A2": scores_A2, "scores_B": scores_B,
        "all_same_AB": all_same_AB,
        "diff_scores": diff_scores,
        "wall_s": time.time() - t0,
    }


@app.local_entrypoint()
def main():
    lib = json.loads(LIB_LOCAL.read_text())
    sig = json.loads(SIG_LOCAL.read_text())
    r = run.remote(lib, sig)
    out = LOCAL_ROOT / "logs" / "e4_diag2.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\n[save] {out}")
    print(f"B norm init→A→B: {r['B_norm_init']:.4f} → {r['B_norm_stageA']:.4f} → {r['B_norm_stageB']:.4f}")
    print(f"stage A and B scores byte-identical: {r['all_same_AB']}")
