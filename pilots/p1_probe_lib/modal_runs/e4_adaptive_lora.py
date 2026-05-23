"""E4: category-aware adaptive LoRA attack (C3).

The attacker knows the top-k feature ids per probe (from the public library)
and fine-tunes a LoRA adapter on Qwen3-0.6B so that, after projecting via a
linear map φ learned on public corpus, the substitute's layer-14 activation
encodes the *same* top-k SAE features as the honest Qwen3-1.7B would.

2-stage attack:
  - Stage A (weak):       N=500 labels on mixed-category probes
  - Stage B (robust-mix): N=2500 labels, with robust categories over-sampled
                          (ioi/induction/coref per P3b finding)

After each stage we score per-category detection rate at k=32 on a held-out
half of the library.

Budget: ~1 GPU-h on Modal L4 (~$0.80).

Output: logs/e4_adaptive_lora.json
"""
import json
from pathlib import Path
import modal

app = modal.App("e4-adaptive-lora")
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
    timeout=3600,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(lib: dict, sig: dict) -> dict:
    import time, random
    import torch
    import torch.nn as nn
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset

    torch.manual_seed(0); random.seed(0)
    t0 = time.time()

    probes = lib["probes"]
    sig_by_id = {r["probe_id"]: r for r in sig["calibration"]}

    # ---- split probes 50/50 by category ----
    by_cat = {}
    for i, p in enumerate(probes):
        by_cat.setdefault(p["category"], []).append(i)
    train_ids, held_ids = [], []
    rng = random.Random(0)
    for c, ids in by_cat.items():
        rng.shuffle(ids)
        h = len(ids) // 2
        train_ids += ids[:h]; held_ids += ids[h:]
    train_ids.sort(); held_ids.sort()
    print(f"[split] train={len(train_ids)} held={len(held_ids)}")

    # ---- load target + SAE (frozen) ----
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    tgt.eval()
    for p in tgt.parameters(): p.requires_grad = False
    print(f"[load] target in {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{TARGET_LAYER}", device="cuda")
    sae.eval()

    # capture hooks
    cap = {}
    def hook_tgt(_m, inputs): cap["tgt"] = inputs[0].detach()
    tgt.model.layers[TARGET_LAYER].mlp.register_forward_pre_hook(hook_tgt)

    # ---- load sub + LoRA ----
    sub_tok = AutoTokenizer.from_pretrained(SUBSTITUTE_MODEL)
    if sub_tok.pad_token is None: sub_tok.pad_token = sub_tok.eos_token
    sub = AutoModelForCausalLM.from_pretrained(SUBSTITUTE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    sub = get_peft_model(sub, lora_cfg)
    sub.print_trainable_parameters()

    def hook_sub(_m, inputs): cap["sub"] = inputs[0]  # NO detach (needs grad)
    sub.base_model.model.model.layers[SUB_LAYER].mlp.register_forward_pre_hook(hook_sub)

    # ---- linear map φ: h_sub[1024] → h_tgt[2048] from public corpus (no probes) ----
    print("[calib] loading 1000 pile samples")
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    pile_texts = []
    for ex in ds:
        if len(ex["text"]) > 50:
            pile_texts.append(ex["text"])
        if len(pile_texts) >= 1000:
            break

    def get_last_act(model, tokenizer, prompts, key):
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
        attn = enc["attention_mask"]
        idx = attn.sum(dim=1) - 1
        with torch.no_grad():
            _ = model(**enc)
        h = cap[key]
        return torch.stack([h[i, idx[i]] for i in range(h.shape[0])])

    X_s, X_t = [], []
    sub.eval()
    for i in range(0, len(pile_texts), 16):
        batch = pile_texts[i:i+16]
        s = get_last_act(sub, sub_tok, batch, "sub").float().cpu()
        t = get_last_act(tgt, tok, batch, "tgt").float().cpu()
        X_s.append(s); X_t.append(t)
    X_s = torch.cat(X_s); X_t = torch.cat(X_t)
    n_tr = int(0.8 * X_s.shape[0])
    A_tr = torch.cat([X_s[:n_tr], torch.ones(n_tr, 1)], dim=1)
    phi, *_ = torch.linalg.lstsq(A_tr, X_t[:n_tr])
    phi = phi.to("cuda").to(torch.bfloat16)  # [d_sub+1, d_tgt]
    # eval held-out R²
    A_ho = torch.cat([X_s[n_tr:], torch.ones(X_s.shape[0]-n_tr, 1)], dim=1).to("cuda").to(torch.bfloat16)
    pred = (A_ho @ phi).float().cpu()
    r2_ho = 1.0 - ((pred - X_t[n_tr:]).norm()**2 / X_t[n_tr:].norm()**2).item()
    print(f"[calib] φ held-out R² = {r2_ho:.3f}  paired_n={X_s.shape[0]}")

    # ---- training labels: pre-compute target raw MLP-input vector per probe.
    # Rationale: matching the RAW activation is differentiable; matching SAE
    # top-k magnitudes directly has zero gradient on inactive features
    # (transcoder uses sparse gating). Matching raw implies matching top-k
    # downstream after the frozen SAE.
    def target_act_for(prompt):
        enc = tok([prompt], return_tensors="pt", truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            _ = tgt(**enc)
            h = cap["tgt"][0, -1].float()   # [d_tgt], last token
        return h.detach()

    labels = {}
    for pid in train_ids:
        labels[pid] = target_act_for(probes[pid]["prompt"])
    print(f"[label] {len(labels)} target raw-activation vectors computed")

    # ---- project sub's activation into target space, keep gradients ----
    def sub_projected_act(prompt):
        enc = sub_tok([prompt], return_tensors="pt", truncation=True, max_length=64).to("cuda")
        _ = sub(**enc)
        h = cap["sub"][0, -1]  # [d_sub], retains grad
        h_aug = torch.cat([h, torch.ones(1, device="cuda", dtype=h.dtype)])
        proj = h_aug @ phi     # [d_tgt], float bf16 → cast for loss
        return proj.float()

    def sub_projected_z(prompt):
        """Eval-only: project + SAE encode (no grad needed here)."""
        proj = sub_projected_act(prompt)
        z = sae.encode(proj.unsqueeze(0).to(sae.dtype))[0].float()
        return z

    opt = torch.optim.AdamW([p for p in sub.parameters() if p.requires_grad], lr=5e-4)

    def score_held():
        """Compute per-category mean maha on the held-out probes."""
        by_cat_maha = {}
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
                by_cat_maha.setdefault(probes[pid]["category"], []).append(maha)
        return {c: float(np.median(v)) for c, v in by_cat_maha.items()}

    # ---- stage A: weak, 500 steps uniform sampling over training probes ----
    print("\n[stage A] weak, 500 steps")
    sub.train()
    tr_pool = list(train_ids)
    loss_hist = []
    for step in range(500):
        pid = rng.choice(tr_pool)
        pred_act = sub_projected_act(probes[pid]["prompt"])
        target_act = labels[pid]
        loss = nn.functional.mse_loss(pred_act, target_act)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_hist.append(float(loss.item()))
        if (step + 1) % 100 == 0:
            print(f"  [{step+1}/500] loss={np.mean(loss_hist[-100:]):.3f} elapsed={time.time()-t0:.0f}s")
    stageA_scores = score_held()
    print(f"[stage A] per-cat med maha: {stageA_scores}")

    # ---- stage B: robust-mix oversampling ----
    print("\n[stage B] robust-mix × 2500 steps (ioi/induction/coref oversampled 3×)")
    robust_cats = {"ioi", "induction", "coref"}
    weighted_pool = []
    for pid in train_ids:
        w = 3 if probes[pid]["category"] in robust_cats else 1
        weighted_pool += [pid] * w
    sub.train()
    loss_hist = []
    for step in range(2500):
        pid = rng.choice(weighted_pool)
        pred_act = sub_projected_act(probes[pid]["prompt"])
        target_act = labels[pid]
        loss = nn.functional.mse_loss(pred_act, target_act)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_hist.append(float(loss.item()))
        if (step + 1) % 500 == 0:
            print(f"  [{step+1}/2500] loss={np.mean(loss_hist[-500:]):.3f} elapsed={time.time()-t0:.0f}s")
    stageB_scores = score_held()
    print(f"[stage B] per-cat med maha: {stageB_scores}")

    return {
        "phi_heldout_r2": r2_ho,
        "stageA_per_cat_med_maha": stageA_scores,
        "stageB_per_cat_med_maha": stageB_scores,
        "train_probes": len(train_ids),
        "held_probes": len(held_ids),
        "wall_s": time.time() - t0,
    }


@app.local_entrypoint()
def main():
    lib = json.loads(LIB_LOCAL.read_text())
    sig = json.loads(SIG_LOCAL.read_text())
    result = run.remote(lib, sig)
    out = LOCAL_ROOT / "logs" / "e4_adaptive_lora.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[save] {out}")
    print(f"held-out φ R² = {result['phi_heldout_r2']:.3f}")
    print(f"Stage A per-cat med maha: {result['stageA_per_cat_med_maha']}")
    print(f"Stage B per-cat med maha: {result['stageB_per_cat_med_maha']}")
    print(f"wall = {result['wall_s']:.0f}s")
