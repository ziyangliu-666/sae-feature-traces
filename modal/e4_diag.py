"""E4 diagnostic: why didn't LoRA train in the full run.

Minimal fork of e4_adaptive_lora.py that tracks:
  (1) pred_act.requires_grad after sub_projected_act
  (2) LoRA B-matrix grad norm after loss.backward()
  (3) LoRA B-matrix weight norm before/after opt.step()
  (4) dtype of trainable params

If (2) shows zero grad → broken grad flow through hook.
If (3) shows no change → optimizer issue.
If (1) is False → detach bug.

Runs 50 steps of stage A only. ~60s on L4, ~$0.02.
"""
import json
from pathlib import Path
import modal

app = modal.App("e4-diag")
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

LOCAL_ROOT = Path(__file__).parent.parent
LIB_LOCAL = LOCAL_ROOT / "results" / "probe_library_qwen3_1.7b_L14_k96.json"


@app.function(gpu=GPU, image=image, timeout=900, volumes={"/cache": VOL},
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])])
def run(lib: dict) -> dict:
    import time, random
    import torch
    import torch.nn as nn
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0); random.seed(0)
    t0 = time.time()
    probes = lib["probes"][:8]   # only need a few probes for diagnosis

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    tgt.eval()
    for p in tgt.parameters(): p.requires_grad = False

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

    # dtype inventory
    dtypes = {}
    for n, p in sub.named_parameters():
        if p.requires_grad:
            key = str(p.dtype)
            dtypes[key] = dtypes.get(key, 0) + p.numel()
    print(f"[dtype] trainable param dtypes: {dtypes}")

    # find one LoRA B matrix for tracking
    target_B_name, target_B_param = None, None
    for n, p in sub.named_parameters():
        if p.requires_grad and "lora_B" in n:
            target_B_name = n; target_B_param = p
            break
    print(f"[track] monitoring: {target_B_name} shape={tuple(target_B_param.shape)}")

    cap = {}
    def hook_tgt(_m, inputs): cap["tgt"] = inputs[0].detach()
    def hook_sub(_m, inputs): cap["sub"] = inputs[0]
    tgt.model.layers[TARGET_LAYER].mlp.register_forward_pre_hook(hook_tgt)
    sub.base_model.model.model.layers[SUB_LAYER].mlp.register_forward_pre_hook(hook_sub)

    # simple phi: sub→tgt dim via random projection (just to get any signal)
    d_sub = sub.base_model.model.config.hidden_size
    d_tgt = tgt.config.hidden_size
    torch.manual_seed(1)
    phi = torch.randn(d_sub + 1, d_tgt, device="cuda", dtype=torch.bfloat16) * 0.01

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

    labels = {i: target_act_for(probes[i]["prompt"]) for i in range(len(probes))}
    opt = torch.optim.AdamW([p for p in sub.parameters() if p.requires_grad], lr=5e-4)

    sub.train()
    print("\n[train] 50 diagnostic steps")
    log = []
    for step in range(50):
        pid = step % len(probes)
        pred_act = sub_projected_act(probes[pid]["prompt"])

        rec = {"step": step,
               "pred_requires_grad": bool(pred_act.requires_grad),
               "pred_has_gradfn": pred_act.grad_fn is not None}

        target_act = labels[pid]
        loss = nn.functional.mse_loss(pred_act, target_act)

        B_before = float(target_B_param.detach().float().norm())
        opt.zero_grad()
        loss.backward()
        B_grad_norm = float(target_B_param.grad.detach().float().norm()) if target_B_param.grad is not None else -1.0

        # collect overall grad norm across all trainable
        total_g2 = 0.0; have_grad = 0; total_params = 0
        for p in sub.parameters():
            if p.requires_grad:
                total_params += 1
                if p.grad is not None:
                    have_grad += 1
                    total_g2 += float(p.grad.detach().float().pow(2).sum())
        total_grad_norm = float(total_g2 ** 0.5)

        opt.step()
        B_after = float(target_B_param.detach().float().norm())

        rec.update({
            "loss": float(loss.item()),
            "B_before": B_before, "B_after": B_after,
            "B_delta": B_after - B_before,
            "B_grad_norm": B_grad_norm,
            "total_grad_norm": total_grad_norm,
            "have_grad_frac": have_grad / max(1, total_params),
        })
        log.append(rec)

        if step < 5 or step % 10 == 0:
            print(f"  [{step:>2}] loss={rec['loss']:.4f} "
                  f"pred_grad={rec['pred_requires_grad']} "
                  f"B_grad={B_grad_norm:.2e} "
                  f"totalG={total_grad_norm:.2e} "
                  f"B_delta={rec['B_delta']:.2e} "
                  f"have_grad={have_grad}/{total_params}")

    return {"log": log, "target_B_name": target_B_name,
            "trainable_dtypes": dtypes,
            "wall_s": time.time() - t0}


@app.local_entrypoint()
def main():
    lib = json.loads(LIB_LOCAL.read_text())
    r = run.remote(lib)
    out = LOCAL_ROOT / "results" / "e4_diag.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\n[save] {out}")
    # summary
    log = r["log"]
    print(f"loss first/last: {log[0]['loss']:.4f} / {log[-1]['loss']:.4f}")
    print(f"B_grad_norm first/last: {log[0]['B_grad_norm']:.2e} / {log[-1]['B_grad_norm']:.2e}")
    print(f"B_delta cumulative: {sum(l['B_delta'] for l in log):.3e}")
    print(f"pred_requires_grad consistently: {all(l['pred_requires_grad'] for l in log)}")
