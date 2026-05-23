"""E4 diag3: is the bug sub.eval() disabling LoRA?

Score held probes in both eval-mode and train-mode (with torch.no_grad).
If scores differ: PEFT-eval-mode is the bug.
"""
import json
from pathlib import Path
import modal

app = modal.App("e4-diag3")
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


@app.function(gpu=GPU, image=image, timeout=600, volumes={"/cache": VOL},
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])])
def run(lib: dict) -> dict:
    import time, random
    import torch, torch.nn as nn
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0); random.seed(0)
    probes = lib["probes"][:8]

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    tgt.eval()
    for p in tgt.parameters(): p.requires_grad = False

    sub_tok = AutoTokenizer.from_pretrained(SUBSTITUTE_MODEL)
    if sub_tok.pad_token is None: sub_tok.pad_token = sub_tok.eos_token
    sub = AutoModelForCausalLM.from_pretrained(SUBSTITUTE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    sub = get_peft_model(sub, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    ))

    cap = {}
    def hook_tgt(_m, inputs): cap["tgt"] = inputs[0].detach()
    def hook_sub(_m, inputs): cap["sub"] = inputs[0]
    tgt.model.layers[TARGET_LAYER].mlp.register_forward_pre_hook(hook_tgt)
    sub.base_model.model.model.layers[SUB_LAYER].mlp.register_forward_pre_hook(hook_sub)

    d_tgt = tgt.config.hidden_size
    d_sub = sub.base_model.model.config.hidden_size
    torch.manual_seed(1)
    phi = (torch.randn(d_sub + 1, d_tgt, device="cuda", dtype=torch.bfloat16) * 0.01)

    def target_act(prompt):
        enc = tok([prompt], return_tensors="pt", truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            _ = tgt(**enc)
            return cap["tgt"][0, -1].float().detach()

    def sub_proj(prompt):
        enc = sub_tok([prompt], return_tensors="pt", truncation=True, max_length=64).to("cuda")
        _ = sub(**enc)
        h = cap["sub"][0, -1]
        h_aug = torch.cat([h, torch.ones(1, device="cuda", dtype=h.dtype)])
        return (h_aug @ phi).float()

    labels = {i: target_act(probes[i]["prompt"]) for i in range(4)}
    opt = torch.optim.AdamW([p for p in sub.parameters() if p.requires_grad], lr=5e-4)

    B_params = [(n, p) for n, p in sub.named_parameters() if p.requires_grad and "lora_B" in n]
    def B_norm(): return float(sum(p.detach().float().pow(2).sum() for _, p in B_params) ** 0.5)

    def capture_eval(mode):
        """Run forward + capture cap['sub'][0,-1] for 4 held prompts."""
        if mode == "eval": sub.eval()
        elif mode == "train_nograd": sub.train()
        out = {}
        with torch.no_grad():
            for i in range(4, 8):
                _ = sub(**sub_tok([probes[i]["prompt"]], return_tensors="pt", truncation=True, max_length=64).to("cuda"))
                h = cap["sub"][0, -1].float().detach().cpu()
                out[i] = float(h.norm()), float(h.abs().max())
        return out

    print(f"[init] B_norm={B_norm():.4f}")
    cap_init_eval  = capture_eval("eval")
    cap_init_train = capture_eval("train_nograd")
    print(f"[init eval]  cap norms: {cap_init_eval}")
    print(f"[init train] cap norms: {cap_init_train}")

    # train 200 steps
    sub.train()
    for step in range(200):
        pid = random.choice(range(4))
        pred = sub_proj(probes[pid]["prompt"])
        loss = nn.functional.mse_loss(pred, labels[pid])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"\n[after 200 steps] B_norm={B_norm():.4f}")

    cap_trained_eval  = capture_eval("eval")
    cap_trained_train = capture_eval("train_nograd")
    print(f"[trained eval]  cap norms: {cap_trained_eval}")
    print(f"[trained train] cap norms: {cap_trained_train}")

    # bit-diff check
    eval_changed = any(cap_init_eval[i] != cap_trained_eval[i] for i in [4,5,6,7])
    train_changed = any(cap_init_train[i] != cap_trained_train[i] for i in [4,5,6,7])
    modes_match = all(cap_trained_eval[i] == cap_trained_train[i] for i in [4,5,6,7])

    print(f"\n[ verdict ]")
    print(f"  eval-mode cap['sub'] changed after training? {eval_changed}")
    print(f"  train-mode cap['sub'] changed after training? {train_changed}")
    print(f"  eval vs train-mode produce same output? {modes_match}")

    return {
        "init_eval": cap_init_eval, "init_train": cap_init_train,
        "trained_eval": cap_trained_eval, "trained_train": cap_trained_train,
        "eval_changed": eval_changed, "train_changed": train_changed,
        "modes_match": modes_match,
        "B_norm_after": B_norm(),
    }


@app.local_entrypoint()
def main():
    lib = json.loads(LIB_LOCAL.read_text())
    r = run.remote(lib)
    out = LOCAL_ROOT / "results" / "e4_diag3.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\n[save] {out}")
    print(f"eval_changed={r['eval_changed']}  train_changed={r['train_changed']}  modes_match={r['modes_match']}")
