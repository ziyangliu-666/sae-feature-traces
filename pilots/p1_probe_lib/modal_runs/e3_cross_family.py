"""E3: cross-family separability under public-corpus calibration.

The attacker runs a different-family model M' (Llama / Gemma / Phi),
trains a linear map φ: h_{M'}(x) → h_M(x) on a PUBLIC English corpus
(no access to our probe prompts), then forwards probe prompts through M',
applies φ, encodes via Qwen3 SAE, and commits the resulting trace.

We measure whether our scorer can distinguish those committed traces from
the honest Qwen3-1.7B trace.

Claim tested:  C2 (cross-family detection under public-corpus calibration).

Runs on Modal L4 (24 GB, ~$0.80/hr). Estimated wall-clock per attacker:
  - model load + warm:         ~60 s
  - fit linear map on 2k pile: ~90 s
  - 96 probe forwards × 4 pos: ~60 s
  - total per attacker:         ~4 min
Four attackers × ~4 min ≈ 16 min = $0.22.

Inputs (mounted from local):
  logs/probe_library_qwen3_1.7b_L14_k96.json    (E1 output)
  logs/sigma_calibration_qwen3_1.7b_L14.json    (E7 output)

Outputs (downloaded back):
  logs/e3_cross_family_results.json
"""
import modal
import json
from pathlib import Path

app = modal.App("e3-cross-family")

# Cheap GPU that fits 3B bf16.
GPU = "L4"
VOL = modal.Volume.from_name("e3-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.56.2",
        "sae_lens==6.39.0",
        "datasets==3.1.0",
        "numpy<2",
        "scikit-learn==1.5.2",
        "zstandard",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

# Attacker families. Matched-depth layer ≈ 50% of target depth.
ATTACKERS = [
    # All fully open (no gated access). Matched-depth layer ≈ 50% of target.
    {"model_id": "Qwen/Qwen2.5-1.5B",                "layer_frac": 0.50},  # diff-version same-family
    {"model_id": "microsoft/Phi-3.5-mini-instruct",  "layer_frac": 0.50},  # cross-family 3.8B
    {"model_id": "allenai/OLMo-2-1124-7B",           "layer_frac": 0.50},  # fully-open 7B
    {"model_id": "Qwen/Qwen3-0.6B",                  "layer_frac": 0.50},  # same-family control
]

TARGET_MODEL = "Qwen/Qwen3-1.7B"
TARGET_LAYER = 14                    # Qwen3-1.7B: 28 layers → 50% = layer 14
TARGET_DMODEL = 2048
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"
TOP_K = 32
N_PILE = 2000      # tokens used to fit linear map
N_POSITIONS = 4    # batch position repeats per probe

LOCAL_ROOT = Path(__file__).parent.parent
LIB_LOCAL = LOCAL_ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"


def _extract_target_hook(model, layer: int):
    captured = {}
    def hook(_m, inputs):
        captured["mlp_in"] = inputs[0].detach()
    h = model.model.layers[layer].mlp.register_forward_pre_hook(hook)
    return captured, h


def _extract_generic_hook(model, layer: int):
    """Try common HF decoder container names to grab layer-level pre-hook on mlp."""
    captured = {}
    def hook(_m, inputs):
        captured["mlp_in"] = inputs[0].detach()
    # Gemma / Llama / Qwen all expose model.layers[i].mlp
    for attr in ("model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None:
            layers = getattr(base, "layers", None) or getattr(base, "h", None)
            if layers is not None:
                h = layers[layer].mlp.register_forward_pre_hook(hook)
                return captured, h
    raise RuntimeError(f"could not find layers on {type(model)}")


@app.function(
    gpu=GPU,
    image=image,
    timeout=1800,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run_attacker(probe_lib: dict, attacker_cfg: dict) -> dict:
    """Load attacker M', fit φ on public pile, forward probes, return SAE feature activations."""
    import torch, time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from sae_lens import SAE
    import numpy as np

    t0 = time.time()
    hf_id = attacker_cfg["model_id"]
    print(f"[attacker] {hf_id}")

    tok_att = AutoTokenizer.from_pretrained(hf_id)
    if tok_att.pad_token is None:
        tok_att.pad_token = tok_att.eos_token
    att = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="cuda")
    att.eval()
    n_layers_att = att.config.num_hidden_layers
    d_model_att = att.config.hidden_size
    layer_att = max(1, int(attacker_cfg["layer_frac"] * n_layers_att))
    print(f"  attacker layer={layer_att}/{n_layers_att}, d_model={d_model_att}")

    tok_tgt = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok_tgt.pad_token is None:
        tok_tgt.pad_token = tok_tgt.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    tgt.eval()
    print(f"  target   layer={TARGET_LAYER}/{tgt.config.num_hidden_layers}, d_model={TARGET_DMODEL}")
    print(f"  vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

    sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{TARGET_LAYER}", device="cuda")
    sae.eval()

    # --- Fit linear map φ on public pile (N_PILE texts, 80/20 train/held-out) ---
    print("[fit] loading pile samples")
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    texts = []
    for ex in ds:
        t = ex["text"]
        if len(t) > 50:
            texts.append(t)
        if len(texts) >= N_PILE:
            break
    print(f"  {len(texts)} pile snippets")

    cap_att, h_att = _extract_generic_hook(att, layer_att)
    cap_tgt, h_tgt = _extract_generic_hook(tgt, TARGET_LAYER)

    def batch_forward(model, hook_cap, tokenizer, prompts):
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            _ = model(**enc)
        h = hook_cap["mlp_in"]
        attn = enc["attention_mask"]
        idx = attn.sum(dim=1) - 1
        return torch.stack([h[i, idx[i]] for i in range(h.shape[0])])

    X_att, X_tgt = [], []
    BS = 16
    for i in range(0, len(texts), BS):
        batch = texts[i:i+BS]
        a = batch_forward(att, cap_att, tok_att, batch).float().cpu()
        t = batch_forward(tgt, cap_tgt, tok_tgt, batch).float().cpu()
        X_att.append(a); X_tgt.append(t)
    X_att = torch.cat(X_att); X_tgt = torch.cat(X_tgt)
    n = X_att.shape[0]
    n_tr = int(0.8 * n)
    X_att_tr, X_att_ho = X_att[:n_tr], X_att[n_tr:]
    X_tgt_tr, X_tgt_ho = X_tgt[:n_tr], X_tgt[n_tr:]
    print(f"  paired activations: train {X_att_tr.shape[0]} / held-out {X_att_ho.shape[0]}")

    A_tr = torch.cat([X_att_tr, torch.ones(X_att_tr.shape[0], 1)], dim=1)
    phi, *_ = torch.linalg.lstsq(A_tr, X_tgt_tr)
    phi = phi.to("cuda").to(torch.bfloat16)

    def r2(X_a, X_t):
        A_ = torch.cat([X_a, torch.ones(X_a.shape[0], 1)], dim=1).to("cuda").to(torch.bfloat16)
        p = (A_ @ phi).float().cpu()
        return 1.0 - ((p - X_t).norm()**2 / X_t.norm()**2).item()
    fit_r2 = r2(X_att_tr, X_tgt_tr)
    fit_r2_ho = r2(X_att_ho, X_tgt_ho)
    print(f"  linear map R²: train={fit_r2:.3f}  held-out={fit_r2_ho:.3f}")

    # --- Forward probes through attacker, apply φ, encode via SAE ---
    print("[forward] 96 probes × 4 positions")
    FILLER = [
        "The weather today is particularly",
        "Climate change affects biodiversity in multiple ways, including",
        "A typical morning routine often includes",
        "The history of cryptography spans several",
    ]
    per_probe = []
    for pidx, p in enumerate(probe_lib["probes"]):
        prompt = p["prompt"]
        feat_ids = p["top_k_feature_ids"]
        snaps = []
        for pos in range(N_POSITIONS):
            batch = [FILLER[(pos + k) % 4] for k in range(N_POSITIONS - 1)]
            batch.insert(pos, prompt)
            act = batch_forward(att, cap_att, tok_att, batch)[pos]   # d_att
            with torch.no_grad():
                act_b = torch.cat([act.float(), torch.ones(1, device="cuda")]).to(torch.bfloat16)
                proj = act_b @ phi   # d_tgt
                z = sae.encode(proj.unsqueeze(0).to(sae.dtype))[0]
            snaps.append(z[feat_ids].float().cpu())
        S = torch.stack(snaps)
        per_probe.append({
            "probe_id": p["probe_id"],
            "category": p["category"],
            "mean": S.mean(dim=0).tolist(),
            "std": S.std(dim=0).tolist(),
        })
        if (pidx + 1) % 16 == 0:
            print(f"  [{pidx+1}/96] elapsed {time.time()-t0:.0f}s")

    h_att.remove(); h_tgt.remove()
    print(f"[done] attacker {hf_id}: {time.time()-t0:.0f}s")
    return {
        "attacker": hf_id,
        "attacker_layer": layer_att,
        "attacker_d_model": d_model_att,
        "linear_fit_r2_train": fit_r2,
        "linear_fit_r2_heldout": fit_r2_ho,
        "n_pile_train": int(X_att_tr.shape[0]),
        "n_pile_heldout": int(X_att_ho.shape[0]),
        "per_probe": per_probe,
        "wall_s": time.time() - t0,
    }


@app.local_entrypoint()
def main():
    if not LIB_LOCAL.exists():
        raise SystemExit(f"E1 library not found: {LIB_LOCAL}")
    probe_lib = json.loads(LIB_LOCAL.read_text())
    print(f"loaded library: {probe_lib['metadata']['n_probes']} probes")

    # fan out one Modal job per attacker
    out = {}
    for cfg in ATTACKERS:
        print(f"\n>>> dispatching {cfg['model_id']}")
        out[cfg["model_id"]] = run_attacker.remote(probe_lib, cfg)

    out_path = LOCAL_ROOT / "logs" / "e3_cross_family_results_v2.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}")
    for k, v in out.items():
        print(f"  {k}: R²_train={v['linear_fit_r2_train']:.3f}  R²_held={v['linear_fit_r2_heldout']:.3f}  wall={v['wall_s']:.0f}s")
