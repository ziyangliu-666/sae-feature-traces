"""E23: per-head ablation on Qwen3-1.7B layer 14.

Reviewer Q2 (post-rewrite): "Can you provide any per-head or per-MLP
ablations (even small-scale) that more directly link the low-rank
attention-pattern collapse to specific components or attention geometry?"

For each attention head h in layer 14, we zero head h's output at
that layer's o_proj-input (i.e., before the output projection),
recompute SAE features, and measure the per-class joint-z shift
vs the clean forward.

Causal claim sought: surface circuits (factual/syntax/lang) should
be relatively insensitive to any single attention head, while
attention-pattern circuits (induction/IOI/coref) should be carried
by a small subset of identifiable heads. If we see (head, class)
effect concentration at attention-pattern classes and ~uniform
absence at surface classes, that is causal evidence for the
capacity argument in §4.3.

Qwen3-1.7B: 16 heads × 8 (head_dim=128) at each layer. Layer 14
is the published probe layer. We ablate one head at a time. ~5min
on L4 (96 probes × 16 ablations × 1 forward each = 1536 forwards).

Output: logs/e23_perhead_ablation.json
"""
import json
import modal
from pathlib import Path

app = modal.App("e23-perhead-ablation")
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
N_HEADS = 16
HEAD_DIM = 128
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"
TOP_K = 32

LOCAL_ROOT = Path(__file__).parent.parent
LIB_LOCAL = LOCAL_ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_LOCAL = LOCAL_ROOT / "logs" / "sigma_calibration_qwen3_1.7b_L14.json"


@app.function(gpu=GPU, image=image, timeout=3600, volumes={"/cache": VOL},
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])])
def run(lib: dict, sig: dict) -> dict:
    import time
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    t0 = time.time()
    probes = lib["probes"]
    sig_by_pid = {r["probe_id"]: r for r in sig["calibration"]}

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    sae = SAE.from_pretrained(
        release=TRANSCODER_RELEASE, sae_id=f"layer_{TARGET_LAYER}", device="cuda")
    sae.eval()

    target_layer = model.model.layers[TARGET_LAYER]
    attn = target_layer.self_attn
    mlp = target_layer.mlp

    # Hook to capture mlp input (residual after attention)
    cap = {"x": None}
    def mlp_pre_hook(_m, inputs):
        cap["x"] = inputs[0]
    mlp_h = mlp.register_forward_pre_hook(mlp_pre_hook)

    # Attention output ablation hook: zero head h before o_proj.
    # In Qwen3 (HF), attention computes context = attn_output then
    # o_proj(context). We need to zero a head's slice of the
    # pre-o_proj context. The cleanest path: hook o_proj.forward to
    # intercept its input and zero head h.
    o_proj = attn.o_proj
    ablate_head = {"h": None}

    def o_proj_pre_hook(_m, inputs):
        x = inputs[0]  # (B, T, n_heads*head_dim)
        h = ablate_head["h"]
        if h is None:
            return None
        x = x.clone()
        x[..., h*HEAD_DIM:(h+1)*HEAD_DIM] = 0
        return (x,)
    o_h = o_proj.register_forward_pre_hook(o_proj_pre_hook)

    def get_top32(prompt: str) -> tuple[np.ndarray, np.ndarray]:
        enc = tok([prompt], return_tensors="pt", truncation=True,
                  max_length=64).to("cuda")
        with torch.no_grad():
            model(**enc)
            x = cap["x"]  # mlp input
            z = sae.encode(x.to(sae.dtype))[0, -1].float().cpu()
        vals, ids = torch.topk(z, TOP_K)
        return ids.numpy(), vals.numpy()

    # ---- Clean baseline: top-32 features per probe ----
    print("[clean] forward all probes")
    ablate_head["h"] = None
    clean_feats = {}
    for p in probes:
        ids, vals = get_top32(p["prompt"])
        clean_feats[p["probe_id"]] = (ids, vals)
    print(f"  clean done in {time.time()-t0:.0f}s")

    # ---- Per-head ablation ----
    classes = sorted({p["category"] for p in probes})
    head_class_effect: dict[int, dict[str, float]] = {}
    head_class_min_z: dict[int, dict[str, float]] = {}

    for h in range(N_HEADS):
        ablate_head["h"] = h
        per_probe_z = {}
        for p in probes:
            ids, vals = get_top32(p["prompt"])
            pid = p["probe_id"]
            cal = sig_by_pid[pid]
            ref_ids = np.asarray(cal["feature_ids"])
            ref_mu = np.asarray(cal["mean_cross_backend"])
            ref_sig = np.asarray(cal["sigma_cross_backend"])
            ablated_top32_vals = vals
            ablated_top32_ids = ids
            id_to_val = dict(zip(ablated_top32_ids.tolist(), ablated_top32_vals.tolist()))
            v_hat = np.array([id_to_val.get(int(fid), 0.0) for fid in ref_ids])
            z_i = float(np.mean(np.abs(v_hat - ref_mu) / np.maximum(ref_sig, 1e-6)))
            per_probe_z[pid] = (p["category"], z_i)
        # Per-class shift: relative joint-z increase vs clean
        # First compute clean per-probe z:
        per_probe_clean_z = {}
        for p in probes:
            pid = p["probe_id"]
            ids, vals = clean_feats[pid]
            cal = sig_by_pid[pid]
            ref_ids = np.asarray(cal["feature_ids"])
            ref_mu = np.asarray(cal["mean_cross_backend"])
            ref_sig = np.asarray(cal["sigma_cross_backend"])
            id_to_val = dict(zip(ids.tolist(), vals.tolist()))
            v_hat = np.array([id_to_val.get(int(fid), 0.0) for fid in ref_ids])
            z_clean = float(np.mean(np.abs(v_hat - ref_mu) / np.maximum(ref_sig, 1e-6)))
            per_probe_clean_z[pid] = (p["category"], z_clean)

        by_cls_delta = {}
        by_cls_clean = {}
        for pid, (c, z_abl) in per_probe_z.items():
            _, z_clean = per_probe_clean_z[pid]
            by_cls_delta.setdefault(c, []).append(z_abl - z_clean)
            by_cls_clean.setdefault(c, []).append(z_clean)
        head_class_effect[h] = {c: float(np.mean(vs)) for c, vs in by_cls_delta.items()}
        head_class_min_z[h] = {c: float(np.mean(vs)) for c, vs in by_cls_clean.items()}
        if h == 0:
            # sanity: clean per-class means
            print(f"  clean per-cls z: {head_class_min_z[h]}")
        print(f"  head {h:2d}: delta_z per class = "
              f"{ {k: round(v, 3) for k,v in head_class_effect[h].items()} }  "
              f"elapsed={time.time()-t0:.0f}s")

    mlp_h.remove(); o_h.remove()

    return {
        "metadata": {
            "target": TARGET_MODEL,
            "layer": TARGET_LAYER,
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "top_k": TOP_K,
            "n_probes": len(probes),
            "classes": classes,
            "wall_s": time.time() - t0,
        },
        "clean_per_class_mean_z": head_class_min_z[0],  # same across heads
        "head_class_delta_z": head_class_effect,
    }


@app.local_entrypoint()
def main():
    lib = json.loads(LIB_LOCAL.read_text())
    sig = json.loads(SIG_LOCAL.read_text())
    r = run.remote(lib, sig)
    out = LOCAL_ROOT / "logs" / "e23_perhead_ablation.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\n[save] {out}")
    print(f"\n=== Per-head class deltas (positive = ablation hurts probe) ===")
    classes = r["metadata"]["classes"]
    print(f"head  " + "  ".join(f"{c:>10}" for c in classes))
    for h, eff in sorted(r["head_class_delta_z"].items(), key=lambda x: int(x[0])):
        print(f"{int(h):>3}   " + "  ".join(f"{eff[c]:>10.3f}" for c in classes))
