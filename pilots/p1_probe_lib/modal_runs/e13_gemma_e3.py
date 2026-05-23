"""E13: cross-family separability on Gemma-2-2B (Phase-2b second-backbone evidence).

Same protocol shape as E3 but on Gemma-2-2B + Gemma-Scope residual SAE @ layer 12.
Attackers fit a public-corpus linear map phi from their own residual into Gemma's
residual space, then commit Gemma-side SAE traces of probe forwards through their
substitute. We ask whether the verifier's joint-Mahalanobis statistic still
separates.

Inputs (from E12 pilot):
  /cache/e12_out/probe_library_gemma2-2b-L12_k96.json
  /cache/e12_out/sigma_calibration_gemma2-2b-L12.json
Outputs:
  logs/e13_gemma_cross_family.json (downloaded)
  /cache/e13_out/e13_gemma_cross_family.json (volume-persistent backup)
"""
import json
import modal
from pathlib import Path

app = modal.App("e13-gemma-e3")
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
        "accelerate>=0.33",
        "zstandard",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

TARGET_MODEL = "google/gemma-2-2b"
SAE_RELEASE  = "gemma-scope-2b-pt-res-canonical"
SAE_ID       = "layer_12/width_16k/canonical"
LAYER        = 12
D_MODEL_TGT  = 2304

ATTACKERS = [
    {"model_id": "google/gemma-2-2b-it",          "layer_frac": 0.50},  # same-family
    {"model_id": "EleutherAI/pythia-1.4b",        "layer_frac": 0.50},  # cross-family d=2048
    {"model_id": "Qwen/Qwen2.5-1.5B",             "layer_frac": 0.50},  # cross-family d=1536
    {"model_id": "microsoft/Phi-3.5-mini-instruct","layer_frac": 0.50}, # cross-family 3.8B
]

TOP_K = 32
N_PILE = 2000
N_POSITIONS = 4

LOCAL_ROOT = Path(__file__).parent.parent
OUT_LOCAL  = LOCAL_ROOT / "logs" / "e13_gemma_cross_family.json"


def _residual_post_hook(model, layer: int):
    """Capture residual stream right after layer block (Gemma-style)."""
    captured = {}
    def hook(_m, _inp, output):
        captured["resid"] = (output[0] if isinstance(output, tuple) else output).detach()
    h = model.model.layers[layer].register_forward_hook(hook)
    return captured, h


def _generic_pre_mlp_hook(model, layer: int):
    """Pre-MLP hook for substitute forwards (matches what attacker would commit)."""
    captured = {}
    def hook(_m, inputs):
        captured["x"] = inputs[0].detach()
    for attr in ("model", "transformer", "gpt_neox"):
        base = getattr(model, attr, None)
        if base is not None:
            layers = getattr(base, "layers", None) or getattr(base, "h", None)
            if layers is not None:
                # use a residual-post hook on the layer for consistency
                def post(_m, _inp, out):
                    captured["x"] = (out[0] if isinstance(out, tuple) else out).detach()
                h = layers[layer].register_forward_hook(post)
                return captured, h
    raise RuntimeError(f"could not find layers on {type(model)}")


@app.function(
    gpu=GPU,
    image=image,
    timeout=2400,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run_attacker(probe_lib: dict, attacker_cfg: dict) -> dict:
    import torch, time, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from sae_lens import SAE

    t0 = time.time()
    hf_id = attacker_cfg["model_id"]
    print(f"[attacker] {hf_id}")

    tok_a = AutoTokenizer.from_pretrained(hf_id)
    if tok_a.pad_token is None: tok_a.pad_token = tok_a.eos_token
    att = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="cuda")
    att.eval()
    n_a = att.config.num_hidden_layers
    d_a = att.config.hidden_size
    layer_a = max(1, int(attacker_cfg["layer_frac"] * n_a))
    print(f"  attacker layer={layer_a}/{n_a}, d_a={d_a}")

    tok_t = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok_t.pad_token is None: tok_t.pad_token = tok_t.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    tgt.eval()
    print(f"  target layer={LAYER}/{tgt.config.num_hidden_layers}, d_t={D_MODEL_TGT}")

    sae_obj = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    sae = sae_obj[0] if isinstance(sae_obj, tuple) else sae_obj
    sae.eval()

    # --- Fit phi on public pile ---
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    texts = []
    for ex in ds:
        if len(ex["text"]) > 50:
            texts.append(ex["text"])
        if len(texts) >= N_PILE: break

    cap_a, h_a = _generic_pre_mlp_hook(att, layer_a)
    cap_t, h_t = _residual_post_hook(tgt, LAYER)

    def fwd(model, cap, tok, prompts):
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            _ = model(**enc)
        h = cap["x"] if "x" in cap else cap["resid"]
        attn = enc["attention_mask"]
        idx = attn.sum(dim=1) - 1
        return torch.stack([h[i, idx[i]] for i in range(h.shape[0])])

    Xa, Xt = [], []
    BS = 16
    for i in range(0, len(texts), BS):
        batch = texts[i:i+BS]
        a = fwd(att, cap_a, tok_a, batch).float().cpu()
        t = fwd(tgt, cap_t, tok_t, batch).float().cpu()
        Xa.append(a); Xt.append(t)
    Xa = torch.cat(Xa); Xt = torch.cat(Xt)
    n = Xa.shape[0]; ntr = int(0.8*n)
    A_tr = torch.cat([Xa[:ntr], torch.ones(ntr,1)], dim=1)
    phi, *_ = torch.linalg.lstsq(A_tr, Xt[:ntr])
    phi = phi.to("cuda").to(torch.bfloat16)

    def r2(Xa_, Xt_):
        A = torch.cat([Xa_, torch.ones(Xa_.shape[0], 1)], dim=1).to("cuda").to(torch.bfloat16)
        p = (A @ phi).float().cpu()
        return 1.0 - ((p-Xt_).norm()**2 / Xt_.norm()**2).item()
    fit_r2_tr = r2(Xa[:ntr], Xt[:ntr])
    fit_r2_ho = r2(Xa[ntr:], Xt[ntr:])
    print(f"  R2 train={fit_r2_tr:.3f} held={fit_r2_ho:.3f}")

    # --- Probe forwards through attacker, project, encode ---
    FILLER = [
        "The weather today is particularly",
        "Climate change affects biodiversity in multiple ways, including",
        "A typical morning routine often includes",
        "The history of cryptography spans several",
    ]
    per_probe = []
    for pidx, p in enumerate(probe_lib["probes"]):
        prompt = p["prompt"]; feat_ids = p["top_k_feature_ids"]
        snaps = []
        for pos in range(N_POSITIONS):
            batch = [FILLER[(pos+k) % 4] for k in range(N_POSITIONS-1)]
            batch.insert(pos, prompt)
            act = fwd(att, cap_a, tok_a, batch)[pos]
            with torch.no_grad():
                ab = torch.cat([act.float(), torch.ones(1, device="cuda")]).to(torch.bfloat16)
                proj = ab @ phi
                z = sae.encode(proj.unsqueeze(0).to(sae.dtype))[0]
            snaps.append(z[feat_ids].float().cpu())
        S = torch.stack(snaps)
        per_probe.append({
            "probe_id": p["probe_id"],
            "category": p["category"],
            "mean": S.mean(dim=0).tolist(),
            "std": S.std(dim=0).tolist(),
        })
        if (pidx+1) % 16 == 0:
            print(f"  [{pidx+1}/96] {time.time()-t0:.0f}s")

    h_a.remove(); h_t.remove()
    return {
        "attacker": hf_id,
        "attacker_layer": layer_a,
        "attacker_d_model": d_a,
        "linear_fit_r2_train": fit_r2_tr,
        "linear_fit_r2_heldout": fit_r2_ho,
        "n_pile_train": ntr,
        "per_probe": per_probe,
        "wall_s": time.time() - t0,
    }


@app.function(image=image, volumes={"/cache": VOL}, timeout=300)
def load_inputs():
    """Read consolidated pilot output from volume."""
    import json as J
    pilot = J.loads(open("/cache/e12_out/e12_gemma_pilot.json").read())
    # E12 packs probes + sigma into one dict; reshape to E3-shaped probe_lib.
    lib = {"metadata": {"n_probes": len(pilot["probes"])}, "probes": pilot["probes"]}
    sig = {"calibration": pilot["sigma_calibration"]}
    return lib, sig


@app.local_entrypoint()
def main():
    print("[load] reading pilot outputs from /cache/e12_out/...")
    lib, sig = load_inputs.remote()
    print(f"  loaded library: {lib['metadata']['n_probes']} probes")

    out = {}
    for cfg in ATTACKERS:
        print(f"\n>>> dispatch {cfg['model_id']}")
        out[cfg["model_id"]] = run_attacker.remote(lib, cfg)

    # Persist locally + volume-side
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOCAL.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {OUT_LOCAL}")
    for k, v in out.items():
        print(f"  {k}: R2_train={v['linear_fit_r2_train']:.3f} R2_held={v['linear_fit_r2_heldout']:.3f} wall={v['wall_s']:.0f}s")
