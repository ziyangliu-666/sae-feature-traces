"""E13-9B: cross-family separability on Gemma-2-9B (P0-1 scale-up).

Mirrors e13_gemma_e3.py (2B target) but:
  - Target: google/gemma-2-9b @ layer 20 + width_131k canonical SAE
  - Single cross-family attacker: Qwen/Qwen2.5-7B (scale-matched)
  - Public-corpus phi_map: 2000 Pile samples, 80/20 LSQ split

Reads:
  /cache/e12_9b_out/e12_gemma_9b_pilot.json (from E12-9B)
Writes:
  results/e13_gemma_9b_cross_family.json
  /cache/e13_9b_out/e13_gemma_9b_cross_family.json (volume backup)

Est. wall on A100-40GB: ~1.0-1.5h -> ~$2.1-3.2.
"""
import json
import modal
from pathlib import Path

app = modal.App("e13-gemma-9b-cross-family")
GPU = "A100-40GB"
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

TARGET_MODEL = "google/gemma-2-9b"
SAE_RELEASE  = "gemma-scope-9b-pt-res-canonical"
SAE_ID       = "layer_20/width_131k/canonical"
LAYER        = 20
D_MODEL_TGT  = 3584

ATTACKERS = [
    {"model_id": "Qwen/Qwen2.5-7B", "layer_frac": 0.50},
]

TOP_K = 32
N_PILE = 2000
N_POSITIONS = 4

LOCAL_ROOT = Path(__file__).parent.parent
OUT_LOCAL  = LOCAL_ROOT / "results" / "e13_gemma_9b_cross_family.json"


@app.function(
    gpu=GPU,
    image=image,
    timeout=5400,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run_attacker(probe_lib: dict, attacker_cfg: dict) -> dict:
    """Sequential load: 9B+7B won't fit on A100-40GB simultaneously.
    Phase 1: load TARGET, cache Xt (Pile) and target last-token acts. Unload.
    Phase 2: load ATTACKER, cache Xa (same Pile) + attacker probe acts.
    Phase 3: fit phi, project attacker probe acts, SAE encode.
    """
    import torch, time, gc, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from sae_lens import SAE

    t0 = time.time()
    hf_id = attacker_cfg["model_id"]
    print(f"[attacker] {hf_id}")

    # --- Pile corpus (download once) ---
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    texts = []
    for ex in ds:
        if len(ex["text"]) > 50:
            texts.append(ex["text"])
        if len(texts) >= N_PILE: break
    print(f"[pile] collected {len(texts)} samples")

    FILLER = [
        "The weather today is particularly",
        "Climate change affects biodiversity in multiple ways, including",
        "A typical morning routine often includes",
        "The history of cryptography spans several",
    ]

    # ==================================================================
    # Phase 1: TARGET forwards (Pile only — probe acts not needed from target)
    # ==================================================================
    print(f"\n[phase 1] load target {TARGET_MODEL}")
    t1 = time.time()
    tok_t = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok_t.pad_token is None: tok_t.pad_token = tok_t.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    tgt.eval()
    print(f"  target loaded in {time.time()-t1:.1f}s, layers={tgt.config.num_hidden_layers}, d={D_MODEL_TGT}, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    cap_t = {}
    def hook_t(_m, _inp, out):
        cap_t["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    h_t = tgt.model.layers[LAYER].register_forward_hook(hook_t)

    def fwd_last(model, cap, tok, prompts, bs=8):
        outs = []
        for i in range(0, len(prompts), bs):
            b = prompts[i:i+bs]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
            with torch.no_grad():
                _ = model(**enc)
            h = cap["h"]
            idx = enc["attention_mask"].sum(dim=1) - 1
            outs.append(torch.stack([h[j, idx[j]] for j in range(h.shape[0])]).float().cpu())
        return torch.cat(outs)

    print(f"[phase 1] target fwd on {N_PILE} Pile samples")
    Xt = fwd_last(tgt, cap_t, tok_t, texts, bs=8)
    print(f"  Xt shape={Xt.shape}, wall={time.time()-t1:.0f}s")

    h_t.remove()
    del tgt, tok_t, cap_t
    gc.collect(); torch.cuda.empty_cache()
    print(f"  target unloaded, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ==================================================================
    # Phase 2: ATTACKER forwards (Pile + probe prompts)
    # ==================================================================
    print(f"\n[phase 2] load attacker {hf_id}")
    t2 = time.time()
    tok_a = AutoTokenizer.from_pretrained(hf_id)
    if tok_a.pad_token is None: tok_a.pad_token = tok_a.eos_token
    att = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="cuda")
    att.eval()
    n_a = att.config.num_hidden_layers
    d_a = att.config.hidden_size
    layer_a = max(1, int(attacker_cfg["layer_frac"] * n_a))
    print(f"  attacker loaded in {time.time()-t2:.1f}s, layer={layer_a}/{n_a}, d_a={d_a}, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    cap_a = {}
    def hook_a(_m, _inp, out):
        cap_a["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    h_a = att.model.layers[layer_a].register_forward_hook(hook_a)

    print(f"[phase 2] attacker fwd on {N_PILE} Pile samples")
    Xa = fwd_last(att, cap_a, tok_a, texts, bs=8)
    print(f"  Xa shape={Xa.shape}, wall={time.time()-t2:.0f}s")

    # Attacker probe acts (N_POSITIONS per probe)
    print(f"[phase 2b] attacker probe acts ({len(probe_lib['probes'])} probes x {N_POSITIONS} positions)")
    t2b = time.time()
    probe_acts = {}  # probe_id -> tensor [N_POSITIONS, d_a]
    for pidx, p in enumerate(probe_lib["probes"]):
        prompt = p["prompt"]
        per_pos = []
        for pos in range(N_POSITIONS):
            batch = [FILLER[(pos+k) % 4] for k in range(N_POSITIONS-1)]
            batch.insert(pos, prompt)
            enc = tok_a(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
            with torch.no_grad():
                _ = att(**enc)
            h = cap_a["h"]
            idx = enc["attention_mask"].sum(dim=1) - 1
            per_pos.append(h[pos, idx[pos]].float().cpu())
        probe_acts[p["probe_id"]] = torch.stack(per_pos)
        if (pidx+1) % 32 == 0:
            print(f"  [{pidx+1}/{len(probe_lib['probes'])}] el={time.time()-t2b:.0f}s")

    h_a.remove()
    del att, tok_a, cap_a
    gc.collect(); torch.cuda.empty_cache()
    print(f"  attacker unloaded, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ==================================================================
    # Phase 3: fit phi, load SAE, project probe acts, SAE encode
    # ==================================================================
    print(f"\n[phase 3] fit phi + SAE encode")
    t3 = time.time()

    n = Xa.shape[0]; ntr = int(0.8*n)
    A_tr = torch.cat([Xa[:ntr], torch.ones(ntr,1)], dim=1)
    phi_cpu, *_ = torch.linalg.lstsq(A_tr, Xt[:ntr])
    phi = phi_cpu.to("cuda").to(torch.bfloat16)

    def r2(Xa_, Xt_):
        A = torch.cat([Xa_, torch.ones(Xa_.shape[0], 1)], dim=1).to("cuda").to(torch.bfloat16)
        p = (A @ phi).float().cpu()
        return 1.0 - ((p-Xt_).norm()**2 / Xt_.norm()**2).item()
    fit_r2_tr = r2(Xa[:ntr], Xt[:ntr])
    fit_r2_ho = r2(Xa[ntr:], Xt[ntr:])
    print(f"  R2 train={fit_r2_tr:.3f} held={fit_r2_ho:.3f}")

    sae_obj = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    sae = sae_obj[0] if isinstance(sae_obj, tuple) else sae_obj
    sae.eval()
    print(f"  SAE loaded vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    per_probe = []
    for p in probe_lib["probes"]:
        feat_ids = p["top_k_feature_ids"]
        acts = probe_acts[p["probe_id"]].to("cuda")  # [N_POSITIONS, d_a]
        ones = torch.ones(acts.shape[0], 1, device="cuda", dtype=acts.dtype)
        aug = torch.cat([acts, ones], dim=1).to(torch.bfloat16)   # [N_POSITIONS, d_a+1]
        proj = aug @ phi  # [N_POSITIONS, D_MODEL_TGT]
        with torch.no_grad():
            z_full = sae.encode(proj.to(sae.dtype))  # [N_POSITIONS, d_sae]
        snaps = z_full[:, feat_ids].float().cpu()  # [N_POSITIONS, TOP_K]
        per_probe.append({
            "probe_id": p["probe_id"],
            "category": p["category"],
            "mean": snaps.mean(dim=0).tolist(),
            "std": snaps.std(dim=0).tolist(),
        })

    print(f"[phase 3] done in {time.time()-t3:.0f}s")

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
    import json as J
    pilot = J.loads(open("/cache/e12_9b_out/e12_gemma_9b_pilot.json").read())
    lib = {"metadata": {"n_probes": len(pilot["probes"])}, "probes": pilot["probes"]}
    sig = {"calibration": pilot["sigma_calibration"]}
    return lib, sig


@app.local_entrypoint()
def main():
    print("[load] reading pilot outputs from /cache/e12_9b_out/...")
    lib, sig = load_inputs.remote()
    print(f"  loaded library: {lib['metadata']['n_probes']} probes")

    out = {}
    for cfg in ATTACKERS:
        print(f"\n>>> dispatch {cfg['model_id']}")
        out[cfg["model_id"]] = run_attacker.remote(lib, cfg)

    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOCAL.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {OUT_LOCAL}")
    for k, v in out.items():
        print(f"  {k}: R2_train={v['linear_fit_r2_train']:.3f} R2_held={v['linear_fit_r2_heldout']:.3f} wall={v['wall_s']:.0f}s")
