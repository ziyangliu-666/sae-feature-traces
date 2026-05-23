"""E12: Phase-2a Gemma pilot — E2-equivalent separability on a second backbone.

Consolidated Modal L4 run:
  (1) Build a probe library on Gemma-2-2B + Gemma-Scope residual SAE @ layer 12
      using the 96 prompts shared with Qwen3 (reuse from E1 library).
  (2) Calibrate sigma across 2 dtypes x 2 kernels x 2 positions (8 configs/probe).
  (3) Run E2-style separability with one same-family and one cross-family
      substitute plus 3 lifts (NATIVE_IT, ZERO_PAD_PYTHIA, RAND_PROJ_PYTHIA,
      LEARNED_LIN_PYTHIA).

Gate (plan):
  honest joint-Mahalanobis 99th pct  <= 2.0
  TPR_joint @ FPR 1%                  >= 0.9 on ALL 4 attackers

Est. wall: ~30-45 min on L4 -> ~$0.6.
"""
import json
import modal
from pathlib import Path

app = modal.App("e12-gemma-pilot")
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
        "accelerate>=0.33",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

# Backbone
TARGET_MODEL   = "google/gemma-2-2b"
SAE_RELEASE    = "gemma-scope-2b-pt-res-canonical"
SAE_ID         = "layer_12/width_16k/canonical"
LAYER          = 12
D_MODEL_TARGET = 2304

# Attackers
SAMEFAM_MODEL  = "google/gemma-2-2b-it"     # same d_model, different weights
CROSSFAM_MODEL = "EleutherAI/pythia-1.4b"   # d_model=2048 -> lift to 2304

TOP_K = 32
N_ATTACK_RUNS = 20
N_REPEAT_LIB = 30
BATCH_COMPANIONS = 3

LOCAL_ROOT = Path(__file__).parent.parent
LIB_K96_QWEN = LOCAL_ROOT / "results" / "probe_library_qwen3_1.7b_L14_k96.json"
OUT_JSON = LOCAL_ROOT / "results" / "e12_gemma_pilot.json"


@app.function(
    gpu=GPU,
    image=image,
    timeout=5400,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(shared_prompts: list) -> dict:
    import os, time, random
    import torch
    import numpy as np
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

    random.seed(0); torch.manual_seed(0)

    FILLER = [
        "The weather today is particularly",
        "Climate change affects biodiversity in multiple ways, including",
        "A typical morning routine often includes",
        "The history of cryptography spans several",
        "Modern cities face challenges such as",
        "Healthy cooking often involves fresh",
        "Space exploration has yielded numerous",
        "Online privacy has become a pressing",
        "Machine learning research has advanced rapidly and",
        "Economic policy debates often center on",
    ]
    CALIB = [
        "Many modern cities suffer from chronic air pollution,",
        "The advent of the printing press revolutionized communication by",
        "Traditional Japanese cuisine places strong emphasis on seasonal",
        "When studying economics, one must carefully distinguish between",
        "Recent advances in renewable energy technology suggest that",
        "The phenomenon of consciousness remains one of the most",
        "International trade agreements have historically shaped the trajectory of",
        "During the Industrial Revolution, urbanization increased dramatically as",
        "Bioinformatics combines tools from biology, statistics, and computer science to",
        "Philosophers since Kant have debated whether morality is ultimately grounded in",
        "The formation of planetary systems from protoplanetary disks involves",
        "Quantum entanglement, one of the most counterintuitive predictions of quantum mechanics,",
        "Early research on neural networks in the 1980s laid the foundation for",
        "The legal concept of intellectual property includes several distinct categories, such as",
        "A balanced diet typically contains proportional amounts of carbohydrates, proteins, and",
        "In classical music theory, the distinction between major and minor keys",
    ]

    # ------------------------------------------------------------------
    # Load target + SAE
    # ------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    target.eval()
    target_layers = target.config.num_hidden_layers
    print(f"[load] target Gemma-2-2B in {time.time()-t0:.1f}s, layers={target_layers}, d={target.config.hidden_size}, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
    assert target.config.hidden_size == D_MODEL_TARGET

    t1 = time.time()
    sae, sae_cfg, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    sae.eval()
    print(f"[load] Gemma-Scope SAE layer_{LAYER} width_16k in {time.time()-t1:.1f}s, d_sae={sae.cfg.d_sae}")

    captured = {}

    def hook_target(_m, _inputs, outputs):
        # Gemma-2 residual post: outputs is a tuple (hidden_states, ...)
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        captured["tgt"] = h.detach()

    h_tgt = target.model.layers[LAYER].register_forward_hook(hook_target)

    def forward_target(prompt, pos, companions, dtype, kernel):
        batch = companions.copy(); batch.insert(pos, prompt)
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        last = enc["attention_mask"].sum(dim=1) - 1
        with sdpa_kernel([kernel, SDPBackend.MATH]), torch.no_grad():
            if dtype == torch.bfloat16:
                _ = target(**enc)
            else:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = target(**enc)
        return captured["tgt"][pos, last[pos]]

    def encode_topk(act):
        with torch.no_grad():
            z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
        vals, idx = torch.topk(z.abs(), k=TOP_K)
        return set(idx.tolist()), z[idx].float().cpu(), z.float()

    # ------------------------------------------------------------------
    # Phase 1: build probe library on Gemma (reuse 96 prompts from Qwen3 E1)
    # ------------------------------------------------------------------
    print(f"\n[phase 1] build probe library on Gemma, {len(shared_prompts)} probes x {N_REPEAT_LIB} repeats")
    t2 = time.time()
    probes = []
    for pidx, (cat, prompt) in enumerate(shared_prompts):
        latents = []
        for rep in range(N_REPEAT_LIB):
            companions = random.sample(FILLER, BATCH_COMPANIONS)
            batch = [prompt] + companions
            random.shuffle(batch)
            tp = batch.index(prompt)
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
            last = enc["attention_mask"].sum(dim=1) - 1
            with torch.no_grad():
                _ = target(**enc)
            act = captured["tgt"][tp, last[tp]]
            with torch.no_grad():
                z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
            latents.append(z.float().cpu())
        L = torch.stack(latents)
        mean = L.mean(dim=0); std = L.std(dim=0)
        topk_vals, topk_idx = torch.topk(mean.abs(), k=TOP_K)
        probes.append({
            "probe_id": pidx, "category": cat, "prompt": prompt,
            "top_k_feature_ids": topk_idx.tolist(),
            "top_k_means": mean[topk_idx].tolist(),
            "top_k_stds": std[topk_idx].tolist(),
        })
        if (pidx+1) % 16 == 0:
            el = time.time() - t2
            eta = el / (pidx+1) * (len(shared_prompts) - pidx - 1)
            print(f"  [{pidx+1:2d}/{len(shared_prompts)}] top1={topk_idx[0].item()} mag={mean[topk_idx[0]]:.2f} (elapsed {el:.0f}s eta {eta:.0f}s)")

    print(f"[phase 1] library built in {time.time()-t2:.0f}s")

    # ------------------------------------------------------------------
    # Phase 2: sigma calibration across honest backends (8 configs/probe)
    # ------------------------------------------------------------------
    PRECISIONS = [torch.bfloat16, torch.float16]
    KERNELS = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    POSITIONS = [0, 2]
    print(f"\n[phase 2] sigma calibration, {len(probes)} x {len(PRECISIONS)*len(KERNELS)*len(POSITIONS)} cfgs")
    t3 = time.time()
    sig_by_probe = {}
    for p in probes:
        snaps = []
        feat_ids = p["top_k_feature_ids"]
        for dtype in PRECISIONS:
            for kernel in KERNELS:
                for pos in POSITIONS:
                    companions = random.sample(FILLER, 3)
                    act = forward_target(p["prompt"], pos, companions, dtype, kernel)
                    with torch.no_grad():
                        z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
                    snaps.append(z[feat_ids].float().cpu())
        S = torch.stack(snaps)
        sig_by_probe[p["probe_id"]] = {
            "mean_cross_backend": S.mean(dim=0).tolist(),
            "sigma_cross_backend": S.std(dim=0).tolist(),
        }
    print(f"[phase 2] sigma done in {time.time()-t3:.0f}s")

    # ------------------------------------------------------------------
    # Phase 3: E2-equivalent separability
    # ------------------------------------------------------------------
    # Load both substitutes
    print(f"\n[phase 3] load substitutes: {SAMEFAM_MODEL}, {CROSSFAM_MODEL}")
    same_tok = AutoTokenizer.from_pretrained(SAMEFAM_MODEL)
    if same_tok.pad_token is None: same_tok.pad_token = same_tok.eos_token
    t4 = time.time()
    samefam = AutoModelForCausalLM.from_pretrained(SAMEFAM_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    samefam.eval()
    samefam_layer = int(round(LAYER / target_layers * samefam.config.num_hidden_layers))
    print(f"  {SAMEFAM_MODEL}: layers={samefam.config.num_hidden_layers}, d={samefam.config.hidden_size}, probe_layer={samefam_layer}  ({time.time()-t4:.0f}s)")

    t5 = time.time()
    cross_tok = AutoTokenizer.from_pretrained(CROSSFAM_MODEL)
    if cross_tok.pad_token is None: cross_tok.pad_token = cross_tok.eos_token
    cross = AutoModelForCausalLM.from_pretrained(CROSSFAM_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    cross.eval()
    cross_layers = cross.config.num_hidden_layers
    cross_d = cross.config.hidden_size
    cross_layer = int(round(LAYER / target_layers * cross_layers))
    print(f"  {CROSSFAM_MODEL}: layers={cross_layers}, d={cross_d}, probe_layer={cross_layer}  ({time.time()-t5:.0f}s)")

    def hook_samefam(_m, _inputs, outputs):
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        captured["same"] = h.detach()

    def hook_cross(_m, _inputs, outputs):
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        captured["cross"] = h.detach()

    h_same = samefam.model.layers[samefam_layer].register_forward_hook(hook_samefam)
    # Pythia: GPT-NeoX style, has `gpt_neox.layers`
    if hasattr(cross, "gpt_neox"):
        h_cross = cross.gpt_neox.layers[cross_layer].register_forward_hook(hook_cross)
    else:
        h_cross = cross.model.layers[cross_layer].register_forward_hook(hook_cross)

    def forward_samefam(prompt, pos, companions):
        batch = companions.copy(); batch.insert(pos, prompt)
        enc = same_tok(batch, return_tensors="pt", padding=True).to("cuda")
        last = enc["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            _ = samefam(**enc)
        return captured["same"][pos, last[pos]]

    def forward_cross(prompt, pos, companions):
        batch = companions.copy(); batch.insert(pos, prompt)
        enc = cross_tok(batch, return_tensors="pt", padding=True).to("cuda")
        last = enc["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            _ = cross(**enc)
        return captured["cross"][pos, last[pos]]

    # Fit LEARNED_LIN on public corpus: Pythia -> Gemma(d=2304)
    print("  fit LEARNED_LIN Pythia -> Gemma")
    tgt_acts, sub_acts = [], []
    for s in CALIB:
        at = forward_target(s, 0, random.sample(FILLER, 3), torch.bfloat16, SDPBackend.MATH)
        asu = forward_cross(s, 0, random.sample(FILLER, 3))
        tgt_acts.append(at.float().cpu()); sub_acts.append(asu.float().cpu())
    X_tgt = torch.stack(tgt_acts)                  # [N, 2304]
    X_sub = torch.stack(sub_acts)                  # [N, 2048]
    A = torch.cat([X_sub, torch.ones(X_sub.shape[0], 1)], dim=1)  # [N, 2049]
    phi, *_ = torch.linalg.lstsq(A, X_tgt)          # [2049, 2304]
    phi = phi.to("cuda").to(torch.bfloat16)
    pred = (A.to("cuda").to(torch.bfloat16) @ phi).float().cpu()
    r2_lin = 1.0 - ((pred - X_tgt).norm()**2 / X_tgt.norm()**2).item()
    print(f"  linear R2 (Pythia->Gemma, {len(CALIB)} prompts): {r2_lin:.3f}")

    def score_maha(zs_committed):
        vals = []
        for p, z in zip(probes, zs_committed):
            ids = p["top_k_feature_ids"]
            mu = torch.tensor(sig_by_probe[p["probe_id"]]["mean_cross_backend"])
            sd = torch.tensor(sig_by_probe[p["probe_id"]]["sigma_cross_backend"]).clamp(min=1e-3)
            z_at = z[ids].float().cpu()
            vals.append(float(((z_at - mu) / sd).abs().mean().item()))
        return float(np.mean(vals))

    def score_overlap(committed):
        return float(np.mean([
            len(ids & set(p["top_k_feature_ids"])) / TOP_K
            for p, (ids, _) in zip(probes, committed)
        ]))

    # Honest runs
    print(f"\n[phase 3a] honest runs: {len(PRECISIONS)*len(KERNELS)*len(POSITIONS)}")
    honest = []
    for dtype in PRECISIONS:
        for kernel in KERNELS:
            for pos in POSITIONS:
                companions = random.sample(FILLER, 3)
                committed, zs = [], []
                for p in probes:
                    act = forward_target(p["prompt"], pos, companions, dtype, kernel)
                    ids, _, z = encode_topk(act)
                    committed.append((ids, _)); zs.append(z)
                ov = score_overlap(committed); mh = score_maha(zs)
                honest.append({"overlap": ov, "maha": mh,
                               "dtype": str(dtype).split(".")[-1],
                               "kernel": kernel.name, "pos": pos})

    tau_ov = float(np.quantile([r["overlap"] for r in honest], 0.01))
    tau_mh = float(np.quantile([r["maha"] for r in honest], 0.99))
    print(f"  tau_overlap={tau_ov:.4f} tau_maha={tau_mh:.3f}")

    # Attacker runs
    def attacker_commit(prompt, pos, companions, mode):
        if mode == "NATIVE_IT":
            act = forward_samefam(prompt, pos, companions)
            act_tgt = act.to(torch.bfloat16)
        else:
            act_sub = forward_cross(prompt, pos, companions).float()
            if mode == "ZERO_PAD_PYTHIA":
                act_tgt = torch.zeros(D_MODEL_TARGET, device="cuda", dtype=torch.bfloat16)
                act_tgt[:cross_d] = act_sub.to(torch.bfloat16)
            elif mode == "RAND_PROJ_PYTHIA":
                W = torch.randn(D_MODEL_TARGET, cross_d, device="cuda") / (cross_d ** 0.5)
                act_tgt = (W @ act_sub).to(torch.bfloat16)
            elif mode == "LEARNED_LIN_PYTHIA":
                aug = torch.cat([act_sub, torch.ones(1, device="cuda")])
                act_tgt = (aug.to(torch.bfloat16) @ phi)
        return encode_topk(act_tgt)

    ATTACKS = ["NATIVE_IT", "ZERO_PAD_PYTHIA", "RAND_PROJ_PYTHIA", "LEARNED_LIN_PYTHIA"]
    print(f"\n[phase 3b] attacker runs x {N_ATTACK_RUNS} per mode ({len(ATTACKS)} modes)")
    attack_runs = {m: [] for m in ATTACKS}
    for mode in ATTACKS:
        t_a = time.time()
        for ri in range(N_ATTACK_RUNS):
            pos = POSITIONS[ri % len(POSITIONS)]
            random.seed(100 + ri)
            companions = random.sample(FILLER, 3)
            committed, zs = [], []
            for p in probes:
                ids, _, z = attacker_commit(p["prompt"], pos, companions, mode)
                committed.append((ids, _)); zs.append(z)
            ov = score_overlap(committed); mh = score_maha(zs)
            attack_runs[mode].append({"overlap": ov, "maha": mh, "run": ri})
        print(f"  [{mode}] 20 runs in {time.time()-t_a:.0f}s")

    h_tgt.remove(); h_same.remove(); h_cross.remove()

    # Classifier
    classifier = {}
    for mode, runs in attack_runs.items():
        ov = np.array([r["overlap"] for r in runs])
        mh = np.array([r["maha"] for r in runs])
        tpr_ov = float((ov < tau_ov).mean())
        tpr_mh = float((mh > tau_mh).mean())
        tpr_joint = float(((ov < tau_ov) | (mh > tau_mh)).mean())
        classifier[mode] = {
            "overlap_mean": float(ov.mean()), "maha_mean": float(mh.mean()),
            "tpr_overlap": tpr_ov, "tpr_maha": tpr_mh, "tpr_joint": tpr_joint,
        }

    gate_pass = all(v["tpr_joint"] >= 0.9 for v in classifier.values()) and tau_mh <= 2.0

    print("\n=== Summary ===")
    for m, v in classifier.items():
        print(f"  {m:<22} ov={v['overlap_mean']:.4f} mh={v['maha_mean']:.2f} TPR_joint={v['tpr_joint']:.2f}")
    print(f"  tau_maha={tau_mh:.3f}  gate_pass={gate_pass}")

    out = {
        "metadata": {
            "experiment": "E12 (Gemma pilot, Phase 2a)",
            "target": TARGET_MODEL, "samefam_sub": SAMEFAM_MODEL, "crossfam_sub": CROSSFAM_MODEL,
            "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
            "layer": LAYER, "d_model": D_MODEL_TARGET,
            "d_sae": int(sae.cfg.d_sae), "top_k": TOP_K,
            "n_probes": len(probes), "n_attack_per_mode": N_ATTACK_RUNS,
            "linear_fit_r2": r2_lin,
            "wall_s": time.time() - t0,
            "gpu": GPU,
        },
        "thresholds": {"tau_overlap": tau_ov, "tau_maha": tau_mh},
        "probes": probes,
        "sigma_calibration": sig_by_probe,
        "honest": honest,
        "attacks": attack_runs,
        "classifier": classifier,
        "gate": {
            "passed": gate_pass,
            "criteria": "all 4 attackers TPR_joint>=0.9 and tau_mh<=2.0",
        },
    }

    import os
    os.makedirs("/cache/e12_out", exist_ok=True)
    with open("/cache/e12_out/e12_gemma_pilot.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[persist] /cache/e12_out/e12_gemma_pilot.json ({os.path.getsize('/cache/e12_out/e12_gemma_pilot.json')} bytes)")
    return out


@app.local_entrypoint()
def main():
    if not LIB_K96_QWEN.exists():
        raise SystemExit(f"Missing Qwen3 library for prompts: {LIB_K96_QWEN}")
    lib = json.loads(LIB_K96_QWEN.read_text())
    shared = [(p["category"], p["prompt"]) for p in lib["probes"]]
    print(f"reusing {len(shared)} probe prompts from Qwen3 E1 library")
    out = run.remote(shared)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {OUT_JSON}")
    print(f"wall: {out['metadata']['wall_s']:.0f}s, gate_pass={out['gate']['passed']}")
