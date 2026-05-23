"""E2 on Modal L4: same-family separability on committed SAE traces (C1).

Runs honest Qwen3-1.7B vs Qwen3-0.6B substitute with 3 lift attacks:
  ZERO_PAD, RAND_PROJ, LEARNED_LIN (fit on public corpus).

Emits honest / attack runs, thresholds at FPR=1%, reports TPR per lift.
Estimated wall-clock ≈ 8-12 min on L4 → ~$0.15.
"""
import json
import modal
from pathlib import Path

app = modal.App("e2-same-family")
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
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

TARGET_MODEL = "Qwen/Qwen3-1.7B"
SUBSTITUTE_MODEL = "Qwen/Qwen3-0.6B"
TARGET_LAYER = 14
TOP_K = 32
N_ATTACK_RUNS = 20
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"

LOCAL_ROOT = Path(__file__).parent.parent
LIB_LOCAL = LOCAL_ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_LOCAL = LOCAL_ROOT / "logs" / "sigma_calibration_qwen3_1.7b_L14.json"
OUT_LOCAL = LOCAL_ROOT / "logs" / "e2_separability.json"


@app.function(
    gpu=GPU,
    image=image,
    timeout=1800,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(lib: dict, sig: dict) -> dict:
    import time, random
    import torch
    import numpy as np
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    random.seed(0); torch.manual_seed(0)

    probes = lib["probes"]
    sig_by_probe = {r["probe_id"]: r for r in sig["calibration"]}
    print(f"[load] {len(probes)} probes")

    FILLER = [
        "The weather today is particularly",
        "Climate change affects biodiversity in multiple ways, including",
        "A typical morning routine often includes",
        "The history of cryptography spans several",
        "Modern cities face challenges such as",
        "Healthy cooking often involves fresh",
        "Space exploration has yielded numerous",
        "Online privacy has become a pressing",
    ]

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    target.eval()
    print(f"[load] target in {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    sub_tok = AutoTokenizer.from_pretrained(SUBSTITUTE_MODEL)
    if sub_tok.pad_token is None: sub_tok.pad_token = sub_tok.eos_token
    t1 = time.time()
    sub = AutoModelForCausalLM.from_pretrained(SUBSTITUTE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    sub.eval()
    sub_layers = sub.config.num_hidden_layers
    sub_dim = sub.config.hidden_size
    sub_layer = int(round(TARGET_LAYER / target.config.num_hidden_layers * sub_layers))
    print(f"[load] substitute in {time.time()-t1:.1f}s, {sub_layers} layers, d={sub_dim}, probe_layer={sub_layer}, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{TARGET_LAYER}", device="cuda")
    sae.eval()

    captured = {}
    def hook_tgt(_m, inputs):
        captured["tgt"] = inputs[0].detach()
    def hook_sub(_m, inputs):
        captured["sub"] = inputs[0].detach()
    h_tgt = target.model.layers[TARGET_LAYER].mlp.register_forward_pre_hook(hook_tgt)
    h_sub = sub.model.layers[sub_layer].mlp.register_forward_pre_hook(hook_sub)

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

    def forward_sub(prompt, pos, companions):
        batch = companions.copy(); batch.insert(pos, prompt)
        enc = sub_tok(batch, return_tensors="pt", padding=True).to("cuda")
        last = enc["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            _ = sub(**enc)
        return captured["sub"][pos, last[pos]]

    def commit_and_fullz(act):
        with torch.no_grad():
            z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
        vals, idx = torch.topk(z.abs(), k=TOP_K)
        return set(idx.tolist()), z[idx].float().cpu(), z.float()

    def score_overlap(committed):
        return float(np.mean([
            len(ids & set(p["top_k_feature_ids"])) / TOP_K
            for p, (ids, _) in zip(probes, committed)
        ]))

    def score_maha(zs):
        vals = []
        for p, z in zip(probes, zs):
            ids = p["top_k_feature_ids"]
            mu = torch.tensor(sig_by_probe[p["probe_id"]]["mean_cross_backend"])
            sd = torch.tensor(sig_by_probe[p["probe_id"]]["sigma_cross_backend"]).clamp(min=1e-3)
            z_at = z[ids].float().cpu()
            vals.append(float(((z_at - mu) / sd).abs().mean().item()))
        return float(np.mean(vals))

    # Phase 1: honest runs across 2 dtypes × 2 kernels × 4 positions = 16 configs
    PRECISIONS = [torch.bfloat16, torch.float16]
    KERNELS = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    POSITIONS = [0, 1, 2, 3]
    print(f"\n[phase 1] honest runs × {len(PRECISIONS)*len(KERNELS)*len(POSITIONS)}")
    honest = []
    t2 = time.time()
    for dtype in PRECISIONS:
        for kernel in KERNELS:
            for pos in POSITIONS:
                companions = random.sample(FILLER, 3)
                committed, zs = [], []
                for p in probes:
                    act = forward_target(p["prompt"], pos, companions, dtype, kernel)
                    ids, _, z = commit_and_fullz(act)
                    committed.append((ids, _)); zs.append(z)
                ov = score_overlap(committed); mh = score_maha(zs)
                honest.append({"overlap": ov, "maha": mh,
                               "dtype": str(dtype).split(".")[-1], "kernel": kernel.name, "pos": pos})
                if len(honest) % 4 == 0:
                    print(f"  [{len(honest):2d}/16] overlap={ov:.4f} maha={mh:.3f} (elapsed {time.time()-t2:.0f}s)")

    # Phase 2: fit linear map on public corpus (NOT probes)
    print(f"\n[phase 2] fit LEARNED_LIN on public corpus")
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
    tgt_acts, sub_acts = [], []
    for s in CALIB:
        at = forward_target(s, 0, random.sample(FILLER, 3), torch.bfloat16, SDPBackend.MATH)
        asu = forward_sub(s, 0, random.sample(FILLER, 3))
        tgt_acts.append(at.float().cpu()); sub_acts.append(asu.float().cpu())
    X_tgt = torch.stack(tgt_acts)
    X_sub = torch.stack(sub_acts)
    A = torch.cat([X_sub, torch.ones(X_sub.shape[0], 1)], dim=1)
    phi, *_ = torch.linalg.lstsq(A, X_tgt)
    phi = phi.to("cuda").to(torch.bfloat16)
    pred = (A.to("cuda").to(torch.bfloat16) @ phi).float().cpu()
    r2 = 1.0 - ((pred - X_tgt).norm()**2 / X_tgt.norm()**2).item()
    print(f"  linear R² on public corpus: {r2:.3f}")

    def attacker_commit(prompt, pos, companions, lift):
        act_sub = forward_sub(prompt, pos, companions).float()
        if lift == "ZERO_PAD":
            act_tgt = torch.zeros(2048, device="cuda", dtype=torch.bfloat16)
            act_tgt[:sub_dim] = act_sub.to(torch.bfloat16)
        elif lift == "RAND_PROJ":
            W = torch.randn(2048, sub_dim, device="cuda") / (sub_dim ** 0.5)
            act_tgt = (W @ act_sub).to(torch.bfloat16)
        elif lift == "LEARNED_LIN":
            aug = torch.cat([act_sub, torch.ones(1, device="cuda")])
            act_tgt = (aug.to(torch.bfloat16) @ phi)
        return commit_and_fullz(act_tgt)

    print(f"\n[phase 3] attacker runs × {N_ATTACK_RUNS} per lift")
    attack_runs = {"ZERO_PAD": [], "RAND_PROJ": [], "LEARNED_LIN": []}
    for lift in attack_runs:
        t3 = time.time()
        for ri in range(N_ATTACK_RUNS):
            pos = POSITIONS[ri % len(POSITIONS)]
            random.seed(100 + ri)
            companions = random.sample(FILLER, 3)
            committed, zs = [], []
            for p in probes:
                ids, _, z = attacker_commit(p["prompt"], pos, companions, lift)
                committed.append((ids, _)); zs.append(z)
            ov = score_overlap(committed); mh = score_maha(zs)
            attack_runs[lift].append({"overlap": ov, "maha": mh, "run": ri})
        print(f"  [{lift}] 20 runs in {time.time()-t3:.0f}s")

    h_tgt.remove(); h_sub.remove()

    # Phase 4: classifier
    ho = np.array([r["overlap"] for r in honest])
    hm = np.array([r["maha"] for r in honest])
    tau_ov = float(np.quantile(ho, 0.01))
    tau_mh = float(np.quantile(hm, 0.99))
    print(f"\n[phase 4] τ_overlap={tau_ov:.4f} (1% FPR), τ_maha={tau_mh:.3f}")
    print(f"  honest overlap: min={ho.min():.4f} med={np.median(ho):.4f} max={ho.max():.4f}")
    print(f"  honest maha:    min={hm.min():.3f} med={np.median(hm):.3f} max={hm.max():.3f}")

    results = {}
    for lift, runs in attack_runs.items():
        ov = np.array([r["overlap"] for r in runs])
        mh = np.array([r["maha"] for r in runs])
        tpr_ov = float((ov < tau_ov).mean())
        tpr_mh = float((mh > tau_mh).mean())
        tpr_joint = float(((ov < tau_ov) | (mh > tau_mh)).mean())
        results[lift] = {
            "overlap_mean": float(ov.mean()), "overlap_std": float(ov.std()),
            "maha_mean": float(mh.mean()), "maha_std": float(mh.std()),
            "tpr_overlap": tpr_ov, "tpr_maha": tpr_mh, "tpr_joint": tpr_joint,
            "n_runs": len(runs),
        }
        print(f"  {lift:<14} ov={ov.mean():.4f}±{ov.std():.4f} mh={mh.mean():.2f}±{mh.std():.2f} TPR_ov={tpr_ov:.2f} TPR_mh={tpr_mh:.2f} TPR_joint={tpr_joint:.2f}")

    return {
        "metadata": {
            "experiment": "E2",
            "target": TARGET_MODEL,
            "substitute": SUBSTITUTE_MODEL,
            "layer": TARGET_LAYER,
            "n_probes": len(probes),
            "top_k": TOP_K,
            "n_honest": len(honest),
            "n_attack_per_lift": N_ATTACK_RUNS,
            "linear_fit_r2": r2,
            "wall_s": time.time() - t0,
            "gpu": GPU,
        },
        "thresholds": {"tau_overlap": tau_ov, "tau_maha": tau_mh},
        "honest": honest,
        "attacks": attack_runs,
        "classifier": results,
    }


@app.local_entrypoint()
def main():
    if not (LIB_LOCAL.exists() and SIG_LOCAL.exists()):
        raise SystemExit("E1 library or E7 sigma calibration missing")
    lib = json.loads(LIB_LOCAL.read_text())
    sig = json.loads(SIG_LOCAL.read_text())
    print(f"loaded: {lib['metadata']['n_probes']} probes + {len(sig['calibration'])} sigma records")
    out = run.remote(lib, sig)
    OUT_LOCAL.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {OUT_LOCAL}")
    print(f"wall: {out['metadata']['wall_s']:.0f}s")
    for lift, r in out["classifier"].items():
        print(f"  {lift:<14} TPR_joint={r['tpr_joint']:.3f}  overlap={r['overlap_mean']:.4f}  maha={r['maha_mean']:.2f}")
