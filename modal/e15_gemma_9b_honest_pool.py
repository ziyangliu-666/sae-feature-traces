"""E15-9B: Gemma-2-9B honest pool with recalibrated sigma (P0-1 scale-up).

Mirrors e15_gemma_honest_pool_v2.py (2B) but on Gemma-2-9B @ L20.
Phase A: 32 configs per probe (2 dtypes x 2 kernels x 4 positions x 2 seeds).
Phase B: 64 configs per probe (2 x 2 x 4 x 4, disjoint seeds).

Produces tau_9B (p99 pool max) -> primary deployment threshold for the
9B scale-up experiment.

Reads: results/e12_gemma_9b_pilot.json (probe library from E12-9B).
Writes:
  results/sigma_calibration_gemma2_9b_L20.json
  results/recipe1_gemma2_9b_honest_pool.json

Est. wall on A100-40GB: ~1.0-1.5h -> ~$2.1-3.2.
"""
import json
import modal
from pathlib import Path

app = modal.App("e15-gemma-9b-honest-pool")
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
        "scipy",
        "accelerate>=0.33",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

TARGET_MODEL = "google/gemma-2-9b"
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_20/width_131k/canonical"
LAYER = 20
D_MODEL = 3584
TOP_K = 32

LOCAL_ROOT = Path(__file__).parent.parent
E12_JSON = LOCAL_ROOT / "results" / "e12_gemma_9b_pilot.json"
SIG_OUT = LOCAL_ROOT / "results" / "sigma_calibration_gemma2_9b_L20.json"
POOL_OUT = LOCAL_ROOT / "results" / "recipe1_gemma2_9b_honest_pool.json"


@app.function(
    gpu=GPU,
    image=image,
    timeout=10800,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(probes: list) -> dict:
    import os, time, random
    import torch
    import numpy as np
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    from scipy.stats import beta

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
        "Literary criticism in the 20th century",
        "The evolution of music streaming has",
    ]

    print(f"[load] {TARGET_MODEL}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    t1 = time.time()
    sae_res = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res
    sae.eval()
    print(f"  SAE loaded in {time.time()-t1:.1f}s, d_sae={sae.cfg.d_sae} vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    captured = {}
    def hook(_m, _inputs, outputs):
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        captured["h"] = h.detach()

    handle = model.model.layers[LAYER].register_forward_hook(hook)

    def forward_probe(prompt, pos, companions, dtype, kernel):
        batch = companions.copy()
        batch.insert(pos, prompt)
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        last = enc["attention_mask"].sum(dim=1) - 1
        with sdpa_kernel([kernel, SDPBackend.MATH]), torch.no_grad():
            if dtype == torch.bfloat16:
                _ = model(**enc)
            else:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = model(**enc)
        return captured["h"][pos, last[pos]]

    def encode_topk_at_ids(act, ids):
        with torch.no_grad():
            z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
        return z[ids].float().cpu().numpy()

    # =========================================================
    # Phase A: Sigma recalibration (32 configs per probe)
    # =========================================================
    PRECISIONS = [torch.bfloat16, torch.float16]
    KERNELS = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    POSITIONS = [0, 1, 2, 3]
    CAL_SEEDS = [50, 51]

    n_cal = len(PRECISIONS) * len(KERNELS) * len(POSITIONS) * len(CAL_SEEDS)
    print(f"\n[phase A] sigma recalibration: {n_cal} configs x {len(probes)} probes")
    tA = time.time()
    sig_snaps = {p["probe_id"]: [] for p in probes}
    cal_idx = 0
    for dtype in PRECISIONS:
        for kernel in KERNELS:
            for pos in POSITIONS:
                for seed in CAL_SEEDS:
                    rng = random.Random(seed + pos * 17)
                    companions = rng.sample(FILLER, 3)
                    for p in probes:
                        act = forward_probe(p["prompt"], pos, companions, dtype, kernel)
                        sig_snaps[p["probe_id"]].append(
                            encode_topk_at_ids(act, p["top_k_feature_ids"])
                        )
                    cal_idx += 1
                    if cal_idx % 4 == 0:
                        el = time.time() - tA
                        eta = el / cal_idx * (n_cal - cal_idx)
                        print(f"  [A {cal_idx:2d}/{n_cal}] (el {el:.0f}s eta {eta:.0f}s)")

    sig_by_probe = {}
    for p in probes:
        S = np.stack(sig_snaps[p["probe_id"]])
        sig_by_probe[p["probe_id"]] = {
            "mean_cross_backend": S.mean(axis=0).tolist(),
            "sigma_cross_backend": S.std(axis=0).tolist(),
            "n_calibration": n_cal,
        }
    print(f"[phase A] sigma done in {time.time()-tA:.0f}s")

    sigma_out = {
        "metadata": {
            "target": TARGET_MODEL, "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
            "layer": LAYER, "top_k": TOP_K,
            "n_calibration_configs": n_cal,
            "precisions": [str(p).split(".")[-1] for p in PRECISIONS],
            "kernels": [k.name for k in KERNELS],
            "positions": POSITIONS,
            "calibration_seeds": CAL_SEEDS,
        },
        "calibration": [
            {"probe_id": pid, **sig_by_probe[pid]}
            for pid in sorted(sig_by_probe.keys())
        ],
    }
    os.makedirs("/cache/e15_9b_out", exist_ok=True)
    with open("/cache/e15_9b_out/sigma_calibration_gemma2_9b_L20.json", "w") as f:
        json.dump(sigma_out, f, indent=2)

    all_sds = np.concatenate([
        np.array(sig_by_probe[p["probe_id"]]["sigma_cross_backend"]) for p in probes
    ])
    print(f"[phase A] sigma stats: p01={np.quantile(all_sds, 0.01):.4f} "
          f"p50={np.quantile(all_sds, 0.5):.4f} "
          f"p95={np.quantile(all_sds, 0.95):.4f} "
          f"max={all_sds.max():.4f}")
    print(f"[phase A] frac sigma < 1e-3: {(all_sds < 1e-3).mean():.4f}; "
          f"< 1e-2: {(all_sds < 1e-2).mean():.4f}")

    # =========================================================
    # Phase B: Honest pool (64 configs, disjoint seeds)
    # =========================================================
    n_probes = len(probes)
    mu = np.zeros((n_probes, TOP_K), dtype=np.float64)
    sd = np.zeros((n_probes, TOP_K), dtype=np.float64)
    for p in probes:
        mu[p["probe_id"]] = np.array(sig_by_probe[p["probe_id"]]["mean_cross_backend"], dtype=np.float64)
        sd[p["probe_id"]] = np.clip(
            np.array(sig_by_probe[p["probe_id"]]["sigma_cross_backend"], dtype=np.float64),
            1e-3, None,
        )

    POOL_SEEDS = [200, 201, 202, 203]
    n_pool = len(PRECISIONS) * len(KERNELS) * len(POSITIONS) * len(POOL_SEEDS)
    print(f"\n[phase B] honest pool: {n_pool} configs x {n_probes} probes")
    tB = time.time()
    configs = []
    per_probe_maha_all = []
    pool_idx = 0
    for dtype in PRECISIONS:
        for kernel in KERNELS:
            for pos in POSITIONS:
                for seed in POOL_SEEDS:
                    rng = random.Random(seed + pos * 17)
                    companions = rng.sample(FILLER, 3)
                    per_probe_z = np.zeros((n_probes, TOP_K), dtype=np.float64)
                    for p in probes:
                        act = forward_probe(p["prompt"], pos, companions, dtype, kernel)
                        per_probe_z[p["probe_id"]] = encode_topk_at_ids(
                            act, p["top_k_feature_ids"]
                        )
                    per_probe_maha = np.abs((per_probe_z - mu) / sd).mean(-1)
                    joint_z = float(per_probe_maha.mean())
                    per_probe_maha_all.append(per_probe_maha)
                    configs.append({
                        "run_id": pool_idx,
                        "dtype": str(dtype).split(".")[-1],
                        "kernel": kernel.name,
                        "pos": pos,
                        "companion_seed": seed,
                        "joint_z": joint_z,
                        "per_probe_max": float(per_probe_maha.max()),
                        "per_probe_median": float(np.median(per_probe_maha)),
                    })
                    pool_idx += 1
                    if pool_idx % 4 == 0:
                        el = time.time() - tB
                        eta = el / pool_idx * (n_pool - pool_idx)
                        print(f"  [B {pool_idx:2d}/{n_pool}] joint_z={joint_z:.4f} "
                              f"max={per_probe_maha.max():.3f} (el {el:.0f}s eta {eta:.0f}s)")

    handle.remove()
    print(f"[phase B] honest pool in {time.time()-tB:.0f}s")

    joints = np.array([c["joint_z"] for c in configs])
    per_probe_maha_arr = np.stack(per_probe_maha_all)
    tau_real_99 = float(np.quantile(joints, 0.99))
    tau_real_max = float(np.max(joints))

    def cp_upper(x, n_, alpha=0.05):
        if x == n_:
            return 1.0
        return float(beta.ppf(1 - alpha, x + 1, n_ - x))

    cp_at_zero_n64 = cp_upper(0, 64)
    exceed_real_99 = int(np.sum(joints > tau_real_99))
    cp_at_real_99 = cp_upper(exceed_real_99, 64)

    print(f"\n=== Gemma-9B honest pool summary (n={len(joints)}) ===")
    print(f"joint-z: min={joints.min():.4f} med={np.median(joints):.4f} "
          f"mean={joints.mean():.4f} max={joints.max():.4f}")
    print(f"p25/50/75/90/95/99: {np.quantile(joints,[.25,.5,.75,.9,.95,.99])}")
    print(f"tau_9B_p99 = {tau_real_99:.4f}, CP 95% upper at 0/64: {cp_at_zero_n64:.4f}")

    out = {
        "metadata": {
            "experiment": "Recipe1: Gemma-2-9B honest pool (P0-1 scale-up)",
            "target": TARGET_MODEL,
            "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
            "layer": LAYER, "top_k": TOP_K,
            "precisions": [str(p).split(".")[-1] for p in PRECISIONS],
            "kernels": [k.name for k in KERNELS],
            "positions": POSITIONS,
            "calibration_seeds": CAL_SEEDS,
            "pool_seeds": POOL_SEEDS,
            "n_calibration_configs": n_cal,
            "n_pool_configs": n_pool,
            "gpu": GPU,
            "wall_s": time.time() - t0,
        },
        "joint_z_values": joints.tolist(),
        "per_probe_maha": per_probe_maha_arr.tolist(),
        "configs": configs,
        "statistics": {
            "min": float(joints.min()),
            "median": float(np.median(joints)),
            "mean": float(joints.mean()),
            "max": float(joints.max()),
            "p90": float(np.quantile(joints, 0.9)),
            "p95": float(np.quantile(joints, 0.95)),
            "p99_empirical": tau_real_99,
        },
        "thresholds": {
            "tau_9B_p99": tau_real_99,
            "tau_9B_max": tau_real_max,
        },
        "exceedances": {
            "at_tau_9B_p99": {"count": exceed_real_99, "n": 64, "cp95_upper_ci": cp_at_real_99},
        },
        "clopper_pearson_upper_ci_at_zero_violations": {
            "n64": cp_at_zero_n64,
        },
        "sigma_stats": {
            "p01": float(np.quantile(all_sds, 0.01)),
            "p50": float(np.quantile(all_sds, 0.5)),
            "p95": float(np.quantile(all_sds, 0.95)),
            "max": float(all_sds.max()),
            "frac_below_1e3": float((all_sds < 1e-3).mean()),
            "frac_below_1e2": float((all_sds < 1e-2).mean()),
        },
    }
    with open("/cache/e15_9b_out/recipe1_gemma2_9b_honest_pool.json", "w") as f:
        json.dump(out, f, indent=2)
    return {"sigma": sigma_out, "pool": out}


@app.local_entrypoint()
def main():
    if not E12_JSON.exists():
        raise SystemExit(f"Missing Gemma-9B pilot artifact: {E12_JSON}")
    e12 = json.loads(E12_JSON.read_text())
    probes = e12["probes"]
    print(f"reusing {len(probes)} Gemma-9B probes from e12_gemma_9b_pilot.json")
    out = run.remote(probes)
    SIG_OUT.write_text(json.dumps(out["sigma"], indent=2))
    POOL_OUT.write_text(json.dumps(out["pool"], indent=2))
    print(f"\n[save] {SIG_OUT}")
    print(f"[save] {POOL_OUT}")
    print(f"wall: {out['pool']['metadata']['wall_s']:.0f}s  "
          f"tau_9B_p99={out['pool']['statistics']['p99_empirical']:.4f}")
