"""E15: Gemma-2-2B honest pool expansion (n=64), analog of 17_honest_pool_expansion.

Addresses reviewer W5 Gemma leg: the current Gemma hold-out is n_hon=8
(Clopper-Pearson upper 0.312). We fire 64 fresh honest configs that are
NOT used in the E12 sigma calibration, and save:
  (i) joint-z per config
  (ii) raw per-probe Mahalanobis array [n, 96]  -- feeds E-C MVN bootstrap
  (iii) empirical tau_real (99th percentile) + CP upper CI at 0 exceedances
  (iv) kernel/dtype/position metadata for position-dependence analysis

Reuses Gemma probe library + sigma calibration from e12_gemma_pilot.json.

Companion seeds 200-215 are disjoint from E12 calibration seeds.

Est. wall on L4: ~45-70 min -> ~$0.8-1.2.
"""
import json
import modal
from pathlib import Path

app = modal.App("e15-gemma-honest-pool")
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
        "scipy",
        "accelerate>=0.33",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

TARGET_MODEL = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
LAYER = 12
D_MODEL = 2304
TOP_K = 32

LOCAL_ROOT = Path(__file__).parent.parent
E12_JSON = LOCAL_ROOT / "results" / "e12_gemma_pilot.json"
OUT_JSON = LOCAL_ROOT / "results" / "recipe1_gemma2_honest_pool.json"


@app.function(
    gpu=GPU,
    image=image,
    timeout=5400,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(probes: list, sigma_calibration: dict) -> dict:
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
    print(f"  loaded in {time.time()-t0:.1f}s, d_model={model.config.hidden_size}")

    print(f"[load] {SAE_RELEASE} / {SAE_ID}")
    t1 = time.time()
    sae_res = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res
    sae.eval()
    print(f"  loaded in {time.time()-t1:.1f}s, d_sae={sae.cfg.d_sae}")

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
        # Gemma-2 requires MATH as fallback for EFFICIENT on some kernels
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

    # Reconstruct mu/sd arrays
    mu = np.zeros((96, TOP_K), dtype=np.float64)
    sd = np.zeros((96, TOP_K), dtype=np.float64)
    for p in probes:
        pid = str(p["probe_id"])
        mu[p["probe_id"]] = np.array(sigma_calibration[pid]["mean_cross_backend"], dtype=np.float64)
        sd[p["probe_id"]] = np.clip(
            np.array(sigma_calibration[pid]["sigma_cross_backend"], dtype=np.float64),
            1e-3, None,
        )

    # 64 honest configs disjoint from E12 calibration
    # 2 dtypes x 2 kernels x 4 positions x 4 seeds = 64
    PRECISIONS = [torch.bfloat16, torch.float16]
    KERNELS = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    POSITIONS = [0, 1, 2, 3]
    COMPANION_SEEDS = [200, 201, 202, 203]

    n_total = len(PRECISIONS) * len(KERNELS) * len(POSITIONS) * len(COMPANION_SEEDS)
    print(f"\n[run] {n_total} honest configs x {len(probes)} probes")

    t2 = time.time()
    configs = []
    per_probe_maha_all = []  # [n, 96]
    run_idx = 0
    for dtype in PRECISIONS:
        for kernel in KERNELS:
            for pos in POSITIONS:
                for seed in COMPANION_SEEDS:
                    rng = random.Random(seed + pos * 17)
                    companions = rng.sample(FILLER, 3)
                    per_probe_z = np.zeros((96, TOP_K), dtype=np.float64)
                    for p in probes:
                        act = forward_probe(p["prompt"], pos, companions, dtype, kernel)
                        per_probe_z[p["probe_id"]] = encode_topk_at_ids(
                            act, p["top_k_feature_ids"]
                        )
                    per_probe_maha = np.abs((per_probe_z - mu) / sd).mean(-1)  # [96]
                    joint_z = float(per_probe_maha.mean())
                    per_probe_maha_all.append(per_probe_maha)
                    configs.append({
                        "run_id": run_idx,
                        "dtype": str(dtype).split(".")[-1],
                        "kernel": kernel.name,
                        "pos": pos,
                        "companion_seed": seed,
                        "joint_z": joint_z,
                        "per_probe_max": float(per_probe_maha.max()),
                        "per_probe_median": float(np.median(per_probe_maha)),
                    })
                    run_idx += 1
                    if run_idx % 4 == 0:
                        el = time.time() - t2
                        eta = el / run_idx * (n_total - run_idx)
                        print(f"  [{run_idx:2d}/{n_total}] joint_z={joint_z:.4f} "
                              f"max={per_probe_maha.max():.3f} (el {el:.0f}s eta {eta:.0f}s)")

    handle.remove()
    print(f"\n[done] honest pool in {time.time()-t2:.1f}s")

    joints = np.array([c["joint_z"] for c in configs])
    per_probe_maha_arr = np.stack(per_probe_maha_all)  # [n, 96]
    tau_real_99 = float(np.quantile(joints, 0.99))
    tau_real_max = float(np.max(joints))
    n = len(joints)

    def cp_upper(x, n_, alpha=0.05):
        if x == n_:
            return 1.0
        return float(beta.ppf(1 - alpha, x + 1, n_ - x))

    cp_at_zero_n64 = cp_upper(0, 64)
    cp_at_zero_n8 = cp_upper(0, 8)
    exceed_real_99 = int(np.sum(joints > tau_real_99))
    cp_at_real_99 = cp_upper(exceed_real_99, 64)

    print(f"\n=== Gemma-2-2B honest pool summary (n={n}) ===")
    print(f"joint-z: min={joints.min():.4f} med={np.median(joints):.4f} "
          f"mean={joints.mean():.4f} max={joints.max():.4f}")
    print(f"p25/p50/p75/p90/p95/p99: {np.quantile(joints,[.25,.5,.75,.9,.95,.99])}")
    print(f"tau_real_99 = {tau_real_99:.4f}, tau_max={tau_real_max:.4f}")
    print(f"CP 95% upper at 0/64: {cp_at_zero_n64:.4f} (vs 0/8 which was {cp_at_zero_n8:.4f})")

    out = {
        "metadata": {
            "experiment": "Recipe1: Gemma-2-2B honest pool expansion (E-A)",
            "target": TARGET_MODEL,
            "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
            "layer": LAYER, "top_k": TOP_K,
            "precisions": [str(p).split(".")[-1] for p in PRECISIONS],
            "kernels": [k.name for k in KERNELS],
            "positions": POSITIONS,
            "companion_seeds": COMPANION_SEEDS,
            "n_honest_total": n,
            "gpu": GPU,
            "wall_s": time.time() - t0,
        },
        "joint_z_values": joints.tolist(),
        "per_probe_maha": per_probe_maha_arr.tolist(),  # [n, 96]
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
            "tau_real_p99": tau_real_99,
            "tau_real_max": tau_real_max,
        },
        "exceedances": {
            "at_tau_real_p99": {"count": exceed_real_99, "n": n, "cp95_upper_ci": cp_at_real_99},
        },
        "clopper_pearson_upper_ci_at_zero_violations": {
            "n64": cp_at_zero_n64,
            "n8_reference": cp_at_zero_n8,
        },
    }

    os.makedirs("/cache/e15_out", exist_ok=True)
    with open("/cache/e15_out/recipe1_gemma2_honest_pool.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[persist] /cache/e15_out/recipe1_gemma2_honest_pool.json")
    return out


@app.local_entrypoint()
def main():
    if not E12_JSON.exists():
        raise SystemExit(f"Missing Gemma pilot artifact: {E12_JSON}")
    e12 = json.loads(E12_JSON.read_text())
    probes = e12["probes"]
    sigma = e12["sigma_calibration"]
    print(f"reusing {len(probes)} probes and sigma from e12_gemma_pilot.json")
    out = run.remote(probes, sigma)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {OUT_JSON}")
    print(f"wall: {out['metadata']['wall_s']:.0f}s  tau_real_99={out['statistics']['p99_empirical']:.4f}")
