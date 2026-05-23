"""E-B: Qwen3-1.7B honest pool across MATH + FLASH + EFFICIENT kernels.

Reviewer W5 flagged that recipe1_qwen3_honest_pool was MATH-only
(WSL2/CUDA could not expose FLASH/EFFICIENT SDPA at pool time).
This script re-runs the honest pool on Modal L4, where all three
kernels are available, and compares the resulting empirical joint-z
distributions + tau_real_99.

Design:
  - Reuse the existing sigma calibration (already includes all three
    kernels, see sigma_calibration_qwen3_1.7b_L14.json metadata).
  - 24 fresh pool configs per kernel: 2 dtypes x 4 positions x 3 seeds.
  - Seeds 300-302 disjoint from recipe1's 100-107.
  - Total 72 configs x 96 probes; est. wall L4 ~20 min, ~$0.27.

Output: logs/recipe1_qwen3_honest_pool_multikernel.json
"""
import json
import modal
from pathlib import Path

app = modal.App("e17-qwen3-multikernel-pool")
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

TARGET_MODEL = "Qwen/Qwen3-1.7B"
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"
LAYER = 14
TOP_K = 32

LOCAL_ROOT = Path(__file__).parent.parent
LIB_PATH = LOCAL_ROOT / "logs" / "probe_library_qwen3_1.7b_L14_k96.json"
SIG_PATH = LOCAL_ROOT / "logs" / "sigma_calibration_qwen3_1.7b_L14.json"
POOL_OUT = LOCAL_ROOT / "logs" / "recipe1_qwen3_honest_pool_multikernel.json"


@app.function(
    gpu=GPU,
    image=image,
    timeout=5400,
    volumes={"/cache": VOL},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def run(probes: list, sigma_cal: list) -> dict:
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
    print(f"  loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    sae_res = SAE.from_pretrained(
        release=TRANSCODER_RELEASE, sae_id=f"layer_{LAYER}", device="cuda"
    )
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res
    sae.eval()
    print(f"  SAE loaded in {time.time()-t1:.1f}s, d_sae={sae.cfg.d_sae}")

    captured = {}
    def mlp_in_hook(_mod, inputs):
        captured["mlp_in"] = inputs[0].detach()

    handle = model.model.layers[LAYER].mlp.register_forward_pre_hook(mlp_in_hook)

    def forward_probe(prompt, pos, companions, dtype, kernel):
        batch = companions.copy()
        batch.insert(pos, prompt)
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        last = enc["attention_mask"].sum(dim=1) - 1
        with sdpa_kernel([kernel]), torch.no_grad():
            if dtype == torch.bfloat16:
                _ = model(**enc)
            else:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = model(**enc)
        return captured["mlp_in"][pos, last[pos]]

    def encode_topk_at_ids(act, ids):
        with torch.no_grad():
            z = sae.encode(act.unsqueeze(0).to(sae.dtype))[0]
        return z[ids].float().cpu().numpy()

    # Load sigma into mu/sd arrays keyed by probe_id
    mu = np.zeros((len(probes), TOP_K), dtype=np.float64)
    sd = np.zeros((len(probes), TOP_K), dtype=np.float64)
    probe_by_id = {p["probe_id"]: p for p in probes}
    for c in sigma_cal:
        pid = c["probe_id"]
        if pid in probe_by_id:
            mu[pid] = np.array(c["mean_cross_backend"], dtype=np.float64)
            sd[pid] = np.clip(np.array(c["sigma_cross_backend"], dtype=np.float64), 1e-3, None)

    PRECISIONS = [torch.bfloat16, torch.float16]
    KERNELS_ALL = [
        ("MATH", SDPBackend.MATH),
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
    ]
    POSITIONS = [0, 1, 2, 3]
    POOL_SEEDS = [300, 301, 302]  # disjoint from recipe1 (100-107)

    per_kernel_results = {}
    all_configs = []
    pool_idx_global = 0

    for kname, kernel in KERNELS_ALL:
        print(f"\n[{kname}] beginning pool leg")
        tK = time.time()
        # Smoke-test: does this kernel actually fire on this backend?
        try:
            _ = forward_probe(probes[0]["prompt"], 0, [FILLER[0], FILLER[1], FILLER[2]],
                              torch.bfloat16, kernel)
            available = True
        except Exception as e:
            print(f"  [{kname}] kernel unavailable: {e}")
            per_kernel_results[kname] = {"available": False, "error": str(e)}
            continue

        configs = []
        joints_k = []
        for dtype in PRECISIONS:
            for pos in POSITIONS:
                for seed in POOL_SEEDS:
                    rng = random.Random(seed + pos * 17)
                    companions = rng.sample(FILLER, 3)
                    per_probe_z = np.zeros((len(probes), TOP_K), dtype=np.float64)
                    for p in probes:
                        act = forward_probe(p["prompt"], pos, companions, dtype, kernel)
                        per_probe_z[p["probe_id"]] = encode_topk_at_ids(
                            act, p["top_k_feature_ids"]
                        )
                    per_probe_maha = np.abs((per_probe_z - mu) / sd).mean(-1)
                    joint_z = float(per_probe_maha.mean())
                    joints_k.append(joint_z)
                    cfg = {
                        "run_id": pool_idx_global,
                        "dtype": str(dtype).split(".")[-1],
                        "kernel": kname,
                        "pos": pos,
                        "companion_seed": seed,
                        "joint_z": joint_z,
                        "per_probe_max": float(per_probe_maha.max()),
                        "per_probe_median": float(np.median(per_probe_maha)),
                        "per_probe_z": per_probe_z.tolist(),
                    }
                    configs.append(cfg)
                    all_configs.append(cfg)
                    pool_idx_global += 1
                    if pool_idx_global % 4 == 0:
                        el = time.time() - tK
                        print(f"  [{kname} {len(configs):2d}] joint_z={joint_z:.4f} "
                              f"max={per_probe_maha.max():.3f} (el {el:.0f}s)")

        joints_arr = np.array(joints_k)
        per_kernel_results[kname] = {
            "available": True,
            "n_configs": len(joints_k),
            "joint_z_values": joints_arr.tolist(),
            "statistics": {
                "min": float(joints_arr.min()),
                "median": float(np.median(joints_arr)),
                "mean": float(joints_arr.mean()),
                "max": float(joints_arr.max()),
                "p99": float(np.quantile(joints_arr, 0.99)),
            },
            "wall_s": time.time() - tK,
        }
        print(f"[{kname}] done in {time.time()-tK:.0f}s; "
              f"med={np.median(joints_arr):.3f} max={joints_arr.max():.3f}")

    handle.remove()

    # Pooled statistics across all available kernels
    all_joints = np.array([c["joint_z"] for c in all_configs])
    def cp_upper(x, n_, alpha=0.05):
        if x == n_:
            return 1.0
        return float(beta.ppf(1 - alpha, x + 1, n_ - x))

    TAU_RECIPE1 = 1.13
    exceed_recipe1 = int(np.sum(all_joints > TAU_RECIPE1))
    cp_at_recipe1 = cp_upper(exceed_recipe1, len(all_joints))

    out = {
        "metadata": {
            "experiment": "E-B: Qwen3 honest pool across MATH+FLASH+EFFICIENT kernels",
            "target": TARGET_MODEL,
            "transcoder_release": TRANSCODER_RELEASE,
            "layer": LAYER, "top_k": TOP_K,
            "precisions": [str(p).split(".")[-1] for p in PRECISIONS],
            "kernels_attempted": [kn for kn, _ in KERNELS_ALL],
            "positions": POSITIONS,
            "pool_seeds": POOL_SEEDS,
            "gpu": GPU,
            "wall_s": time.time() - t0,
            "tau_recipe1": TAU_RECIPE1,
        },
        "per_kernel": per_kernel_results,
        "all_configs": all_configs,
        "pooled_statistics": {
            "n": len(all_joints),
            "min": float(all_joints.min()) if len(all_joints) else None,
            "median": float(np.median(all_joints)) if len(all_joints) else None,
            "mean": float(all_joints.mean()) if len(all_joints) else None,
            "max": float(all_joints.max()) if len(all_joints) else None,
            "p99": float(np.quantile(all_joints, 0.99)) if len(all_joints) else None,
            "exceed_tau_recipe1_1p13": exceed_recipe1,
            "cp95_upper_at_recipe1_exceedances": cp_at_recipe1,
        },
    }
    return out


@app.local_entrypoint()
def main():
    if not LIB_PATH.exists():
        raise SystemExit(f"Missing library: {LIB_PATH}")
    if not SIG_PATH.exists():
        raise SystemExit(f"Missing sigma: {SIG_PATH}")
    lib = json.loads(LIB_PATH.read_text())
    sig = json.loads(SIG_PATH.read_text())
    probes = lib["probes"]
    sigma_cal = sig["calibration"]
    print(f"[main] {len(probes)} probes, {len(sigma_cal)} sigma entries")
    out = run.remote(probes, sigma_cal)
    POOL_OUT.write_text(json.dumps(out, indent=2))
    print(f"[save] {POOL_OUT}")
    ps = out["pooled_statistics"]
    print(f"  pooled n={ps['n']} med={ps['median']:.4f} max={ps['max']:.4f} "
          f"p99={ps['p99']:.4f}")
    print(f"  exceedances above tau=1.13: {ps['exceed_tau_recipe1_1p13']}/{ps['n']} "
          f"(CP95 upper {ps['cp95_upper_at_recipe1_exceedances']:.4f})")
    for kname, r in out["per_kernel"].items():
        avail = r.get("available")
        if avail:
            s = r["statistics"]
            print(f"  {kname}: avail med={s['median']:.3f} max={s['max']:.3f} p99={s['p99']:.3f}")
        else:
            print(f"  {kname}: UNAVAILABLE ({r.get('error','?')[:60]})")
