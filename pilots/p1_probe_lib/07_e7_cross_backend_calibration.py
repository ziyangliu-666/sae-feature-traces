"""E7: cross-backend σ_i calibration for honest M.

Goal: for every (probe p, feature i) pair, measure the activation std across
legitimate backend variations that an honest provider might switch between.
The per-feature σ_i is then used by every downstream scorer to normalize
"acceptable drift" from malicious drift.

Variations we sweep (all represent *honest* execution of M):
  - precision:      bfloat16 vs float16 (two common prod serving dtypes)
  - sdpa kernel:    MATH vs FLASH_ATTENTION vs EFFICIENT_ATTENTION
  - batch position: probe placed at position 0, 1, 2, 3 in a 4-sample batch
                    (4 different companion-prompt compositions per position)

For each probe we therefore collect:  2 precisions × 3 kernels × 4 positions
= 24 honest snapshots. We emit the top-32 feature IDs from the k=96 library
and record the mean + std of each such feature across those 24 snapshots.

Output: logs/sigma_calibration_qwen3_1.7b_L14.json
  - per-probe, per-feature mean/std across honest backends
  - per-probe L2 drift from the canonical E1 library mean
  - global summary: how large is honest σ_i vs E1 library σ (intra-composition)

Estimated runtime: ~5-8 min on RTX 3090 (96 probes × 24 configs).
"""
import json
import time
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

OUT_DIR = Path(__file__).parent / "logs"
OUT_DIR.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"
LAYER = 14
LIBRARY_PATH = OUT_DIR / "probe_library_qwen3_1.7b_L14_k96.json"

PRECISIONS = [torch.bfloat16, torch.float16]
KERNELS = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
N_POSITIONS = 4

random.seed(0)
torch.manual_seed(0)

print("[load] library")
lib = json.loads(LIBRARY_PATH.read_text())
PROBES = [(p["category"], p["prompt"], p["position_desc"], p["top_k_feature_ids"]) for p in lib["probes"]]
print(f"  {len(PROBES)} probes loaded from {LIBRARY_PATH.name}")

FILLER_PROMPTS = [
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

print(f"\n[load] {MODEL_ID}")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# load once in bf16 on cuda, we'll cast per-pass via autocast
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()
print(f"  model loaded in {time.time()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

print(f"[load] transcoder layer_{LAYER}")
t1 = time.time()
sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{LAYER}", device="cuda")
sae.eval()
print(f"  loaded in {time.time()-t1:.1f}s, d_sae={sae.cfg.d_sae}")

captured = {}
def mlp_in_hook(_mod, inputs):
    captured["mlp_in"] = inputs[0].detach()

mlp_module = model.model.layers[LAYER].mlp
handle = mlp_module.register_forward_pre_hook(mlp_in_hook)

# precompute 4 fixed batch compositions (one per target position), so the only
# thing that varies at a given position is precision × kernel
BATCH_COMPOSITIONS = []
for pos in range(N_POSITIONS):
    companions = random.sample(FILLER_PROMPTS, N_POSITIONS - 1)
    BATCH_COMPOSITIONS.append((pos, companions))

def run_probe(prompt: str, pos: int, companions: list[str], dtype, kernel) -> torch.Tensor:
    """Return the top-layer pre-MLP activation vector at the probe position."""
    batch = companions.copy()
    batch.insert(pos, prompt)
    enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
    attn = enc["attention_mask"]
    last_pos = attn.sum(dim=1) - 1
    with sdpa_kernel([kernel]), torch.no_grad():
        if dtype == torch.bfloat16:
            _ = model(**enc)
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = model(**enc)
    mlp_in = captured["mlp_in"]
    return mlp_in[pos, last_pos[pos]]

def encode_features(vec: torch.Tensor, feature_ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        z = sae.encode(vec.unsqueeze(0).to(sae.dtype))
    return z.squeeze(0)[feature_ids].float().cpu()

N_CONFIGS = len(PRECISIONS) * len(KERNELS) * N_POSITIONS
print(f"\n[calibrate] {len(PROBES)} probes × {N_CONFIGS} backend configs ({len(PRECISIONS)} dtypes × {len(KERNELS)} kernels × {N_POSITIONS} positions)")
t2 = time.time()

results = []
for pidx, (cat, prompt, pos_desc, feat_ids) in enumerate(PROBES):
    snapshots = []
    cfg_labels = []
    for dtype in PRECISIONS:
        for kernel in KERNELS:
            for (pos, companions) in BATCH_COMPOSITIONS:
                try:
                    act = run_probe(prompt, pos, companions, dtype, kernel)
                    feats = encode_features(act, feat_ids)
                    snapshots.append(feats)
                    cfg_labels.append((str(dtype).split(".")[-1], kernel.name, pos))
                except Exception as e:
                    print(f"  [warn] probe {pidx} dtype={dtype} kernel={kernel.name} pos={pos}: {e}")
    if not snapshots:
        continue
    S = torch.stack(snapshots)  # [n_configs, top_k]
    mean = S.mean(dim=0)
    std = S.std(dim=0)
    # compare to E1 canonical mean (first snapshot in snapshots is our reference)
    ref = lib["probes"][pidx]["top_k_means"]
    ref_t = torch.tensor(ref, dtype=torch.float32)
    l2_drift = (mean - ref_t).norm().item() / (ref_t.norm().item() + 1e-6)
    cv_med = float(torch.median(std / (mean.abs() + 1e-4)).item())
    results.append({
        "probe_id": pidx,
        "category": cat,
        "feature_ids": feat_ids,
        "sigma_cross_backend": std.tolist(),
        "mean_cross_backend": mean.tolist(),
        "e1_mean": ref,
        "l2_relative_drift_from_e1": l2_drift,
        "cv_median": cv_med,
        "n_configs": len(snapshots),
    })
    if (pidx + 1) % 8 == 0 or pidx == 0:
        elapsed = time.time() - t2
        eta = elapsed / (pidx + 1) * (len(PROBES) - pidx - 1)
        print(f"  [{pidx+1:2d}/{len(PROBES)}] {cat:>12}: drift_L2={l2_drift:.4f} cv_med={cv_med:.3f}  (elapsed {elapsed:.0f}s eta {eta:.0f}s)")

handle.remove()
print(f"\n[done] calibration in {time.time()-t2:.1f}s")

out_path = OUT_DIR / "sigma_calibration_qwen3_1.7b_L14.json"
metadata = {
    "model": MODEL_ID,
    "transcoder_release": TRANSCODER_RELEASE,
    "layer": LAYER,
    "n_probes": len(results),
    "n_configs_per_probe": N_CONFIGS,
    "precisions": [str(p).split(".")[-1] for p in PRECISIONS],
    "kernels": [k.name for k in KERNELS],
    "n_positions": N_POSITIONS,
    "total_runtime_sec": time.time() - t0,
    "experiment": "E7",
}
out_path.write_text(json.dumps({"metadata": metadata, "calibration": results}, indent=2))
print(f"[save] {out_path}")

print("\n=== Summary ===")
drifts = [r["l2_relative_drift_from_e1"] for r in results]
cvs = [r["cv_median"] for r in results]
drifts.sort(); cvs.sort()
print(f"L2 relative drift honest-backend vs E1 canonical:")
print(f"  min={drifts[0]:.4f}  med={drifts[len(drifts)//2]:.4f}  p95={drifts[int(0.95*len(drifts))]:.4f}  max={drifts[-1]:.4f}")
print(f"Per-feature coefficient-of-variation (σ/|μ|, median over top-32):")
print(f"  min={cvs[0]:.3f}  med={cvs[len(cvs)//2]:.3f}  p95={cvs[int(0.95*len(cvs))]:.3f}  max={cvs[-1]:.3f}")

by_cat = defaultdict(list)
for r in results:
    by_cat[r["category"]].append(r["l2_relative_drift_from_e1"])
print(f"\nPer-category median L2 drift:")
for c in sorted(by_cat):
    s = sorted(by_cat[c])
    print(f"  {c:>12}: n={len(s):2d}  med={s[len(s)//2]:.4f}  max={max(s):.4f}")

print(f"\nTotal: {time.time()-t0:.1f}s")
