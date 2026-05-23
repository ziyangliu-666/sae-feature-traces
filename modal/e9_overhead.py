"""E9: overhead + batched-FLOPs (C6-FLOPs).

Measure wall-clock latency for three service paths on Qwen3-1.7B, varying
batch size. Report commit overhead ratio and per-request payload size.

Paths:
  A:  forward-only                       (baseline serving)
  B:  forward + SAE encode               (analysis)
  C:  forward + SAE encode + top-32 + hash (full commit)

Each batch size × path → 20 trials (drop first 3 warmup), torch.cuda.synchronize
around the measured region.

Runs on Modal L4 (24GB fits Qwen3-1.7B bf16 + SAE). Estimated ≈ 20 min, $0.30.
"""
import json
import modal
from pathlib import Path

app = modal.App("e9-overhead")
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
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"
TOP_K = 32
SEQ_LEN = 128
BATCHES = [1, 4, 16, 32]       # L4 should fit up to 32 with seq 128
TRIALS = 20
WARMUP = 3

LOCAL_ROOT = Path(__file__).parent.parent


@app.function(gpu=GPU, image=image, timeout=1800, volumes={"/cache": VOL},
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])])
def run() -> dict:
    import time, hashlib
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{TARGET_LAYER}", device="cuda")
    sae.eval()
    print(f"[load] model + SAE ready, vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    captured = {}
    def hook(_m, inputs): captured["x"] = inputs[0].detach()
    model.model.layers[TARGET_LAYER].mlp.register_forward_pre_hook(hook)

    # synthetic input (fixed tokens, random pad) — content doesn't matter for timing
    def make_batch(B):
        ids = torch.randint(0, 50000, (B, SEQ_LEN), device="cuda")
        attn = torch.ones((B, SEQ_LEN), device="cuda")
        return {"input_ids": ids, "attention_mask": attn}

    def run_path(path: str, enc):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            _ = model(**enc)
            if path in ("B", "C"):
                x = captured["x"][:, -1]     # last-token acts, [B, d_model]
                z = sae.encode(x.to(sae.dtype))
            if path == "C":
                vals, idx = torch.topk(z.abs(), k=TOP_K, dim=-1)
                z_top = torch.gather(z, -1, idx)
                # commit: serialize (idx + vals) → bytes → sha256 per row
                idx_bytes = idx.to(torch.int32).cpu().numpy().tobytes()
                # bf16 → view as int16 (same 2 bytes) to skip numpy's bf16 gap
                val_bytes = z_top.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes()
                h = hashlib.sha256(idx_bytes + val_bytes).digest()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000.0  # ms

    results = []
    for B in BATCHES:
        enc = make_batch(B)
        timings = {"A": [], "B": [], "C": []}
        for path in ["A", "B", "C"]:
            for t in range(WARMUP + TRIALS):
                ms = run_path(path, enc)
                if t >= WARMUP:
                    timings[path].append(ms)
        medA = sorted(timings["A"])[TRIALS // 2]
        medB = sorted(timings["B"])[TRIALS // 2]
        medC = sorted(timings["C"])[TRIALS // 2]
        payload_bytes = TOP_K * (4 + 2) + 32  # int32 idx + bf16 val + sha256
        print(f"B={B:>3}  A={medA:>6.2f}ms  B={medB:>6.2f}ms  C={medC:>6.2f}ms"
              f"  C/A={medC/medA:.3f}  payload={payload_bytes*B}B")
        results.append({
            "batch": B,
            "A_forward_ms": medA,
            "B_forward_sae_ms": medB,
            "C_full_commit_ms": medC,
            "C_over_A_ratio": medC / medA,
            "B_over_A_ratio": medB / medA,
            "payload_bytes_total": payload_bytes * B,
            "payload_bytes_per_req": payload_bytes,
            "A_trials_ms": timings["A"],
            "B_trials_ms": timings["B"],
            "C_trials_ms": timings["C"],
        })

    return {
        "model": TARGET_MODEL, "seq_len": SEQ_LEN, "top_k": TOP_K,
        "trials": TRIALS, "warmup": WARMUP,
        "per_batch": results,
    }


@app.local_entrypoint()
def main():
    r = run.remote()
    out = LOCAL_ROOT / "results" / "e9_overhead.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\n[save] {out}")
    for row in r["per_batch"]:
        print(f"  batch={row['batch']:>3} A={row['A_forward_ms']:.2f}ms "
              f"C={row['C_full_commit_ms']:.2f}ms  ratio={row['C_over_A_ratio']:.3f}x"
              f"  payload={row['payload_bytes_per_req']}B/req")
