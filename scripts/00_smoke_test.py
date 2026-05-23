"""P1 smoke test: load Qwen3-1.7B + mwhanna transcoder, run 1 prompt, dump top-k latent activations.

Goal: verify end-to-end plumbing before we invest in the full probe-library extraction.
Expected runtime: <5 min. Expected VRAM: ~6 GB (1.7B bf16 + transcoder).
"""
import torch
import json
import time
from pathlib import Path

OUT = Path(__file__).parent.parent / "results" / "smoke.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("P1 smoke test — Qwen3-1.7B + transcoder")
print("=" * 60)

# --- 1. Load base model ------------------------------------------------------
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
print(f"\n[1/4] Loading {MODEL_ID}...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s")
print(f"  layers: {model.config.num_hidden_layers}")
print(f"  hidden: {model.config.hidden_size}")
print(f"  vram: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# --- 2. Load transcoder ------------------------------------------------------
t1 = time.time()
print(f"\n[2/4] Loading transcoder (mwhanna-qwen3-1.7b-transcoders-lowl0)...")
from sae_lens import SAE

# Try canonical layer 14 (mid-depth for 28-layer model)
# transcoder SAE_IDs are typically "blocks.{L}.hook_mlp_in" or similar
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="mwhanna-qwen3-1.7b-transcoders-lowl0",
    sae_id="layer_14",
    device="cuda",
)
print(f"  loaded via sae_id=layer_14 in {time.time()-t1:.1f}s")
print(f"  d_sae: {sae.cfg.d_sae}")
print(f"  d_in:  {sae.cfg.d_in}")
print(f"  architecture: {type(sae).__name__}")
# dump the full cfg dict to see what keys exist
cfg_dict_raw = sae.cfg.to_dict() if hasattr(sae.cfg, "to_dict") else sae.cfg.__dict__
print("  cfg keys:", sorted(cfg_dict_raw.keys()))
for k in ("hook_layer", "hook_name", "hook_name_in", "hook_name_out", "metadata", "architecture"):
    v = cfg_dict_raw.get(k)
    if v is not None:
        print(f"  cfg.{k}: {v}")
print(f"  vram: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# --- 3. Capture activations and encode via SAE -------------------------------
t2 = time.time()
print(f"\n[3/4] Running one probe prompt and capturing activations...")

PROMPT = "When John and Mary went to the store, John gave a book to"

# Hook the mlp_in of layer 14
captured = {}
def hook(module, inputs, output):
    # mlp takes (hidden_states,) as input; we want the input
    # inputs is a tuple; first element is the hidden state tensor
    captured["mlp_in"] = inputs[0].detach()

target_layer = 14
handle = model.model.layers[target_layer].mlp.register_forward_pre_hook(
    lambda mod, inputs: captured.__setitem__("mlp_in", inputs[0].detach())
)

ids = tok(PROMPT, return_tensors="pt").to("cuda")
with torch.no_grad():
    _ = model(**ids)
handle.remove()

h = captured["mlp_in"]  # [1, T, d_model]
print(f"  captured mlp_in shape: {tuple(h.shape)}, dtype: {h.dtype}")

with torch.no_grad():
    latents = sae.encode(h.to(sae.dtype))  # [1, T, d_sae]
print(f"  SAE latents shape: {tuple(latents.shape)}")
print(f"  sparsity: {(latents.abs() > 1e-6).float().mean().item():.4f}")

# Top-k at final position
final_pos = latents[0, -1]  # [d_sae]
topv, topi = torch.topk(final_pos.abs(), k=20)
print(f"  top-20 latent indices at final pos: {topi.tolist()[:10]} ...")
print(f"  top-20 magnitudes: {[f'{v.item():.2f}' for v in topv[:10]]}")

# --- 4. Dump --------------------------------------------------------------------
print(f"\n[4/4] Dumping results to {OUT}")
result = {
    "model": MODEL_ID,
    "sae_release": "mwhanna-qwen3-1.7b-transcoders-lowl0",
    "sae_id": "blocks.14.hook_mlp_in",
    "d_sae": int(sae.cfg.d_sae),
    "d_in": int(sae.cfg.d_in),
    "prompt": PROMPT,
    "top20_indices": topi.tolist(),
    "top20_magnitudes": [float(v) for v in topv.tolist()],
    "sparsity_final_pos": float((final_pos.abs() > 1e-6).float().mean().item()),
    "seconds": time.time() - t0,
}
OUT.write_text(json.dumps(result, indent=2))
print(f"  total: {time.time()-t0:.1f}s")
print("SMOKE TEST PASSED" if latents.shape[-1] == sae.cfg.d_sae else "SMOKE TEST FAIL")
