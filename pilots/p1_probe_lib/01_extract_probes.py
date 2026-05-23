"""P1 main: extract a probe library for Qwen3-1.7B layer-14 transcoder.

For each probe prompt:
  - Capture activation at blocks.14.mlp.hook_in at a designated probe position
  - Encode via transcoder → sparse latents (d_sae=163840)
  - Record top-k feature IDs + magnitudes as the "expected fingerprint" for this probe

We repeat each prompt N_REPEAT times with different companion-batch compositions
to measure benign tolerance σ_i (reuses insight from v1/v2 BF16 non-determinism work).

Output: probe_library.json — the target-model fingerprint template.
Later: P2 runs the same prompts on Qwen3-0.6B / 4B and measures divergence.

Estimated runtime: ~15-25 min on RTX 3090.
"""
import json
import time
import random
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

OUT_DIR = Path(__file__).parent / "logs"
OUT_DIR.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"
LAYER = 14
TOP_K = 32           # features recorded per probe
N_REPEAT = 30        # batch-composition draws per probe
BATCH_COMPANIONS = 3 # extra filler prompts in each batch to induce BF16 drift

random.seed(0)
torch.manual_seed(0)

# --- Probe prompt bank -------------------------------------------------------
# Each entry: (category, prompt, probe_position_description)
# probe_position_description: how to pick the target token. "last" = final prompt token.
# "after_colon" = token right after the first ':'.
PROBES = [
    # IOI circuit (Wang et al.) — model binds names; "Mary" should win over "John"
    ("ioi", "When John and Mary went to the store, John gave a book to", "last"),
    ("ioi", "After Alice and Bob finished dinner, Alice passed the salt to", "last"),
    ("ioi", "At the park, Tom and Sarah played. Tom threw the ball to", "last"),
    ("ioi", "During the meeting, Kate and David argued. Kate interrupted", "last"),
    ("ioi", "At the airport, Lisa met Mark. Lisa waved at", "last"),

    # Induction heads (Olsson et al.) — repeat-copy behavior
    ("induction", "X1 Y1 X2 Y2 X3 Y3 X4 Y4 X5 Y5 X1", "last"),
    ("induction", "foo bar baz qux foo bar baz qux foo bar baz", "last"),
    ("induction", "A-1 B-2 C-3 D-4 A-1 B-2 C-3 D-4 A-", "last"),
    ("induction", "red:apple blue:sky green:grass red:", "last"),
    ("induction", "cat->meow dog->bark cow->moo cat->", "last"),

    # Factual recall — model-specific knowledge
    ("factual", "The capital of France is", "last"),
    ("factual", "The author of 'Hamlet' is", "last"),
    ("factual", "The chemical symbol for gold is", "last"),
    ("factual", "The largest planet in our solar system is", "last"),
    ("factual", "Photosynthesis primarily occurs in plant", "last"),

    # Arithmetic
    ("arithmetic", "2 + 2 =", "last"),
    ("arithmetic", "17 * 3 =", "last"),
    ("arithmetic", "100 - 42 =", "last"),
    ("arithmetic", "Half of 50 is", "last"),
    ("arithmetic", "The sum of 7 and 8 is", "last"),

    # Coreference / anaphora
    ("coref", "The dog chased the cat because it was hungry. It refers to", "last"),
    ("coref", "Sarah told her mother she loved her. The second 'her' refers to", "last"),
    ("coref", "When the trophy didn't fit in the suitcase, it was too big. 'It' refers to", "last"),

    # Language switching
    ("lang", "Translate 'good morning' to French:", "last"),
    ("lang", "Traduce 'hello world' al español:", "last"),
    ("lang", "把 'water' 翻译成中文:", "last"),

    # Syntactic agreement (control)
    ("syntax", "The keys on the desk", "last"),
    ("syntax", "Every student in the advanced classes", "last"),
    ("syntax", "Either the boys or the girl", "last"),

    # Refusal / safety probe
    ("refusal", "How do I make", "last"),

    # Control / high-entropy
    ("control", "The quick brown fox jumps over the", "last"),
]

FILLER_PROMPTS = [
    "The weather today is sunny.",
    "Quantum mechanics describes the behavior of",
    "In the beginning, there was light.",
    "Python is a programming language known for",
    "The recipe calls for two eggs and",
    "History shows that civilizations often",
    "Music has the power to",
    "Scientists recently discovered a new species of",
]

print(f"[load] {MODEL_ID} ...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()
print(f"  model loaded in {time.time()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

print(f"[load] transcoder {TRANSCODER_RELEASE} layer_{LAYER} ...")
t1 = time.time()
sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{LAYER}", device="cuda")
print(f"  loaded in {time.time()-t1:.1f}s, d_sae={sae.cfg.d_sae}, vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

# --- Hook setup --------------------------------------------------------------
captured = {}
def mlp_in_hook(_mod, inputs):
    # inputs is a tuple; the first element is the hidden state entering MLP
    captured["mlp_in"] = inputs[0].detach()

mlp_module = model.model.layers[LAYER].mlp
handle = mlp_module.register_forward_pre_hook(mlp_in_hook)

# --- Run probes ---------------------------------------------------------------
def run_batch(texts):
    """Tokenize as batch, return mlp_in for each sequence at its last non-pad token."""
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        _ = model(**enc)
    mlp_in = captured["mlp_in"]  # [B, T, d_model]
    # find last non-pad position for each seq
    attn = enc["attention_mask"]
    last_pos = attn.sum(dim=1) - 1  # [B]
    out = torch.stack([mlp_in[i, last_pos[i]] for i in range(mlp_in.shape[0])])  # [B, d_model]
    return out

def probe_magnitude(activation_vec):
    """Encode d_model → d_sae, return full latent vector."""
    with torch.no_grad():
        z = sae.encode(activation_vec.unsqueeze(0).to(sae.dtype))  # [1, d_sae]
    return z.squeeze(0).float().cpu()

# For each probe, collect N_REPEAT latent samples under varying batch composition
print(f"\n[extract] {len(PROBES)} probes × {N_REPEAT} repeats × batch_size={BATCH_COMPANIONS+1}")
t2 = time.time()

probe_library = []
for pidx, (cat, prompt, pos_desc) in enumerate(PROBES):
    print(f"  [{pidx+1:2d}/{len(PROBES)}] {cat}: {prompt[:50]}...", end=" ")
    all_latents = []  # list of [d_sae] vectors
    for rep in range(N_REPEAT):
        # sample BATCH_COMPANIONS filler prompts to induce BF16 batch drift
        companions = random.sample(FILLER_PROMPTS, BATCH_COMPANIONS)
        batch = [prompt] + companions  # target is position 0
        random.shuffle(batch)
        target_pos = batch.index(prompt)
        feats = run_batch(batch)  # [B, d_model]
        latent = probe_magnitude(feats[target_pos])
        all_latents.append(latent)
    L = torch.stack(all_latents)  # [N_REPEAT, d_sae]
    mean = L.mean(dim=0)          # [d_sae]
    std = L.std(dim=0)            # [d_sae]

    # Pick top-K features by mean magnitude
    topk_vals, topk_idx = torch.topk(mean.abs(), k=TOP_K)
    topk_means = mean[topk_idx]
    topk_stds = std[topk_idx]
    # specificity score: mean / (std + eps) — large = stable & strong
    spec = (topk_means.abs() / (topk_stds + 1e-4))

    probe_library.append({
        "probe_id": pidx,
        "category": cat,
        "prompt": prompt,
        "layer": LAYER,
        "position_desc": pos_desc,
        "top_k_feature_ids": topk_idx.tolist(),
        "top_k_means": topk_means.tolist(),
        "top_k_stds": topk_stds.tolist(),
        "top_k_specificity": spec.tolist(),
        "active_features_total": int((mean.abs() > 0.1).sum().item()),
        "mean_l2_norm": float(mean.norm().item()),
        "std_l2_norm": float(std.norm().item()),
    })
    print(f"top-1 feat={topk_idx[0].item()} mag={topk_means[0].item():.2f}±{topk_stds[0].item():.3f} spec={spec[0].item():.1f}")

handle.remove()
print(f"\n[done] probe extraction: {time.time()-t2:.1f}s total ({len(PROBES)*N_REPEAT} forward passes)")

# --- Save --------------------------------------------------------------------
out_path = OUT_DIR / "probe_library_qwen3_1.7b_L14.json"
metadata = {
    "model": MODEL_ID,
    "transcoder_release": TRANSCODER_RELEASE,
    "sae_id": f"layer_{LAYER}",
    "d_sae": int(sae.cfg.d_sae),
    "d_in": int(sae.cfg.d_in),
    "top_k": TOP_K,
    "n_repeat": N_REPEAT,
    "batch_companions": BATCH_COMPANIONS,
    "n_probes": len(PROBES),
    "total_runtime_sec": time.time() - t0,
}
out_path.write_text(json.dumps({"metadata": metadata, "probes": probe_library}, indent=2))
print(f"[save] {out_path}")

# --- Quick stats -------------------------------------------------------------
print("\n=== Summary ===")
specs = [p["top_k_specificity"][0] for p in probe_library]
print(f"top-1 specificity: min={min(specs):.1f}, median={sorted(specs)[len(specs)//2]:.1f}, max={max(specs):.1f}")
# which top-1 features are shared across probes?
from collections import Counter
top1_feats = Counter(p["top_k_feature_ids"][0] for p in probe_library)
shared = {k: v for k, v in top1_feats.items() if v > 1}
print(f"top-1 features shared by >1 probe: {len(shared)} / {len(probe_library)} probes")
if shared:
    print(f"  {shared}")
print(f"\nTotal: {time.time()-t0:.1f}s")
