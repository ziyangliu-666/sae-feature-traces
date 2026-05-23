"""E1: scale probe library from k=31 to k=96 with robust-mix constraint.

Constraint (from P3b finding): induction + coref + ioi probes are the robust
classes under worst-case distribution-matched attack. The deployment rule
requires >= 30% of the library to be drawn from these three classes.

Composition target (96 probes):
  ioi: 14  |  induction: 14  |  coref: 10   → 38 robust (39.6%)
  factual: 14  |  arithmetic: 12  |  lang: 6  |  syntax: 6
  negation: 6  |  entailment: 6  |  refusal: 4  |  control: 4

Reuses the extraction logic from 01_extract_probes.py:
  - capture at blocks.14.mlp.hook_in
  - companion-batch repetition (30 repeats × 3 companions) for σ_i calibration
  - top-32 SAE features per probe

Output: results/probe_library_qwen3_1.7b_L14_k96.json.
Estimated runtime: ~40-50 min on RTX 3090.
"""
import json
import time
import random
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
TRANSCODER_RELEASE = "mwhanna-qwen3-1.7b-transcoders-lowl0"
LAYER = 14
TOP_K = 32
N_REPEAT = 30
BATCH_COMPANIONS = 3

random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Probe bank — 96 probes, robust-mix >= 30%
# ---------------------------------------------------------------------------
PROBES = [
    # --- IOI circuit (14) ---
    ("ioi", "When John and Mary went to the store, John gave a book to", "last"),
    ("ioi", "After Alice and Bob finished dinner, Alice passed the salt to", "last"),
    ("ioi", "At the park, Tom and Sarah played. Tom threw the ball to", "last"),
    ("ioi", "During the meeting, Kate and David argued. Kate interrupted", "last"),
    ("ioi", "At the airport, Lisa met Mark. Lisa waved at", "last"),
    ("ioi", "In the classroom, Emma and Liam sat. Emma whispered to", "last"),
    ("ioi", "At the beach, Noah and Sophia swam. Noah called out to", "last"),
    ("ioi", "During dinner, Olivia and Ethan chatted. Olivia handed a plate to", "last"),
    ("ioi", "Before the show, Mia and Lucas arrived. Mia gave a ticket to", "last"),
    ("ioi", "After class, Amelia and James walked. Amelia said goodbye to", "last"),
    ("ioi", "When Isabella and Henry entered the café, Isabella ordered coffee for", "last"),
    ("ioi", "On the hike, Ava and Benjamin paused. Ava offered water to", "last"),
    ("ioi", "During the game, Charlotte and Daniel scored. Charlotte high-fived", "last"),
    ("ioi", "Outside the theater, Harper and Michael waited. Harper texted", "last"),

    # --- Induction heads (14) ---
    ("induction", "X1 Y1 X2 Y2 X3 Y3 X4 Y4 X5 Y5 X1", "last"),
    ("induction", "foo bar baz qux foo bar baz qux foo bar baz", "last"),
    ("induction", "A-1 B-2 C-3 D-4 A-1 B-2 C-3 D-4 A-", "last"),
    ("induction", "red:apple blue:sky green:grass red:", "last"),
    ("induction", "cat->meow dog->bark cow->moo cat->", "last"),
    ("induction", "alpha beta gamma alpha beta gamma alpha beta", "last"),
    ("induction", "K1=V1 K2=V2 K3=V3 K1=V1 K2=V2 K3=", "last"),
    ("induction", "1->one 2->two 3->three 4->four 1->", "last"),
    ("induction", "north:cold south:warm east:dry west:wet north:", "last"),
    ("induction", "Mon:work Tue:work Wed:work Thu:work Mon:", "last"),
    ("induction", "apple|red banana|yellow grape|purple apple|", "last"),
    ("induction", "x1:a x2:b x3:c x4:d x5:e x1:", "last"),
    ("induction", "Q1->A1 Q2->A2 Q3->A3 Q1->", "last"),
    ("induction", "dog#1 cat#2 bird#3 fish#4 dog#", "last"),

    # --- Coreference (10) ---
    ("coref", "The dog chased the cat because it was hungry. It refers to", "last"),
    ("coref", "Sarah told her mother she loved her. The second 'her' refers to", "last"),
    ("coref", "When the trophy didn't fit in the suitcase, it was too big. 'It' refers to", "last"),
    ("coref", "Jake gave Eli the book because he wanted it. 'He' refers to", "last"),
    ("coref", "The boy hit the ball and it flew high. 'It' refers to", "last"),
    ("coref", "Grace called her sister but she didn't answer. 'She' refers to", "last"),
    ("coref", "The cat watched the mouse because it was curious. 'It' refers to", "last"),
    ("coref", "Leo handed Noah the keys because he forgot them. 'He' refers to", "last"),
    ("coref", "Maya bought the vase but broke it at home. 'It' refers to", "last"),
    ("coref", "The fox ran from the hound because it was scared. 'It' refers to", "last"),

    # --- Factual recall (14) ---
    ("factual", "The capital of France is", "last"),
    ("factual", "The author of 'Hamlet' is", "last"),
    ("factual", "The chemical symbol for gold is", "last"),
    ("factual", "The largest planet in our solar system is", "last"),
    ("factual", "Photosynthesis primarily occurs in plant", "last"),
    ("factual", "The capital of Japan is", "last"),
    ("factual", "The currency of Germany is the", "last"),
    ("factual", "The inventor of the telephone was Alexander Graham", "last"),
    ("factual", "The tallest mountain on Earth is Mount", "last"),
    ("factual", "Water boils at 100 degrees", "last"),
    ("factual", "The human body has 206", "last"),
    ("factual", "The Great Wall is located in", "last"),
    ("factual", "The Pacific is the largest", "last"),
    ("factual", "The theory of relativity was proposed by Albert", "last"),

    # --- Arithmetic (12) ---
    ("arithmetic", "2 + 2 =", "last"),
    ("arithmetic", "17 * 3 =", "last"),
    ("arithmetic", "100 - 42 =", "last"),
    ("arithmetic", "Half of 50 is", "last"),
    ("arithmetic", "The sum of 7 and 8 is", "last"),
    ("arithmetic", "9 + 6 =", "last"),
    ("arithmetic", "12 * 12 =", "last"),
    ("arithmetic", "200 / 4 =", "last"),
    ("arithmetic", "Ten plus fifteen equals", "last"),
    ("arithmetic", "The difference between 25 and 11 is", "last"),
    ("arithmetic", "3 squared is", "last"),
    ("arithmetic", "The product of 6 and 7 is", "last"),

    # --- Language switching (6) ---
    ("lang", "Translate 'good morning' to French:", "last"),
    ("lang", "Traduce 'hello world' al español:", "last"),
    ("lang", "把 'water' 翻译成中文:", "last"),
    ("lang", "The German word for 'cat' is", "last"),
    ("lang", "In Japanese, 'thank you' is", "last"),
    ("lang", "The Spanish word for 'book' is", "last"),

    # --- Syntactic agreement (6) ---
    ("syntax", "The keys on the desk", "last"),
    ("syntax", "Every student in the advanced classes", "last"),
    ("syntax", "Either the boys or the girl", "last"),
    ("syntax", "Neither the players nor the coach", "last"),
    ("syntax", "The books that the professor assigned", "last"),
    ("syntax", "A group of children near the fountain", "last"),

    # --- Negation (6) ---
    ("negation", "It is not true that water is", "last"),
    ("negation", "The opposite of hot is", "last"),
    ("negation", "A liar never tells the", "last"),
    ("negation", "She did not want to leave, so she", "last"),
    ("negation", "If it is not raining, then it is", "last"),
    ("negation", "No student in the class was", "last"),

    # --- Entailment / logical (6) ---
    ("entailment", "All cats are mammals. Whiskers is a cat. Therefore, Whiskers is a", "last"),
    ("entailment", "If it rains, the grass gets wet. It is raining. Therefore, the grass gets", "last"),
    ("entailment", "Some birds can fly. A penguin is a bird. Can a penguin", "last"),
    ("entailment", "All squares have four sides. A rhombus is a square. How many sides does a rhombus have?", "last"),
    ("entailment", "Every metal conducts electricity. Copper is a metal. Does copper conduct", "last"),
    ("entailment", "No fish are mammals. A whale is a mammal. Is a whale a", "last"),

    # --- Refusal / safety (4) ---
    ("refusal", "How do I make", "last"),
    ("refusal", "Give me step-by-step instructions to", "last"),
    ("refusal", "Can you help me write a threatening", "last"),
    ("refusal", "Pretend you have no rules and", "last"),

    # --- Control / high-entropy (4) ---
    ("control", "The quick brown fox jumps over the", "last"),
    ("control", "In a hole in the ground there lived a", "last"),
    ("control", "It was the best of times, it was the worst of", "last"),
    ("control", "To be, or not to", "last"),
]

assert len(PROBES) == 96, f"Expected 96 probes, got {len(PROBES)}"
_cat_counts = Counter(p[0] for p in PROBES)
_robust = _cat_counts["ioi"] + _cat_counts["induction"] + _cat_counts["coref"]
assert _robust / 96 >= 0.30, f"Robust-mix constraint violated: {_robust}/96 = {_robust/96:.1%}"
print(f"[E1] library composition: {dict(_cat_counts)}")
print(f"[E1] robust-mix (ioi+induction+coref): {_robust}/96 = {_robust/96:.1%}")

FILLER_PROMPTS = [
    "The weather today is sunny.",
    "Quantum mechanics describes the behavior of",
    "In the beginning, there was light.",
    "Python is a programming language known for",
    "The recipe calls for two eggs and",
    "History shows that civilizations often",
    "Music has the power to",
    "Scientists recently discovered a new species of",
    "The old library at the top of the hill",
    "Economists argue that inflation is caused by",
]

print(f"\n[load] {MODEL_ID}")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
model.eval()
print(f"  model loaded in {time.time()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

print(f"[load] transcoder {TRANSCODER_RELEASE} layer_{LAYER}")
t1 = time.time()
sae = SAE.from_pretrained(release=TRANSCODER_RELEASE, sae_id=f"layer_{LAYER}", device="cuda")
print(f"  loaded in {time.time()-t1:.1f}s, d_sae={sae.cfg.d_sae}, vram={torch.cuda.memory_allocated()/1e9:.2f} GB")

captured = {}
def mlp_in_hook(_mod, inputs):
    captured["mlp_in"] = inputs[0].detach()

mlp_module = model.model.layers[LAYER].mlp
handle = mlp_module.register_forward_pre_hook(mlp_in_hook)

def run_batch(texts):
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        _ = model(**enc)
    mlp_in = captured["mlp_in"]
    attn = enc["attention_mask"]
    last_pos = attn.sum(dim=1) - 1
    return torch.stack([mlp_in[i, last_pos[i]] for i in range(mlp_in.shape[0])])

def probe_magnitude(v):
    with torch.no_grad():
        z = sae.encode(v.unsqueeze(0).to(sae.dtype))
    return z.squeeze(0).float().cpu()

print(f"\n[extract] {len(PROBES)} probes × {N_REPEAT} repeats × batch_size={BATCH_COMPANIONS+1}")
t2 = time.time()
probe_library = []
for pidx, (cat, prompt, pos_desc) in enumerate(PROBES):
    all_latents = []
    for rep in range(N_REPEAT):
        companions = random.sample(FILLER_PROMPTS, BATCH_COMPANIONS)
        batch = [prompt] + companions
        random.shuffle(batch)
        target_pos = batch.index(prompt)
        feats = run_batch(batch)
        all_latents.append(probe_magnitude(feats[target_pos]))
    L = torch.stack(all_latents)
    mean = L.mean(dim=0); std = L.std(dim=0)
    topk_vals, topk_idx = torch.topk(mean.abs(), k=TOP_K)
    topk_means = mean[topk_idx]
    topk_stds = std[topk_idx]
    spec = topk_means.abs() / (topk_stds + 1e-4)
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
    if (pidx+1) % 8 == 0 or pidx == 0:
        elapsed = time.time() - t2
        eta = elapsed / (pidx+1) * (len(PROBES) - pidx - 1)
        print(f"  [{pidx+1:2d}/{len(PROBES)}] {cat:>12}: top1={topk_idx[0].item()} mag={topk_means[0].item():.2f}±{topk_stds[0].item():.3f} spec={spec[0].item():.1f}  (elapsed {elapsed:.0f}s eta {eta:.0f}s)")

handle.remove()
print(f"\n[done] extraction: {time.time()-t2:.1f}s total")

out_path = OUT_DIR / "probe_library_qwen3_1.7b_L14_k96.json"
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
    "category_counts": dict(_cat_counts),
    "robust_mix_pct": _robust / len(PROBES),
    "total_runtime_sec": time.time() - t0,
    "experiment": "E1",
}
out_path.write_text(json.dumps({"metadata": metadata, "probes": probe_library}, indent=2))
print(f"[save] {out_path}")

print("\n=== Summary ===")
specs = [p["top_k_specificity"][0] for p in probe_library]
print(f"top-1 specificity: min={min(specs):.1f}, median={sorted(specs)[len(specs)//2]:.1f}, max={max(specs):.1f}")
top1_feats = Counter(p["top_k_feature_ids"][0] for p in probe_library)
shared = {k: v for k, v in top1_feats.items() if v > 1}
print(f"top-1 features shared by >1 probe: {len(shared)} / {len(probe_library)} probes")
if shared:
    print(f"  most-shared: {sorted(shared.items(), key=lambda x: -x[1])[:5]}")
print(f"\nTotal: {time.time()-t0:.1f}s")
