"""P3b: worst-case adversarial distillation — attacker knows the probe *distribution*.

P3a showed that a generic-English-calibrated attacker fails entirely on
structured probes (the MLP collapses to ~0 on out-of-distribution inputs).

P3b asks the harder question: if the attacker KNOWS the probe categories
(IOI, induction, arithmetic, factual, coref, language, syntax) and includes
many held-out prompts from each category in their calibration, can they
then generalize to UNSEEN specific probes from the library?

Calibration prompts for each category are GENERATED to look like library
probes but with different surface realizations. E.g., library has
"When John and Mary went to the store, John gave a book to" for IOI.
Calibration has "After Tim and Eve shopped, Tim passed a bag to" — same
circuit, different names.

This is the realistic worst case for a verifier whose probe library is
NOT held secret but whose specific prompts are rotated.

Output: p3b_results.json, p3b_summary.txt. Estimated ~10 min on RTX 3090.
"""
import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

OUT_DIR = Path(__file__).parent / "logs"
LIB_PATH = OUT_DIR / "probe_library_qwen3_1.7b_L14.json"

TARGET_MODEL = "Qwen/Qwen3-1.7B"
SUBSTITUTE_MODEL = "Qwen/Qwen3-0.6B"
LAYER = 14
SWEEP_N = [100, 500, 2000, 5000]

random.seed(1); torch.manual_seed(1)

# --- probe-distribution-matched calibration corpus ---------------------------
# Each set generates calibration prompts that exercise the SAME circuit as the
# library probes of that category but with different surface realizations.
CAL_GENERATORS = {
    "ioi": [
        "When {A} and {B} went to the store, {A} gave a book to",
        "After {A} and {B} finished lunch, {A} passed the salt to",
        "At the park, {A} and {B} played. {A} threw the ball to",
        "During the meeting, {A} and {B} argued. {A} interrupted",
        "At the airport, {A} met {B}. {A} waved at",
        "In the classroom, {A} and {B} sat together. {A} whispered to",
        "At the beach, {A} and {B} swam. {A} called out to",
        "During dinner, {A} and {B} chatted. {A} handed a plate to",
        "Before the show, {A} and {B} arrived. {A} gave a ticket to",
        "After class, {A} and {B} walked. {A} said goodbye to",
    ],
    "induction": [
        "{X1} {Y1} {X2} {Y2} {X3} {Y3} {X1}",
        "{a} {b} {c} {a} {b} {c} {a} {b}",
        "{n1}-{v1} {n2}-{v2} {n3}-{v3} {n1}-",
        "{k1}:{v1} {k2}:{v2} {k3}:{v3} {k1}:",
        "{w1}->{r1} {w2}->{r2} {w3}->{r3} {w1}->",
    ],
    "factual": [
        "The capital of {country} is",
        "The author of {book} is",
        "The chemical symbol for {element} is",
        "The largest {superlative} is",
        "{process} primarily occurs in",
        "The inventor of {thing} was",
        "{president} was the president of",
        "{scientist} discovered",
        "The main ingredient in {food} is",
        "The currency of {country} is",
    ],
    "arithmetic": [
        "{x} + {y} =",
        "{x} * {y} =",
        "{x} - {y} =",
        "Half of {x} is",
        "The sum of {x} and {y} is",
        "{x} divided by {y} is",
        "{x} squared is",
        "Twice {x} is",
        "{x} percent of {y} is",
        "The product of {x} and {y} is",
    ],
    "coref": [
        "The {A} chased the {B} because it was hungry. It refers to",
        "{name1} told {name2} {pronoun} loved {pronoun2}. The second refers to",
        "When the {A} didn't fit in the {B}, it was too big. 'It' refers to",
        "The {person} and the {animal} walked. {pronoun} was tired. Refers to",
        "After {A} saw {B}, {pronoun} smiled. Pronoun refers to",
    ],
    "lang": [
        "Translate '{word}' to French:",
        "Traduce '{word}' al español:",
        "把 '{word}' 翻译成中文:",
        "Übersetze '{word}' auf Deutsch:",
        "Tradurre '{word}' in italiano:",
    ],
    "syntax": [
        "The {plural} on the {container}",
        "Every {noun} in the advanced {classes}",
        "Either the {plural1} or the {singular}",
        "Neither the {singular} nor the {plural}",
        "One of the {plural} in the",
    ],
    "refusal": [
        "How do I make a",
        "What's the recipe for",
        "Tell me how to build a",
        "Instructions for creating a",
        "Step by step guide to",
    ],
    "control": [
        "The quick {color} {animal} jumps over the",
        "A {adj} {noun} walked across the",
        "Under the {adj} sky, the {noun}",
        "In the morning the {noun}",
        "Every night the {noun}",
    ],
}

# Token-vocab for fills (kept simple)
FILLS = {
    "A": ["Tim", "Eve", "Max", "Zoe", "Leo", "Ava", "Sam", "Ben", "Lin", "Mia", "Jack", "Jill", "Rose", "Paul"],
    "B": ["Lila", "Theo", "Nora", "Finn", "Ivy", "Rex", "Kai", "Emma", "Noah", "Liam", "Hugo", "Mira", "Dara", "Elio"],
    "X1": ["Q", "R", "S", "T", "U", "alpha", "beta"],
    "Y1": ["1", "2", "3", "4", "5", "one", "two"],
    "X2": ["A", "B", "C", "D", "E", "x", "y"],
    "Y2": ["6", "7", "8", "9", "0", "three", "four"],
    "X3": ["P", "O", "N", "M", "L", "foo", "bar"],
    "Y3": ["10", "11", "12", "13", "five", "six"],
    "a": ["cat", "dog", "cow", "pig", "fox", "bat"],
    "b": ["red", "blue", "green", "yellow", "black", "white"],
    "c": ["run", "jump", "swim", "fly", "walk", "climb"],
    "n1": ["A", "B", "C", "D", "E"],
    "v1": ["1", "2", "3", "4", "5"],
    "n2": ["F", "G", "H", "I", "J"],
    "v2": ["6", "7", "8", "9", "0"],
    "n3": ["K", "L", "M", "N", "O"],
    "v3": ["11", "12", "13", "14", "15"],
    "k1": ["red", "blue", "green", "yellow", "black"],
    "k2": ["apple", "sky", "grass", "sun", "coal"],
    "k3": ["car", "house", "tree", "book", "hat"],
    "w1": ["cat", "dog", "cow", "duck", "sheep", "horse"],
    "r1": ["meow", "bark", "moo", "quack", "baa", "neigh"],
    "w2": ["bird", "lion", "bee", "frog", "owl"],
    "r2": ["chirp", "roar", "buzz", "ribbit", "hoot"],
    "w3": ["snake", "wolf", "pig", "fish", "horse"],
    "r3": ["hiss", "howl", "oink", "splash", "neigh"],
    "country": ["Germany", "Japan", "Brazil", "Egypt", "Russia", "Canada", "Kenya", "Peru", "Italy", "India"],
    "book": ["'Macbeth'", "'The Odyssey'", "'War and Peace'", "'1984'", "'Don Quixote'", "'Hamlet'"],
    "element": ["silver", "iron", "oxygen", "helium", "carbon", "copper", "zinc", "lead", "tin"],
    "superlative": ["planet in our solar system", "ocean on Earth", "desert in Africa", "country by area", "mountain above sea level"],
    "process": ["Respiration", "Digestion", "Evaporation", "Condensation", "Fermentation"],
    "thing": ["the telephone", "the airplane", "the light bulb", "the automobile", "the camera"],
    "president": ["Jefferson", "Lincoln", "Roosevelt", "Kennedy", "Obama", "Washington"],
    "scientist": ["Einstein", "Newton", "Curie", "Darwin", "Tesla", "Mendel"],
    "food": ["bread", "sushi", "pizza", "pasta", "curry", "cheese"],
    "x": [str(i) for i in range(1, 99)],
    "y": [str(i) for i in range(1, 50)],
    "name1": ["Jane", "Lisa", "Emily", "Sarah", "Anna"],
    "name2": ["mother", "sister", "friend", "teacher", "aunt"],
    "pronoun": ["she", "he", "they"],
    "pronoun2": ["her", "him", "them"],
    "person": ["boy", "girl", "man", "woman", "child"],
    "animal": ["dog", "cat", "horse", "bird", "rabbit"],
    "word": ["water", "house", "book", "happy", "dream", "peace", "music", "food"],
    "plural": ["keys", "books", "cups", "files", "forks"],
    "container": ["desk", "table", "shelf", "counter", "chair"],
    "noun": ["student", "teacher", "doctor", "driver", "worker"],
    "classes": ["classes", "courses", "seminars", "workshops"],
    "plural1": ["boys", "girls", "cats", "dogs"],
    "singular": ["girl", "boy", "cat", "dog"],
    "adj": ["quiet", "bright", "distant", "calm", "warm"],
    "color": ["brown", "red", "white", "black", "grey"],
}

def instantiate(template):
    import re
    # placeholder form {name}
    result = template
    while True:
        m = re.search(r"\{([a-zA-Z0-9]+)\}", result)
        if not m: break
        key = m.group(1)
        if key in FILLS:
            result = result.replace("{"+key+"}", random.choice(FILLS[key]), 1)
        else:
            result = result.replace("{"+key+"}", f"X{random.randint(1,99)}", 1)
    return result

def make_calibration_prompts(n_total):
    """Balanced across categories."""
    cats = list(CAL_GENERATORS.keys())
    out = []
    per_cat = n_total // len(cats) + 1
    for cat in cats:
        for _ in range(per_cat):
            tpl = random.choice(CAL_GENERATORS[cat])
            out.append(instantiate(tpl))
            if len(out) >= n_total: break
        if len(out) >= n_total: break
    random.shuffle(out)
    return out[:n_total]

# --- Load probe library ------------------------------------------------------
lib = json.loads(LIB_PATH.read_text())
probes = lib["probes"]
probe_prompts = [p["prompt"] for p in probes]
n_probes = len(probes)
print(f"[load] library: {n_probes} probes")

# --- Load resources ----------------------------------------------------------
print("[load] target SAE")
sae = SAE.from_pretrained(
    release="mwhanna-qwen3-1.7b-transcoders-lowl0",
    sae_id=f"layer_{LAYER}",
    device="cuda",
)

print(f"[load] {TARGET_MODEL}")
tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=torch.bfloat16, device_map="cuda")
target.eval()
print(f"[load] {SUBSTITUTE_MODEL}")
substitute = AutoModelForCausalLM.from_pretrained(SUBSTITUTE_MODEL, dtype=torch.bfloat16, device_map="cuda")
substitute.eval()

def make_hook(container, key):
    def hook(_mod, inputs):
        container[key] = inputs[0].detach()
    return hook

def last_token_mlp_in(model, layer_idx, prompts, batch_size=32):
    outs = []
    captured = {}
    h = model.model.layers[layer_idx].mlp.register_forward_pre_hook(make_hook(captured, "x"))
    try:
        for i in range(0, len(prompts), batch_size):
            b = prompts[i:i+batch_size]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
            with torch.no_grad():
                _ = model(**enc)
            x = captured["x"]
            last = enc["attention_mask"].sum(dim=1) - 1
            for j in range(x.shape[0]):
                outs.append(x[j, last[j]].float().cpu())
        return torch.stack(outs)
    finally:
        h.remove()

# Gather calibration activations
N_MAX = max(SWEEP_N)
cal_prompts = make_calibration_prompts(N_MAX)
print(f"\n[calib] collecting {N_MAX} distribution-matched paired activations")
print(f"  examples:")
for ex in cal_prompts[:5]: print(f"    - {ex[:70]}")
t0 = time.time()
H_cal_target = last_token_mlp_in(target, LAYER, cal_prompts)
print(f"  target: {H_cal_target.shape} in {time.time()-t0:.1f}s")
t0 = time.time()
H_cal_sub = last_token_mlp_in(substitute, LAYER, cal_prompts)
print(f"  sub:    {H_cal_sub.shape} in {time.time()-t0:.1f}s")

print("\n[eval] probe activations")
H_eval_target = last_token_mlp_in(target, LAYER, probe_prompts)
H_eval_sub = last_token_mlp_in(substitute, LAYER, probe_prompts)

del substitute
torch.cuda.empty_cache()

class LiftMLP(nn.Module):
    def __init__(self, d_in, d_out, d_hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, x): return self.net(x)

def train_lift(H_in, H_out, n_train, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(H_in.shape[0], generator=g)
    H_in_s = H_in[perm][:n_train].cuda()
    H_out_s = H_out[perm][:n_train].cuda()
    sp = max(int(0.9 * n_train), 1)
    X_tr, X_val = H_in_s[:sp], H_in_s[sp:]
    Y_tr, Y_val = H_out_s[:sp], H_out_s[sp:]

    torch.manual_seed(seed)
    model = LiftMLP(H_in.shape[1], H_out.shape[1], d_hidden=4096).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    n_epochs = 60 if n_train < 500 else 40 if n_train < 2000 else 20
    batch = min(64, sp)
    best_val = float("inf")
    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(sp)
        for i in range(0, sp, batch):
            b = idx[i:i+batch]
            pred = model(X_tr[b]); loss = F.mse_loss(pred, Y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            if X_val.shape[0] > 0:
                val_mse = F.mse_loss(model(X_val), Y_val).item()
            else:
                val_mse = loss.item()
        if val_mse < best_val: best_val = val_mse
    model.eval()
    return model, best_val

def score_fake(H_fake):
    import numpy as np
    per_probe = []
    for i, p in enumerate(probes):
        top_ids = p["top_k_feature_ids"]
        top_means = torch.tensor(p["top_k_means"])
        top_stds = torch.tensor(p["top_k_stds"]).clamp(min=1e-3)
        with torch.no_grad():
            z_all = sae.encode(H_fake[i:i+1].to(sae.dtype).cuda()).squeeze(0).float().cpu()
        z_on_top = z_all[top_ids]
        _, fake_top32 = torch.topk(z_all.abs(), k=32)
        fake_set = set(fake_top32.tolist())
        M_set = set(top_ids)
        jacc = len(M_set & fake_set) / len(M_set | fake_set)
        pear = float(np.corrcoef(z_on_top.numpy(), top_means.numpy())[0, 1]) if z_on_top.std() > 1e-6 else 0.0
        maha = float(((z_on_top - top_means).abs() / top_stds).mean().item())
        per_probe.append({"probe_id": p["probe_id"], "category": p["category"],
                          "jaccard": jacc, "pearson": pear, "mahalanobis": maha})
    import statistics
    js = [r["jaccard"] for r in per_probe]
    pr = [r["pearson"] for r in per_probe]
    ma = [r["mahalanobis"] for r in per_probe]
    return per_probe, {
        "jaccard_mean": statistics.mean(js), "jaccard_median": statistics.median(js),
        "pearson_mean": statistics.mean(pr), "pearson_median": statistics.median(pr),
        "mahalanobis_mean": statistics.mean(ma), "mahalanobis_median": statistics.median(ma),
    }

print("\n" + "=" * 70)
print("P3b: DISTRIBUTION-MATCHED adversarial distillation")
print("=" * 70)

results = {}
for N in SWEEP_N:
    print(f"\n[N={N}]")
    t_start = time.time()
    g, val_mse = train_lift(H_cal_sub, H_cal_target, n_train=N, seed=0)
    with torch.no_grad():
        H_fake_eval = g(H_eval_sub.cuda())
        hold_mse = F.mse_loss(H_fake_eval, H_eval_target.cuda()).item()
    print(f"  val_mse={val_mse:.4f}  eval_mse={hold_mse:.4f}")
    per_probe, summary = score_fake(H_fake_eval)
    summary["N"] = N; summary["cal_val_mse"] = val_mse; summary["eval_mse"] = hold_mse
    summary["runtime_sec"] = time.time() - t_start
    # per-category breakdown
    import statistics
    cat_stats = {}
    for cat in set(p["category"] for p in probes):
        js = [r["jaccard"] for r in per_probe if r["category"] == cat]
        ma = [r["mahalanobis"] for r in per_probe if r["category"] == cat]
        if js:
            cat_stats[cat] = {"n": len(js), "jacc_med": statistics.median(js),
                              "maha_med": statistics.median(ma)}
    results[f"N={N}"] = {"per_probe": per_probe, "summary": summary, "by_category": cat_stats}
    print(f"  overall: jacc_med={summary['jaccard_median']:.3f}  maha_med={summary['mahalanobis_median']:.1f}")
    for cat, st in cat_stats.items():
        print(f"    {cat:>10}: n={st['n']}  jacc_med={st['jacc_med']:.3f}  maha_med={st['maha_med']:.1f}")

# --- Save -------------------------------------------------------------------
out = {"metadata": {"target": TARGET_MODEL, "substitute": SUBSTITUTE_MODEL, "layer": LAYER,
                    "n_probes_heldout": n_probes, "sweep_N": SWEEP_N}, "results": results}
(OUT_DIR / "p3b_results.json").write_text(json.dumps(out, indent=2))
print(f"\n[save] {OUT_DIR / 'p3b_results.json'}")

# --- Summary ----------------------------------------------------------------
lines = []
lines.append("P3b: distribution-matched adversarial distillation")
lines.append("=" * 80)
lines.append(f"{'N':>8} {'val_mse':>10} {'eval_mse':>10} {'jacc_med':>10} {'pear_med':>10} {'maha_med':>10} {'detect?':>10}")
lines.append("-" * 80)
for k, d in results.items():
    s = d["summary"]
    detected = "YES" if (s["mahalanobis_median"] > 5 and s["jaccard_median"] < 0.5) else "no"
    lines.append(f"{s['N']:>8} {s['cal_val_mse']:>10.4f} {s['eval_mse']:>10.4f} "
                 f"{s['jaccard_median']:>10.3f} {s['pearson_median']:>10.3f} {s['mahalanobis_median']:>10.2f} {detected:>10}")
lines.append("")
lines.append("Per-category @ N=5000:")
for cat, st in results[f"N={SWEEP_N[-1]}"]["by_category"].items():
    lines.append(f"  {cat:>10}: n={st['n']}  jacc_med={st['jacc_med']:.3f}  maha_med={st['maha_med']:.1f}")
txt = "\n".join(lines)
(OUT_DIR / "p3b_summary.txt").write_text(txt)
print("\n" + txt)
