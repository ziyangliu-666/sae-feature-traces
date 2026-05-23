"""Recipe 3 (Round 6): extend the SVIP side-by-side from Qwen-only to both
backbones.

Reviewer feedback: the Round-5 SVIP comparison was narrow (3 Qwen3
attackers on one side-channel setup). Extending it to include Gemma
attackers answers "does the parallel-serve side-channel hold on a
second backbone as well?"

Protocol identical to 15_e11_svip_analysis.py, but we now add:
  - Qwen3: existing cheap/cross-family/adaptive_rank64 (3 attackers)
  - Gemma-2-2B: 4 same-family + 4 cross-family attackers from e12/e13
    + Gemma adaptive LoRA r=64 Stage A from e14 (if available)

For each: SVIP routes probe queries to the honest model -> joint-z ~
honest_median regardless of the attacker. Ours (commit-open) uses the
measured joint-z against the backbone's synthetic tau.

Output: results/recipe3_svip_two_backbone.json
"""
import json
from pathlib import Path

LOG = Path(__file__).parent.parent / "results"


def load(name):
    return json.loads((LOG / name).read_text())


def main():
    # --- Qwen3 side ---
    e2 = load("e2_separability.json")
    e3 = load("e3_v2_scored.json")
    e4 = load("e4_v2_library_aware.json")

    qwen_tau = e3["tau_honest_joint_99th_pct"]
    qwen_honest_median = e3["honest_joint_median"]
    e3_by_sub = {a["attacker"]: a for a in e3["per_attacker"]}

    qwen_attackers = [
        {
            "backbone": "qwen3-1.7b-L14",
            "name": "B. cheap (Qwen3-0.6B, LEARNED_LIN lift)",
            "short": "Qwen cheap",
            "ours_joint_z": e2["classifier"]["LEARNED_LIN"]["maha_mean"],
        },
        {
            "backbone": "qwen3-1.7b-L14",
            "name": "C. cross-family (Qwen2.5-1.5B linear fit)",
            "short": "Qwen cross-fam 2.5-1.5B",
            "ours_joint_z": e3_by_sub["Qwen/Qwen2.5-1.5B"]["joint_maha"],
        },
        {
            "backbone": "qwen3-1.7b-L14",
            "name": "C. cross-family (Phi-3.5-mini linear fit)",
            "short": "Qwen cross-fam Phi-3.5",
            "ours_joint_z": e3_by_sub["microsoft/Phi-3.5-mini-instruct"]["joint_maha"],
        },
        {
            "backbone": "qwen3-1.7b-L14",
            "name": "D. adaptive LoRA r64 joint-phi StageA",
            "short": "Qwen adaptive r64 StageA",
            "ours_joint_z": e4["results"]["stageA_pure_probe"]["joint_z"],
        },
    ]

    # --- Gemma side ---
    # The Gemma pilot thresholds use the raw Mahalanobis scale (tau_maha=57.99
    # from n_hon=8 with self-calibrated sigma). The paper standardises reporting
    # via the *synthetic joint-z tau* produced by bootstrap of normal draws
    # from (mu, sigma), which is ~0.82 for both backbones. We use the scored
    # tau from e13_gemma_scored.json (the synthetic 99th-pct on the Gemma noise
    # model).
    e12 = load("e12_gemma_pilot.json")
    e13s = load("e13_gemma_scored.json")
    gemma_tau = e13s["tau_honest_joint_99th_pct"]
    gemma_honest_median = e13s["honest_joint_median"]
    e13_by = {a["attacker"]: a for a in e13s["per_attacker"]}

    gemma_attackers = [
        {
            "backbone": "gemma2-2b-L12",
            "name": "E2-same-family (Gemma-2-2B-it native)",
            "short": "Gemma same-fam IT",
            "ours_joint_z": e12["classifier"]["NATIVE_IT"]["maha_mean"],
        },
        {
            "backbone": "gemma2-2b-L12",
            "name": "E2-cross-family (Pythia-1.4B, LEARNED_LIN)",
            "short": "Gemma cross-fam Pythia",
            "ours_joint_z": e12["classifier"]["LEARNED_LIN_PYTHIA"]["maha_mean"],
        },
    ]
    # Cross-family Gemma substitutes from e13 scored file (four models)
    for sub_key, short_label in [
        ("Qwen/Qwen2.5-1.5B", "Gemma cross-fam Qwen2.5"),
        ("EleutherAI/pythia-1.4b", "Gemma cross-fam Pythia-1.4B-e13"),
        ("microsoft/Phi-3.5-mini-instruct", "Gemma cross-fam Phi-3.5"),
        ("google/gemma-2-2b-it", "Gemma same-fam IT (e13 linear)"),
    ]:
        if sub_key in e13_by:
            gemma_attackers.append({
                "backbone": "gemma2-2b-L12",
                "name": f"C-style cross-family ({sub_key})",
                "short": short_label,
                "ours_joint_z": e13_by[sub_key]["joint_maha"],
            })

    # Optional: Gemma adaptive LoRA (Recipe 2 output)
    e14_path = LOG / "e14_gemma_adaptive_lora.json"
    if e14_path.exists():
        e14 = json.loads(e14_path.read_text())
        gemma_attackers.append({
            "backbone": "gemma2-2b-L12",
            "name": "D. Gemma adaptive LoRA r=64 joint-phi StageA",
            "short": "Gemma adaptive r64 StageA",
            "ours_joint_z": e14["results"]["stageA_pure_probe"]["joint_z"],
        })

    # --- Apply SVIP model: SVIP returns probe responses from M (honest),
    # so joint-z ~ honest median.
    for a in qwen_attackers:
        a["svip_joint_z"] = qwen_honest_median
        a["tau"] = qwen_tau
        a["svip_over_tau"] = a["svip_joint_z"] / qwen_tau
        a["ours_over_tau"] = a["ours_joint_z"] / qwen_tau
        a["svip_detected"] = a["svip_joint_z"] > qwen_tau
        a["ours_detected"] = a["ours_joint_z"] > qwen_tau

    for a in gemma_attackers:
        a["svip_joint_z"] = gemma_honest_median
        a["tau"] = gemma_tau
        a["svip_over_tau"] = a["svip_joint_z"] / gemma_tau
        a["ours_over_tau"] = a["ours_joint_z"] / gemma_tau
        a["svip_detected"] = a["svip_joint_z"] > gemma_tau
        a["ours_detected"] = a["ours_joint_z"] > gemma_tau

    all_attackers = qwen_attackers + gemma_attackers

    out = {
        "metadata": {
            "note": "Two-backbone SVIP parallel-serve vs commit-open",
            "qwen_tau_synthetic_p99": qwen_tau,
            "gemma_tau_synthetic_p99": gemma_tau,
            "qwen_honest_median": qwen_honest_median,
            "gemma_honest_median": gemma_honest_median,
            "n_qwen_attackers": len(qwen_attackers),
            "n_gemma_attackers": len(gemma_attackers),
            "n_total": len(all_attackers),
        },
        "qwen_attackers": qwen_attackers,
        "gemma_attackers": gemma_attackers,
        "summary": {
            "svip_miss_rate": sum(not a["svip_detected"] for a in all_attackers) / len(all_attackers),
            "ours_detect_rate": sum(a["ours_detected"] for a in all_attackers) / len(all_attackers),
            "n_svip_missed": sum(not a["svip_detected"] for a in all_attackers),
            "n_ours_detected": sum(a["ours_detected"] for a in all_attackers),
            "n_total": len(all_attackers),
        },
    }

    out_path = LOG / "recipe3_svip_two_backbone.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[save] {out_path}")

    def _print_block(label, group, tau):
        print(f"\n=== {label} (tau={tau:.3f}) ===")
        for a in group:
            sd = "MISS" if not a["svip_detected"] else "det"
            od = "DET" if a["ours_detected"] else "MISS"
            print(f"  {a['short']:35s}  SVIP={a['svip_joint_z']:>7.3f} ({a['svip_over_tau']:.2f}x, {sd:>4s})   Ours={a['ours_joint_z']:>8.2f} ({a['ours_over_tau']:.2f}x, {od})")

    _print_block("Qwen3-1.7B + transcoder L14", qwen_attackers, qwen_tau)
    _print_block("Gemma-2-2B + Gemma-Scope L12", gemma_attackers, gemma_tau)

    print("\n--- Summary ---")
    s = out["summary"]
    print(f"SVIP miss rate : {s['n_svip_missed']}/{s['n_total']} = {100*s['svip_miss_rate']:.0f}%")
    print(f"Ours detect    : {s['n_ours_detected']}/{s['n_total']} = {100*s['ours_detect_rate']:.0f}%")


if __name__ == "__main__":
    main()
