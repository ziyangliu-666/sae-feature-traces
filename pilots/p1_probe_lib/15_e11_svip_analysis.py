"""E11: SVIP parallel-serve vs. commit-open (ours) — pure-local analysis.

Builds a side-by-side comparison from already-logged experiments:

  SVIP parallel-serve model: the attacker serves M' but *routes probe queries
  to M* (honest model). Probe answers are therefore drawn from the honest
  distribution -> joint-z ~ honest median, well below tau.

  Ours (commit-open): the attacker must commit an SAE trace from the served
  model M'. Verifier opens random positions of that trace and scores them
  against the probe library. Scores are the joint-z values already measured
  in E2/E3/E4.

Output: logs/e11_svip_vs_ours.json
"""
import json
from pathlib import Path

LOG = Path(__file__).parent / "logs"


def load(name):
    return json.loads((LOG / name).read_text())


def main():
    e2 = load("e2_separability.json")
    e3 = load("e3_v2_scored.json")
    e4 = load("e4_v2_library_aware.json")

    tau = e3["tau_honest_joint_99th_pct"]
    honest_median = e3["honest_joint_median"]

    # SVIP parallel-serve: probe responses come from M (honest). Under the
    # honest null we expect joint-z ~ honest_median with fluctuations
    # bounded by tau_{99pct}. We report a single representative point.
    svip_joint = honest_median

    e2_lin = e2["classifier"]["LEARNED_LIN"]

    e3_by_sub = {a["attacker"]: a for a in e3["per_attacker"]}

    def j(atk):
        return e3_by_sub[atk]["joint_maha"]

    attackers = [
        {
            "name": "B. cheap (Qwen3-0.6B, LEARNED_LIN lift)",
            "short": "cheap",
            "source": "E2 LEARNED_LIN (per-probe maha mean)",
            "svip_joint_z": svip_joint,
            "ours_joint_z": e2_lin["maha_mean"],
            "metric": "per-probe mean",
        },
        {
            "name": "C. cross-family (Qwen2.5-1.5B, linear fit)",
            "short": "cross-family (Qwen2.5-1.5B)",
            "source": "E3 linear fit on 96 probes",
            "svip_joint_z": svip_joint,
            "ours_joint_z": j("Qwen/Qwen2.5-1.5B"),
            "metric": "joint-z",
        },
        {
            "name": "C. cross-family (Phi-3.5-mini)",
            "short": "cross-family (Phi-3.5)",
            "source": "E3 linear fit on 96 probes",
            "svip_joint_z": svip_joint,
            "ours_joint_z": j("microsoft/Phi-3.5-mini-instruct"),
            "metric": "joint-z",
        },
        {
            "name": "D. adaptive LoRA r64 joint-phi (stageA pure-probe)",
            "short": "adaptive LoRA r64",
            "source": "E4 v2 stageA joint_z",
            "svip_joint_z": svip_joint,
            "ours_joint_z": e4["results"]["stageA_pure_probe"]["joint_z"],
            "metric": "joint-z",
        },
    ]

    for a in attackers:
        a["ours_over_tau"] = a["ours_joint_z"] / tau
        a["svip_over_tau"] = a["svip_joint_z"] / tau
        a["svip_detected_at_fpr1pct"] = a["svip_joint_z"] > tau
        a["ours_detected_at_fpr1pct"] = a["ours_joint_z"] > tau

    out = {
        "metadata": {
            "backbone": "qwen3-1.7b-L14",
            "note": "SVIP=parallel-serve side-channel. Ours=commit-open over Merkle trace.",
            "honest_joint_median": honest_median,
            "tau_honest_joint_99th_pct": tau,
        },
        "attackers": attackers,
        "summary": {
            "svip_miss_rate": sum(not a["svip_detected_at_fpr1pct"] for a in attackers) / len(attackers),
            "ours_detect_rate": sum(a["ours_detected_at_fpr1pct"] for a in attackers) / len(attackers),
        },
    }

    out_path = LOG / "e11_svip_vs_ours.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"tau = {tau:.3f}")
    for a in attackers:
        print(f"  {a['short']:35s}  SVIP={a['svip_joint_z']:.2f} ({a['svip_over_tau']:.2f}x tau)   Ours={a['ours_joint_z']:.2f} ({a['ours_over_tau']:.2f}x tau)")
    print(f"\nSVIP miss rate = {out['summary']['svip_miss_rate']*100:.0f}%")
    print(f"Ours detect rate = {out['summary']['ours_detect_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
