#!/usr/bin/env python3
"""
validate_gate_reports_v2.py

Schema-aware sanity checks for NCEC reports.
This validates *calibration logic*, not embeddings.
"""

import json
from pathlib import Path

ARTIFACTS = Path("artifacts")

def load_report(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def check_calibration(report):
    p = report["params"]
    cal = report["stats"]["cal"]
    eval_ = report["stats"]["eval"]

    print(f"\nPolicy regime: {p['regime']} | neg_mode: {p['neg_mode']}")
    print(f"Target FAR: {p['far_target']:.3%}")
    print(f"τ_cal: {p['tau_cal']:.4f}")

    # FAR sanity
    if abs(cal["FAR"] - p["far_target"]) > 1e-6:
        print("⚠️  CAL FAR does not match target!")
    else:
        print("✓ CAL FAR matches target")

    if eval_["FAR"] > p["far_target"] * 1.5:
        print("⚠️  EVAL FAR drift too high!")
    else:
        print("✓ EVAL FAR stable")

def check_separation(report):
    p = report["params"]
    tau = p["tau_cal"]
    far = p["far_target"]

    cal = report["stats"]["cal"]
    ev  = report["stats"]["eval"]

    def fmt_split(name, s):
        pos = s["pos"]
        neg = s["neg"]

        # Use quantiles, not min/max
        neg_q = neg.get("p01") if abs(far - 0.01) < 1e-9 else None

        margin_pos = tau - pos["p99"]          # + = accepts almost all POS
        margin_neg = neg["p50"] - tau          # + = rejects most NEG comfortably

        print(f"\n[{name} SEPARATION]")
        print(f"pos p99: {pos['p99']:.4f}")
        print(f"neg p50: {neg['p50']:.4f}")
        if neg_q is not None:
            print(f"neg p01: {neg_q:.4f}  (should be ~ τ on CAL by design)")
        print(f"τ_cal : {tau:.4f}")
        print(f"margin_pos (τ - pos.p99): {margin_pos:+.4f}")
        print(f"margin_neg (neg.p50 - τ): {margin_neg:+.4f}")

        # Interpret
        if margin_pos < 0:
            print("⚠️  τ is inside POS tail → expect noticeable false rejects")
        else:
            print("✓ τ sits beyond POS p99 → low false rejects")

        if margin_neg < 0:
            print("⚠️  τ is above NEG median → expect many false accepts (bad)")
        else:
            print("✓ τ is below NEG median → rejects most negatives")

    fmt_split("CAL", cal)
    fmt_split("EVAL", ev)

def check_hard_negative_behavior(report):
    neg_mode = report["params"]["neg_mode"]
    cal = report["stats"]["cal"]

    if neg_mode == "hard_mined":
        print("\n[HARD-NEGATIVE DIAGNOSTIC]")
        print(f"AUC: {cal['AUC']:.3f}")
        print(f"TPR@FAR: {cal['TPR']:.3%}")

        if cal["TPR"] < 0.1:
            print("✓ Expected collapse: hard negatives are genuinely hard")
        else:
            print("⚠️  Hard negatives separating too well — check mining")

def main():
    for p in sorted(ARTIFACTS.glob("feverous_negcal_*.json")):
        print("\n" + "=" * 80)
        print(f"VALIDATING {p.name}")

        rpt = load_report(p)
        check_calibration(rpt)
        check_separation(rpt)
        check_hard_negative_behavior(rpt)

if __name__ == "__main__":
    main()
