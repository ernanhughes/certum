#!/usr/bin/env python3
"""
Comprehensive analysis of hard-negative failures in hallucination energy gating.

Goal: Classify false accepts (hard negatives with energy < tau) into failure patterns
to guide augmentation strategy selection.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_entities(text: str) -> List[str]:
    """Simple entity extraction (replace with spaCy/NLP pipeline for production)"""
    # Dates
    dates = re.findall(r"\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", text)
    # Numbers/percentages
    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    # Proper nouns (crude heuristic: capitalized words not at sentence start)
    proper = re.findall(r"(?<!^)[A-Z][a-z]+", text)
    return dates + nums + proper


def detect_failure_patterns(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify failure mode based on claim/evidence characteristics.
    
    Returns pattern scores (higher = more likely this failure mode)
    """
    claim = ex["claim"]
    evidence = ex["evidence"]
    energy = ex["energy"]
    
    patterns = {
        "numerical_synthesis": 0.0,    # Claim adds precision not in evidence
        "temporal_inference": 0.0,     # Claim adds timeline not in evidence
        "causal_leap": 0.0,            # Claim adds causation not in evidence
        "entity_conflation": 0.0,      # Claim merges distinct entities
        "multi_fragment_synthesis": 0.0,  # Claim combines disjoint evidence fragments
        "qualifier_addition": 0.0,     # Claim adds "significantly", "dramatically", etc.
    }
    
    # 1. Numerical synthesis detection
    claim_nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", claim)
    ev_nums = []
    for ev in evidence:
        ev_nums.extend(re.findall(r"\b\d+(?:\.\d+)?%?\b", ev))
    
    if claim_nums and not ev_nums:
        patterns["numerical_synthesis"] = 0.8
    elif claim_nums and ev_nums:
        # Check if claim adds precision (evidence: "increased", claim: "increased 50%")
        if any("%" in c for c in claim_nums) and not any("%" in e for e in ev_nums):
            patterns["numerical_synthesis"] = 0.7
    
    # 2. Temporal inference detection
    temporal_words = ["before", "after", "during", "since", "until", "when", "year", "decade"]
    claim_has_temporal = any(w in claim.lower() for w in temporal_words)
    ev_has_temporal = any(any(w in ev.lower() for w in temporal_words) for ev in evidence)
    
    if claim_has_temporal and not ev_has_temporal:
        patterns["temporal_inference"] = 0.6
    
    # 3. Causal leap detection
    causal_words = ["caused", "led to", "resulted in", "triggered", "because", "therefore"]
    claim_has_causal = any(w in claim.lower() for w in causal_words)
    ev_has_causal = any(any(w in ev.lower() for w in causal_words) for ev in evidence)
    
    if claim_has_causal and not ev_has_causal:
        patterns["causal_leap"] = 0.7
    
    # 4. Entity conflation detection
    claim_ents = set(extract_entities(claim))
    ev_ents = set()
    for ev in evidence:
        ev_ents.update(extract_entities(ev))
    
    # If claim mentions entities not in evidence → potential conflation
    missing_ents = claim_ents - ev_ents
    if missing_ents:
        patterns["entity_conflation"] = min(0.5 + len(missing_ents) * 0.1, 0.9)
    
    # 5. Multi-fragment synthesis detection
    # Count how many evidence fragments contribute entities to the claim
    fragments_touched = 0
    for ev in evidence:
        ev_ents = set(extract_entities(ev))
        if claim_ents & ev_ents:  # Non-empty intersection
            fragments_touched += 1
    
    if fragments_touched >= 2 and len(evidence) >= 3:
        patterns["multi_fragment_synthesis"] = 0.6
    
    # 6. Qualifier addition detection
    intensifiers = ["significantly", "dramatically", "greatly", "substantially", "major", "key"]
    claim_has_intensifier = any(w in claim.lower() for w in intensifiers)
    
    if claim_has_intensifier:
        patterns["qualifier_addition"] = 0.5
    
    # Determine dominant pattern
    dominant = max(patterns.items(), key=lambda x: x[1])
    return {
        "energy": energy,
        "patterns": patterns,
        "dominant_pattern": dominant[0] if dominant[1] > 0.4 else "ambiguous",
        "dominant_score": dominant[1],
        "claim_entities": list(claim_ents),
        "evidence_entities": list(ev_ents),
        "fragments_touched": fragments_touched,
        "evidence_count": len(evidence),
    }


def analyze_false_accepts(
    neg_path: Path,
    tau: float,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Analyze false accepts in hard negatives (energy < tau despite being mismatched).
    
    Returns comprehensive diagnostics for augmentation strategy selection.
    """
    negatives = load_jsonl(neg_path)
    
    # Filter false accepts: negatives with energy < tau (should be rejected but weren't)
    false_accepts = [ex for ex in negatives if ex.get("energy", 1.0) < tau]
    
    print(f"📊 Analyzing {len(false_accepts)} false accepts (out of {len(negatives)} hard negatives)")
    print(f"   Threshold τ = {tau:.4f}")
    print(f"   False accept rate = {len(false_accepts)/len(negatives):.1%}")
    print()
    
    if not false_accepts:
        return {"summary": "No false accepts found", "patterns": {}, "examples": []}
    
    # Pattern distribution
    pattern_counts = Counter()
    pattern_energies = defaultdict(list)
    examples_by_pattern = defaultdict(list)
    
    for ex in tqdm(false_accepts[:500], desc="Classifying failure patterns"):
        analysis = detect_failure_patterns(ex)
        pattern = analysis["dominant_pattern"]
        pattern_counts[pattern] += 1
        pattern_energies[pattern].append(analysis["energy"])
        
        if len(examples_by_pattern[pattern]) < 3:  # Keep 3 examples per pattern
            examples_by_pattern[pattern].append({
                "claim": ex["claim"],
                "evidence_sample": ex["evidence"][0] if ex["evidence"] else "",
                "energy": ex["energy"],
                "analysis": analysis,
            })
    
    # Build report
    total = sum(pattern_counts.values())
    pattern_distribution = {
        pat: {
            "count": cnt,
            "percent": cnt / total * 100,
            "mean_energy": np.mean(pattern_energies[pat]),
            "examples": examples_by_pattern[pat],
        }
        for pat, cnt in pattern_counts.most_common()
    }
    
    # Energy distribution analysis
    energies = [ex["energy"] for ex in false_accepts]
    energy_bins = np.histogram(energies, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    report = {
        "summary": {
            "total_false_accepts": len(false_accepts),
            "false_accept_rate": len(false_accepts) / len(negatives),
            "tau": tau,
            "mean_energy": np.mean(energies),
            "median_energy": np.median(energies),
            "energy_std": np.std(energies),
        },
        "pattern_distribution": pattern_distribution,
        "energy_histogram": {
            "bins": energy_bins[1].tolist(),
            "counts": energy_bins[0].tolist(),
        },
        "recommendations": generate_recommendations(pattern_distribution),
    }
    
    return report


def generate_recommendations(pattern_dist: Dict[str, Any]) -> List[str]:
    """
    Generate augmentation strategy recommendations based on failure pattern distribution.
    """
    recommendations = []
    
    # Check for dominant patterns
    dominant_patterns = [
        (pat, stats["percent"]) 
        for pat, stats in pattern_dist.items() 
        if stats["percent"] > 25
    ]
    
    if not dominant_patterns:
        recommendations.append(
            "⚠️ No dominant failure pattern detected. Consider max-clause energy augmentation "
            "(detects buried unsupported constituents regardless of pattern)."
        )
        return recommendations
    
    # Pattern-specific recommendations
    for pattern, pct in dominant_patterns:
        if pattern == "numerical_synthesis" and pct > 30:
            recommendations.append(
                f"✅ NUMERICAL SYNTHESIS ({pct:.0f}% of failures): "
                "Implement numerical grounding check—reject claims adding precision (%) "
                "not present in evidence. High ROI augmentation."
            )
        elif pattern == "causal_leap" and pct > 30:
            recommendations.append(
                f"✅ CAUSAL LEAP ({pct:.0f}% of failures): "
                "Add causal language detector—flag claims with 'caused/led to' when evidence "
                "lacks causal markers. Medium ROI."
            )
        elif pattern == "multi_fragment_synthesis" and pct > 25:
            recommendations.append(
                f"✅ MULTI-FRAGMENT SYNTHESIS ({pct:.0f}% of failures): "
                "Implement evidence fragmentation analysis—reject claims touching >2 disjoint "
                "evidence fragments without explicit bridging language. HIGH ROI for FEVEROUS."
            )
        elif pattern == "entity_conflation" and pct > 25:
            recommendations.append(
                f"✅ ENTITY CONFLATION ({pct:.0f}% of failures): "
                "Augment with entity grounding score—require each claim entity to have direct "
                "evidence anchor. Medium ROI."
            )
    
    # Universal recommendation
    recommendations.append(
        "\n🔧 UNIVERSAL STRATEGY (works regardless of pattern):\n"
        "   Implement MAX-CLAUSE ENERGY:\n"
        "   - Split claim into semantic constituents (clauses/entities)\n"
        "   - Compute energy per constituent\n"
        "   - Use MAX constituent energy as gate signal (not mean)\n"
        "   Why it works: Catches buried unsupported clauses even when overall claim energy is low."
    )
    
    return recommendations


def print_report(report: Dict[str, Any], out_path: Path):
    """Pretty-print analysis report to console and file"""
    lines = []
    lines.append("=" * 80)
    lines.append("HARD NEGATIVE FAILURE ANALYSIS")
    lines.append("=" * 80)
    lines.append()
    
    # Summary
    summ = report["summary"]
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total false accepts: {summ['total_false_accepts']}")
    lines.append(f"False accept rate:   {summ['false_accept_rate']:.1%}")
    lines.append(f"Threshold τ:         {summ['tau']:.4f}")
    lines.append(f"Mean energy:         {summ['mean_energy']:.3f}")
    lines.append(f"Median energy:       {summ['median_energy']:.3f}")
    lines.append()
    
    # Pattern distribution
    lines.append("FAILURE PATTERNS (dominant)")
    lines.append("-" * 80)
    for pat, stats in report["pattern_distribution"].items():
        lines.append(f"\n{pat.replace('_', ' ').title()}")
        lines.append(f"  Occurrence:  {stats['count']} ({stats['percent']:.1f}%)")
        lines.append(f"  Mean energy: {stats['mean_energy']:.3f}")
        lines.append(f"  Examples:")
        for ex in stats["examples"][:2]:
            claim = ex["claim"][:100] + "..." if len(ex["claim"]) > 100 else ex["claim"]
            ev = ex["evidence_sample"][:80] + "..." if len(ex["evidence_sample"]) > 80 else ex["evidence_sample"]
            lines.append(f"    • Energy {ex['energy']:.3f}: \"{claim}\"")
            lines.append(f"      Evidence: \"{ev}\"")
    
    # Recommendations
    lines.append("\n" + "=" * 80)
    lines.append("AUGMENTATION STRATEGY RECOMMENDATIONS")
    lines.append("=" * 80)
    for rec in report["recommendations"]:
        lines.append(rec)
    
    lines.append("\n" + "=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append("""
Your gate fails on hard negatives NOT because the metric is broken—but because
plausible synthesis creates directional alignment (low residual energy) despite
factual non-support.

This is a FUNDAMENTAL LIMITATION of subspace projection:
  → It measures geometric alignment, not logical entailment
  → Plausible-but-wrong claims often point in similar semantic directions

Your architecture's strength is POLICY AUGMENTATION:
  → Don't try to "fix" the energy metric
  → Augment policy with lightweight heuristics that detect synthesis patterns
  → Combine signals: (energy < τ) AND (no synthesis patterns) → ACCEPT

This is your contribution: deterministic policy enforcement that combines multiple
weak signals into a strong gate.
""")
    
    # Write to file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    
    # Print to console
    print("\n".join(lines))
    print(f"\n✅ Full report written to: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Analyze hard-negative failures in hallucination energy gating")
    ap.add_argument("--neg_path", type=Path, required=True,
                    help="Path to neg_hard_mined.jsonl scored negatives")
    ap.add_argument("--tau", type=float, required=True,
                    help="Threshold τ from calibration (e.g., 0.126 from your report)")
    ap.add_argument("--out_report", type=Path, 
                    default=Path("artifacts/hard_failure_analysis.md"),
                    help="Output report path")
    args = ap.parse_args()
    
    report = analyze_false_accepts(args.neg_path, args.tau)
    print_report(report, args.out_report)


if __name__ == "__main__":
    main()