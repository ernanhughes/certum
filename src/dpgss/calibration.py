from typing import List, Dict, Optional
import numpy as np
from .gate import VerifiabilityGate

class AdaptiveCalibrator:
    """
    Learns percentile thresholds from data distribution.
    Replaces hand-tuned thresholds with data-calibrated policy.
    """
    
    def __init__(self, gate: VerifiabilityGate):
        self.gate = gate
    
    def run_sweep(
        self,
        claims: List[str],
        evidence_sets: List[List[str]],
        percentiles: List[int] = [1, 5, 10, 20, 30],
        oracle_claims: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Returns: {
            "tau_by_percentile": {1: 0.33, 5: 0.39, ...},
            "energy_gaps": [...],  # Raw distribution for analysis
            "acceptance_rates": {"P1": 0.01, "P5": 0.05, ...}
        }
        """
        if len(claims) != len(evidence_sets):
            raise ValueError("Claims and evidence sets must align")
        
        # Compute energy gaps for all samples
        gaps = []
        for i, (claim, evidence) in enumerate(zip(claims, evidence_sets)):
            oc = oracle_claims[i] if oracle_claims else evidence[0] if evidence else None
            if not evidence or not oc:
                continue
            
            # Use fixed policy just to get energy values (not decisions)
            from .policy import FixedThresholdPolicy
            dummy_policy = FixedThresholdPolicy(1.0)  # Always accept for measurement
            
            result = self.gate.evaluate(claim, evidence, dummy_policy, oracle_claim=oc)
            if result.oracle_calibration and result.oracle_calibration.is_valid:
                gaps.append(result.oracle_calibration.energy_gap)
        
        if len(gaps) < 10:
            raise ValueError(f"Insufficient valid samples for calibration: {len(gaps)}")
        
        # Compute thresholds
        tau_by_percentile = {
            p: float(np.percentile(gaps, p)) for p in percentiles
        }
        
        # Compute actual acceptance rates (sanity check against linear scaling)
        acceptance_rates = {}
        for p in percentiles:
            tau = tau_by_percentile[p]
            accepts = sum(1 for g in gaps if g <= tau)
            acceptance_rates[f"P{p}"] = accepts / len(gaps)
        
        return {
            "tau_by_percentile": tau_by_percentile,
            "energy_gaps": gaps,
            "acceptance_rates": acceptance_rates,
            "sample_count": len(gaps),
            "oracle_validity_rate": sum(1 for g in gaps if g > 0) / len(gaps)  # Should be ~1.0
        }