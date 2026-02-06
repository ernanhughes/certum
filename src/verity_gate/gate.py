# src/verity_gate/gate.py
from verity_gate.energy.hallucination import hallucination_energy_svd
from verity_gate.policy import apply_policy

def evaluate_claim(
    claim_vec,
    evidence_vecs,
    regime: str,
    **energy_kwargs,
):
    result = hallucination_energy_svd(
        claim_vec,
        evidence_vecs,
        **energy_kwargs,
    )
    decision = apply_policy(result.energy, regime)
    return result, decision
