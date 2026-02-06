from verity_gate.energy.hallucination import hallucination_energy_svd
from verity_gate.policy import apply_policy


def evaluate_claim(
    claim_vec,
    evidence_vecs,
    regime: str,
    **energy_kwargs,
):
    # Base computation
    base = hallucination_energy_svd(
        claim_vec,
        evidence_vecs,
        **energy_kwargs,
    )

    # Robustness probe (hyperparameter sensitivity)
    probe = []
    for k in (8, 12, 20):
        r = hallucination_energy_svd(
            claim_vec,
            evidence_vecs,
            top_k=k,
            rank_r=energy_kwargs.get("rank_r", 8),
            return_debug=False,
        )
        probe.append(r.energy)

    decision = apply_policy(base.energy, regime)

    return base, decision, probe
