from verity_gate.energy.hallucination import hallucination_energy_svd
from verity_gate.policy import apply_policy
import numpy as np


def evaluate_claim(
    claim_vec,
    evidence_vecs,
    regime: str,
    *,
    oracle_vec: np.ndarray | None = None,
    **energy_kwargs,
):
    """
    Returns:
        base_result      : EnergyResult
        decision         : str
        probe_energies   : list[float]
        oracle_energy    : float | None
        energy_gap       : float | None
    """

    # --------------------------------------------------
    # 1) Base computation (main claim energy)
    # --------------------------------------------------
    base = hallucination_energy_svd(
        claim_vec,
        evidence_vecs,
        **energy_kwargs,
    )

    # --------------------------------------------------
    # 2) Robustness probe (hyperparameter sensitivity)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 3) Deterministic policy decision
    # --------------------------------------------------
    decision = apply_policy(base.energy, regime)

    # --------------------------------------------------
    # 4) Oracle energy + energy gap (NEW)
    # --------------------------------------------------
    oracle_energy = None
    energy_gap = None

    if oracle_vec is not None:
        oracle = hallucination_energy_svd(
            oracle_vec,
            evidence_vecs,
            top_k=energy_kwargs.get("top_k", 12),
            rank_r=energy_kwargs.get("rank_r", 8),
            return_debug=False,
        )
        oracle_energy = oracle.energy
        energy_gap = base.energy - oracle_energy

    return base, decision, probe, oracle_energy, energy_gap
