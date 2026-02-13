# src/certum/gate.py

from typing import List, Optional
import numpy as np
import logging

from certum.custom_types import DecisionAxes, EnergyResult, EvaluationResult
from certum.energy import HallucinationEnergyComputer
from certum.protocols.embedder import Embedder
from certum.policy.policy import Policy
from certum.policy.decision_trace import DecisionTrace

logger = logging.getLogger(__name__)


class VerifiabilityGate:
    """
    Certum Core Gate

    Deterministic wrapper around:
        - Embedding
        - Energy computation
        - Policy decision

    No threshold tuning happens here.
    Pure execution layer.
    """

    def __init__(
        self,
        embedder: Embedder,
        energy_computer: HallucinationEnergyComputer,
    ):
        self.embedder = embedder
        self.energy_computer = energy_computer

    # ---------------------------------------------------------
    # Energy-only computation (used for calibration)
    # ---------------------------------------------------------

    def compute_energy(
        self,
        claim: str,
        evidence_texts: List[str],
        *,
        claim_vec: Optional[np.ndarray] = None,
        evidence_vecs: Optional[np.ndarray] = None,
    ) -> EnergyResult:
        """
        Compute hallucination energy WITHOUT policy decision.
        Used during calibration sweeps.
        """

        if claim_vec is None:
            claim_vec_raw = self.embedder.embed([claim])
            if claim_vec_raw.shape[0] != 1:
                raise ValueError(
                    f"Unexpected claim embedding shape: {claim_vec_raw.shape}"
                )
            claim_vec = claim_vec_raw[0]

        if evidence_vecs is None:
            evidence_vecs = self.embedder.embed(evidence_texts)

        return self.energy_computer.compute(claim_vec, evidence_vecs)

    # ---------------------------------------------------------
    # Full evaluation (energy + policy decision)
    # ---------------------------------------------------------

    def evaluate(
        self,
        claim: str,
        evidence_texts: List[str],
        policy: Policy,
        *,
        run_id: str,
        split: str = "pos",
        neg_mode: Optional[str] = None,
        claim_vec: Optional[np.ndarray] = None,
        ev_vecs: Optional[np.ndarray] = None,
    ) -> EvaluationResult:

        # --- Embedding (strict shape handling) ---

        if claim_vec is None:
            claim_vec_raw = self.embedder.embed([claim])
            if claim_vec_raw.shape[0] != 1:
                raise ValueError(
                    f"Unexpected claim embedding shape: {claim_vec_raw.shape}"
                )
            claim_vec = claim_vec_raw[0]

        if ev_vecs is None:
            ev_vecs = self.embedder.embed(evidence_texts)

        # --- Energy computation ---

        base = self.energy_computer.compute(claim_vec, ev_vecs)

        # --- Build Decision Axes (explicit 3D surface) ---

        axes = DecisionAxes(
            energy=base.energy,
            participation_ratio=base.geometry.participation_ratio,
            sensitivity=base.geometry.sensitivity,
            alignment=base.geometry.alignment_to_sigma1,
            sim_margin=base.geometry.sim_margin,
        )

        # --- Margin-based effectiveness (diagnostic only) ---

        effectiveness = self._accept_margin_ratio(
            energy=base.energy,
            tau=policy.tau_accept,
        )

        # --- Policy decision ---

        verdict = policy.decide(axes, effectiveness)

        logger.debug(
            f"[Gate] E={axes.energy:.4f} "
            f"PR={axes.participation_ratio:.4f} "
            f"S={axes.sensitivity:.4f} "
            f"| tauE={policy.tau_accept:.4f} "
            f"tauPR={policy.pr_threshold:.4f} "
            f"tauS={policy.sensitivity_threshold:.4f} "
            f"=> {verdict.value}"
        )

        # --- Trace object (full transparency) ---

        decision_trace = DecisionTrace(
            energy=base.energy,
            alignment=base.geometry.alignment_to_sigma1,
            participation_ratio=base.geometry.participation_ratio,
            sensitivity=base.geometry.sensitivity,
            tau_accept=policy.tau_accept,
            tau_review=policy.tau_review,
            pr_threshold=policy.pr_threshold,
            sensitivity_threshold=policy.sensitivity_threshold,
            effectiveness=effectiveness,
            margin_band=0.1 * policy.tau_accept if policy.tau_accept else None,
            policy_name=policy.name,
            hard_negative_gap=policy.hard_negative_gap,
            verdict=verdict.value,
        )

        embedding_info = {
            "claim_dim": int(claim_vec.shape[0]),
            "evidence_count": int(ev_vecs.shape[0]),
            "embedding_backend": self.embedder.name,
        }

        return EvaluationResult(
            run_id=run_id,
            claim=claim,
            evidence=evidence_texts,
            decision_trace=decision_trace,
            embedding_info=embedding_info,
            energy_result=base,
            effectiveness=effectiveness,
            verdict=verdict,
            policy_applied=policy.name,
            split=split,
            neg_mode=neg_mode,
        )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    @staticmethod
    def policy_log_row(result: EvaluationResult) -> dict:
        """
        Minimal structured log row for analysis.
        """
        return {
            "energy": result.energy_result.energy,
            "verdict": result.verdict.value,
            "policy": result.policy_applied,
        }

    @staticmethod
    def _accept_margin_ratio(energy: float, tau: float) -> float:
        """
        Normalized margin relative to acceptance threshold.
        """
        return max(0.0, (tau - energy) / max(tau, 1e-6))
