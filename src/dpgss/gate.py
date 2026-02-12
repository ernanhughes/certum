# src/dpgss/gate.py
from typing import List, Optional
from dpgss.policy.decision_trace import DecisionTrace


from dpgss.custom_types import DecisionAxes, EnergyResult, EvaluationResult
from dpgss.energy import HallucinationEnergyComputer
from dpgss.difficulty.difficulty_metrics import DifficultyMetrics
from dpgss.protocols.embedder import Embedder
from dpgss.policy.policy import Policy
import numpy as np

class VerifiabilityGate:
    def __init__(
        self,
        embedder: Embedder,
        energy_computer: HallucinationEnergyComputer,
    ):
        self.embedder = embedder
        self.energy_computer = energy_computer

    
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
        Used for calibration/sweeping — no dummy policy needed.
        """
        if claim_vec is None:
            claim_vec_raw = self.embedder.embed([claim])
            claim_vec = claim_vec_raw[0] if claim_vec_raw.shape[0] == 1 else claim_vec_raw
        
        if evidence_vecs is None:
            evidence_vecs = self.embedder.embed(evidence_texts)
        
        return self.energy_computer.compute(claim_vec, evidence_vecs)
    
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
        # 1. Embed with STRICT shape handling
        if claim_vec is None:
            claim_vec_raw = self.embedder.embed([claim])
            if claim_vec_raw.shape[0] == 1:
                claim_vec = claim_vec_raw[0]
            else:
                raise ValueError(f"Unexpected claim embedding shape: {claim_vec_raw.shape}")
        
        if ev_vecs is None:
            ev_vecs = self.embedder.embed(evidence_texts)
        
        # 2. Compute base energy
        base = self.energy_computer.compute(claim_vec, ev_vecs)
        
        # 3. Build metrics (data only)
        metrics = DifficultyMetrics(
            sensitivity=base.geometry.sensitivity,
            sim_margin=base.geometry.sim_margin,
            evidence_count=int(ev_vecs.shape[0]),
            effective_rank=int(base.geometry.effective_rank),
            participation_ratio=float(base.geometry.participation_ratio),
        )

        # 4. Policy decision
        axes = DecisionAxes(
            energy=base.energy,
            participation_ratio=base.geometry.participation_ratio,
            sensitivity=base.geometry.sensitivity,
            alignment=base.geometry.alignment_to_sigma1,
            sim_margin=base.geometry.sim_margin,
        )

        # 5. Compute difficulty index (scalar)
        effectiveness = self.effectiveness_score(base.energy, policy.tau_accept)

        verdict = policy.decide(axes, effectiveness)

        embedding_info = {
            "claim_dim": int(claim_vec.shape[0]),
            "evidence_count": int(ev_vecs.shape[0]),
            "embedding_backend": self.embedder.name,
        }

        geom = base.geometry

        decision_trace = DecisionTrace(
            energy=base.energy,
            alignment=geom.alignment_to_sigma1,
            participation_ratio=geom.participation_ratio,
            sensitivity=geom.sensitivity,
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
    
    def policy_log_row(result: EvaluationResult) -> dict:
        return {
            "energy": result.energy_result.energy,
            "difficulty_bucket": result.difficulty_bucket,
            "verdict": result.verdict.value,
            "policy": result.policy_applied,
        }
   
    def effectiveness_score(self, energy: float, tau: float) -> float:
        """
        How much margin we have relative to policy threshold.
        """
        return max(0.0, (tau - energy) / max(tau, 1e-6))
