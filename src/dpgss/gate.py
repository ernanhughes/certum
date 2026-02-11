# src/dpgss/gate.py
from typing import List, Optional
from dpgss.policy.decision_trace import DecisionTrace


from dpgss.custom_types import EnergyResult, EvaluationResult
from dpgss.energy import HallucinationEnergyComputer
from dpgss.policy.difficulty_metrics import DifficultyMetrics
from dpgss.policy.difficulty import Difficulty, DifficultyRanges
from dpgss.protocols.embedder import Embedder
from dpgss.policy.policy import Policy
import numpy as np

class VerifiabilityGate:
    def __init__(
        self,
        embedder: Embedder,
        energy_computer: HallucinationEnergyComputer,
        difficulty_index: Difficulty
    ):
        self.embedder = embedder
        self.energy_computer = energy_computer
        self.difficulty_index = difficulty_index

    
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
        
        # 4. Robustness probe
        probe = self.energy_computer.compute_robustness_probe(claim_vec, ev_vecs)
        robust_var = float(np.var(probe))

        # 5. Build metrics (data only)
        metrics = DifficultyMetrics(
            sensitivity=base.sensitivity,
            sim_margin=base.sim_margin,
            evidence_count=int(ev_vecs.shape[0]),
            effective_rank=int(base.effective_rank),
        )

        # 6. Compute difficulty index (scalar)
        effectiveness = self.effectiveness_score(base.energy, policy.tau_accept)

        # 7. Policy decision
        difficulty_value = self.difficulty_index.compute(metrics)

        # 8. Final verdict All right something All right so let's start    
        verdict = policy.decide(base, difficulty_value, effectiveness)

        embedding_info = {
            "claim_dim": int(claim_vec.shape[0]),
            "evidence_count": int(ev_vecs.shape[0]),
            "embedding_backend": self.embedder.name,
        }

        tau_accept = policy.tau_accept
        tau_review = policy.tau_review
        margin = 0.1 * tau_accept


        decision_trace = DecisionTrace(
            energy=base.energy,
            difficulty=difficulty_value,
            effectiveness=effectiveness,
            tau_accept=tau_accept,
            tau_review=tau_review,
            margin_band=margin,
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
            robustness_probe=probe,
            difficulty_value=difficulty_value,
            difficulty_bucket=self.bucket_difficulty(difficulty_value)
        ) 
    
    def policy_log_row(result: EvaluationResult) -> dict:
        return {
            "energy": result.energy_result.energy,
            "difficulty": result.difficulty_value,
            "difficulty_bucket": result.difficulty_bucket,
            "verdict": result.verdict.value,
            "policy": result.policy_applied,
        }

    def effectiveness_score(self, energy: float, tau: float) -> float:
        """
        How much margin we have relative to policy threshold.
        """
        return max(0.0, (tau - energy) / max(tau, 1e-6))

    def bucket_difficulty(self, value: float) -> str:
        if value <= 0.40:
            return "easy"
        if value <= 0.75:
            return "medium"
        return "hard"
