from abc import ABC, abstractmethod
from typing import List, Optional
from .types import EnergyResult, OracleCalibration, Verdict
import numpy as np

class Policy(ABC):
    """Abstract policy interface - deterministic decision boundary."""
    
    @abstractmethod
    def decide(
        self,
        energy_result: EnergyResult,
        oracle_calibration: Optional[OracleCalibration] = None
    ) -> Verdict:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

class FixedThresholdPolicy(Policy):
    """Static energy threshold (legacy systems)."""
    
    def __init__(self, tau_accept: float, tau_review: Optional[float] = None):
        self.tau_accept = tau_accept
        self.tau_review = tau_review or (tau_accept * 1.25)
    
    @property
    def name(self) -> str:
        return f"fixed.tau{self.tau_accept:.2f}"
    
    def decide(self, energy_result: EnergyResult, oracle_calibration: Optional[OracleCalibration] = None) -> Verdict:
        if energy_result.energy <= self.tau_accept:
            return Verdict.ACCEPT
        if energy_result.energy <= self.tau_review:
            return Verdict.REVIEW
        return Verdict.REJECT

class AdaptivePercentilePolicy(Policy):
    """
    Percentile-calibrated policy (your key innovation).
    Thresholds learned from data distribution, not hand-tuned.
    """
    
    def __init__(self, percentile: int, calibration_energies: List[float]):
        if not (1 <= percentile <= 100):
            raise ValueError("Percentile must be 1-100")
        self.percentile = percentile
        self.tau_accept = float(np.percentile(calibration_energies, percentile))
        self.tau_review = self.tau_accept * 1.25
    
    @property
    def name(self) -> str:
        return f"adaptive.P{self.percentile}"
    
    def decide(
        self,
        energy_result: EnergyResult,
        oracle_calibration: Optional[OracleCalibration] = None
    ) -> Verdict:
        # Use energy_gap if oracle calibration available
        energy = oracle_calibration.energy_gap if oracle_calibration else energy_result.energy
        
        if energy <= self.tau_accept:
            return Verdict.ACCEPT
        if energy <= self.tau_review:
            return Verdict.REVIEW
        return Verdict.REJECT

class PolicyRegistry:
    """Factory for policy instantiation."""
    
    _fixed_presets = {
        "editorial": FixedThresholdPolicy(0.30),
        "standard": FixedThresholdPolicy(0.45),
        "strict": FixedThresholdPolicy(0.55)
    }
    
    @classmethod
    def get_fixed(cls, name: str) -> FixedThresholdPolicy:
        if name not in cls._fixed_presets:
            raise ValueError(f"Unknown fixed policy: {name}. Options: {list(cls._fixed_presets.keys())}")
        return cls._fixed_presets[name]
    
    @classmethod
    def get_adaptive(cls, percentile: int, calibration_energies: List[float]) -> AdaptivePercentilePolicy:
        return AdaptivePercentilePolicy(percentile, calibration_energies)