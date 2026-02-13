from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from certum.custom_types import Verdict, DecisionAxes


class Policy(ABC):
    """
    Deterministic multi-axis decision boundary.
    """

    tau_accept: float = 0.0
    tau_review: Optional[float] = None
    hard_negative_gap: float = 0.0

    @abstractmethod
    def decide(
        self,
        axes: DecisionAxes,
        effectiveness_score: float
    ) -> Verdict:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class FixedThresholdPolicy(Policy):
    """
    Pure energy-only baseline.
    """

    def __init__(self, tau_accept: float, tau_review: Optional[float] = None):
        self.tau_accept = tau_accept
        self.tau_review = tau_review or (tau_accept * 1.25)
        self.pr_threshold: float = 8.0,
        self.sensitivity_threshold: float = 0.4,

    @property
    def name(self) -> str:
        return f"fixed.tau{self.tau_accept:.2f}"

    def decide(
        self,
        axes: DecisionAxes
    ) -> Verdict:

        if axes.energy <= self.tau_accept:
            return Verdict.ACCEPT

        if axes.energy <= self.tau_review:
            return Verdict.REVIEW

        return Verdict.REJECT

@dataclass
class RoadGateThresholds:
    tau_energy: float
    tau_energy_review: float
    tau_pr: float
    tau_sensitivity: float


class RoadGatePolicy(Policy):

    def __init__(self, thresholds: RoadGateThresholds):
        self.t = thresholds

    @property
    def name(self):
        return "roadgate.v3"

    def decide(self, axes: DecisionAxes, effectiveness_score: float) -> Verdict:

        E = axes.energy
        R = axes.participation_ratio
        S = axes.sensitivity

        # Hard reject
        if E > self.t.tau_energy_review:
            return Verdict.REJECT

        # Accept region
        if (
            E <= self.t.tau_energy
            and R <= self.t.tau_pr
            and S <= self.t.tau_sensitivity
        ):
            return Verdict.ACCEPT

        return Verdict.REVIEW


class AdaptivePolicy(Policy):
    """
    3D RoadGate Surface:
        Energy + Participation Ratio + Sensitivity
    """

    def __init__(
        self,
        *,
        tau_energy: float,
        tau_pr: float,
        tau_sensitivity: float,
        tau_review: Optional[float] = None,
        hard_negative_gap: float = 0.0,
    ):
        self.tau_accept = tau_energy
        self.tau_review = tau_review or (tau_energy * 1.25)
        self.hard_negative_gap = hard_negative_gap

        self.pr_threshold = tau_pr
        self.sensitivity_threshold = tau_sensitivity

    @property
    def name(self) -> str:
        return f"AdaptivePolicy(tau_energy={self.tau_accept:.2f}, tau_pr={self.pr_threshold:.2f}, tau_sensitivity={self.sensitivity_threshold:.2f})"

    def decide(
        self,
        axes: DecisionAxes,
        effectiveness_score: float

    ) -> Verdict:

        # Hard reject region
        if axes.energy > self.tau_review:
            return Verdict.REJECT

        # Secondary axes
        if axes.participation_ratio > self.pr_threshold:
            return Verdict.REVIEW

        if axes.sensitivity > self.sensitivity_threshold:
            return Verdict.REVIEW

        # Border band
        margin = 0.1 * self.tau_accept
        if abs(axes.energy - self.tau_accept) <= margin:
            return Verdict.REVIEW

        # Accept region
        if axes.energy <= self.tau_accept:
            return Verdict.ACCEPT

        return Verdict.REJECT


class PolicyRegistry:
    """Factory for policy instantiation."""

    _fixed_presets = {
        "editorial": FixedThresholdPolicy(0.30),
        "standard": FixedThresholdPolicy(0.45),
        "strict": FixedThresholdPolicy(0.55),
    }

    @classmethod
    def get_fixed(cls, name: str) -> FixedThresholdPolicy:
        if name not in cls._fixed_presets:
            raise ValueError(
                f"Unknown fixed policy: {name}. Options: {list(cls._fixed_presets.keys())}"
            )
        return cls._fixed_presets[name]

    @classmethod
    def get_adaptive(
        cls,
        percentile: int,
        calibration_energies: List[float],
    ) -> AdaptivePolicy:
        return AdaptivePolicy(percentile, calibration_energies)
