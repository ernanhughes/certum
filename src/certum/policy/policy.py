
from certum.axes.bundle import AxisBundle
from certum.custom_types import Verdict
    

class AdaptivePolicy:

    def __init__(
        self,
        *,
        tau_energy: float,
        tau_pr: float,
        tau_sensitivity: float,
        tau_review: float | None = None,
        hard_negative_gap: float = 0.0,
    ):
        self.tau_accept = tau_energy
        self.tau_review = tau_review or (tau_energy * 1.25)
        self.hard_negative_gap = hard_negative_gap
        self.pr_threshold = tau_pr
        self.sensitivity_threshold = tau_sensitivity

        self.thresholds = {
            "participation_ratio": tau_pr,
            "sensitivity": tau_sensitivity,
        }

    @property
    def name(self) -> str:
        return f"AdaptivePolicy(tau_energy={self.tau_accept:.2f}, tau_pr={self.pr_threshold:.2f}, tau_sensitivity={self.sensitivity_threshold:.2f})"

    def decide(
        self,
        axes: AxisBundle,
        effectiveness_score: float
    ) -> Verdict:
        
        energy = axes.get("energy")

        # Hard reject region
        if energy > self.tau_review:
            return Verdict.REJECT

        # Secondary axes
        if axes.get("participation_ratio") > self.pr_threshold:
            return Verdict.REVIEW

        if axes.get("sensitivity") > self.sensitivity_threshold:
            return Verdict.REVIEW

        # Border band
        margin = 0.1 * self.tau_accept
        if abs(energy - self.tau_accept) <= margin:
            return Verdict.REVIEW

        # Accept region
        if energy <= self.tau_accept:
            return Verdict.ACCEPT

        return Verdict.REJECT
