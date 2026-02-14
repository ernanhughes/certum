from certum.axes.bundle import AxisBundle
from certum.custom_types import Verdict


class GapAdaptivePolicy:
    """
    Energy-dominant policy with gap-conditioned geometric refinement.

    Behavior:
        - Outside gap: energy decides
        - Inside gap: axes can override
    """

    def __init__(
        self,
        *,
        tau_energy: float,
        tau_pr: float,
        tau_sensitivity: float,
        gap_width: float,
        tau_review: float | None = None,
    ):
        self.tau_accept = tau_energy
        self.tau_review = tau_review or (tau_energy * 1.25)
        self.gap_width = gap_width

        self.pr_threshold = tau_pr
        self.sensitivity_threshold = tau_sensitivity

    @property
    def name(self) -> str:
        return (
            f"GapAdaptive("
            f"tau={self.tau_accept:.3f}, "
            f"gap={self.gap_width:.3f}, "
            f"pr={self.pr_threshold:.3f}, "
            f"sens={self.sensitivity_threshold:.3f})"
        )

    def decide(
        self,
        axes: AxisBundle,
        effectiveness_score: float
    ) -> Verdict:

        energy = axes.get("energy")

        # -------------------------------------------------
        # 1️⃣ Hard reject region
        # -------------------------------------------------
        if energy > self.tau_review:
            return Verdict.REJECT

        # -------------------------------------------------
        # 2️⃣ Outside gap → pure energy decision
        # -------------------------------------------------
        if abs(energy - self.tau_accept) > self.gap_width:
            if energy <= self.tau_accept:
                return Verdict.ACCEPT
            else:
                return Verdict.REJECT

        # -------------------------------------------------
        # 3️⃣ Inside gap → geometric refinement
        # -------------------------------------------------
        if axes.get("participation_ratio") > self.pr_threshold:
            return Verdict.REVIEW

        if axes.get("sensitivity") > self.sensitivity_threshold:
            return Verdict.REVIEW

        if energy <= self.tau_accept:
            return Verdict.ACCEPT

        return Verdict.REJECT
