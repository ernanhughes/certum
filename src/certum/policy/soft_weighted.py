from certum.axes.bundle import AxisBundle
from certum.custom_types import Verdict


class SoftWeightedPolicy:
    """
    Continuous geometric scoring model.
    No hard vetoes.
    """

    def __init__(
        self,
        *,
        tau_score: float,
        w_energy: float = 1.0,
        w_pr: float = 0.5,
        w_sensitivity: float = 0.5,
    ):
        self.tau_score = tau_score
        self.w_energy = w_energy
        self.w_pr = w_pr
        self.w_sensitivity = w_sensitivity

    @property
    def name(self) -> str:
        return (
            f"SoftWeighted("
            f"tau={self.tau_score:.2f}, "
            f"wE={self.w_energy:.2f}, "
            f"wPR={self.w_pr:.2f}, "
            f"wS={self.w_sensitivity:.2f})"
        )

    def decide(self, axes: AxisBundle, effectiveness_score: float):

        energy = axes.get("energy")
        pr = axes.get("participation_ratio")
        sens = axes.get("sensitivity")

        score = (
            self.w_energy * energy
            + self.w_pr * pr
            + self.w_sensitivity * sens
        )

        return Verdict.ACCEPT if score <= self.tau_score else Verdict.REJECT
