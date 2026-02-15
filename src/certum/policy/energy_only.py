from certum.axes.bundle import AxisBundle
from certum.custom_types import Verdict


class EnergyOnlyPolicy:
    """
    Pure 1D energy gate.
    Accept if energy <= tau_accept.
    Reject otherwise.
    """

    def __init__(self, tau_energy: float, tau_review: float | None = None):
        self.tau_accept = tau_energy
        self.tau_review = tau_review or (tau_energy * 1.25)

    @property
    def name(self) -> str:
        return f"EnergyOnlyPolicy(tau_energy={self.tau_accept:.2f})"

    def decide(
        self,
        axes: AxisBundle,
        effectiveness_score: float
    ) -> Verdict:

        energy = axes.get("energy")

        if energy > self.tau_review:
            return Verdict.REJECT

        if energy <= self.tau_accept:
            return Verdict.ACCEPT

        return Verdict.REJECT
