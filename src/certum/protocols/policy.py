# certum/protocols/policy.py

from typing import Protocol, Optional
from certum.custom_types import Verdict
from certum.axes.bundle import AxisBundle


class Policy(Protocol):

    # Required attributes
    tau_accept: float
    tau_review: Optional[float]
    hard_negative_gap: float

    # Required behavior
    def decide(
        self,
        axes: AxisBundle,
        effectiveness_score: float
    ) -> Verdict:
        ...

    @property
    def name(self) -> str:
        ...
