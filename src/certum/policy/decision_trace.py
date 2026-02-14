from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class DecisionTrace:
    """
    Deterministic explanation of a 3-axis geometry-aware policy decision.
    """

    # === Core Energy Axis ===
    energy: float
    alignment: float  # |dot(claim, v1)|

    # === Geometry Axis ===
    participation_ratio: float
    sensitivity: float
    effectiveness: float
    difficulty: float

    # Policy thresholds
    tau_accept: Optional[float]
    tau_review: Optional[float]
    pr_threshold: Optional[float]
    sensitivity_threshold: Optional[float]
    margin_band: Optional[float]

    # Policy metadata
    policy_name: str
    hard_negative_gap: float

    # Final action
    verdict: str  # expected: "accept" | "review" | "reject"

    def to_dict(self) -> dict:
        return asdict(self)


def why_rejected(trace: DecisionTrace) -> str:
    """
    Deterministic explanation for REJECT verdicts.
    """
    if trace.verdict != "reject":
        return "Not rejected."

    reasons = []

    # Energy hard-reject (most common)
    if trace.tau_review is not None and trace.energy > trace.tau_review:
        reasons.append("Energy exceeds review threshold.")

    # If you ever make PR/Sensitivity hard-reject in policy, these become exact.
    # Otherwise they are informational: they explain why the sample is risky.
    if trace.pr_threshold is not None and trace.participation_ratio > trace.pr_threshold:
        reasons.append("High PR (diffuse evidence manifold).")

    if trace.sensitivity_threshold is not None and trace.sensitivity > trace.sensitivity_threshold:
        reasons.append("High sensitivity (brittle evidence dependence).")

    if trace.effectiveness < 0.05:
        reasons.append("Insufficient effectiveness margin.")

    if not reasons:
        reasons.append("Rejected by policy fallback.")

    return " | ".join(reasons)


def why_reviewed(trace: DecisionTrace) -> str:
    """
    Deterministic explanation for REVIEW verdicts.
    """
    if trace.verdict != "review":
        return "Not reviewed."

    reasons = []

    if (
        trace.margin_band is not None
        and trace.tau_accept is not None
        and abs(trace.energy - trace.tau_accept) <= trace.margin_band
    ):
        reasons.append("Within policy margin band.")

    # These are “human-facing” interpretations; keep them stable.
    if trace.difficulty > 0.4:
        reasons.append("Moderate difficulty region.")

    if trace.effectiveness < 0.25:
        reasons.append("Low effectiveness margin.")

    # Optional: geometry triggers (only if policy actually uses these to trigger REVIEW)
    if trace.pr_threshold is not None and trace.participation_ratio > trace.pr_threshold:
        reasons.append("PR exceeds threshold.")

    if trace.sensitivity_threshold is not None and trace.sensitivity > trace.sensitivity_threshold:
        reasons.append("Sensitivity exceeds threshold.")

    if not reasons:
        reasons.append("Reviewed by policy fallback.")

    return " | ".join(reasons)
