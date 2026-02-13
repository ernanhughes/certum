from dataclasses import dataclass
from typing import List

from certum.policy.decision_trace import DecisionTrace


@dataclass(frozen=True)
class ValidationIssue:
    level: str          # "error" | "warn"
    code: str           # machine-stable identifier
    message: str        # human-readable


class DecisionTraceValidator:
    """
    Validates that a DecisionTrace is consistent with the policy region rules.

    This is a governance invariant: if it fails, something drifted (code change,
    config mismatch, or incorrect trace capture).
    """

    def __init__(
        self,
        *,
        difficulty_low: float = 0.40,
        difficulty_high: float = 0.75,
        eff_min_review: float = 0.05,
        eff_min_accept: float = 0.25,
    ):
        self.difficulty_low = float(difficulty_low)
        self.difficulty_high = float(difficulty_high)
        self.eff_min_review = float(eff_min_review)
        self.eff_min_accept = float(eff_min_accept)

    def expected_verdict(self, t: DecisionTrace) -> str:
        """
        Recompute verdict from the trace + validator thresholds.
        This must match the policy implementation.
        """
        # If thresholds absent, we can only do partial validation.
        if t.tau_accept is None or t.tau_review is None:
            # Fallback: difficulty/effectiveness-only check
            if t.difficulty > self.difficulty_high or t.effectiveness < self.eff_min_review:
                return "reject"
            if t.difficulty > self.difficulty_low or t.effectiveness < self.eff_min_accept:
                return "review"
            return "accept"

        tau = t.tau_accept
        tau_review = t.tau_review
        margin = t.margin_band if t.margin_band is not None else 0.1 * tau

        # Region C: Unsafe
        if (
            t.difficulty > self.difficulty_high
            or t.energy > tau_review
            or t.effectiveness < self.eff_min_review
        ):
            return "reject"

        # Region B: Hard / ambiguous
        if (
            t.difficulty > self.difficulty_low
            or abs(t.energy - tau) <= margin
            or t.effectiveness < self.eff_min_accept
        ):
            return "review"

        # Region A: Safe
        return "accept"

    def validate(self, t: DecisionTrace) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # Basic range checks
        if not (0.0 <= t.energy <= 1.0):
            issues.append(ValidationIssue("error", "range.energy", f"Energy out of [0,1]: {t.energy}"))
        if not (0.0 <= t.difficulty <= 1.0):
            issues.append(ValidationIssue("error", "range.difficulty", f"Difficulty out of [0,1]: {t.difficulty}"))
        if t.effectiveness < 0.0:
            issues.append(ValidationIssue("error", "range.effectiveness", f"Effectiveness < 0: {t.effectiveness}"))

        # Threshold sanity
        if t.tau_accept is not None and not (0.0 < t.tau_accept <= 1.0):
            issues.append(ValidationIssue("error", "range.tau_accept", f"tau_accept invalid: {t.tau_accept}"))
        if t.tau_review is not None and t.tau_accept is not None and t.tau_review < t.tau_accept:
            issues.append(ValidationIssue("error", "order.tau", f"tau_review < tau_accept: {t.tau_review} < {t.tau_accept}"))

        # Verdict consistency
        expected = self.expected_verdict(t)
        if t.verdict != expected:
            issues.append(
                ValidationIssue(
                    "error",
                    "verdict.mismatch",
                    f"Verdict mismatch: trace={t.verdict} expected={expected} "
                    f"(E={t.energy:.4f}, D={t.difficulty:.4f}, eff={t.effectiveness:.4f})"
                )
            )

        return issues
