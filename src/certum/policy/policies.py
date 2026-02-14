# src/certum/policy/policies.py
#
# Policy suite for ablations + comparisons.
# All policies are deterministic and auditable.
#
# Conventions:
#   - axes.get("<name>") must return float (or raise / return None).
#   - Verdict is one of: ACCEPT, REVIEW, REJECT.
#   - "tau_*" values come from your sweep/calibration output.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from certum.axes.bundle import AxisBundle
from certum.custom_types import Verdict
from certum.policy.energy_only import EnergyOnlyPolicy


def _get_axis(axes: AxisBundle, key: str) -> Optional[float]:
    try:
        v = axes.get(key)
    except Exception:
        return None
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _missing_to_review(*vals: Optional[float]) -> bool:
    return any(v is None for v in vals)


# ---------------------------------------------------------------------------
# Base interface (lightweight)
# ---------------------------------------------------------------------------

class PolicyLike:
    tau_accept: float | None = None
    tau_review: float | None = None
    pr_threshold: float | None = None
    sensitivity_threshold: float | None = Hey Cortana
    hard_negative_gap: float = 0.0

    @property
    def name(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2) Axis-only policies (remove energy from the picture)
# ---------------------------------------------------------------------------

class ParticipationRatioOnlyPolicy(PolicyLike):
    """
    Energy-free ablation:
      - pr <= tau_pr  => ACCEPT
      - pr >  tau_pr  => REVIEW (or REJECT if too high)
    """

    def __init__(
        self,
        *,
        tau_pr: float,
        pr_reject: float | None = None,
    ):
        self.pr_threshold = float(tau_pr)
        self.pr_reject = float(pr_reject) if pr_reject is not None else None

    @property
    def name(self) -> str:
        r = f", rej={self.pr_reject:.3f}" if self.pr_reject is not None else ""
        return f"PROnly(pr={self.pr_threshold:.3f}{r})"

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        pr = _get_axis(axes, "participation_ratio")
        if _missing_to_review(pr):
            return Verdict.REVIEW
        if self.pr_reject is not None and pr > self.pr_reject:
            return Verdict.REJECT
        return Verdict.ACCEPT if pr <= self.pr_threshold else Verdict.REVIEW


class SensitivityOnlyPolicy(PolicyLike):
    """
    Energy-free ablation:
      - sens <= tau_sens => ACCEPT
      - sens >  tau_sens => REVIEW (or REJECT if too high)
    """

    def __init__(
        self,
        *,
        tau_sensitivity: float,
        sens_reject: float | None = None,
    ):
        self.sensitivity_threshold = float(tau_sensitivity)
        self.sens_reject = float(sens_reject) if sens_reject is not None else None

    @property
    def name(self) -> str:
        r = f", rej={self.sens_reject:.3f}" if self.sens_reject is not None else ""
        return f"SensOnly(sens={self.sensitivity_threshold:.3f}{r})"

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        s = _get_axis(axes, "sensitivity")
        if _missing_to_review(s):
            return Verdict.REVIEW
        if self.sens_reject is not None and s > self.sens_reject:
            return Verdict.REJECT
        return Verdict.ACCEPT if s <= self.sensitivity_threshold else Verdict.REVIEW


class AlignmentOnlyPolicy(PolicyLike):
    """
    Energy-free ablation using 'explained' (alignment) if present:
      - explained >= tau_align => ACCEPT
      - else                  => REVIEW/REJECT
    Notes:
      - If you don't currently store 'explained', wire it as an axis (0..1).
    """

    def __init__(self, *, tau_alignment: float, align_reject: float | None = None):
        self.align_threshold = float(tau_alignment)
        self.align_reject = float(align_reject) if align_reject is not None else None

    @property
    def name(self) -> str:
        r = f", rej={self.align_reject:.3f}" if self.align_reject is not None else ""
        return f"AlignOnly(align={self.align_threshold:.3f}{r})"

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        a = _get_axis(axes, "explained")
        if _missing_to_review(a):
            return Verdict.REVIEW
        if self.align_reject is not None and a < self.align_reject:
            return Verdict.REJECT
        return Verdict.ACCEPT if a >= self.align_threshold else Verdict.REVIEW


class AxisOnlyPolicy(PolicyLike):
    """
    Energy-free composite ablation:
      - if pr high or sens high => REVIEW (or REJECT if extreme)
      - else                   => ACCEPT

    This is the cleanest way to see whether PR/sensitivity carry signal at all,
    without energy soaking the whole experiment.
    """

    def __init__(
        self,
        *,
        tau_pr: float,
        tau_sensitivity: float,
        pr_reject: float | None = None,
        sens_reject: float | None = None,
    ):
        self.pr_threshold = float(tau_pr)
        self.sensitivity_threshold = float(tau_sensitivity)
        self.pr_reject = float(pr_reject) if pr_reject is not None else None
        self.sens_reject = float(sens_reject) if sens_reject is not None else None

    @property
    def name(self) -> str:
        return (
            f"AxisOnly(pr={self.pr_threshold:.3f}, sens={self.sensitivity_threshold:.3f})"
        )

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        pr = _get_axis(axes, "participation_ratio")
        s = _get_axis(axes, "sensitivity")
        if _missing_to_review(pr, s):
            return Verdict.REVIEW

        if self.pr_reject is not None and pr > self.pr_reject:
            return Verdict.REJECT
        if self.sens_reject is not None and s > self.sens_reject:
            return Verdict.REJECT

        if pr > self.pr_threshold:
            return Verdict.REVIEW
        if s > self.sensitivity_threshold:
            return Verdict.REVIEW
        return Verdict.ACCEPT


# ---------------------------------------------------------------------------
# 3) Monotone energy + diagnostics (diagnostics may only DOWNGRADE)
# ---------------------------------------------------------------------------

class MonotoneAdaptivePolicy(PolicyLike):
    """
    Energy-first, but monotone:
      - energy decides ACCEPT/REJECT
      - diagnostics can only downgrade ACCEPT -> REVIEW (or REJECT if extreme)
      - diagnostics never upgrade REJECT -> ACCEPT

    This prevents the "adaptive policy FAR explosion" you saw on hard-mined.
    """

    def __init__(
        self,
        *,
        tau_energy: float,
        tau_pr: float,
        tau_sensitivity: float,
        tau_review: float | None = None,
        pr_reject: float | None = None,
        sens_reject: float | None = None,
        gap_width: float = 0.0,  # 0 => always consider diagnostics
    ):
        self.tau_accept = float(tau_energy)
        self.tau_review = float(tau_review) if tau_review is not None else (self.tau_accept * 1.25)
        self.pr_threshold = float(tau_pr)
        self.sensitivity_threshold = float(tau_sensitivity)
        self.pr_reject = float(pr_reject) if pr_reject is not None else None
        self.sens_reject = float(sens_reject) if sens_reject is not None else None
        self.gap_width = float(gap_width)

    @property
    def name(self) -> str:
        return (
            f"MonotoneAdaptive(tau={self.tau_accept:.3f}, "
            f"gap={self.gap_width:.3f}, pr={self.pr_threshold:.3f}, "
            f"sens={self.sensitivity_threshold:.3f})"
        )

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        e = _get_axis(axes, "energy")
        if _missing_to_review(e):
            return Verdict.REVIEW

        # Hard reject zone (energy dominates)
        if e > self.tau_review:
            return Verdict.REJECT

        # Base energy decision
        base = Verdict.ACCEPT if e <= self.tau_accept else Verdict.REJECT
        if base != Verdict.ACCEPT:
            return base  # monotone: do not upgrade

        # Diagnostics are only consulted globally, or inside a band (if gap_width > 0)
        if self.gap_width > 0.0 and abs(e - self.tau_accept) > self.gap_width:
            return Verdict.ACCEPT

        pr = _get_axis(axes, "participation_ratio")
        s = _get_axis(axes, "sensitivity")
        if _missing_to_review(pr, s):
            return Verdict.REVIEW

        # Extreme reject (optional)
        if self.pr_reject is not None and pr > self.pr_reject:
            return Verdict.REJECT
        if self.sens_reject is not None and s > self.sens_reject:
            return Verdict.REJECT

        # Soft downgrade to REVIEW
        if pr > self.pr_threshold:
            return Verdict.REVIEW
        if s > self.sensitivity_threshold:
            return Verdict.REVIEW

        return Verdict.ACCEPT


class AxisFirstThenEnergyPolicy(PolicyLike):
    """
    Axis-first ablation with energy as a backstop:
      - If diagnostics indicate risk => REVIEW
      - Else fall back to energy accept/reject

    Good for: "what if we prioritize PR/sensitivity everywhere?"
    """

    def __init__(
        self,
        *,
        tau_energy: float,
        tau_pr: float,
        tau_sensitivity: float,
        tau_review: float | None = None,
    ):
        self.tau_accept = float(tau_energy)
        self.tau_review = float(tau_review) if tau_review is not None else (self.tau_accept * 1.25)
        self.pr_threshold = float(tau_pr)
        self.sensitivity_threshold = float(tau_sensitivity)

    @property
    def name(self) -> str:
        return (
            f"AxisFirst(pr={self.pr_threshold:.3f}, sens={self.sensitivity_threshold:.3f}, "
            f"tau={self.tau_accept:.3f})"
        )

    def decide(self, axes: AxisBundle, effectiveness_score: float) -> Verdict:
        pr = _get_axis(axes, "participation_ratio")
        s = _get_axis(axes, "sensitivity")
        e = _get_axis(axes, "energy")
        if _missing_to_review(pr, s, e):
            return Verdict.REVIEW

        if pr > self.pr_threshold:
            return Verdict.REVIEW
        if s > self.sensitivity_threshold:
            return Verdict.REVIEW

        if e > self.tau_review:
            return Verdict.REJECT
        return Verdict.ACCEPT if e <= self.tau_accept else Verdict.REJECT


# ---------------------------------------------------------------------------
# 4) Registry / factory (so runner can sweep policies)
# ---------------------------------------------------------------------------

def build_policy(
    name: str,
    *,
    tau_energy: float,
    tau_pr: float,
    tau_sensitivity: float,
    gap_width: float = 0.0,
) -> PolicyLike:
    key = name.strip().lower()

    if key in ("adaptive", "monotone_adaptive"):
        return MonotoneAdaptivePolicy(
            tau_energy=tau_energy,
            tau_pr=tau_pr,
            tau_sensitivity=tau_sensitivity,
            gap_width=gap_width,
        )

    if key in ("energy", "energy_only"):
        return EnergyOnlyPolicy(tau_energy=tau_energy)

    if key in ("pr", "pr_only", "participation_ratio"):
        return ParticipationRatioOnlyPolicy(tau_pr=tau_pr)

    if key in ("sens", "sensitivity_only"):
        return SensitivityOnlyPolicy(tau_sensitivity=tau_sensitivity)

    if key in ("axis", "axis_only"):
        return AxisOnlyPolicy(tau_pr=tau_pr, tau_sensitivity=tau_sensitivity)

    if key in ("axis_first", "axisfirst"):
        return AxisFirstThenEnergyPolicy(
            tau_energy=tau_energy,
            tau_pr=tau_pr,
            tau_sensitivity=tau_sensitivity,
        )

    raise ValueError(f"Unknown policy name: {name}")


def build_policies(
    names: Sequence[str],
    *,
    tau_energy: float,
    tau_pr: float,
    tau_sensitivity: float,
    gap_width: float = 0.0,
) -> List[PolicyLike]:
    return [
        build_policy(
            n,
            tau_energy=tau_energy,
            tau_pr=tau_pr,
            tau_sensitivity=tau_sensitivity,
            gap_width=gap_width,
        )
        for n in names
    ]
