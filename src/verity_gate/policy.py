# src/verity_gate/policy.py
POLICIES = {
    "editorial": 0.55,
    "standard": 0.45,
    "strict": 0.30,
}


def apply_policy(energy: float, regime: str, delta: float = 0.10) -> str:
    tau = POLICIES[regime]
    if energy <= tau:
        return "accept"
    if energy <= tau + delta:
        return "review"
    return "reject"
