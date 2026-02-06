# src/verity_gate/policy.py
POLICIES = {
    "editorial": 0.25,
    "standard": 0.15,
    "strict": 0.05,
}

def apply_policy(energy: float, regime: str) -> str:
    tau = POLICIES[regime]
    if energy <= tau:
        return "accept"
    if energy <= tau + 0.10:
        return "review"
    return "reject"
