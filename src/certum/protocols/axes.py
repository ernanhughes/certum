from typing import Protocol, Dict, Any


class AxisCalculator(Protocol):
    """
    Structural typing: any object with these members is an Axis.
    """

    name: str

    def compute(self, context: Dict[str, Any]) -> float:
        ...
