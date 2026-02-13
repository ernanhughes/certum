# certum/axes/bundle.py

from typing import Dict


class AxisBundle:
    def __init__(self, axes: Dict[str, float]):
        self._axes = axes

    def get(self, name: str) -> float:
        return self._axes.get(name, 0.0)

    def items(self):
        return self._axes.items()

    def __repr__(self):
        return f"AxisBundle({self._axes})"
