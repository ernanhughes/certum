from dataclasses import dataclass

@dataclass(frozen=True)
class DifficultyMetrics:
    sim_margin: float
    sensitivity: float
    evidence_count: int  
    effective_rank: int  
    participation_ratio: float  
