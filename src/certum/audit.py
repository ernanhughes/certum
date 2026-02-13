import json
from pathlib import Path
from typing import List, Dict
from .custom_types import EvaluationResult, Verdict
import numpy as np

class AuditLogger:
    """Produces human + machine readable audit trails."""
    
    @staticmethod
    def write_evaluation_dump(results: List[EvaluationResult], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + '\n')
    
    @staticmethod
    def generate_summary_report(results: List[EvaluationResult]) -> Dict:
        """Statistical summary for paper/report."""
        verdicts = [r.verdict for r in results]
        energies = [r.energy_result.energy for r in results]
        
        return {
            "total_samples": len(results),
            "verdict_distribution": {
                "accept": verdicts.count(Verdict.ACCEPT),
                "review": verdicts.count(Verdict.REVIEW),
                "reject": verdicts.count(Verdict.REJECT)
            },
            "energy_stats": {
                "mean": float(np.mean(energies)),
                "std": float(np.std(energies)),
                "p50": float(np.percentile(energies, 50)),
                "p90": float(np.percentile(energies, 90))
            },
            "stability": {
                "stable_count": sum(1 for r in results if r.energy_result.is_stable()),
                "unstable_count": sum(1 for r in results if not r.energy_result.is_stable())
            }
        }
    
    @staticmethod
    def write_run_header(path: Path, header: Dict):
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(header, f, indent=2)

    @staticmethod
    def generate_pathology_report(results):
        return {
            "high_energy_low_difficulty": sum(
                1 for r in results
                if r.energy_result.energy > 0.6 and r.difficulty_value < 0.3
            ),
            "low_energy_high_difficulty": sum(
                1 for r in results
                if r.energy_result.energy < 0.4 and r.difficulty_value > 0.7
            ),
            "review_due_to_margin": sum(
                1 for r in results
                if r.verdict == Verdict.REVIEW and abs(
                    r.energy_result.energy - r.decision_trace["tau_accept"]
                ) < r.decision_trace["margin_band"]
            )
        }
