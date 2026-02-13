from pathlib import Path

from certum.calibration import AdaptiveCalibrator
from certum.protocols.calibration import Calibrator
from certum.protocols.embedder import Embedder
from certum.protocols.gate import Gate
from certum.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from certum.embedding.hf_embedder import HFEmbedder
from certum.geometry.claim_evidence import ClaimEvidenceGeometry
from certum.gate import VerifiabilityGate
from certum.protocols.geometry import GeometryComputer


class CertumFactory:
    """
    Responsible ONLY for constructing concrete implementations.
    No execution logic.
    """

    # -------------------------------------------------
    # Embedding
    # -------------------------------------------------

    def build_embedder(self, model: str, embedding_db: Path) -> Embedder:
        backend = SQLiteEmbeddingBackend(str(embedding_db))
        return HFEmbedder(model_name=model, backend=backend)

    # -------------------------------------------------
    # Energy
    # -------------------------------------------------

    def build_energy_computer(self) -> GeometryComputer:
        return ClaimEvidenceGeometry(top_k=12, rank_r=8)

    # -------------------------------------------------
    # Gate
    # -------------------------------------------------

    def build_gate(self, embedder: Embedder, energy_computer: GeometryComputer) -> Gate:
        return VerifiabilityGate(embedder, energy_computer)


    def build_calibrator(self, gate: Gate, embedder: Embedder) -> Calibrator:
        return AdaptiveCalibrator(gate, embedder)