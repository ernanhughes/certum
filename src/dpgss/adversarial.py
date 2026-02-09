from abc import ABC, abstractmethod
from typing import List, Optional
import random

class AdversarialClaimGenerator(ABC):
    """Generates semantically drifted claims to stress-test the gate."""
    
    @abstractmethod
    def transform(self, claim: str, evidence: List[str], seed: Optional[int] = None) -> str:
        """Return adversarial version of claim."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

# ---- 5 modes matching your original suite ----

class DerangedGenerator(AdversarialClaimGenerator):
    """Shuffle words to destroy semantic coherence while preserving vocabulary."""
    @property
    def name(self) -> str: return "deranged"
    
    def transform(self, claim: str, evidence: List[str], seed: Optional[int] = None) -> str:
        words = claim.split()
        rng = random.Random(seed) if seed else random
        rng.shuffle(words)
        return " ".join(words)

class OffsetGenerator(AdversarialClaimGenerator):
    """Rotate words by fixed offset (preserves some structure but breaks meaning)."""
    def __init__(self, offset: int = 37):
        self.offset = offset
    
    @property
    def name(self) -> str: return f"offset_{self.offset}"
    
    def transform(self, claim: str, evidence: List[str], seed: Optional[int] = None) -> str:
        words = claim.split()
        if not words: return claim
        offset = self.offset % len(words)
        return " ".join(words[offset:] + words[:offset])

class CyclicGenerator(AdversarialClaimGenerator):
    """Reverse word order (preserves vocabulary, inverts syntax)."""
    @property
    def name(self) -> str: return "cyclic"
    
    def transform(self, claim: str, evidence: List[str], seed: Optional[int] = None) -> str:
        words = claim.split()
        return " ".join(reversed(words))

class PermuteGenerator(AdversarialClaimGenerator):
    """Swap adjacent word pairs (mild syntactic corruption)."""
    @property
    def name(self) -> str: return "permute"
    
    def transform(self, claim: str, evidence: List[str], seed: Optional[int] = None) -> str:
        words = claim.split()
        if len(words) < 2: return claim
        rng = random.Random(seed) if seed else random
        pairs = [(words[i], words[i+1]) for i in range(0, len(words)-1, 2)]
        swapped = [pair[::-1] if rng.random() < 0.7 else pair for pair in pairs]
        flat = [w for pair in swapped for w in pair]
        if len(words) % 2 == 1:
            flat.append(words[-1])
        return " ".join(flat)

class HardMinedGenerator(AdversarialClaimGenerator):
    """
    REAL adversarial test: inject domain-mismatched concepts.
    This is what will break "too good" results.
    """
    def __init__(self, mismatch_phrases: Optional[List[str]] = None):
        self.mismatch_phrases = mismatch_phrases or [
            "quantum entanglement", "blockchain consensus", "neural lace", 
            "dark matter", "supernova collapse", "CRISPR editing"
        ]
    
    @property
    def name(self) -> str: return "hard_mined"
    
    def transform(self, claim: str, evidence: List[str], seed: Optional[int] = None) -> str:
        # Inject domain-mismatched phrase at random position
        rng = random.Random(seed) if seed else random
        phrase = rng.choice(self.mismatch_phrases)
        words = claim.split()
        if not words: return phrase
        pos = rng.randint(0, len(words))
        words.insert(pos, phrase)
        return " ".join(words)