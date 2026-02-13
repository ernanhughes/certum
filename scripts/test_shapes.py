from certum.embedding.embedder import HFEmbedder
from certum.energy import HallucinationEnergyComputer
import numpy as np

embedder = HFEmbedder()
computer = HallucinationEnergyComputer()

# Test 1: Single claim + evidence
claim = "Paris is the capital of France."
evidence = ["Paris is a city in France.", "France has a capital city named Paris."]

claim_vec = embedder.embed([claim])[0]  # MUST index [0]
ev_vecs = embedder.embed(evidence)

print(f"claim_vec shape: {claim_vec.shape}")  # Should be (384,)
print(f"ev_vecs shape: {ev_vecs.shape}")      # Should be (2, 384)

res = computer.compute(claim_vec, ev_vecs)
print(f"✅ Energy computed: {res.energy:.4f}")

# Test 2: Deliberately trigger bug (for validation)
try:
    buggy_claim_vec = embedder.embed([claim])  # Forgot [0] → (1, 384)
    computer.compute(buggy_claim_vec, ev_vecs)
    print("❌ FAILED: Should have raised shape error!")
except (ValueError, RuntimeError) as e:
    print(f"✅ Shape validation working: {type(e).__name__}")