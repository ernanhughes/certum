## 2026-02-09 — FEVEROUS gate suite (cache-based evidence)
- commit: <git hash>
- dataset: E:\data\feverous_dev_complete.jsonl
- cache:   E:\data\feverous_cache.db
- model:   all-MiniLM-L6-v2
- n=4000, cal_frac=0.5, FAR=0.01, top_k=12, rank_r=8

### Results
- deranged: tau=..., FAR=..., TPR=..., AUC=...
- offset(37): ...
- cyclic: ...
- permute: ...
- hard_mined: ...

### Notes / Next action
- [ ] hard_mined excludes same-claim candidates
- [ ] add seed salt by mode to avoid identical files
- [ ] add same_page negatives
