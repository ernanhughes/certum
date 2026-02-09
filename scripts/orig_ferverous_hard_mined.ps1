$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force "artifacts" | Out-Null

py scripts\gate_suite.py `
  --kind feverous `
  --in_path E:\data\feverous_dev_complete.jsonl `
  --cache_db E:\data\feverous_cache.db `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  --regime standard `
  --far 0.01 `
  --cal_frac 0.5 `
  --neg_mode hard_mined `
  --n 4000 `
  --seed 1337 `
  --out_report artifacts\feverous_negcal_hard_mined.json `
  --out_pos_scored artifacts\pos_hard_mined.jsonl `
  --out_neg_scored artifacts\neg_hard_mined.jsonl `
  --plot_png artifacts\hard_mined.png
