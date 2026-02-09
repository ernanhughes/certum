# This script runs the "deranged" negative sampling regime on the Feverous dataset.
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
  --neg_mode deranged `
  --n 4000 `
  --seed 1337 `
  --out_report artifacts\feverous_negcal_deranged.json `
  --out_pos_scored artifacts\pos_deranged.jsonl `
  --out_neg_scored artifacts\neg_deranged.jsonl `
  --plot_png artifacts\deranged.png
