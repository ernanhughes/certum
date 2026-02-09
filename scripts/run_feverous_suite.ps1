$ErrorActionPreference = "Stop"
$env:TRANSFORMERS_VERBOSITY = "error"
# ---- config (edit once) ----
$DATA    = "E:\data\feverous_dev_complete.jsonl"
$CACHEDB = "E:\data\feverous_cache.db"
$MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
$REGIME  = "standard"
$FAR     = "0.01"
$CALFRAC = "0.5"
$N       = "100"
$SEED    = "1337"

$RUNID = (Get-Date -Format "yyyyMMdd_HHmmss")
$OUTDIR = "artifacts\runs\$RUNID"
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

function Run-One($MODE, $EXTRA_ARGS) {
  $report = "$OUTDIR\feverous_negcal_$MODE.json"
  $pos    = "$OUTDIR\pos_$MODE.jsonl"
  $neg    = "$OUTDIR\neg_$MODE.jsonl"
  $plot   = "$OUTDIR\$MODE.png"

  py scripts\run_gate_suite.py `
    --kind feverous `
    --in_path $DATA `
    --cache_db $CACHEDB `
    --model $MODEL `
    --regime $REGIME `
    --far $FAR `
    --cal_frac $CALFRAC `
    --n $N `
    --seed $SEED `
    --neg_mode $MODE `
    --out_report $report `
    --out_pos_scored $pos `
    --out_neg_scored $neg `
    --plot_png $plot `
    @EXTRA_ARGS
}

# Run all 5 adversarial modes (critical for falsification testing)
Run-One "deranged" @()
Run-One "offset"   @("--neg_offset","37")
Run-One "cyclic"   @()
Run-One "permute"  @()
Run-One "hard_mined" @()

# Validate artifacts
py scripts\validate_gate_artifacts.py `
  --artifacts_dir $OUTDIR `
  --cache_db $CACHEDB `
  --model $MODEL `
  --strict

Write-Host ""
Write-Host "DONE. Run folder: $OUTDIR"
Write-Host "⚠️  Check oracle validity rates in reports — if >95% near-zero, your gate is trivial!"