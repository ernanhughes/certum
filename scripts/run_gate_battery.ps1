$ErrorActionPreference = "Stop"

# ---- config (edit once) ----
$DATA    = "E:\data\feverous_dev_complete.jsonl"
$CACHEDB = "E:\data\feverous_cache.db"
$MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
$REGIME  = "standard"
$FAR     = "0.01"
$CALFRAC = "0.5"
$N       = "4000"
$SEED    = "1337"

$RUNID = (Get-Date -Format "yyyyMMdd_HHmmss")
$OUTDIR = "artifacts\runs\$RUNID"
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

function Run-One($MODE, $EXTRA_ARGS) {
  $report = "$OUTDIR\feverous_negcal_$MODE.json"
  $pos    = "$OUTDIR\pos_$MODE.jsonl"
  $neg    = "$OUTDIR\neg_$MODE.jsonl"
  $plot   = "$OUTDIR\$MODE.png"

  py scripts\gate_suite.py `
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

Run-One "deranged" @()
Run-One "offset"   @("--neg_offset","37")
Run-One "cyclic"   @()
Run-One "permute"  @()
Run-One "hard_mined" @()

# ---- validate the whole run folder ----
py scripts\validate_gate_artifacts.py `
  --artifacts_dir $OUTDIR `
  --cache_db $CACHEDB `
  --model $MODEL `
  --strict

Write-Host ""
Write-Host "DONE. Run folder: $OUTDIR"
