# ==========================================
# Policy Ablation Runner (Certum)
# ==========================================

param(
    [string]$InputPath = "datasets/pubmedqa/pubmedqa_train.jsonl",
    [string]$EmbeddingDB = "E:\data\global_embeddings_test.db",
    [string]$CacheDB = "E:\data\feverous_cache.db",
    [string]$Model = "sentence-transformers/all-MiniLM-L6-v2",
    [int]$N = 5000,
    [double]$FAR = 0.01,
    [string]$NegMode = "hard_mined_v2",
    [string]$OutDir = "artifacts\ablation"
)

Write-Host ""
Write-Host "==========================================="
Write-Host " CERTUM Policy Ablation Test Runner"
Write-Host "==========================================="
Write-Host ""

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutDir $timestamp

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
New-Item -ItemType Directory -Force -Path "$runDir\plots" | Out-Null

$reportPath = "$runDir\report.json"

Write-Host "Run directory: $runDir"
Write-Host "Model: $Model"
Write-Host "FAR target: $FAR"
Write-Host "Negative mode: $NegMode"
Write-Host ""

# Run policy_ablation.py
python .\scripts\policy_ablation.py `
    --in_path $InputPath `
    --embedding_db $EmbeddingDB `
    --cache_db $CacheDB `
    --model $Model `
    --n $N `
    --far $FAR `
    --neg_mode $NegMode `
    --out_report $reportPath `
    --plot_dir "$runDir\plots"

Write-Host ""
Write-Host "==========================================="
Write-Host " Run Complete"
Write-Host "==========================================="
Write-Host ""
Write-Host "Report:"
Write-Host "  $reportPath"
Write-Host ""
Write-Host "Plots:"
Write-Host "  $runDir\plots"
Write-Host ""
