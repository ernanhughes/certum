param(
    [Parameter(Mandatory=$true)]
    [string]$Target
)

$ErrorActionPreference = "Stop"
$env:TRANSFORMERS_VERBOSITY = "error"

# ============================================================
# Global Defaults
# ============================================================

$gitHash = (git rev-parse HEAD)

$GLOBAL = @{
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDB         = "E:\data\global_embeddings.db"
    ENTAILMENT_DB   = "E:\data\entailment_cache.db"
    NLI_MODEL       = "MoritzLaurer/deberta-v3-base-mnli-fever-anli"
    TOP_K           = "3"
    N               = "3000"
    SEED            = "1337"
    GEOMETRY_TOP_K   = "1000"
    RANK_R          = "32"
}

# ============================================================
# Dataset Configurations
# ============================================================

$DATASETS = @{

    halueval = @{
        data_source = "halueval"
        data        = "E:\data\halueval_test_v1.jsonl"
    }

    pubmed = @{
        data_source = "pubmedqa"
        data        = "E:\data\pubmedqa_train.jsonl"
    }

    casehold = @{
        data_source = "casehold"
        data        = "E:\data\casehold_pos.jsonl"
    }

    scifact = @{
        data_source = "scifact"
        data        = "E:\data\scifact_dev_rationale.jsonl"
    }
}

# ============================================================
# Validate Target
# ============================================================

if (-not $DATASETS.ContainsKey($Target)) {
    Write-Host ""
    Write-Host "Invalid target: $Target"
    Write-Host "Available targets:"
    $DATASETS.Keys | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

$CFG = $DATASETS[$Target]

# ============================================================
# Build Run Folder
# ============================================================

$RUNID  = (Get-Date -Format "yyyyMMdd_HHmmss")
$OUTDIR = "artifacts\runs\summarization\$RUNID"
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

# ============================================================
# Write Metadata File
# ============================================================

$MetaFile = "$OUTDIR\config.json"

$meta = @{
    run_id    = $RUNID
    timestamp = (Get-Date).ToUniversalTime().ToString("o")
    target    = $Target
    git_commit = $gitHash

    dataset = @{
        path = $CFG.data
        n    = [int]$GLOBAL.N
    }

    embedding = @{
        model = $GLOBAL.EMBEDDING_MODEL
        embedding_db = $GLOBAL.EMBEDDB
    }

    entailment = @{
        model = $GLOBAL.NLI_MODEL
        top_k = [int]$GLOBAL.TOP_K
    }

    random = @{
        seed = [int]$GLOBAL.SEED
    }
}

$meta | ConvertTo-Json -Depth 6 | Out-File -Encoding UTF8 $MetaFile

# ============================================================
# Output Paths
# ============================================================

$resultsJsonl = "$OUTDIR\summary_results.jsonl"
$reportJson   = "$OUTDIR\evaluation_report.json"
$coeffGeo     = "$OUTDIR\coefficients_geometry.csv"
$coeffEnt     = "$OUTDIR\coefficients_entailment.csv"
$coeffFull    = "$OUTDIR\coefficients_full.csv"

$rocGeo       = "$OUTDIR\roc_geometry.png"
$rocEnt       = "$OUTDIR\roc_entailment.png"
$rocFull      = "$OUTDIR\roc_full.png"
$calibration  = "$OUTDIR\calibration_curve.png"
$prCurve      = "$OUTDIR\precision_recall_curve.png"
$corrCsv      = "$OUTDIR\feature_correlation_matrix.csv"

# ============================================================
# Execute Evaluation Runner
# ============================================================

Write-Host ""
Write-Host "Running summarization evaluation..."
Write-Host "Dataset: $Target"
Write-Host "Run folder: $OUTDIR"

py -m certum.evaluation.runner `
    --input_jsonl $CFG.data `
    --out_dir $OUTDIR `
    --dataset_name $Target `
    --embedding_model $GLOBAL.EMBEDDING_MODEL `
    --embedding_db $GLOBAL.EMBEDDB `
    --nli_model $GLOBAL.NLI_MODEL `
    --top_k $GLOBAL.TOP_K `
    --limit $GLOBAL.N `
    --seed $GLOBAL.SEED `
    --entailment_db $GLOBAL.ENTAILMENT_DB `
    --geometry_top_k $GLOBAL.GEOMETRY_TOP_K `
    --rank_r $GLOBAL.RANK_R

Write-Host ""
Write-Host "DONE."
Write-Host "Run folder: $OUTDIR"
