# ---------- 1. Global Configuration ----------
$env:PYTHONPATH="D:\LLM\model"

$inputDir = "D:\LLM\data\Foundation\Foundation_train_aligned"   
$outRoot  = "D:\LLM\outputs\teacher_outputs"
$modelId  = "ep-20260227163508-kxp4m"
$knowledgeRoot = "D:\LLM\model\teacher\knowledge"
$groupby  = "leiden"

# Check if input directory exists
if (-not (Test-Path $inputDir)) {
    Write-Host "ERROR: Input directory not found -> $inputDir" -ForegroundColor Red
    exit
}

# ---------- 2. Execution Loop ----------
# Ensure the root output directory exists
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

# Get all aligned h5ad files in the directory
$files = Get-ChildItem $inputDir -Filter "*.aligned.h5ad"

Write-Host "`nDetected $($files.Count) files to process..." -ForegroundColor Yellow

foreach ($file in $files) {
    $h5ad = $file.FullName
    $name = $file.BaseName.Replace(".aligned","")
    $outDir = Join-Path $outRoot $name

    # Create a unique output subdirectory for each dataset
    if (-not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }

    Write-Host ("`n" + "="*50) -ForegroundColor Gray
    Write-Host "[RUNNING] Dataset: " -NoNewline; Write-Host $name -ForegroundColor Cyan
    
    # Execute Teacher CLI
    # Fix: Used ${name} to prevent PowerShell variable ambiguity with underscores
    python -m teacher.cli run `
      --h5ad $h5ad `
      --out-dir $outDir `
      --model-id $modelId `
      --knowledge-root $knowledgeRoot `
      --groupby $groupby `
      --output-name "${name}_llm_teacher_tokens.json"

    # Error handling and Logging
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAILED]  " -NoNewline; Write-Host $name -ForegroundColor Red
        # Log error to file
        $logEntry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): Failed on $name"
        Add-Content -Path (Join-Path $outRoot "batch_error_log.txt") -Value $logEntry
    } else {
        Write-Host "[SUCCESS] " -NoNewline; Write-Host $name -ForegroundColor Green
    }
}

Write-Host ("`n" + "="*50) -ForegroundColor Gray
Write-Host "🎉 Batch processing completed!" -ForegroundColor Yellow
Write-Host "Outputs saved in: $outRoot"