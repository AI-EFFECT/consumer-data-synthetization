# 1. Normalize working directory to the script location
$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir

$EnvFilePath = Join-Path $ScriptDir "..\.env"
$Device = "CPU"

if (Test-Path $EnvFilePath) {
    Write-Host "Loading environment variables from $EnvFilePath..." -ForegroundColor Cyan
    Get-Content $EnvFilePath | Where-Object { $_ -notmatch "^#" -and $_ -ne "" } | ForEach-Object {
        if ($_ -match "=") {
            $parts = $_.Trim() -split '=', 2
            $key = $parts[0].Trim()
            $value = $parts[1].Trim().Trim('"').Trim("'")
            [System.Environment]::SetEnvironmentVariable($key, $value, [System.EnvironmentVariableTarget]::Process)
        }
    }
}

if (-not [string]::IsNullOrWhiteSpace($env:SYNTHETIC_DATA_DEVICE)) {
    $Device = $env:SYNTHETIC_DATA_DEVICE
}

$ReqFile = Join-Path $ScriptDir "..\synthetic_data_generation\requirements$Device.txt"
if (-not (Test-Path $ReqFile)) {
    Write-Error "Requirements file '$ReqFile' not found. Please set SYNTHETIC_DATA_DEVICE to CPU or GPU and ensure the file exists."
    exit 1
}

$VenvPath = Join-Path $ScriptDir "venv"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

if (-not (Test-Path $ActivateScript)) {
    Write-Host "Virtual environment not found. Creating venv..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
. $ActivateScript

Write-Host "Installing dependencies from $ReqFile..." -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install -r "$ReqFile"

$RootPath = (Resolve-Path (Join-Path $ScriptDir "..\..") ).Path
$ServicesPath = Join-Path $RootPath "services"
$SyntheticDataPath = (Resolve-Path (Join-Path $ScriptDir "..\synthetic_data_generation") ).Path
$env:PYTHONPATH = "$ScriptDir;$RootPath;$ServicesPath;$SyntheticDataPath"

$Port = if (-not [string]::IsNullOrWhiteSpace($env:SYNTHETIC_TRAINING_PORT)) { $env:SYNTHETIC_TRAINING_PORT } else { "6001" }

Write-Host "PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor Gray
Write-Host "Starting Synthetic Model Training API on port $Port..." -ForegroundColor Green
python -m uvicorn main:app --host 0.0.0.0 --port $Port --reload
