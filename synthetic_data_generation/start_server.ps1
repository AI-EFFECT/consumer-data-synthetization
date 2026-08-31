# Get the current directory and path to the parent .env file
$ServiceDir = Get-Location
$EnvFilePath = Join-Path $ServiceDir "..\.env"

# 1. Load environment variables from ../.env
if (Test-Path $EnvFilePath) {
    Write-Host "Loading environment variables from $EnvFilePath..." -ForegroundColor Cyan
    Get-Content $EnvFilePath | Where-Object { $_ -notmatch "^#" -and $_ -ne "" } | ForEach-Object {
        $line = $_.Trim()
        $key, $value = $line -split '=', 2
        
        # Trim whitespace and quotes
        $key = $key.Trim()
        $value = $value.Trim().Trim('"').Trim("'")
        
        # Set variable in the current process scope
        [System.Environment]::SetEnvironmentVariable($key, $value, [System.EnvironmentVariableTarget]::Process)
    }
}
else {
    Write-Warning ".env file not found at $EnvFilePath. Using defaults."
}

# 2. Determine Requirements File based on Device (CPU vs GPU)
$Device = [System.Environment]::GetEnvironmentVariable("SYNTHETIC_DATA_DEVICE")
if ([string]::IsNullOrWhiteSpace($Device)) { 
    $Device = "CPU" 
    Write-Host "SYNTHETIC_DATA_DEVICE not set, defaulting to CPU." -ForegroundColor Yellow
}

$ReqFile = "requirements${Device}.txt"

if (-not (Test-Path $ReqFile)) {
    Write-Error "Requirements file '$ReqFile' not found! Please check your SYNTHETIC_DATA_DEVICE setting."
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  - Device: $Device"
Write-Host "  - Requirements: $ReqFile"
Write-Host "  - Auth Enabled: $([System.Environment]::GetEnvironmentVariable('AUTH_ENABLED'))"

# 3. Setup Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# 4. Activate Virtual Environment
Write-Host "Activating venv..."
. .\venv\Scripts\Activate.ps1

# 5. Install Dependencies
Write-Host "Installing dependencies..."
pip install -r $ReqFile

# 6. Start Server
# We use 'python main.py' instead of 'uvicorn' directly to ensure
# the custom port (600) and reload logic defined in main.py is used.
Write-Host "Starting Synthetic Data Service..."
python main.py