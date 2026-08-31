# 1. Move to the services directory context
$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir
Push-Location ..

try {
    $EnvFilePath = ".env"
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
    } else {
        Write-Warning ".env file not found in $(Get-Location). Using defaults."
    }

    $Device = $env:SYNTHETIC_DATA_DEVICE
    if ([string]::IsNullOrWhiteSpace($Device)) {
        $Device = "CPU"
        Write-Host "SYNTHETIC_DATA_DEVICE not set, defaulting to CPU." -ForegroundColor Yellow
    }

    $ReqFile = "requirements$Device.txt"
    $ImageName = "synthetic-model-training-service"
    $ContainerName = "platform-synthetic-model-training-standalone"
    $PortMapping = "8007:601"

    Write-Host "------------------------------------------------" -ForegroundColor Yellow
    Write-Host " Starting Synthetic Model Training Service (Docker)" -ForegroundColor Yellow
    Write-Host " Requirements: $ReqFile" -ForegroundColor Yellow
    Write-Host "------------------------------------------------" -ForegroundColor Yellow

    docker stop $ContainerName 2>$null
    docker rm $ContainerName 2>$null

    Write-Host "Building Docker image ($ImageName) using synthetic_model_training/Dockerfile..." -ForegroundColor Cyan
    docker build -t $ImageName -f synthetic_model_training/Dockerfile --build-arg REQUIREMENTS_FILE=$ReqFile .

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed."
        exit 1
    }

    $RunArgs = @(
        "--rm"
        "-d"
        "--name"
        $ContainerName
        "-p"
        $PortMapping
        $ImageName
    )

    if (Test-Path $EnvFilePath) {
        $RunArgs.Insert(4, $EnvFilePath)
        $RunArgs.Insert(4, "--env-file")
    }

    Write-Host "Starting container on port $PortMapping..." -ForegroundColor Green
    docker run @RunArgs
    Write-Host "Service started! Access Swagger UI at http://localhost:8007/docs" -ForegroundColor Green
}
finally {
    Pop-Location
}
