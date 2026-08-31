# 1. Move up to the 'services' directory (Parent of synthetic_data_generation)
Write-Host "Moving to parent directory context..." -ForegroundColor Cyan
Push-Location ..

try {
    # 2. Load environment variables from .env (Now in the current directory)
    $EnvFilePath = ".env"
    if (Test-Path $EnvFilePath) {
        Write-Host "Loading environment variables from $EnvFilePath..." -ForegroundColor Cyan
        Get-Content $EnvFilePath | Where-Object { $_ -notmatch "^#" -and $_ -ne "" } | ForEach-Object {
            $line = $_.Trim()
            $key, $value = $line -split '=', 2
            $key = $key.Trim()
            $value = $value.Trim().Trim('"').Trim("'")
            [System.Environment]::SetEnvironmentVariable($key, $value, [System.EnvironmentVariableTarget]::Process)
        }
    }
    else {
        Write-Warning ".env file not found in $(Get-Location). Using defaults."
    }

    # 3. Determine Requirements File based on Device
    $Device = [System.Environment]::GetEnvironmentVariable("SYNTHETIC_DATA_DEVICE")
    if ([string]::IsNullOrWhiteSpace($Device)) { 
        $Device = "CPU" 
        Write-Host "SYNTHETIC_DATA_DEVICE not set, defaulting to CPU." -ForegroundColor Yellow
    }
    $ReqFile = "requirements${Device}.txt"
    
    # 4. Docker Configuration
    $ImageName = "synthetic-data-service"
    $ContainerName = "platform-synthetic-data-standalone"
    $PortMapping = "8004:600"

    # 5. Cleanup Old Containers
    Write-Host "Stopping and removing old container..."
    docker stop $ContainerName 2>$null
    docker rm $ContainerName 2>$null

    # 6. Build Image
    # Context is now '.' (services folder)
    # File is inside the subdirectory: synthetic_data_generation/Dockerfile
    Write-Host "Building Docker image ($ImageName)..." -ForegroundColor Cyan
    docker build -t $ImageName -f synthetic_data_generation/Dockerfile --build-arg REQUIREMENTS_FILE=$ReqFile .

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed."
        exit 1
    }

    # 7. Run Container
    Write-Host "Running container on port $PortMapping..." -ForegroundColor Cyan
    docker run --rm -d `
        --name $ContainerName `
        --env-file $EnvFilePath `
        -p $PortMapping `
        $ImageName

    Write-Host "Service started! Access Swagger UI at http://localhost:8004/docs" -ForegroundColor Green

}
finally {
    # 8. Always return to the original directory
    Pop-Location
}