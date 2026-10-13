# InferPath — 1-Command Reviewer Demo (Windows PowerShell)
# Launches FastAPI backend in background, executes a test request, and opens the Streamlit dashboard.

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  🧭 InferPath — Research Prototype 1-Command Demo" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Determine Python and Streamlit binaries
if (Test-Path ".venv\Scripts\python.exe") {
    $PYTHON = ".venv\Scripts\python.exe"
    $STREAMLIT = ".venv\Scripts\streamlit.exe"
} else {
    $PYTHON = "python"
    $STREAMLIT = "streamlit"
}

$pyVer = & $PYTHON --version
Write-Host "Using Python: $pyVer" -ForegroundColor Gray

# Check if port 8000 is already active
$apiJob = $null
$isApiUp = $false

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/health" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($health.status -eq "healthy") {
        Write-Host "✔ FastAPI backend is already running on http://127.0.0.1:8000" -ForegroundColor Green
        $isApiUp = $true
    }
} catch {
    $isApiUp = $false
}

if (-not $isApiUp) {
    Write-Host "Starting FastAPI backend in background (simulated/offline mode)..." -ForegroundColor Yellow
    $env:ADAPTER_MODE = "simulated"
    $apiJob = Start-Process -FilePath $PYTHON -ArgumentList "-m", "uvicorn", "inferpath.api.main:app", "--host", "127.0.0.1", "--port", "8000" -PassThru -WindowStyle Hidden

    # Poll for readiness
    Write-Host "Waiting for API readiness at http://127.0.0.1:8000/v1/health..." -ForegroundColor Gray
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/health" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($health.status -eq "healthy") {
                Write-Host "✔ FastAPI backend ready." -ForegroundColor Green
                $isApiUp = $true
                break
            }
        } catch {}
    }
}

try {
    Write-Host ""
    Write-Host "Sending sample verification request to POST http://127.0.0.1:8000/v1/respond..." -ForegroundColor Cyan
    $body = @{
        prompt = "Calculate: (15 * 8) + (240 / 6)"
        privacy_class = "public"
    } | ConvertTo-Json

    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/respond" -Method Post -Body $body -ContentType "application/json"
    Write-Host "Response received:" -ForegroundColor Green
    $resp | ConvertTo-Json -Depth 4 | Write-Host -ForegroundColor White
    Write-Host ""

    Write-Host "Launching Streamlit Research Dashboard..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to terminate the demo." -ForegroundColor Gray
    $env:INFERPATH_API_URL = "http://127.0.0.1:8000"
    & $STREAMLIT run app/dashboard.py
}
finally {
    if ($null -ne $apiJob -and -not $apiJob.HasExited) {
        Write-Host "Shutting down background FastAPI backend (PID: $($apiJob.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $apiJob.Id -Force -ErrorAction SilentlyContinue
    }
}
