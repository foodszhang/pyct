# ============================================================
# PyCT Windows Build Script (Portable mode)
# Usage: cd to repo root, run:
#   .\scripts\build_win.ps1
#
# Prerequisites: uv
# First run will auto-setup Python 3.13 venv + dependencies
# Subsequent runs use -SkipVenv to skip dependency installation
# ============================================================

param(
    [string]$AstraWhl = "",
    [switch]$SkipVenv,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$Version = "1.0.0"
$BuildDir = "$RepoRoot\build_output"
$DistDir = "$BuildDir\dist\PyCT"
$VenvDir = "$RepoRoot\.venv_build"
$VenvPython = "$VenvDir\Scripts\python.exe"
$VenvPyInstaller = "$VenvDir\Scripts\pyinstaller.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PyCT Build Script v$Version (Portable)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ============================================================
# Clean mode
# ============================================================
if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
    if (Test-Path "$RepoRoot\build") { Remove-Item -Recurse -Force "$RepoRoot\build" }
    Write-Host "Done." -ForegroundColor Green
    exit 0
}

# ============================================================
# Find astra wheel
# ============================================================
function Find-AstraWhl {
    param([string]$ManualPath)
    if ($ManualPath -and (Test-Path $ManualPath)) { return $ManualPath }

    $searchPaths = @(
        "$RepoRoot\astra_pkg",
        "$RepoRoot\deps",
        "$RepoRoot\scripts\deps",
        "$env:USERPROFILE\Downloads"
    )
    foreach ($dir in $searchPaths) {
        if (Test-Path $dir) {
            $whl = Get-ChildItem $dir -Recurse -Filter "astra_toolbox*cp313*win_amd64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($whl) {
                Write-Host "  Found astra wheel: $($whl.FullName)" -ForegroundColor Green
                return $whl.FullName
            }
        }
    }
    return $null
}

# ============================================================
# [0/4] Check uv
# ============================================================
Write-Host "`n[0/4] Checking prerequisites..." -ForegroundColor Yellow
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Error "uv not found. Install: winget install astral-sh.uv"
    exit 1
}
Write-Host "  uv: $($uvCmd.Source)" -ForegroundColor Green

# ============================================================
# [1/4] Create or reuse venv
# ============================================================
if (-not $SkipVenv) {
    Write-Host "`n[1/4] Setting up Python 3.13 build environment..." -ForegroundColor Yellow

    $ErrorActionPreference = "Continue"
    & uv python install 3.13 2>&1 | Out-Null
    $ErrorActionPreference = "Stop"

    if (Test-Path $VenvDir) {
        Remove-Item -Recurse -Force $VenvDir
    }
    & uv venv $VenvDir --python 3.13
    Write-Host "  venv created" -ForegroundColor Green

    Write-Host "  Installing dependencies..."
    & uv pip install -r "$RepoRoot\requirements_qt.txt" -p $VenvPython
    & uv pip install pyinstaller -p $VenvPython

    Write-Host "  Looking for astra-toolbox wheel..."
    $whl = Find-AstraWhl -ManualPath $AstraWhl
    if ($whl) {
        Write-Host "  Installing astra from: $whl"
        & uv pip install $whl -p $VenvPython
    } else {
        Write-Warning "  astra wheel not found! CUDA reconstruction will not work."
        Write-Warning "  Put .whl in $RepoRoot\astra_pkg\ or pass -AstraWhl path"
    }

    # Environment check
    Write-Host "`n  --- Environment ---" -ForegroundColor Cyan
    & $VenvPython -c "import sys; print(f'  Python: {sys.version}')"
    $ErrorActionPreference = "Continue"
    & $VenvPython -c "import astra; print(f'  ASTRA: {astra.__version__}, CUDA: {astra.use_cuda()}')" 2>&1 | Out-Host
    & $VenvPython -c "import PySide6; print(f'  PySide6: {PySide6.__version__}')" 2>&1 | Out-Host
    $ErrorActionPreference = "Stop"
    Write-Host "  --------------------" -ForegroundColor Cyan

} else {
    Write-Host "`n[1/4] Reusing existing venv (no dependency download)" -ForegroundColor Gray
    if (-not (Test-Path $VenvPython)) {
        Write-Error "No venv found. Run once without -SkipVenv first."
        exit 1
    }
    if (-not (Test-Path $VenvPyInstaller)) {
        Write-Error "PyInstaller not found in venv. Run once without -SkipVenv first."
        exit 1
    }
    Write-Host "  venv: $VenvDir" -ForegroundColor Green
}

# ============================================================
# [2/4] PyInstaller
# ============================================================
Write-Host "`n[2/4] PyInstaller packaging..." -ForegroundColor Yellow
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

& $VenvPyInstaller `
    --noconfirm `
    --distpath "$BuildDir\dist" `
    --workpath "$BuildDir\work" `
    "$RepoRoot\scripts\pyct.spec"

if (-not (Test-Path "$DistDir\PyCT.exe")) {
    Write-Error "PyInstaller failed: PyCT.exe not found"
    exit 1
}
Write-Host "  PyCT.exe created" -ForegroundColor Green

# ============================================================
# [3/4] Assemble portable directory
# ============================================================
Write-Host "`n[3/4] Assembling portable package..." -ForegroundColor Yellow

# --- config ---
if (-not (Test-Path "$DistDir\config")) { New-Item -ItemType Directory "$DistDir\config" | Out-Null }
Copy-Item "$RepoRoot\config.yaml" "$DistDir\config\default_config.yaml" -Force

# --- detector_bridge + py34 ---
$DetBridge = "$DistDir\detector_bridge"
if (-not (Test-Path $DetBridge)) { New-Item -ItemType Directory $DetBridge | Out-Null }
Copy-Item "$RepoRoot\detector.py" "$DetBridge\detector.py" -Force -ErrorAction SilentlyContinue

$Py34Sources = @(
    "$RepoRoot\detector_bridge\py34",
    "C:\Python34",
    "D:\Python34"
)
$Py34Copied = $false
foreach ($src in $Py34Sources) {
    if (Test-Path "$src\python.exe") {
        Write-Host "  Copying Python 3.4 from: $src" -ForegroundColor Yellow
        $dest = "$DetBridge\py34"
        if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
        Copy-Item $src $dest -Recurse -Force
        $Py34Copied = $true
        Write-Host "  Python 3.4 copied" -ForegroundColor Green
        break
    }
}
if (-not $Py34Copied) {
    Write-Warning "  Python 3.4 not found. Place in: $RepoRoot\detector_bridge\py34\"
}

@"
# Detector Bridge (Python 3.4)
Place Python 3.4 portable runtime in py34\ subdirectory.
Structure: detector_bridge\py34\python.exe
Without py34, calibration and reconstruction still work (offline mode).
"@ | Out-File -Encoding utf8 "$DetBridge\README.txt"

# --- vc_redist ---
$VcRedist = Get-ChildItem "$RepoRoot\astra_pkg" -Recurse -Filter "vc_redist.x64.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($VcRedist) {
    if (-not (Test-Path "$DistDir\redist")) { New-Item -ItemType Directory "$DistDir\redist" | Out-Null }
    Copy-Item $VcRedist.FullName "$DistDir\redist\vc_redist.x64.exe" -Force
    Write-Host "  vc_redist.x64.exe copied" -ForegroundColor Green
}

# --- docs ---
if (-not (Test-Path "$DistDir\docs")) { New-Item -ItemType Directory "$DistDir\docs" | Out-Null }
if (Test-Path "$RepoRoot\scripts\Offline_Install_Guide.md") {
    Copy-Item "$RepoRoot\scripts\Offline_Install_Guide.md" "$DistDir\docs\" -Force
}

# --- SDK ---
$SdkDir = "$RepoRoot\detector_bridge\sdk"
if (Test-Path $SdkDir) {
    Copy-Item $SdkDir "$DetBridge\sdk" -Recurse -Force
    Write-Host "  Detector SDK copied" -ForegroundColor Green
}

Write-Host "  Assembly complete" -ForegroundColor Green

# ============================================================
# [4/4] Create portable zip
# ============================================================
Write-Host "`n[4/4] Creating portable zip..." -ForegroundColor Yellow
$ZipPath = "$BuildDir\PyCT_Portable_$Version.zip"
if (Test-Path $ZipPath) { Remove-Item $ZipPath }
Compress-Archive -Path "$DistDir\*" -DestinationPath $ZipPath

$zipSize = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "  Portable zip: $ZipPath ($zipSize MB)" -ForegroundColor Green

# ============================================================
# Done
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Build complete!" -ForegroundColor Cyan
Write-Host "  Folder: $DistDir" -ForegroundColor White
Write-Host "  Zip:    $ZipPath" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nUsage:" -ForegroundColor Gray
Write-Host "  First build:    .\scripts\build_win.ps1" -ForegroundColor Gray
Write-Host "  Rebuild (fast): .\scripts\build_win.ps1 -SkipVenv" -ForegroundColor Gray
Write-Host "  Clean:          .\scripts\build_win.ps1 -Clean" -ForegroundColor Gray
