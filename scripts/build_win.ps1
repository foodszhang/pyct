# ============================================================
# PyCT Windows Build Script (Portable mode)
# Usage:
#   .\scripts\build_win.ps1 -Stage setup    # 首次：创建 venv + 装依赖
#   .\scripts\build_win.ps1 -Stage build    # PyInstaller 构建（清理后）
#   .\scripts\build_win.ps1 -Stage rebuild  # PyInstaller 增量构建
#   .\scripts\build_win.ps1 -Stage package  # 组装便携目录 + zip
#   .\scripts\build_win.ps1 -Stage full     # setup + build + package
#   .\scripts\build_win.ps1 -Stage clean    # 清理
#   .\scripts\build_win.ps1                 # 默认 = full
# ============================================================

param(
    [ValidateSet("setup", "build", "rebuild", "package", "full", "clean")]
    [string]$Stage = "full",
    [string]$AstraWhl = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$RepoRoot       = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$Version        = "1.0.0"
$BuildDir       = "$RepoRoot\build_output"
$DistDir        = "$BuildDir\dist\PyCT"
$VenvDir        = "$RepoRoot\.venv_build"
$VenvPython     = "$VenvDir\Scripts\python.exe"
$VenvPyInstaller = "$VenvDir\Scripts\pyinstaller.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " PyCT Build Script v$Version"            -ForegroundColor Cyan
Write-Host " Stage: $Stage"                          -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ============================================================
# Helper: Find astra wheel
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
# Stage: setup — 创建 venv + 安装依赖
# ============================================================
function Invoke-Setup {
    Write-Host "`n[Setup] Checking prerequisites..." -ForegroundColor Yellow
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Write-Error "uv not found. Install: winget install astral-sh.uv"
        exit 1
    }
    Write-Host "  uv: $($uvCmd.Source)" -ForegroundColor Green

    Write-Host "`n[Setup] Creating Python 3.13 build venv..." -ForegroundColor Yellow
    $ErrorActionPreference = "Continue"
    & uv python install 3.13 2>&1 | Out-Null
    $ErrorActionPreference = "Stop"

    if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir }
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

    Write-Host "`n  --- Environment ---" -ForegroundColor Cyan
    & $VenvPython -c "import sys; print(f'  Python: {sys.version}')"
    $ErrorActionPreference = "Continue"
    & $VenvPython -c "import astra; print(f'  ASTRA: {astra.__version__}, CUDA: {astra.use_cuda()}')" 2>&1 | Out-Host
    & $VenvPython -c "import PySide6; print(f'  PySide6: {PySide6.__version__}')" 2>&1 | Out-Host
    $ErrorActionPreference = "Stop"
    Write-Host "  --------------------" -ForegroundColor Cyan
    Write-Host "[Setup] Done." -ForegroundColor Green
}

# ============================================================
# Stage: build — 清理 + PyInstaller
# ============================================================
function Invoke-Build {
    Assert-Venv
    Write-Host "`n[Build] Running PyInstaller (clean)..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
    Invoke-PyInstaller
    Write-Host "[Build] Done: $DistDir\PyCT.exe" -ForegroundColor Green
}

# ============================================================
# Stage: rebuild — 增量 PyInstaller（不清理 work 目录）
# ============================================================
function Invoke-Rebuild {
    Assert-Venv
    Write-Host "`n[Rebuild] Running PyInstaller (incremental)..." -ForegroundColor Yellow
    Invoke-PyInstaller
    Write-Host "[Rebuild] Done: $DistDir\PyCT.exe" -ForegroundColor Green
}

# ============================================================
# Stage: package — 组装便携目录 + zip
# ============================================================
function Invoke-Package {
    if (-not (Test-Path "$DistDir\PyCT.exe")) {
        Write-Error "PyCT.exe not found. Run 'just build' first."
        exit 1
    }
    Write-Host "`n[Package] Assembling portable directory..." -ForegroundColor Yellow

    # config
    if (-not (Test-Path "$DistDir\config")) { New-Item -ItemType Directory "$DistDir\config" | Out-Null }
    Copy-Item "$RepoRoot\config.yaml" "$DistDir\config\default_config.yaml" -Force

    # detector_bridge + py34
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

    # SDK
    $SdkDir = "$RepoRoot\detector_bridge\sdk"
    if (Test-Path $SdkDir) {
        Copy-Item $SdkDir "$DetBridge\sdk" -Recurse -Force
        Write-Host "  Detector SDK copied" -ForegroundColor Green
    }

    # vc_redist
    $VcRedist = Get-ChildItem "$RepoRoot\astra_pkg" -Recurse -Filter "vc_redist.x64.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($VcRedist) {
        if (-not (Test-Path "$DistDir\redist")) { New-Item -ItemType Directory "$DistDir\redist" | Out-Null }
        Copy-Item $VcRedist.FullName "$DistDir\redist\vc_redist.x64.exe" -Force
        Write-Host "  vc_redist.x64.exe copied" -ForegroundColor Green
    }

    # docs
    if (-not (Test-Path "$DistDir\docs")) { New-Item -ItemType Directory "$DistDir\docs" | Out-Null }
    if (Test-Path "$RepoRoot\scripts\Offline_Install_Guide.md") {
        Copy-Item "$RepoRoot\scripts\Offline_Install_Guide.md" "$DistDir\docs\" -Force
    }

    # README
    @"
# Detector Bridge (Python 3.4)
Place Python 3.4 portable runtime in py34\ subdirectory.
Structure: detector_bridge\py34\python.exe
Without py34, calibration and reconstruction still work (offline mode).
"@ | Out-File -Encoding utf8 "$DetBridge\README.txt"

    Write-Host "  Assembly complete" -ForegroundColor Green

    # zip
    Write-Host "`n[Package] Creating portable zip..." -ForegroundColor Yellow
    $ZipPath = "$BuildDir\PyCT_Portable_$Version.zip"
    if (Test-Path $ZipPath) { Remove-Item $ZipPath }
    Compress-Archive -Path "$DistDir\*" -DestinationPath $ZipPath
    $zipSize = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
    Write-Host "[Package] Done: $ZipPath ($zipSize MB)" -ForegroundColor Green
}

# ============================================================
# Stage: clean
# ============================================================
function Invoke-Clean {
    Write-Host "`n[Clean] Removing build artifacts..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
    if (Test-Path "$RepoRoot\build") { Remove-Item -Recurse -Force "$RepoRoot\build" }
    Write-Host "[Clean] Done." -ForegroundColor Green
}

# ============================================================
# Internal helpers
# ============================================================
function Assert-Venv {
    if (-not (Test-Path $VenvPython)) {
        Write-Error "No venv found. Run 'just setup' first."
        exit 1
    }
    if (-not (Test-Path $VenvPyInstaller)) {
        Write-Error "PyInstaller not found in venv. Run 'just setup' first."
        exit 1
    }
    Write-Host "  venv: $VenvDir" -ForegroundColor Green
}

function Invoke-PyInstaller {
    & $VenvPyInstaller `
        --noconfirm `
        --distpath "$BuildDir\dist" `
        --workpath "$BuildDir\work" `
        "$RepoRoot\scripts\pyct.spec"

    if (-not (Test-Path "$DistDir\PyCT.exe")) {
        Write-Error "PyInstaller failed: PyCT.exe not found"
        exit 1
    }
}

# ============================================================
# Dispatch
# ============================================================
switch ($Stage) {
    "setup"   { Invoke-Setup }
    "build"   { Invoke-Build }
    "rebuild" { Invoke-Rebuild }
    "package" { Invoke-Package }
    "full"    { Invoke-Setup; Invoke-Build; Invoke-Package }
    "clean"   { Invoke-Clean }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Stage '$Stage' complete!"               -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
