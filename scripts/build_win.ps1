# ============================================================
# PyCT Windows Build Script
# Usage: cd to repo root, run:
#   .\scripts\build_win.ps1
#
# Prerequisites: uv, Inno Setup 6
# First run will auto-setup Python 3.13 venv + astra wheel
# ============================================================

param(
    [string]$InnoSetupExe = "",
    [string]$AstraWhl = "",
    [switch]$SkipVenv,
    [switch]$SkipInstaller,
    [switch]$Portable
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
Write-Host "  PyCT Build Script v$Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ============================================================
# Find Inno Setup
# ============================================================
function Find-InnoSetup {
    param([string]$ManualPath)

    if ($ManualPath -and (Test-Path $ManualPath)) { return $ManualPath }

    $candidates = @(
        "D:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "D:\Program Files\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:USERPROFILE\scoop\apps\inno-setup\current\ISCC.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            Write-Host "  Found Inno Setup: $c" -ForegroundColor Green
            return $c
        }
    }
    $inPath = Get-Command ISCC -ErrorAction SilentlyContinue
    if ($inPath) {
        Write-Host "  Found Inno Setup (PATH): $($inPath.Source)" -ForegroundColor Green
        return $inPath.Source
    }
    return $null
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
# [0/5] Check uv
# ============================================================
Write-Host "`n[0/5] Checking prerequisites..." -ForegroundColor Yellow
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Error "uv not found. Install it first: winget install astral-sh.uv"
    exit 1
}
Write-Host "  uv: $($uvCmd.Source)" -ForegroundColor Green

# ============================================================
# [1/5] Create venv (Python 3.13 via uv)
# ============================================================
if (-not $SkipVenv) {
    Write-Host "`n[1/5] Creating Python 3.13 build venv..." -ForegroundColor Yellow

    # Ensure Python 3.13 is available
    Write-Host "  Ensuring Python 3.13 is installed..."
    $ErrorActionPreference = "Continue"
    & uv python install 3.13 2>&1 | Out-Null
    $ErrorActionPreference = "Stop"

    # Create venv
    if (Test-Path $VenvDir) {
        Remove-Item -Recurse -Force $VenvDir
    }
    & uv venv $VenvDir --python 3.13
    Write-Host "  venv created at $VenvDir" -ForegroundColor Green

    # Install dependencies via uv pip
    Write-Host "  Installing project dependencies..."
    & uv pip install -r "$RepoRoot\requirements_qt.txt" -p $VenvPython
    & uv pip install pyinstaller -p $VenvPython

    # Install astra wheel
    Write-Host "  Looking for astra-toolbox wheel..."
    $whl = Find-AstraWhl -ManualPath $AstraWhl
    if ($whl) {
        Write-Host "  Installing astra from: $whl"
        & uv pip install $whl -p $VenvPython
    } else {
        Write-Warning "  astra wheel not found!"
        Write-Warning "  Download from: https://github.com/astra-toolbox/astra-toolbox/releases"
        Write-Warning "  Then either:"
        Write-Warning "    - Put the .whl in $RepoRoot\astra_pkg\"
        Write-Warning "    - Or pass -AstraWhl 'path\to\astra_toolbox-xxx.whl'"
    }

    # Verify environment
    Write-Host "`n  --- Environment Check ---" -ForegroundColor Cyan
    & $VenvPython -c "import sys; print(f'  Python: {sys.version}')"

    $ErrorActionPreference = "Continue"
    & $VenvPython -c "import astra; print(f'  ASTRA: {astra.__version__}, CUDA: {astra.use_cuda()}')" 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  ASTRA not available - CUDA reconstruction will not work in packaged app"
    }
    & $VenvPython -c "import PySide6; print(f'  PySide6: {PySide6.__version__}')" 2>&1 | Out-Host
    $ErrorActionPreference = "Stop"

    Write-Host "  --------------------------" -ForegroundColor Cyan

} else {
    Write-Host "`n[1/5] Skipping venv creation (using existing)" -ForegroundColor Gray
    if (-not (Test-Path $VenvPython)) {
        Write-Error "No existing venv found at $VenvDir. Run without -SkipVenv."
        exit 1
    }
}

# ============================================================
# [2/5] PyInstaller
# ============================================================
Write-Host "`n[2/5] PyInstaller packaging..." -ForegroundColor Yellow
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
# [3/5] Copy resources
# ============================================================
Write-Host "`n[3/5] Copying resources..." -ForegroundColor Yellow

# --- config ---
if (-not (Test-Path "$DistDir\config")) { New-Item -ItemType Directory "$DistDir\config" | Out-Null }
Copy-Item "$RepoRoot\config.yaml" "$DistDir\config\default_config.yaml" -Force

# --- detector_bridge + py34 ---
$DetBridge = "$DistDir\detector_bridge"
if (-not (Test-Path $DetBridge)) { New-Item -ItemType Directory $DetBridge | Out-Null }
Copy-Item "$RepoRoot\detector.py" "$DetBridge\detector.py" -Force -ErrorAction SilentlyContinue

# Python 3.4 portable
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
    Write-Warning "  Python 3.4 not found. Scanning feature will not work."
    Write-Warning "  Place Python 3.4 in: $RepoRoot\detector_bridge\py34\"
}

# detector_bridge README
@"
# Detector Bridge (Python 3.4)
Place Python 3.4 portable runtime in py34\ subdirectory.
Structure: detector_bridge\py34\python.exe
Without py34, calibration and reconstruction still work (offline mode).
"@ | Out-File -Encoding utf8 "$DetBridge\README.txt"

# --- vc_redist (from astra package if available) ---
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

# --- SDK DLLs ---
$SdkDir = "$RepoRoot\detector_bridge\sdk"
if (Test-Path $SdkDir) {
    Copy-Item $SdkDir "$DetBridge\sdk" -Recurse -Force
    Write-Host "  Detector SDK copied" -ForegroundColor Green
}

Write-Host "  Resources copied" -ForegroundColor Green

# ============================================================
# [4/5] Inno Setup installer
# ============================================================
if (-not $SkipInstaller) {
    Write-Host "`n[4/5] Inno Setup building installer..." -ForegroundColor Yellow
    $ISCC = Find-InnoSetup -ManualPath $InnoSetupExe

    if (-not $ISCC) {
        Write-Warning "  Inno Setup not found!"
        Write-Warning "  Install: winget install JRSoftware.InnoSetup"
        Write-Warning "  Or pass: -InnoSetupExe 'D:\...\ISCC.exe'"
        Write-Warning "  Skipping installer (dist folder is still usable)"
    } else {
        & $ISCC "/DAppVersion=$Version" "/DDistDir=$DistDir" "/DOutputDir=$BuildDir\installer" "$RepoRoot\scripts\setup.iss"
        Write-Host "  Installer created" -ForegroundColor Green
    }
} else {
    Write-Host "`n[4/5] Skipping installer" -ForegroundColor Gray
}

# ============================================================
# [5/5] Portable zip (optional)
# ============================================================
if ($Portable) {
    Write-Host "`n[5/5] Creating portable zip..." -ForegroundColor Yellow
    $ZipPath = "$BuildDir\PyCT_Portable_$Version.zip"
    if (Test-Path $ZipPath) { Remove-Item $ZipPath }
    Compress-Archive -Path "$DistDir\*" -DestinationPath $ZipPath
    Write-Host "  Portable: $ZipPath" -ForegroundColor Green
} else {
    Write-Host "`n[5/5] Skipping portable zip" -ForegroundColor Gray
}

# ============================================================
# Done
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Build complete!" -ForegroundColor Cyan
Write-Host "  dist: $DistDir" -ForegroundColor White
$SetupFiles = Get-ChildItem "$BuildDir\installer\PyCT_Setup_*.exe" -ErrorAction SilentlyContinue
if ($SetupFiles) {
    Write-Host "  Installer: $($SetupFiles[0].FullName)" -ForegroundColor White
}
if (Test-Path "$BuildDir\PyCT_Portable_*.zip") {
    $zip = Get-ChildItem "$BuildDir\PyCT_Portable_*.zip" | Select-Object -First 1
    Write-Host "  Portable: $($zip.FullName)" -ForegroundColor White
}
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nNotes:" -ForegroundColor Gray
Write-Host "  - Build venv in .venv_build\ (reuse with -SkipVenv)" -ForegroundColor Gray
Write-Host "  - Add .venv_build/ and build_output/ to .gitignore" -ForegroundColor Gray
