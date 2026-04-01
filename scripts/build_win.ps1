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
    [ValidateSet("setup", "build", "rebuild", "package", "full", "clean", "conda-setup", "conda-build", "conda-full", "conda-fix")]
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
    # 兼容 onedir（默认）和 onefile 两种 spec 模式
    if (-not (Test-Path "$DistDir\PyCT\PyCT.exe") -and -not (Test-Path "$DistDir\PyCT.exe")) {
        Write-Error "PyCT.exe not found. Run 'just build' or 'just conda-build' first."
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
# Stage: conda-setup — 用 conda 创建 CUDA 11 环境（兼容旧驱动）
# ============================================================
function Invoke-CondaSetup {
    Write-Host "`n[CondaSetup] Creating conda environment for CUDA 11 build..." -ForegroundColor Yellow

    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaCmd) {
        Write-Error "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    }

    $mambaCmd = Get-Command mamba -ErrorAction SilentlyContinue
    if (-not $mambaCmd) {
        Write-Warning "mamba not found in PATH, falling back to conda (slower)."
        Write-Warning "Consider: conda install -n base mamba -y"
        Set-Alias -Name mamba -Value conda -Scope Script
    } else {
        Write-Host " mamba: $($mambaCmd.Source)" -ForegroundColor Green
    }

    # 镜像 + 超时配置（仅 base 环境，写一次即可）
    Write-Host " Configuring conda channels and timeouts..."
    conda config --set channel_priority flexible 2>&1 | Out-Null
    conda config --set remote_read_timeout_secs 600 2>&1 | Out-Null
    conda config --set remote_connect_timeout_secs 60 2>&1 | Out-Null
    # --prepend 确保镜像优先级最高（已有则跳过）
    conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 2>&1 | Out-Null
    conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main 2>&1 | Out-Null

    $CondaEnvName = "pyct-cuda11"

    Write-Host " Removing old conda env if exists..."
    $envExists = conda env list 2>&1 | Select-String $CondaEnvName
    if ($envExists) {
        & conda env remove -n $CondaEnvName -y 2>&1 | Out-Null
    }

    Write-Host " Creating conda env $CondaEnvName with Python 3.11 (using mamba)..."
    mamba create -n $CondaEnvName python=3.11 -y
    if ($LASTEXITCODE -ne 0) { Write-Error "mamba create failed"; exit 1 }

    # 用 conda info --json 获取环境目录，避免 conda run 在 Windows 上的路径问题
    $condaInfo = conda info --json | ConvertFrom-Json
    $condaEnvsDir = $condaInfo.envs_dirs[0]
    $CondaEnvPath = "$condaEnvsDir\$CondaEnvName"
    $condaPython = "$CondaEnvPath\python.exe"
    $condaPip = "$CondaEnvPath\Scripts\pip.exe"
    if (-not (Test-Path $condaPython)) {
        Write-Error "conda env Python not found at $condaPython"
        exit 1
    }
    Write-Host "  Env path: $CondaEnvPath" -ForegroundColor Green
    Write-Host "  Python:   $condaPython" -ForegroundColor Green

    Write-Host " Installing astra-toolbox (CUDA 11) (using mamba)..."
    mamba install -n $CondaEnvName -c astra-toolbox -c nvidia astra-toolbox "cuda-version=11" -y
    if ($LASTEXITCODE -ne 0) { Write-Error "mamba install astra failed"; exit 1 }

    Write-Host " Installing Python dependencies..."
    & $condaPip install -r "$RepoRoot\requirements_qt.txt" pyinstaller
    if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed"; exit 1 }

    Write-Host "`n  --- Conda Env ---" -ForegroundColor Cyan
    $ErrorActionBackup = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $condaPython -c "import sys; print(f'  Python: {sys.version}')"
    & $condaPython -c "import astra; print(f'  ASTRA: {astra.__version__}, CUDA: {astra.use_cuda()}')"
    & $condaPython -c "import PySide6; print(f'  PySide6: {PySide6.__version__}')"
    $ErrorActionPreference = $ErrorActionBackup
    Write-Host "  ------------------" -ForegroundColor Cyan

    $condaPython | Out-File -Encoding utf8 "$RepoRoot\.conda_python_path.txt"
    Write-Host "[CondaSetup] Done. Python path saved to .conda_python_path.txt" -ForegroundColor Green
}

# ============================================================
# Stage: conda-build — 用 conda 环境的 PyInstaller 构建
# ============================================================
function Invoke-CondaBuild {
    $condaPythonFile = "$RepoRoot\.conda_python_path.txt"
    if (-not (Test-Path $condaPythonFile)) {
        Write-Error ".conda_python_path.txt not found. Run 'just conda-setup' first."
        exit 1
    }
    $CondaPython = (Get-Content $condaPythonFile).Trim()
    $CondaPyInstaller = Join-Path (Split-Path $CondaPython) "pyinstaller.exe"

    Write-Host "`n[CondaBuild] Running PyInstaller from conda env..." -ForegroundColor Yellow
    Write-Host " Python: $CondaPython"

    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }

    & $CondaPyInstaller `
        --noconfirm `
        --distpath "$BuildDir\dist" `
        --workpath "$BuildDir\work" `
        "$RepoRoot\scripts\pyct.spec"

    Write-Host "[CondaBuild] dist contents:"
    Get-ChildItem "$DistDir" -ErrorAction SilentlyContinue | Select-Object Name

    # 兼容 onedir（默认）和 onefile 两种 spec 模式
    $exePath = ""
    if (Test-Path "$DistDir\PyCT\PyCT.exe") {
        $exePath = "$DistDir\PyCT\PyCT.exe"
    } elseif (Test-Path "$DistDir\PyCT.exe") {
        $exePath = "$DistDir\PyCT.exe"
    } else {
        Write-Error "PyInstaller failed: PyCT.exe not found in $DistDir"
        exit 1
    }
    Write-Host "[CondaBuild] Done: $exePath" -ForegroundColor Green
}

# ============================================================
# Stage: conda-fix — 修复已有 conda 环境（不重建环境）
# ============================================================
function Invoke-CondaFix {
    Write-Host "`n[CondaFix] Repairing existing conda environment..." -ForegroundColor Yellow

    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaCmd) {
        Write-Error "conda not found."
        exit 1
    }

    $mambaCmd = Get-Command mamba -ErrorAction SilentlyContinue
    if (-not $mambaCmd) {
        Write-Warning "mamba not found, falling back to conda."
        Set-Alias -Name mamba -Value conda -Scope Script
    }

    $CondaEnvName = "pyct-cuda11"
    $CondaPythonFile = "$RepoRoot\.conda_python_path.txt"

    $condaInfo = conda info --json | ConvertFrom-Json
    $condaEnvsDir = $condaInfo.envs_dirs[0]
    $CondaEnvPath = "$condaEnvsDir\$CondaEnvName"
    $condaPython = "$CondaEnvPath\python.exe"
    $condaPip = "$CondaEnvPath\Scripts\pip.exe"

    if (-not (Test-Path $condaPython)) {
        Write-Error "conda env not found at $CondaEnvPath. Run 'just conda-setup' first."
        exit 1
    }
    Write-Host "  Env path: $CondaEnvPath" -ForegroundColor Green

    # 检查 astra 是否已安装
    Write-Host "  Checking astra-toolbox..." -ForegroundColor Yellow
    $ErrorActionBackup2 = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $astraCheck = & $condaPython -c "import astra; print(astra.__version__)" 2>&1
    $astraExitCode = $LASTEXITCODE
    $ErrorActionPreference = $ErrorActionBackup2
    if ($astraExitCode -ne 0) {
        Write-Host "  astra not found, installing via mamba..." -ForegroundColor Yellow
        mamba install -n $CondaEnvName -c astra-toolbox -c nvidia astra-toolbox "cuda-version=11" -y
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install astra-toolbox"
            exit 1
        }
    } else {
        Write-Host "  astra OK: $astraCheck" -ForegroundColor Green
    }

    # 重装 pip 依赖
    Write-Host "  Reinstalling pip dependencies..." -ForegroundColor Yellow
    & $condaPip install -r "$RepoRoot\requirements_qt.txt" pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install failed"
        exit 1
    }

    # 保存 python 路径
    $condaPython | Out-File -Encoding utf8 $CondaPythonFile

    # 验证
    Write-Host "`n  --- Conda Env ---" -ForegroundColor Cyan
    $ErrorActionBackup = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $condaPython -c "import sys; print(f'  Python: {sys.version}')"
    & $condaPython -c "import astra; print(f'  ASTRA: {astra.__version__}, CUDA: {astra.use_cuda()}')"
    & $condaPython -c "import PySide6; print(f'  PySide6: {PySide6.__version__}')"
    $ErrorActionPreference = $ErrorActionBackup
    Write-Host "  ------------------" -ForegroundColor Cyan
    Write-Host "[CondaFix] Done." -ForegroundColor Green
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
    "setup"       { Invoke-Setup }
    "build"       { Invoke-Build }
    "rebuild"     { Invoke-Rebuild }
    "package"     { Invoke-Package }
    "full"        { Invoke-Setup; Invoke-Build; Invoke-Package }
    "clean"       { Invoke-Clean }
    "conda-setup" { Invoke-CondaSetup }
    "conda-fix"   { Invoke-CondaFix }
    "conda-full"  { Invoke-CondaSetup; Invoke-CondaBuild; Invoke-Package }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Stage '$Stage' complete!"               -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
