# ============================================================
# PyCT Windows 构建脚本
# 用法：在仓库根目录打开 PowerShell，执行：
#   .\scripts\build_win.ps1
#
# 首次运行会自动下载 Python 3.10 embeddable 到 .python_build/，
# 后续复用，不污染系统环境。
#
# 前置：Inno Setup 6 已安装（winget install JRSoftware.InnoSetup）
# ============================================================

param(
    [string]$PythonExe = "",          # 留空则自动使用本地便携 Python
    [string]$InnoSetupExe = "",       # 留空则自动搜索
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

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PyCT 构建脚本 v$Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ============================================================
# 自动获取纯净 Python 3.10（不污染系统）
# ============================================================
function Ensure-BuildPython {
    param([string]$ManualPython)

    if ($ManualPython -and (Test-Path $ManualPython)) {
        Write-Host "  使用指定 Python: $ManualPython" -ForegroundColor Green
        return $ManualPython
    }

    $PyBuildDir = "$RepoRoot\.python_build"
    $PyInstaller = "$PyBuildDir\python.exe"

    if (Test-Path $PyInstaller) {
        Write-Host "  使用本地构建 Python: $PyInstaller" -ForegroundColor Green
        return $PyInstaller
    }

    Write-Host "  未找到构建用 Python，正在下载 Python 3.10 安装版..." -ForegroundColor Yellow

    # 下载 Python 3.10 nuget 包（自带 pip，比 embeddable 好用）
    $PyVersion = "3.10.11"
    $PyUrl = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-amd64.exe"
    $PySetup = "$RepoRoot\.python_build_setup.exe"

    if (-not (Test-Path (Split-Path $PyBuildDir))) {
        New-Item -ItemType Directory (Split-Path $PyBuildDir) -Force | Out-Null
    }

    # 下载
    Write-Host "  下载 Python $PyVersion ..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $PyUrl -OutFile $PySetup -UseBasicParsing

    # 静默安装到本地目录（不写注册表、不加 PATH）
    Write-Host "  静默安装到 $PyBuildDir ..." -ForegroundColor Yellow
    & $PySetup /quiet `
        InstallAllUsers=0 `
        TargetDir="$PyBuildDir" `
        DefaultAllUsersTargetDir="$PyBuildDir" `
        AssociateFiles=0 `
        Shortcuts=0 `
        Include_launcher=0 `
        Include_test=0 `
        Include_doc=0 `
        Include_tcltk=0 `
        PrependPath=0 `
        CompileAll=0

    # 等待安装完成
    Start-Sleep -Seconds 5

    # 清理安装包
    Remove-Item $PySetup -Force -ErrorAction SilentlyContinue

    if (-not (Test-Path $PyInstaller)) {
        Write-Error "Python 安装失败：未找到 $PyInstaller"
        Write-Error "请手动安装 Python 3.10 并使用 -PythonExe 参数指定路径"
        exit 1
    }

    # 确保 pip 可用
    & $PyInstaller -m ensurepip --upgrade 2>$null
    & $PyInstaller -m pip install --upgrade pip

    Write-Host "  Python $PyVersion 安装完成: $PyInstaller" -ForegroundColor Green
    return $PyInstaller
}

# ============================================================
# 自动查找 Inno Setup
# ============================================================
function Find-InnoSetup {
    param([string]$ManualPath)

    if ($ManualPath -and (Test-Path $ManualPath)) {
        return $ManualPath
    }

    $candidates = @(
        # D 盘（你的实际安装位置）
        "D:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "D:\Program Files\Inno Setup 6\ISCC.exe",
        # winget 用户级安装
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        # 传统安装（C 盘）
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        # Scoop
        "$env:USERPROFILE\scoop\apps\inno-setup\current\ISCC.exe"
    )

    foreach ($c in $candidates) {
        if (Test-Path $c) {
            Write-Host "  找到 Inno Setup: $c" -ForegroundColor Green
            return $c
        }
    }

    $inPath = Get-Command ISCC -ErrorAction SilentlyContinue
    if ($inPath) {
        Write-Host "  找到 Inno Setup (PATH): $($inPath.Source)" -ForegroundColor Green
        return $inPath.Source
    }

    return $null
}

# ============================================================
# 开始构建
# ============================================================

# ---- 0. 获取纯净 Python ----
Write-Host "`n[0/5] 准备构建用 Python 环境..." -ForegroundColor Yellow
$BuildPython = Ensure-BuildPython -ManualPython $PythonExe

# ---- 1. 创建 venv ----
$VenvDir = "$RepoRoot\.venv_build"
$VenvPython = "$VenvDir\Scripts\python.exe"
$VenvPip = "$VenvDir\Scripts\pip.exe"

if (-not $SkipVenv) {
    Write-Host "`n[1/5] 创建构建虚拟环境..." -ForegroundColor Yellow
    if (Test-Path $VenvDir) {
        Remove-Item -Recurse -Force $VenvDir
    }
    & $BuildPython -m venv $VenvDir

    Write-Host "  安装项目依赖..."
    & $VenvPip install --upgrade pip
    & $VenvPip install -r "$RepoRoot\requirements_qt.txt"
    & $VenvPip install pyinstaller
    Write-Host "  venv 创建完成" -ForegroundColor Green
} else {
    Write-Host "`n[1/5] 跳过 venv 创建（使用已有环境）" -ForegroundColor Gray
    if (-not (Test-Path $VenvPython)) {
        Write-Error "未找到已有 venv，请去掉 -SkipVenv 重新运行"
        exit 1
    }
}

# ---- 2. PyInstaller 打包 ----
Write-Host "`n[2/5] PyInstaller 打包..." -ForegroundColor Yellow
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

& "$VenvDir\Scripts\pyinstaller.exe" `
    --noconfirm `
    --distpath "$BuildDir\dist" `
    --workpath "$BuildDir\work" `
    --specpath "$BuildDir" `
    "$RepoRoot\scripts\pyct.spec"

if (-not (Test-Path "$DistDir\PyCT.exe")) {
    Write-Error "PyInstaller 打包失败：未找到 PyCT.exe"
    exit 1
}
Write-Host "  PyCT.exe 生成成功" -ForegroundColor Green

# ---- 3. 复制额外资源 ----
Write-Host "`n[3/5] 复制额外资源..." -ForegroundColor Yellow

# config
if (-not (Test-Path "$DistDir\config")) { New-Item -ItemType Directory "$DistDir\config" | Out-Null }
Copy-Item "$RepoRoot\config.yaml" "$DistDir\config\default_config.yaml" -Force

# detector_bridge
$DetBridge = "$DistDir\detector_bridge"
if (-not (Test-Path $DetBridge)) { New-Item -ItemType Directory $DetBridge | Out-Null }
Copy-Item "$RepoRoot\detector.py" "$DetBridge\detector.py" -Force -ErrorAction SilentlyContinue

@"
# Detector Bridge (Python 3.4)

此目录用于放置 Python 3.4 运行时和设备 SDK。

## 设置步骤：
1. 将 Python 3.4 便携版解压到 py34\ 子目录
   结构：detector_bridge\py34\python.exe
2. 将设备 SDK 的 DLL 放到 py34\Lib\site-packages\ 或系统 PATH 中
3. 安装 detector 依赖：py34\python.exe -m pip install <所需包>

## 注意：
- 如果没有设备/SDK，校准和重建功能不受影响
- 主程序会自动检测 py34 是否可用，不可用时进入离线模式
"@ | Out-File -Encoding utf8 "$DetBridge\README.txt"

# docs
if (-not (Test-Path "$DistDir\docs")) { New-Item -ItemType Directory "$DistDir\docs" | Out-Null }
if (Test-Path "$RepoRoot\scripts\Offline_Install_Guide.md") {
    Copy-Item "$RepoRoot\scripts\Offline_Install_Guide.md" "$DistDir\docs\" -Force
}

Write-Host "  资源复制完成" -ForegroundColor Green

# ---- 4. 生成安装包 ----
if (-not $SkipInstaller) {
    Write-Host "`n[4/5] Inno Setup 生成安装包..." -ForegroundColor Yellow
    $ISCC = Find-InnoSetup -ManualPath $InnoSetupExe

    if (-not $ISCC) {
        Write-Warning "=========================================="
        Write-Warning "  未找到 Inno Setup！"
        Write-Warning "  安装方式（任选其一）："
        Write-Warning "    winget install JRSoftware.InnoSetup"
        Write-Warning "    https://jrsoftware.org/isdl.php"
        Write-Warning "  或手动指定路径："
        Write-Warning "    .\scripts\build_win.ps1 -InnoSetupExe 'D:\...\ISCC.exe'"
        Write-Warning "=========================================="
        Write-Warning "  跳过安装包生成（dist 目录仍可直接使用）"
    } else {
        & $ISCC "/DAppVersion=$Version" "/DDistDir=$DistDir" "/DOutputDir=$BuildDir\installer" "$RepoRoot\scripts\setup.iss"
        Write-Host "  安装包生成完成" -ForegroundColor Green
    }
} else {
    Write-Host "`n[4/5] 跳过安装包生成" -ForegroundColor Gray
}

# ---- 5. Portable zip（可选） ----
if ($Portable) {
    Write-Host "`n[5/5] 生成便携版 zip..." -ForegroundColor Yellow
    $ZipPath = "$BuildDir\PyCT_Portable_$Version.zip"
    if (Test-Path $ZipPath) { Remove-Item $ZipPath }
    Compress-Archive -Path "$DistDir\*" -DestinationPath $ZipPath
    Write-Host "  便携版：$ZipPath" -ForegroundColor Green
} else {
    Write-Host "`n[5/5] 跳过便携版" -ForegroundColor Gray
}

# ---- 完成 ----
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  构建完成！" -ForegroundColor Cyan
Write-Host "  构建 Python: $BuildPython" -ForegroundColor Gray
Write-Host "  dist 目录：$DistDir" -ForegroundColor White
$SetupFiles = Get-ChildItem "$BuildDir\installer\PyCT_Setup_*.exe" -ErrorAction SilentlyContinue
if ($SetupFiles) {
    Write-Host "  安装包：$($SetupFiles[0].FullName)" -ForegroundColor White
}
Write-Host "========================================" -ForegroundColor Cyan

# ---- 清理提示 ----
Write-Host "`n提示：" -ForegroundColor Gray
Write-Host "  - 构建用 Python 在 .python_build\（可删除或复用）" -ForegroundColor Gray
Write-Host "  - 构建用 venv 在 .venv_build\（可删除或复用）" -ForegroundColor Gray
Write-Host "  - 建议把 .python_build 和 .venv_build 加入 .gitignore" -ForegroundColor Gray
