set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

venv_python := ".venv_build\\Scripts\\python.exe"

# 从源码直接运行（本地开发测试，不需要打包）
run:
    & "venv_python" pyct_app.py

# 首次：创建 venv + 安装全部依赖
setup:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage setup

# PyInstaller → build_output/dist/PyCT/PyCT.exe（清理旧产物后构建）
build:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage build

# 增量 PyInstaller（不清理，复用旧 work 目录，更快）
rebuild:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage rebuild

# 在已有 dist 基础上组装便携目录 + 打 zip
package:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage package

# 完整流程：setup + build + package
full:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage full

# 清理所有构建产物
clean:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage clean

# 用 conda + CUDA 11 创建打包环境（兼容旧驱动 465）
conda-setup:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage conda-setup

# 从已有 conda 环境重新跑 PyInstaller（跳过环境重建）
conda-build:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage conda-build

# 修复已有 conda 环境（补装 astra + 重装 pip 依赖，不重建环境）
conda-fix:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage conda-fix

# conda-setup + conda-build + package
conda-full:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage conda-full

# 在已有 conda dist 基础上组装便携目录 + zip（CUDA 11 版本）
conda-package:
    powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1 -Stage conda-package
