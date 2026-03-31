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
