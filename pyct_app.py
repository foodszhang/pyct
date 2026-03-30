"""
PyCT GUI 统一入口脚本。

用途：
  - 源码运行：python pyct_app.py
  - PyInstaller 打包后：PyCT.exe

主要职责：
  1. Windows 下打包后隐藏控制台黑窗口
  2. 配置日志（控制台 + 文件，logs/pyct_yyyy-mm-dd.log）
  3. 环境检测（Python 版本、ASTRA、CUDA、硬件状态）
  4. 确保 config.yaml 存在
  5. 启动 Qt GUI
"""

import sys
import os

# ---------------------------------------------------------------
# 1. Windows 打包后隐藏控制台黑窗口
#    在 import 任何 GUI 相关模块之前执行。
# ---------------------------------------------------------------
if sys.platform == "win32":
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12
        GetStdHandle = user32.GetStdHandle
        SetStdHandle = kernel32.SetStdHandle
        GetConsoleWindow = user32.GetConsoleWindow
        ShowWindow = user32.ShowWindow
        console_hwnd = GetConsoleWindow()
        if console_hwnd:
            ShowWindow(console_hwnd, 0)
        SetStdHandle(STD_OUTPUT_HANDLE, None)
        SetStdHandle(STD_ERROR_HANDLE, None)
    except Exception:
        pass


# ---------------------------------------------------------------
# 2. 日志配置（控制台 + 文件，按天命名）
# ---------------------------------------------------------------
def _setup_logging():
    from datetime import datetime
    import logging

    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, f"pyct_{datetime.now().strftime('%Y-%m-%d')}.log")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)
    return log_file


# ---------------------------------------------------------------
# 3. 环境检测
# ---------------------------------------------------------------
def _detect_environment():
    import logging

    log = logging.getLogger(__name__)

    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    astra_available = False
    cuda_available = False
    try:
        import astra as ast

        astra_available = True
        try:
            ast.astra_dict("FDK_CUDA")
            cuda_available = True
        except Exception:
            pass
    except ImportError:
        log.warning("[Env] ASTRA 不可用（未安装）")
    except Exception as e:
        log.warning(f"[Env] ASTRA 检查失败：{e}")

    hardware_mode = "离线"
    try:
        import serial_controller

        if hasattr(serial_controller, "UltraBrightController"):
            hardware_mode = "在线"
    except ImportError:
        pass

    log.info("=" * 50)
    log.info(f"[Env] Python 版本: {py_version}")
    log.info(f"[Env] ASTRA 可用: {'是' if astra_available else '否'}")
    log.info(f"[Env] CUDA 可用: {'是' if cuda_available else '否'}")
    log.info(f"[Env] 硬件模式: {hardware_mode}")
    log.info("=" * 50)

    return {
        "python_version": py_version,
        "astra_available": astra_available,
        "cuda_available": cuda_available,
        "hardware_mode": hardware_mode,
    }


# ---------------------------------------------------------------
# 4. 源码运行时确保 CWD 为项目根目录
#    打包后（sys.frozen）不需要，资源已在 sys._MEIPASS
# ---------------------------------------------------------------
def _setup_develop_path():
    if getattr(sys, "frozen", False):
        return
    base = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(base):
        os.chdir(base)


# ---------------------------------------------------------------
# 5. 确保 config.yaml 存在
# ---------------------------------------------------------------
def _ensure_config():
    from utils.paths import ensure_config_exists, get_config_path
    import logging

    log = logging.getLogger(__name__)
    cfg_path = ensure_config_exists()
    log.info(f"[Config] 使用配置文件：{cfg_path}")


# ---------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------
_setup_develop_path()
log_file = _setup_logging()
_ensure_config()
env_info = _detect_environment()

from qt_gui.gui import start_gui

if __name__ == "__main__":
    import logging

    logging.info(f"[App] PyCT 启动，日志文件：{log_file}")
    start_gui()
