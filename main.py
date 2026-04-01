import sys, pathlib, traceback, datetime

# PyInstaller 打包后 stderr/stdout 为 None，需要重定向到文件
_log_path = pathlib.Path.home() / "pyct_crash.log"

if sys.stderr is None:
    sys.stderr = open(_log_path, "a", encoding="utf-8")
if sys.stdout is None:
    sys.stdout = open(_log_path, "a", encoding="utf-8")


def _crash_handler(exc_type, exc_value, exc_tb):
    with open(_log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.datetime.now()}] UNHANDLED EXCEPTION:\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)


sys.excepthook = _crash_handler

# from gui.gui import start_gui
from qt_gui.gui import start_gui

start_gui()
