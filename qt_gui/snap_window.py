# use main.ui to create  main window use pyside6
import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader
from threading import Thread
import subprocess
import pipe
import os
import numpy as np
import cv2
import qt_gui.reconstruction as rec
import pyqtgraph as pg
from concurrent.futures import ThreadPoolExecutor
import yaml
from serial_controller import ZolixMcController
from enum import Enum
from typing import Literal
import time
import queue, threading
from utils.paths import get_config_path, get_ui_path


def _readline_with_timeout(stream, timeout=15):
    """从流中读取一行，支持超时（Windows 兼容）"""
    q = queue.Queue()

    def _reader():
        try:
            q.put(stream.readline())
        except Exception:
            q.put(b"")

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None

loader = QUiLoader()
Config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)


class SnapType(Enum):
    DARK = "dark"
    EMPTY = "empty"


class SnapWindow(QtWidgets.QDialog):
    ImageChanged = QtCore.Signal(np.ndarray)
    ScanOver = QtCore.Signal()
    ProgressBarChanged = QtCore.Signal(int, str)
    error = QtCore.Signal(str)

    def __init__(self, parent, type: Literal[SnapType.DARK, SnapType.EMPTY]):
        super().__init__()
        self.parent_window = parent
        self.ui = loader.load(get_ui_path("snap.ui"), None)
        self.button_box = self.ui.findChild(QtWidgets.QDialogButtonBox, "buttonBox")
        self.button_box.accepted.connect(self.button_start)
        self.file_name_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "filenameLineEdit"
        )
        self.number_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "numberLineEdit")
        self.expose_time_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "exposeTimeLineEdit"
        )
        self.img = None
        self.pool = ThreadPoolExecutor(max_workers=10)
        if type == SnapType.DARK:
            self.setWindowTitle("暗场采集")
            self.file_name_line_edit.setText("dark.tif")
        elif type == SnapType.EMPTY:
            self.setWindowTitle("空场采集")
            self.file_name_line_edit.setText("empty.tif")
        self.snap_type = type

    def detector_receive(self, conn):
        count = 0
        total_array = np.zeros((1944, 1536), dtype=np.float32)
        try:
            while True:
                cnt, buf = conn.recv()
                count += 1
                print(count)
                # logger.log_info(f"recv filename: {filename}")
                w = 1944
                h = 1536
                ar = np.frombuffer(buf, dtype=np.uint16).reshape(w, h)
                ar = np.flip(ar, axis=0)
                total_array += ar
                # ar = cv2.normalize(ar, None, 0, 255, cv2.NORM_MINMAX)
                # width = 512
                ar = cv2.resize(ar, (800, 800))
                # ar = cv2.cvtColor(ar, cv2.COLOR_GRAY2BGR)
                self.ImageChanged.emit(ar)

        except EOFError:
            # logger.log_info("连接断开， 采集结束")
            total_array /= count
            total_array = total_array.astype(np.uint16)
            cv2.imwrite(f"{self.full_filename}", total_array)
            print("close!")

    def _unfreeze_ui(self):
        """通过 signal 通知主线程解冻 UI"""
        self.ProgressBarChanged.emit(-1, "error")

    def scan_thread(self):
        try:
            py34 = os.environ.get("py34", "")
            if not py34:
                self.error.emit("py34环境变量未配置!请检查系统环境变量")
                self._unfreeze_ui()
                return

            full_filename = os.path.join(
                self.parent_window.project_path, self.file_name_line_edit.text().strip()
            )
            self.full_filename = full_filename

            server_thread = Thread(
                target=pipe.detector_server,
                args=(r"\\.\pipe\detectResult", b"ctRestruct", self.detector_receive),
                daemon=True,
            )
            server_thread.start()

            # seq exposeTime gapTime number
            if self.snap_type == SnapType.EMPTY:
                xray_controller = self.parent_window.xray_controller
                xray_controller.xray_on()
                time_out = 20
                t = 0
                while True:
                    if (
                        abs(
                            self.parent_window.xray_current
                            - float(self.parent_window.current_line_edit.text().strip())
                        )
                        < 1
                        and abs(
                            self.parent_window.xray_voltage
                            - float(self.parent_window.voltage_line_edit.text().strip())
                        )
                        < 1
                    ):
                        break
                    time.sleep(0.5)
                    t += 0.5
                    if t > time_out:
                        self.error.emit("x射线管启动失败，请检查x射线管控制器是否连接正常")
                        self._unfreeze_ui()
                        return

            sub = subprocess.Popen(
                [
                    py34,
                    "detector.py",
                    "seq",
                    self.expose_time_line_edit.text().strip(),
                    self.expose_time_line_edit.text().strip(),
                    self.number_line_edit.text().strip(),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            assert sub.stdout
            assert sub.stdin

            ready_cmd = _readline_with_timeout(sub.stdout, timeout=15)
            self.ProgressBarChanged.emit(20, "准备中")
            if ready_cmd is None:
                self.error.emit("探测器启动超时（15s未收到READY）")
                sub.kill()
                self._unfreeze_ui()
                return
            if not ready_cmd.startswith(b"READY"):
                self.error.emit(f"探测器握手失败，收到: {ready_cmd!r}")
                sub.kill()
                self._unfreeze_ui()
                return
            sub.stdin.write("start\n".encode())
            sub.stdin.flush()
            cmd = sub.stdout.readline()
            self.ProgressBarChanged.emit(100, "流程结束")
        except Exception as e:
            import traceback

            print(f"[Error] snap scan_thread crash: {traceback.format_exc()}")
            self.error.emit(str(e))
        finally:
            self._unfreeze_ui()
            if self.snap_type == SnapType.EMPTY:
                try:
                    self.parent_window.xray_off()
                except Exception:
                    pass

    def button_start(self):
        self.ProgressBarChanged.emit(0, "初始化中")
        self.parent_window.tab_widget.setEnabled(False)
        scan_thread = Thread(target=self.scan_thread, daemon=True)
        scan_thread.start()
        self.ProgressBarChanged.emit(5, "")
        print("666666")
