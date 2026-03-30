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
from utils.paths import get_config_path, get_ui_path

loader = QUiLoader()
Config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)


class SnapType(Enum):
    DARK = "dark"
    EMPTY = "empty"


class SnapWindow(QtWidgets.QDialog):
    ImageChanged = QtCore.Signal(np.ndarray)
    ScanOver = QtCore.Signal()
    ProgressBarChanged = QtCore.Signal(int, str)

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

    def scan_thread(self):
        server_thread = Thread(
            target=pipe.detector_server,
            args=(r"\\.\pipe\detectResult", b"ctRestruct", self.detector_receive),
            daemon=True,
        )
        server_thread.start()
        py34 = os.environ.get("py34", "")
        if not py34:
            QtWidgets.QMessageBox.critical(
                self,
                "警告",
                "py34环境变量未配置!请检查系统环境变量, 保证py34环境变量指向3.4版本python.exe的路径",
            )
            return

        full_filename = os.path.join(
            self.parent_window.project_path, self.file_name_line_edit.text().strip()
        )
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
                    print(
                        "555555",
                        abs(
                            self.parent_window.xray_current
                            - float(self.parent_window.current_line_edit.text())
                        ),
                        self.parent_window.xray_voltage,
                    )
                    print(
                        "555555",
                        self.parent_window.current_line_edit.text(),
                        self.parent_window.voltage_line_edit.text(),
                    )
                    break
                time.sleep(0.5)
                t += 0.5
                if t > time_out:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "警告",
                        "x射线管启动失败, 请检查x射线管控制器是否连接正常",
                    )
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
        self.full_filename = full_filename
        ready_cmd = sub.stdout.readline()
        self.ProgressBarChanged.emit(20, "准备中")
        if not ready_cmd.startswith(b"READY"):
            print("ready_cmd333", ready_cmd)
            sys.exit(1)
        sub.stdin.write("start\n".encode())
        sub.stdin.flush()
        # ZC.motion_rotation(70)
        cmd = sub.stdout.readline()
        self.ProgressBarChanged.emit(100, "流程结束")
        self.parent_window.tab_widget.setEnabled(True)
        if self.snap_type == SnapType.EMPTY:
            self.parent_window.xray_off()

    def button_start(self):
        # config = Config.get("ZolixMcController", None)
        # if not config:
        #    QtWidgets.QMessageBox.critical(self, "警告", "转台控制器配置出错!请检查config.yaml文件")
        #    return
        # controller = ZolixMcController(config["port"], config["baudrate"])
        self.ProgressBarChanged.emit(0, "初始化中")
        self.parent_window.tab_widget.setEnabled(False)
        scan_thread = Thread(target=self.scan_thread, daemon=True)
        scan_thread.start()
        self.ProgressBarChanged.emit(5, "")
        print("666666")
