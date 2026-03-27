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
import time

loader = QUiLoader()
Config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)


class ScanWindow(QtWidgets.QDialog):
    ImageChanged = QtCore.Signal(np.ndarray)
    ProgressBarChanged = QtCore.Signal(int, str)

    def __init__(self, parent_window, parent=None):
        super().__init__()
        self.parent_window = parent_window
        self.ui = loader.load("qt_gui/scan.ui", None)
        self.button_box = self.ui.findChild(QtWidgets.QDialogButtonBox, "buttonBox")
        self.button_box.accepted.connect(self.button_start)
        self.number_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "numberLineEdit")
        self.dark_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "darkLineEdit")
        self.empty_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "emptyLineEdit")
        self.defect_map_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "defectMapLineEdit"
        )
        self.expose_time_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "exposeTimeLineEdit"
        )
        self.gap_time_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "gapTimeLineEdit"
        )
        self.rotation_speed_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "rotationSpeedLineEdit"
        )

        self.pool = ThreadPoolExecutor(max_workers=20)
        self.img = None
        self.dark_img = None
        self.empty_img = None
        self.img_dict = {}
        self.fut_list = []

    def detector_receive(self, conn):
        dark = None
        empty = None
        max_dark = None
        if self.dark_img is not None and self.empty_img is not None:
            dark = self.dark_img
            empty = self.empty_img
            max_dark = np.max(dark)
            empty = np.where(empty <= max_dark, max_dark + 1, empty)
        try:
            while True:
                cnt, buf = conn.recv()
                self.ProgressBarChanged.emit(20 + cnt * 0.22, "")
                w = 1944
                h = 1536
                ar = np.frombuffer(buf, dtype=np.uint16).reshape(w, h)
                ar = np.flip(ar, axis=0)
                if self.dark_img is not None and self.empty_img is not None:
                    ar = (ar - dark) / (empty - dark)

                self.img_dict[cnt] = ar
                show_ar = cv2.resize(ar, (800, 800))
                # ar = cv2.normalize(ar, None, 0, 255, cv2.NORM_MINMAX)
                self.ImageChanged.emit(show_ar)
                fut = self.pool.submit(self.save_img, ar, cnt)
                self.fut_list.append(fut)

        except EOFError:
            # logger.log_info("连接断开， 采集结束")
            print("close!")

    def save_img(self, img, cnt):
        full_filename = os.path.join(self.parent_window.project_path, f"{cnt}.tif")
        print("saved", full_filename)
        img = np.clip(img, 0, 1)
        img = (img * 65535).astype(np.uint16)
        cv2.imwrite(full_filename, img)

    def scan_thread(self):
        config = Config.get("ZolixMcController", None)
        if not config:
            QtWidgets.QMessageBox.critical(
                self, "警告", "转台控制器配置出错!请检查config.yaml文件"
            )
            return
        controller = ZolixMcController(config["port"], config["baudrate"])
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

        # seq exposeTime gapTime number
        # 转盘速度与探测器时间 从设置上相互独立
        # 但是为了最终效果需要一定等式
        # 拍摄张数 * (曝光时间 + 间隔时间) = 360 / 转盘速度
        # 采集间隔度数 = 360 / 采集图片数 = 转速 * (曝光时间+间隔时间)
        # 转速2, 曝光时间100ms, 间隔时间400ms
        # 转速4 曝光时间100ms, 间隔时间150ms
        # 转速5 曝光时间100ms, 间隔时间100ms
        speed = int(self.rotation_speed_line_edit.text().strip())
        controller.set_speed(speed)
        controller.set_init_speed(speed)
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
                self.gap_time_line_edit.text().strip(),
                self.number_line_edit.text().strip(),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        # sub = subprocess.Popen(
        #    [
        #        py34,
        #        "detector.py",
        #        "seq",
        #        '100',
        #        '150',
        #        '60',
        #    ],
        #    stdin=subprocess.PIPE,
        #    stdout=subprocess.PIPE,
        # )
        assert sub.stdout
        assert sub.stdin
        if self.dark_line_edit.text().strip():
            dark_path = os.path.join(
                self.parent_window.project_path, self.dark_line_edit.text().strip()
            )
            if os.path.exists(dark_path):
                self.dark_img = cv2.imread(dark_path, -1)
        else:
            self.dark_img = None
        if self.empty_line_edit.text().strip():
            empty_path = os.path.join(
                self.parent_window.project_path, self.empty_line_edit.text().strip()
            )
            if os.path.exists(empty_path):
                self.empty_img = cv2.imread(empty_path, -1)
        else:
            self.empty_img = None

        ready_cmd = sub.stdout.readline()

        self.ProgressBarChanged.emit(20, "采集中")
        if not ready_cmd.startswith(b"READY"):
            print("ready_cmd333", ready_cmd)
            sys.exit(1)
        sub.stdin.write("start\n".encode())
        sub.stdin.flush()
        # 这里稍微多转一会
        controller.motion_rotation(380)
        # controller.motion_rotation(70)
        cmd = sub.stdout.readline()
        print("77777", cmd)
        for fut in self.fut_list:
            fut.result()
        self.ProgressBarChanged.emit(100, "采集完成")
        self.parent_window.tab_widget.setEnabled(True)
        xray_controller.xray_off()

    def button_start(self):
        # self.parent_window.ct_scan_progress.setValue(0)
        self.parent_window.ct_scan_progress_bar.setValue(0)
        self.parent_window.ct_scan_progress_label.setText("初始化中")
        self.parent_window.tab_widget.setEnabled(False)
        self.img = None
        self.dark_img = None
        self.empty_img = None
        self.img_dict = {}
        self.fut_list = []
        scan_thread = Thread(target=self.scan_thread, daemon=True)
        scan_thread.start()
