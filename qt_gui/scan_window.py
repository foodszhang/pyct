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
import queue, threading
from utils.paths import get_config_path, get_ui_path, find_py34, get_detector_bridge_dir


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


class ScanWindow(QtWidgets.QDialog):
    ImageChanged = QtCore.Signal(np.ndarray)
    ProgressBarChanged = QtCore.Signal(int, str)
    error = QtCore.Signal(str)

    def __init__(self, parent_window, parent=None):
        super().__init__()
        self.parent_window = parent_window
        self.ui = loader.load(get_ui_path("scan.ui"), None)
        self.button_box = self.ui.findChild(QtWidgets.QDialogButtonBox, "buttonBox")
        self.button_box.accepted.connect(self._on_accepted)
        self.button_box.rejected.connect(self.ui.close)
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

    def _unfreeze_ui(self):
        """通过 signal 通知主线程解冻 UI"""
        self.ProgressBarChanged.emit(-1, "error")

    def scan_thread(self):
        try:
            sub = None
            config = Config.get("ZolixMcController", None)
            if not config:
                self.error.emit("转台控制器配置出错!请检查config.yaml文件")
                self._unfreeze_ui()
                return
            controller = ZolixMcController(config["port"], config["baudrate"])

            py34 = find_py34()
            if not py34:
                self.error.emit(
                    "找不到 Python 3.4 运行环境。\n"
                    "请确认 detector_bridge/py34/python.exe 存在，"
                    "或设置 py34 环境变量。"
                )
                self._unfreeze_ui()
                return
            print(f"[Detector] 使用 py34: {py34}")

            ready_event = threading.Event()
            server_thread = Thread(
                target=pipe.detector_server,
                args=(r"\\.\pipe\detectResult", b"ctRestruct", self.detector_receive),
                kwargs={"ready_event": ready_event},
                daemon=True,
            )
            server_thread.start()

            if not ready_event.wait(timeout=5):
                self.error.emit("pipe server 启动超时")
                self._unfreeze_ui()
                return

            detector_bridge_dir = get_detector_bridge_dir()

            # seq exposeTime gapTime number
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
                    self.error.emit("x射线管启动失败，请检查x射线管控制器是否连接正常")
                    self._unfreeze_ui()
                    return

            CREATE_NO_WINDOW = 0x08000000
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            sub = subprocess.Popen(
                [
                    str(py34),
                    "detector.py",
                    "seq",
                    self.expose_time_line_edit.text().strip(),
                    self.gap_time_line_edit.text().strip(),
                    self.number_line_edit.text().strip(),
                ],
                cwd=detector_bridge_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=CREATE_NO_WINDOW,
                startupinfo=startupinfo,
            )
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

            ready_cmd = _readline_with_timeout(sub.stdout, timeout=15)
            self.ProgressBarChanged.emit(20, "采集中")
            if ready_cmd is None or not ready_cmd.startswith(b"READY"):
                try:
                    stderr_out = sub.stderr.read(2048) if sub.stderr else b""
                except Exception:
                    stderr_out = b""
                print(f"[Error] detector.py stderr: {stderr_out.decode(errors='replace')}")
                self.error.emit(
                    f"探测器启动失败。\n"
                    f"ready_cmd={ready_cmd!r}\n"
                    f"stderr={stderr_out.decode(errors='replace')}"
                )
                sub.kill()
                self._unfreeze_ui()
                return
            sub.stdin.write("start\n".encode())
            sub.stdin.flush()
            controller.motion_rotation(380)
            # 带超时读取，防止子进程挂住导致永远阻塞
            cmd = _readline_with_timeout(sub.stdout, timeout=60)
            print("77777", cmd)
            # 主动关闭子进程
            try:
                if sub.poll() is None:
                    sub.stdin.close()
                    sub.wait(timeout=10)
            except Exception:
                pass
            for fut in self.fut_list:
                fut.result()
            self.ProgressBarChanged.emit(100, "采集完成")
        except Exception as e:
            import traceback

            print(f"[Error] scan_thread crash: {traceback.format_exc()}")
            self.error.emit(str(e))
        finally:
            # 确保子进程关闭
            try:
                if sub and sub.poll() is None:
                    sub.terminate()
                    sub.wait(timeout=5)
            except Exception:
                try:
                    sub.kill()
                except Exception:
                    pass
            self._unfreeze_ui()
            try:
                self.parent_window.xray_off()
            except Exception:
                pass

    def _on_accepted(self):
        self.ui.close()
        self.button_start()

    def button_start(self):
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
