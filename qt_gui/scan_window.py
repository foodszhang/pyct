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


def _drain_stream(stream):
    if stream is None:
        return
    while True:
        line = stream.readline()
        if not line:
            break
        print(line.decode(errors="replace"), end="")


def _close_subprocess_streams(sub):
    if sub is None:
        return
    for stream in (sub.stdin, sub.stdout, sub.stderr):
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass


def _close_listener(listener_holder):
    if not listener_holder:
        return
    listener = listener_holder.get("listener")
    if listener is None:
        return
    try:
        listener.close()
    except Exception:
        pass


def _format_angle_name(angle_deg):
    return f"{float(angle_deg):.6f}".rstrip("0").rstrip(".")

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
        self.scan_mode = "连续采集"

        self.pool = ThreadPoolExecutor(max_workers=20)
        self.img = None
        self.dark_img = None
        self.empty_img = None
        self.projection_normalized = False
        self.img_dict = {}
        self.fut_list = []

    def detector_receive(self, conn):
        dark = None
        empty = None
        denominator = None
        if self.projection_normalized:
            dark = self.dark_img
            empty = self.empty_img
            denominator = np.maximum(empty - dark, 1.0)
        try:
            while True:
                cnt, buf = conn.recv()
                self._scan_received += 1
                progress = 20 + int(self._scan_received * 75 / max(self.scan_number, 1))
                self.ProgressBarChanged.emit(min(progress, 95), "")
                w = 1944
                h = 1536
                ar = np.frombuffer(buf, dtype=np.uint16).reshape(w, h)
                ar = np.flip(ar, axis=0)
                if self.projection_normalized:
                    ar_norm = np.clip((ar.astype(np.float32) - dark) / denominator, 0, 1)
                    ar_for_recon = ar_norm * 65535.0
                    show_ar = cv2.resize(ar_norm, (800, 800))
                else:
                    ar_for_recon = ar.astype(np.float32)
                    show_ar = cv2.resize(ar, (800, 800))

                angle_key = self._normalize_projection_key(cnt)
                self.img_dict[angle_key] = ar_for_recon.astype(np.float32)
                # ar = cv2.normalize(ar, None, 0, 255, cv2.NORM_MINMAX)
                self.ImageChanged.emit(show_ar)
                fut = self.pool.submit(self.save_img, ar_for_recon, angle_key)
                self.fut_list.append(fut)

        except EOFError:
            # logger.log_info("连接断开， 采集结束")
            print("close!")

    def save_img(self, img, cnt):
        filename = f"{_format_angle_name(cnt)}.tif"
        full_filename = os.path.join(self.parent_window.project_path, filename)
        print("saved", full_filename)
        if self.projection_normalized and np.max(img) <= 1.5:
            img = np.clip(img, 0, 1) * 65535
        else:
            img = np.clip(img, 0, 65535)
        img = img.astype(np.uint16)
        cv2.imwrite(full_filename, img)

    def _normalize_projection_key(self, cnt):
        try:
            value = float(cnt)
        except (TypeError, ValueError):
            return cnt
        if abs(value - round(value)) < 1e-6:
            return int(round(value))
        return value

    def _unfreeze_ui(self):
        """通过 signal 通知主线程解冻 UI"""
        self.ProgressBarChanged.emit(-1, "error")

    def scan_thread(self):
        sub = None
        server_holder = {}
        controller = None
        self.fut_list = []
        try:
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
                kwargs={"ready_event": ready_event, "listener_holder": server_holder},
                daemon=True,
            )
            server_thread.start()

            if not ready_event.wait(timeout=5):
                self.error.emit("pipe server 启动超时")
                _close_listener(server_holder)
                self._unfreeze_ui()
                return

            detector_bridge_dir = get_detector_bridge_dir()
            detector_script = os.path.join(
                os.path.dirname(detector_bridge_dir), "detector.py"
            )
            if not os.path.isfile(detector_script):
                detector_script = "detector.py"

            # seq exposeTime gapTime number
            self.scan_number = int(self.number_line_edit.text().strip())
            expose_time = int(self.expose_time_line_edit.text().strip())
            gap_time = int(self.gap_time_line_edit.text().strip())
            scan_mode = self.scan_mode
            if self.scan_number <= 0:
                raise ValueError("采集图片张数必须大于0")
            speed = int(self.rotation_speed_line_edit.text().strip())
            controller.set_speed(speed)
            controller.set_init_speed(speed)
            self.parent_window.xray_on_with_current_settings()
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

            missing_paths = []
            if self.dark_line_edit.text().strip():
                dark_path = os.path.join(
                    self.parent_window.project_path, self.dark_line_edit.text().strip()
                )
                if os.path.exists(dark_path):
                    self.dark_img = cv2.imread(dark_path, -1)
                    if self.dark_img is None:
                        missing_paths.append(dark_path)
                    else:
                        self.dark_img = self.dark_img.astype(np.float32)
                else:
                    missing_paths.append(dark_path)
            else:
                self.dark_img = None
            if self.empty_line_edit.text().strip():
                empty_path = os.path.join(
                    self.parent_window.project_path, self.empty_line_edit.text().strip()
                )
                if os.path.exists(empty_path):
                    self.empty_img = cv2.imread(empty_path, -1)
                    if self.empty_img is None:
                        missing_paths.append(empty_path)
                    else:
                        self.empty_img = self.empty_img.astype(np.float32)
                else:
                    missing_paths.append(empty_path)
            else:
                self.empty_img = None
            if missing_paths:
                self.error.emit("校正文件不存在:\n" + "\n".join(missing_paths))
                self._unfreeze_ui()
                return
            self.projection_normalized = self.dark_img is not None and self.empty_img is not None
            if self.projection_normalized:
                print("[Scan] Saving normalized transmission projections")
            else:
                print("[Scan] Saving raw detector projections")

            CREATE_NO_WINDOW = 0x08000000
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            if scan_mode == "普通采集":
                detector_args = [str(py34), detector_script, "stepseq", str(expose_time)]
            else:
                detector_args = [
                    str(py34),
                    detector_script,
                    "seq",
                    str(expose_time),
                    str(gap_time),
                    str(self.scan_number),
                ]
            sub = subprocess.Popen(
                detector_args,
                cwd=detector_bridge_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=CREATE_NO_WINDOW,
                startupinfo=startupinfo,
            )
            assert sub.stdout
            assert sub.stdin
            Thread(target=_drain_stream, args=(sub.stderr,), daemon=True).start()

            ready_cmd = _readline_with_timeout(sub.stdout, timeout=15)
            self.ProgressBarChanged.emit(20, "采集中")
            if ready_cmd is None or not ready_cmd.startswith(b"READY"):
                stderr_msg = "探测器错误详情已输出到日志窗口"
                self.error.emit(
                    f"探测器启动失败。\n"
                    f"ready_cmd={ready_cmd!r}\n"
                    f"{stderr_msg}"
                )
                sub.kill()
                _close_subprocess_streams(sub)
                _close_listener(server_holder)
                self._unfreeze_ui()
                return
            sub.stdin.write("start\n".encode())
            sub.stdin.flush()
            if scan_mode == "普通采集":
                step_degree = 360.0 / self.scan_number
                settle_seconds = gap_time / 1000
                for cnt in range(self.scan_number):
                    angle_name = _format_angle_name(cnt * step_degree)
                    sub.stdin.write("snap {}\n".format(angle_name).encode())
                    sub.stdin.flush()
                    snap_cmd = _readline_with_timeout(
                        sub.stdout, timeout=max(30, int(expose_time / 1000 + 20))
                    )
                    if snap_cmd is None or not snap_cmd.startswith(b"ok"):
                        raise RuntimeError("探测器单张采集失败: {}".format(snap_cmd))
                    if cnt < self.scan_number - 1:
                        if not controller.motion_rotation(step_degree):
                            raise RuntimeError("普通采集转台步进失败")
                        time.sleep(settle_seconds)
                if not controller.motion_rotation(step_degree):
                    raise RuntimeError("普通采集结束回到 360/0 度失败")
                print("[Scan] Step scan finished, returned to nominal 360/0 deg")
                sub.stdin.write("exit\n".encode())
                sub.stdin.flush()
            else:
                controller.motion_rotation(380)
            # 带超时读取，防止子进程挂住导致永远阻塞
            timeout = max(
                120, int(self.scan_number * (expose_time + gap_time) / 1000 + 60)
            )
            cmd = _readline_with_timeout(sub.stdout, timeout=timeout)
            print("77777", cmd)
            if cmd is None or not cmd.startswith(b"EXIT"):
                raise RuntimeError("探测器采集未正常结束: {}".format(cmd))
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
            for fut in self.fut_list:
                try:
                    fut.result()
                except Exception as e:
                    print(f"[Error] save image failed: {e}")
            _close_subprocess_streams(sub)
            _close_listener(server_holder)
            self._unfreeze_ui()
            try:
                self.parent_window.xray_off()
            except Exception:
                pass
            try:
                if controller is not None:
                    controller.close()
            except Exception:
                pass

    def _on_accepted(self):
        self.ui.hide()
        self.button_start()

    def set_scan_mode(self, scan_mode: str):
        self.scan_mode = scan_mode
        title = "普通采集" if scan_mode == "普通采集" else "旋转采集"
        self.ui.setWindowTitle(title)

    def button_start(self):
        self.parent_window.ct_scan_progress_bar.setValue(0)
        self.parent_window.ct_scan_progress_label.setText("初始化中")
        self.parent_window.tab_widget.setEnabled(False)
        self.img = None
        self.dark_img = None
        self.empty_img = None
        self.projection_normalized = False
        self.img_dict = {}
        self.fut_list = []
        self._scan_received = 0
        scan_thread = Thread(target=self.scan_thread, daemon=True)
        scan_thread.start()
