# use main.ui to create  main window use pyside6
import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, QObject, Signal
import subprocess
import pipe
import os
import numpy as np
import cv2
import qt_gui.reconstruction as rec
import pyqtgraph as pg
from concurrent.futures import ThreadPoolExecutor
import yaml
from qt_gui.snap_window import SnapWindow, SnapType
from qt_gui.scan_window import ScanWindow
from serial_controller import UltraBrightController
from algorithm.calibration import cal
from utils.paths import get_config_path, get_ui_path

loader = QUiLoader()
Config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)

xray_status_code = [
    "待预热",
    "预热中",
    "待开启",
    "X射线开启",
    "X射线过载",
    "X射线无法开启",
    "自检中",
]


class LogSignal(QObject):
    message = Signal(str)


class StdoutRedirector:
    def __init__(self, signal, original):
        self.signal = signal
        self._original = original

    def write(self, text):
        if text and text != "\n":
            self.signal.message.emit(text)
        if self._original:
            self._original.write(text)

    def flush(self):
        if self._original:
            self._original.flush()


class CalibWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(dict)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, project_path, config):
        super().__init__()
        self.project_path = project_path
        self.config = config

    def run(self):
        try:
            c = cal.Calibration(
                self.project_path,
                float(self.config["detectorPixelSize"]),
                int(self.config["BBNumber"]),
                int(self.config["detectorWidth"]),
                int(self.config["detectorHeight"]),
            )
            self.status.emit("正在加载投影...")
            result = c.calculate_vshift_package(
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    CalImageChanged = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        loader.registerCustomWidget(rec.CTSliceView)
        loader.registerCustomWidget(pg.GraphicsView)
        self.ui = loader.load(get_ui_path("main.ui"), None)
        self.mainSplitter = self.ui.findChild(QtWidgets.QSplitter, "mainSplitter")
        if self.mainSplitter:
            self.mainSplitter.setSizes([1100, 150])
        # 选择项目目录

        self.project_path = None
        self.project_selector_window = ProjectSelectorWindow(parent=self)
        if self.project_path is None:
            self.open_project_selector()

        # tab配置
        self.tab_widget = self.ui.findChild(QtWidgets.QTabWidget, "tabWidget")
        self.tab_1 = self.ui.findChild(QtWidgets.QWidget, "tab_1")
        self.tab_2 = self.ui.findChild(QtWidgets.QWidget, "tab_2")

        # tab1 扫描+xray配置
        self.dark_snap_button = self.ui.findChild(
            QtWidgets.QPushButton, "darkSnapButton"
        )
        self.empty_snap_button = self.ui.findChild(
            QtWidgets.QPushButton, "emptySnapButton"
        )
        self.scan_button = self.ui.findChild(QtWidgets.QPushButton, "scanButton")
        self.scan_window = ScanWindow(parent_window=self)
        self.dark_snap_window = SnapWindow(parent=self, type=SnapType.DARK)
        self.empty_snap_window = SnapWindow(parent=self, type=SnapType.EMPTY)
        self.scan_view = self.ui.findChild(pg.GraphicsView, "scanGraphicsView")
        self.ct_scan_progress_bar = self.ui.findChild(
            QtWidgets.QProgressBar, "ctScanProgressBar"
        )
        self.ct_scan_progress_label = self.ui.findChild(
            QtWidgets.QLabel, "ctScanProgressLabel"
        )
        self.dark_snap_button.clicked.connect(self.open_dark_snap_window)
        self.empty_snap_button.clicked.connect(self.open_empty_snap_window)
        self.empty_snap_button.clicked.connect(self.open_empty_snap_window)
        self.scan_button.clicked.connect(self.open_scan_window)
        self.dark_snap_window.ImageChanged.connect(self.update_scan_view)
        self.empty_snap_window.ImageChanged.connect(self.update_scan_view)
        self.dark_snap_window.ProgressBarChanged.connect(self.update_progress_bar)
        self.empty_snap_window.ProgressBarChanged.connect(self.update_progress_bar)
        self.scan_window.ImageChanged.connect(self.update_scan_view)
        self.scan_window.ProgressBarChanged.connect(self.update_progress_bar)
        self.scan_image_item = pg.ImageItem(np.zeros((800, 800)), autoLevels=True)
        self.scan_view.addItem(self.scan_image_item)

        self.voltage_number = self.ui.findChild(QtWidgets.QLCDNumber, "voltageNumber")
        self.current_number = self.ui.findChild(QtWidgets.QLCDNumber, "currentNumber")
        self.voltage_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "voltageLineEdit"
        )
        self.current_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "currentLineEdit"
        )
        self.xray_status_label = self.ui.findChild(QtWidgets.QLabel, "xrayStatusLabel")
        self.xray_refresh_timer = QtCore.QTimer()
        self.xray_refresh_timer.timeout.connect(self.refresh_xray_status)
        self.xray_refresh_timer.start(1000)

        self.xray_in_control = False

        self.xray_warm_button = self.ui.findChild(
            QtWidgets.QPushButton, "xrayWarmButton"
        )
        self.xray_off_button = self.ui.findChild(QtWidgets.QPushButton, "xrayOffButton")
        self.xray_reconnect_button = self.ui.findChild(
            QtWidgets.QPushButton, "xrayReconnectButton"
        )

        self.xray_warm_button.clicked.connect(self.xray_warm)
        self.xray_off_button.clicked.connect(self.xray_off)
        self.xray_reconnect_button.clicked.connect(self.xray_reconnect)
        self.xray_status = "准备中"
        self.xray_voltage = 0
        self.xray_current = 0

        self.xray_wdigets = [
            self.voltage_number,
            self.current_number,
            self.voltage_line_edit,
            self.current_line_edit,
            self.xray_warm_button,
            self.xray_reconnect_button,
            self.xray_off_button,
            self.scan_button,
            self.dark_snap_button,
            self.empty_snap_button,
        ]

        config = Config.get("BrightController", None)
        if not config:
            QtWidgets.QMessageBox.critical(self, "警告", "未配置X射线控制器")
            sys.exit(1)
        try:
            self.xray_controller = UltraBrightController(
                config["port"], config["baudrate"]
            )
            self.hardware_available = True
        except Exception as e:
            print("[Warn] 硬件不可用，已进入离线模式（可校准/重建，不能扫描）")
            self.xray_controller = None
            self.hardware_available = False
            for w in self.xray_wdigets:
                w.setEnabled(False)
            self.xray_reconnect_button.setEnabled(True)

        # tab2 重建配置
        self.start_reconstruction_button = self.ui.findChild(
            QtWidgets.QPushButton, "startReconstructionButton"
        )
        self.reconstruction_dialog = rec.ReconstrcionDialog(
            parent=self.tab_2, parent_window=self
        )
        self.reconstruction_dialog.ReconDone.connect(self.show_reconstruction_result)
        self.start_reconstruction_button.clicked.connect(self.start_reconstruction)
        self.ct_slice_view_list = []
        self.ct_slice_view_list.append(self.ui.findChild(rec.CTSliceView, "sliceView0"))
        self.ct_slice_view_list.append(self.ui.findChild(rec.CTSliceView, "sliceView1"))
        self.ct_slice_view_list.append(self.ui.findChild(rec.CTSliceView, "sliceView2"))
        self.recon_window_high_spin_box = self.ui.findChild(
            QtWidgets.QDoubleSpinBox, "reconWindowHighSpinBox"
        )
        self.recon_window_low_spin_box = self.ui.findChild(
            QtWidgets.QDoubleSpinBox, "reconWindowLowSpinBox"
        )
        # self.recon_window_high_slider = self.ui.findChild(QtWidgets.QSlider, "reconWindowHighSlider")
        # self.recon_window_low_slider = self.ui.findChild(QtWidgets.QSlider, "reconWindowLowSlider")
        self.recon_window_change_button = self.ui.findChild(
            QtWidgets.QPushButton, "reconWindowChangeButton"
        )
        self.recon_window_high_spin_box.valueChanged.connect(self.set_window_high)
        self.recon_window_low_spin_box.valueChanged.connect(self.set_window_low)
        # self.recon_window_high_slider.valueChanged.connect(self.set_slider_window_high)
        # self.recon_window_low_slider.valueChanged.connect(self.set_slider_window_low)
        self.recon_window_change_button.clicked.connect(self.change_recon_window)
        self.recon_max_value = 255
        self.recon_min_value = 0

        # tab3 校准配置
        self.calButton = self.ui.findChild(QtWidgets.QPushButton, "calButton")
        self.saveCalButton = self.ui.findChild(QtWidgets.QPushButton, "saveCalButton")
        self.cal_detector_x_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calDetectorXLineEdit"
        )
        self.cal_detector_y_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calDetectorYLineEdit"
        )
        self.cal_rotation_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calRotationLineEdit"
        )
        self.cal_sod_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calSODLineEdit"
        )
        self.cal_sdd_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calSDDLineEdit"
        )
        self.cal_eta_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calEtaLineEdit"
        )
        self.cal_vc_raw_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calVcRawLineEdit"
        )
        self.cal_vs_raw_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calVsRawLineEdit"
        )
        self.cal_sx_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "calSxLineEdit")
        self.cal_sy_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "calSyLineEdit")
        self.cal_u0_recon_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calU0ReconLineEdit"
        )
        self.cal_v0_recon_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calV0ReconLineEdit"
        )
        self.cal_vc_recon_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calVcReconLineEdit"
        )
        self.cal_vs_recon_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "calVsReconLineEdit"
        )

        self.calButton.clicked.connect(self.calibrate)
        self.saveCalButton.clicked.connect(self.save_calibrate_result)

        self.log_text_edit = self.ui.findChild(QtWidgets.QTextEdit, "logTextEdit")
        self.clear_log_button = self.ui.findChild(
            QtWidgets.QPushButton, "clearLogButton"
        )
        self.auto_scroll_checkbox = self.ui.findChild(
            QtWidgets.QCheckBox, "autoScrollCheckBox"
        )
        self.cal_progress_bar = self.ui.findChild(
            QtWidgets.QProgressBar, "calProgressBar"
        )
        self.cal_status_label = self.ui.findChild(QtWidgets.QLabel, "calStatusLabel")

        self.log_signal = LogSignal()
        self.log_signal.message.connect(self.append_log)
        sys.stdout = StdoutRedirector(self.log_signal, sys.__stdout__)
        sys.stderr = StdoutRedirector(self.log_signal, sys.__stderr__)

        self.clear_log_button.clicked.connect(self.log_text_edit.clear)

        self.cal_progress_bar.hide()
        self.cal_status_label.hide()

    def append_log(self, text: str):
        color = "#d4d4d4"
        if "[Warn]" in text or "WARNING" in text:
            color = "#e5c07b"
        elif "[Error]" in text or "ERROR" in text:
            color = "#e06c75"
        elif any(tag in text for tag in ["[Env]", "[Config]", "[Geometry]"]):
            color = "#56b6c2"
        elif any(tag in text for tag in ["[Reconstruction]", "[CalibResult]"]):
            color = "#98c379"

        cursor = self.log_text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        if self.auto_scroll_checkbox.isChecked():
            self.log_text_edit.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def change_recon_window(self):
        for v in self.ct_slice_view_list:
            v.change_window()

    def set_slider_window_high(self, value):
        max_value = self.recon_max_value
        min_value = self.recon_min_value
        self.recon_window_high_spin_box.setValue(
            value * (max_value - min_value) / 255 + min_value
        )
        # self.recon_window_high_spin_box.setValue(value)
        for v in self.ct_slice_view_list:
            v.set_window_high(value)

    def set_slider_window_low(self, value):
        max_value = self.recon_max_value
        min_value = self.recon_min_value
        self.recon_window_low_spin_box.setValue(
            value * (max_value - min_value) / 255 + min_value
        )
        # self.recon_window_low_spin_box.setValue(value)
        for v in self.ct_slice_view_list:
            v.set_window_high(value)

    def set_window_high(self, value):
        max_value = self.recon_max_value
        min_value = self.recon_min_value
        self.recon_window_high_slider.setValue(
            int((value - min_value) * 255 / (max_value - min_value))
        )
        for v in self.ct_slice_view_list:
            v.set_window_high(value)

    def set_window_low(self, value):
        max_value = self.recon_max_value
        min_value = self.recon_min_value
        self.recon_window_low_slider.setValue(
            int((value - min_value) * 255 / (max_value - min_value))
        )
        for v in self.ct_slice_view_list:
            v.set_window_low(value)

    def xray_reconnect(self):
        config = Config.get("BrightController", None)
        if not config:
            QtWidgets.QMessageBox.critical(self, "警告", "未配置X射线控制器")
            sys.exit(1)
        try:
            self.xray_controller = UltraBrightController(
                config["port"], config["baudrate"]
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "警告", "X射线控制器连接失败")
            self.xray_controller = None
        if not self.xray_refresh_timer.isActive():
            self.xray_refresh_timer.start()

    def xray_warm(self):
        ret = self.xray_controller.set_voltage(int(self.voltage_line_edit.text()))
        if not ret:
            QtWidgets.QMessageBox.critical(self, "警告", "设置电压失败")
            return
        ret = self.xray_controller.set_current(int(self.current_line_edit.text()))
        if not ret:
            QtWidgets.QMessageBox.critical(self, "警告", "设置电流失败")
            return
        ret = self.xray_controller.set_focus_mode(2)
        if not ret:
            QtWidgets.QMessageBox.critical(self, "警告", "设置电流失败")
            return
        self.xray_controller.xray_on()
        self.xray_in_control = True
        for w in self.xray_wdigets:
            w.setEnabled(False)
        if self.xray_status != "待开启":
            self.scan_button.setEnabled(False)
            self.dark_snap_button.setEnabled(False)
            self.empty_snap_button.setEnabled(False)

    def xray_off(self):
        self.xray_controller.xray_off()

    def refresh_xray_status(self):
        if not self.xray_controller or not self.xray_controller.ser.isOpen():
            self.xray_status_label.setText("未连接")
            self.xray_status_label.setStyleSheet("color: red")
            for w in self.xray_wdigets:
                w.setEnabled(False)
            self.xray_reconnect_button.setEnabled(True)
            if self.xray_refresh_timer.isActive():
                self.xray_refresh_timer.stop()
        self.xray_reconnect_button.setEnabled(False)

        status = self.xray_controller.query_all()
        if not status:
            self.xray_status_label.setText("未连接")
            self.xray_status_label.setStyleSheet("color: red")
            for w in self.xray_wdigets:
                w.setEnabled(False)
            self.xray_reconnect_button.setEnabled(True)
            if self.xray_refresh_timer.isActive():
                self.xray_refresh_timer.stop()
            self.xray_controller.ser.close()
            self.xray_controller = None
            return

        status, voltage, current = status
        self.voltage_number.display(voltage)
        self.current_number.display(current)
        self.xray_status_label.setText(status)
        self.xray_status = status
        self.xray_status_label.setStyleSheet("color: green")
        self.xray_voltage = float(voltage)
        self.xray_current = float(current)
        if self.xray_status == "待预热" or self.xray_status == "待开启":
            for w in self.xray_wdigets:
                w.setEnabled(True)
        if self.xray_status == "待开启":
            self.scan_button.setEnabled(True)
            self.dark_snap_button.setEnabled(True)
            self.empty_snap_button.setEnabled(True)
            self.xray_warm_button.setEnabled(False)

    def calibrate_thread(self):
        pass

    def calibrate(self):
        config = Config.get("CalibrationParam", None)
        if not config:
            QtWidgets.QMessageBox.critical(
                self, "警告", "配置文件错误， 请检查config.yaml文件"
            )
            return
        self.tab_widget.setEnabled(False)
        self.cal_progress_bar.show()
        self.cal_status_label.show()
        self.cal_progress_bar.setValue(0)
        self.cal_status_label.setText("正在校准...")

        self.cal_worker = CalibWorker(self.project_path, config)
        self.cal_worker.progress.connect(
            lambda cur, tot: (
                self.cal_progress_bar.setMaximum(tot),
                self.cal_progress_bar.setValue(cur),
            )
        )
        self.cal_worker.status.connect(self.cal_status_label.setText)
        self.cal_worker.finished.connect(self.on_calibrate_finished)
        self.cal_worker.error.connect(self.on_calibrate_error)
        self.cal_worker.start()

    def on_calibrate_finished(self, result):
        self.cal_result = result
        self.cal_sod_line_edit.setText(str(result["SOD"]))
        self.cal_sdd_line_edit.setText(str(result["SDD"]))
        self.cal_detector_x_line_edit.setText(str(result["u0_raw"]))
        self.cal_detector_y_line_edit.setText(str(result["v0_raw"]))
        self.cal_rotation_line_edit.setText("0")
        self.cal_eta_line_edit.setText(f"{result['eta']:.6f}")
        self.cal_vc_raw_line_edit.setText(f"{result['vc_raw']:.4f}")
        self.cal_vs_raw_line_edit.setText(f"{result['vs_raw']:.4f}")
        self.cal_sx_line_edit.setText(str(result["sx"]))
        self.cal_sy_line_edit.setText(str(result["sy"]))
        self.cal_u0_recon_line_edit.setText(f"{result['u0_recon']:.4f}")
        self.cal_v0_recon_line_edit.setText(f"{result['v0_recon']:.4f}")
        self.cal_vc_recon_line_edit.setText(f"{result['vc_recon']:.4f}")
        self.cal_vs_recon_line_edit.setText(f"{result['vs_recon']:.4f}")
        self.cal_progress_bar.setValue(self.cal_progress_bar.maximum())
        self.cal_status_label.setText("校准完成")
        self.tab_widget.setEnabled(True)
        QtCore.QTimer.singleShot(
            2000,
            lambda: (
                self.cal_progress_bar.hide(),
                self.cal_status_label.hide(),
            ),
        )

    def on_calibrate_error(self, msg):
        self.cal_status_label.setText(f"校准失败: {msg}")
        self.tab_widget.setEnabled(True)
        QtCore.QTimer.singleShot(
            5000,
            lambda: (
                self.cal_progress_bar.hide(),
                self.cal_status_label.hide(),
            ),
        )

    def save_calibrate_result(self):
        cr = self.cal_result
        self.reconstruction_dialog.sod_line_edit.setText(str(cr["SOD"]))
        self.reconstruction_dialog.sdd_line_edit.setText(str(cr["SDD"]))
        self.reconstruction_dialog.detector_x_line_edit.setText(str(cr["u0_raw"]))
        self.reconstruction_dialog.detector_y_line_edit.setText(str(cr["v0_raw"]))
        self.reconstruction_dialog.rotation_line_edit.setText("0")
        self.reconstruction_dialog.eta_line_edit.setText(f"{cr['eta']:.6f}")
        self.reconstruction_dialog.vc_line_edit.setText(f"{cr['vc_recon']:.4f}")
        self.reconstruction_dialog.vs_line_edit.setText(f"{cr['vs_recon']:.4f}")
        self.reconstruction_dialog.sx_line_edit.setText(str(cr["sx"]))
        self.reconstruction_dialog.sy_line_edit.setText(str(cr["sy"]))

        calib_for_yaml = {
            "SOD": cr["SOD"],
            "SDD": cr["SDD"],
            "u0_raw": cr["u0_raw"],
            "v0_raw": cr["v0_raw"],
            "eta": cr["eta"],
            "vc_raw": cr["vc_raw"],
            "vs_raw": cr["vs_raw"],
            "sx": cr["sx"],
            "sy": cr["sy"],
            "u0_recon": cr["u0_recon"],
            "v0_recon": cr["v0_recon"],
            "vc_recon": cr["vc_recon"],
            "vs_recon": cr["vs_recon"],
        }
        Config["CalibResult"] = calib_for_yaml
        yaml.dump(Config, open(get_config_path(), "w"), Dumper=yaml.Dumper)

        self.reconstruction_dialog.save_config()

    def update_progress_bar(self, value, text):
        self.ct_scan_progress_bar.setValue(value)
        if text:
            self.ct_scan_progress_label.setText(text)

    def show_reconstruction_result(self, img: np.ndarray):
        max_value = np.max(img)
        min_value = np.min(img)
        self.recon_max_value = max_value
        self.recon_min_value = min_value
        self.recon_window_high_spin_box.setMaximum(max_value)
        self.recon_window_high_spin_box.setMinimum(min_value)
        self.recon_window_low_spin_box.setMaximum(max_value)
        self.recon_window_low_spin_box.setMinimum(min_value)
        # self.recon_window_high_slider.setMaximum(255)
        # self.recon_window_high_slider.setMinimum(0)
        # self.recon_window_low_slider.setMaximum(255)
        # self.recon_window_low_slider.setMinimum(0)

        self.recon_window_high_spin_box.setValue(max_value)
        self.recon_window_low_spin_box.setValue(min_value)
        # self.recon_window_high_slider.setValue(255)
        # self.recon_window_low_slider.setValue(0)
        if self.reconstruction_dialog.use_ct_hu.isChecked():
            self.recon_window_high_spin_box.setValue(3000)
            self.recon_window_low_spin_box.setValue(-1000)
            # self.recon_window_high_slider.setValue((3000-min_value) /(max_value-min_value) * 255)
            # self.recon_window_low_slider.setValue((0-min_value) /(max_value-min_value) * 255)

        for i in range(3):
            if self.reconstruction_dialog.use_ct_hu.isChecked():
                self.ct_slice_view_list[i].min_window = -1000
                self.ct_slice_view_list[i].max_window = 3000
            else:
                self.ct_slice_view_list[i].min_window = min_value
                self.ct_slice_view_list[i].max_window = max_value
            self.ct_slice_view_list[i].show_img(img, i)

    def start_reconstruction(self):
        self.reconstruction_dialog.ui.show()

    def open_dark_snap_window(self):
        self.dark_snap_window.ui.show()

    def open_empty_snap_window(self):
        self.empty_snap_window.ui.show()

    def open_scan_window(self):
        self.scan_window.ui.show()

    def update_scan_view(self, img):
        self.scan_image_item.setImage(img.T, autoLevels=True)

    def open_project_selector(self):
        self.project_selector_window.ui.show()
        # always focus on the project selector window
        self.project_selector_window.ui.activateWindow()
        self.project_selector_window.ui.exec()


class ProjectSelectorWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__()
        self.parent_window = parent
        loader = QUiLoader()
        self.ui = loader.load(get_ui_path("projectSelector.ui"), None)
        self.choose_project_button = self.ui.findChild(
            QtWidgets.QPushButton, "chooseProjectpushButton"
        )
        self.project_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "projectLineEdit"
        )
        self.ok_button = self.ui.findChild(QtWidgets.QPushButton, "okButton")

        self.choose_project_button.clicked.connect(self.open_project_selector)
        self.ok_button.clicked.connect(self.button_ok)

    def open_project_selector(self):
        self.project_line_edit.setText(
            QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Project Directory")
        )

    def button_ok(self):
        if self.project_line_edit.text() == "":
            QtWidgets.QMessageBox.critical(self, "警告", "必须选择一个目录")
            return
        self.close()
        self.parent_window.project_path = self.project_line_edit.text()
        self.ui.hide()


def start_gui():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.ui.show()
    app.exec()
