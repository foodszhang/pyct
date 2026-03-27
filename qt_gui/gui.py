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
from qt_gui.snap_window import SnapWindow, SnapType
from qt_gui.scan_window import ScanWindow
from serial_controller import UltraBrightController
from algorithm.calibration import cal

loader = QUiLoader()
Config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

xray_status_code = ['待预热', '预热中', '待开启', 'X射线开启', 'X射线过载', 'X射线无法开启', '自检中']

class MainWindow(QtWidgets.QMainWindow):
    CalImageChanged = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        loader.registerCustomWidget(rec.CTSliceView)
        loader.registerCustomWidget(pg.GraphicsView)
        self.ui = loader.load("qt_gui/main.ui", None)
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
        self.xray_status = '准备中'
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
            self.xray_controller = UltraBrightController(config["port"], config["baudrate"])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "警告", "X射线控制器连接失败")
            self.xray_controller = None
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
        self.recon_window_high_spin_box = self.ui.findChild(QtWidgets.QDoubleSpinBox, "reconWindowHighSpinBox")
        self.recon_window_low_spin_box = self.ui.findChild(QtWidgets.QDoubleSpinBox, "reconWindowLowSpinBox")
        #self.recon_window_high_slider = self.ui.findChild(QtWidgets.QSlider, "reconWindowHighSlider")
        #self.recon_window_low_slider = self.ui.findChild(QtWidgets.QSlider, "reconWindowLowSlider")
        self.recon_window_change_button = self.ui.findChild(QtWidgets.QPushButton, "reconWindowChangeButton")
        self.recon_window_high_spin_box.valueChanged.connect(self.set_window_high)
        self.recon_window_low_spin_box.valueChanged.connect(self.set_window_low)
        #self.recon_window_high_slider.valueChanged.connect(self.set_slider_window_high)
        #self.recon_window_low_slider.valueChanged.connect(self.set_slider_window_low)
        self.recon_window_change_button.clicked.connect(self.change_recon_window)
        self.recon_max_value = 255
        self.recon_min_value = 0


        #tab3 校准配置
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

        self.calButton.clicked.connect(self.calibrate)
        self.saveCalButton.clicked.connect(self.save_calibrate_result)

    
    def change_recon_window(self):
        for v in self.ct_slice_view_list:
            v.change_window()

    def set_slider_window_high(self, value):
        max_value = self.recon_max_value
        min_value = self.recon_min_value
        self.recon_window_high_spin_box.setValue(value * (max_value - min_value) / 255 + min_value)
        #self.recon_window_high_spin_box.setValue(value)
        for v in self.ct_slice_view_list:
            v.set_window_high(value)

    def set_slider_window_low(self, value):
        max_value = self.recon_max_value
        min_value = self.recon_min_value
        self.recon_window_low_spin_box.setValue(value * (max_value - min_value) / 255 + min_value)
        #self.recon_window_low_spin_box.setValue(value)
        for v in self.ct_slice_view_list:
            v.set_window_high(value)

    def set_window_high(self, value):
        max_value = self.recon_max_value
        min_value = self.recon_min_value
        self.recon_window_high_slider.setValue(int((value-min_value) * 255 / (max_value - min_value)))
        for v in self.ct_slice_view_list:
            v.set_window_high(value)

    def set_window_low(self, value):
        max_value = self.recon_max_value
        min_value = self.recon_min_value
        self.recon_window_low_slider.setValue(int((value-min_value) * 255 / (max_value - min_value)))
        for v in self.ct_slice_view_list:
            v.set_window_low(value)

    def xray_reconnect(self):
        config = Config.get("BrightController", None)
        if not config:
            QtWidgets.QMessageBox.critical(self, "警告", "未配置X射线控制器")
            sys.exit(1) 
        try:
            self.xray_controller = UltraBrightController(config["port"], config["baudrate"])
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
        if self.xray_status != '待开启':
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
        if self.xray_status == '待预热' or self.xray_status=="待开启":
            for w in self.xray_wdigets:
                w.setEnabled(True)
        if self.xray_status == "待开启":
            self.scan_button.setEnabled(True)
            self.dark_snap_button.setEnabled(True)
            self.empty_snap_button.setEnabled(True)
            self.xray_warm_button.setEnabled(False)


    def calibrate_thread(self):
        config = Config.get("CalibrationParam", None)
        if not config:
            QtWidgets.QMessageBox.critical(self, "警告", "配置文件错误， 请检查config.yaml文件")
        c = cal.Calibration(
            self.project_path,
            float(config["detectorPixelSize"]),
            int(config["BBNumber"]),
            int(config["detectorWidth"]),
            int(config["detectorHeight"]),
        )
        # return SOD, SDD, u0, v0, theta

        self.cal_result = c.calculate()

        self.cal_sod_line_edit.setText(str(self.cal_result[0]))
        self.cal_sdd_line_edit.setText(str(self.cal_result[1]))
        self.cal_detector_x_line_edit.setText(str(self.cal_result[2]))
        self.cal_detector_y_line_edit.setText(str(self.cal_result[3]))
        self.cal_rotation_line_edit.setText(str(self.cal_result[4]))
        self.tab_widget.setEnabled(True)

    def calibrate(self):
        self.tab_widget.setEnabled(False)
        cal_thread = Thread(target=self.calibrate_thread)
        cal_thread.start()

    def save_calibrate_result(self):
        self.reconstruction_dialog.sod_line_edit.setText(str(self.cal_result[0]))
        self.reconstruction_dialog.sdd_line_edit.setText(str(self.cal_result[1]))
        self.reconstruction_dialog.detector_x_line_edit.setText(str(self.cal_result[2]))
        self.reconstruction_dialog.detector_y_line_edit.setText(str(self.cal_result[3]))
        self.reconstruction_dialog.rotation_line_edit.setText(str(self.cal_result[4]))
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
        #self.recon_window_high_slider.setMaximum(255)
        #self.recon_window_high_slider.setMinimum(0)
        #self.recon_window_low_slider.setMaximum(255)
        #self.recon_window_low_slider.setMinimum(0)

        self.recon_window_high_spin_box.setValue(max_value)
        self.recon_window_low_spin_box.setValue(min_value)
        #self.recon_window_high_slider.setValue(255)
        #self.recon_window_low_slider.setValue(0)
        if self.reconstruction_dialog.use_ct_hu.isChecked():
            self.recon_window_high_spin_box.setValue(3000)
            self.recon_window_low_spin_box.setValue(-1000)
            #self.recon_window_high_slider.setValue((3000-min_value) /(max_value-min_value) * 255)
            #self.recon_window_low_slider.setValue((0-min_value) /(max_value-min_value) * 255)
                                                
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
        self.ui = loader.load("qt_gui/projectSelector.ui", None)
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
