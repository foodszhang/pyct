import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, QObject, Signal
import subprocess
import pipe
import os
import numpy as np
import cv2
import pyqtgraph as pg
from algorithm.astra.conebeam import ConeBeam
from typing import Optional
import yaml
import nibabel as nib
from utils.paths import get_config_path, get_ui_path

loader = QUiLoader()
Config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)


class CTSliceView(QtWidgets.QWidget):
    def __init__(self, parent=None, window_sliders=None):
        super().__init__(parent)
        loader.registerCustomWidget(pg.GraphicsView)
        self.ui = loader.load(get_ui_path("ctSlice.ui"), self)
        self.gv = self.ui.findChild(pg.GraphicsView, "ImageView")
        self.slider = self.ui.findChild(QtWidgets.QSlider, "SliceSlider")
        self.slider.valueChanged.connect(self.slider_value_changed)
        self.axis = None
        self.img = None
        self.max_window = None
        self.min_window = None
        self.imageItem = None
        self.slider_value = 0
        # self.gv.setRange(QtCore.QRectF(0, 0, 512, 512))

    def change_window(self):
        self.show_img(self.img, self.axis, self.slider_value)

    def set_window_high(self, value):
        if not self.imageItem:
            return
        self.max_window = value

    def set_window_low(self, value):
        if not self.imageItem:
            return
        self.min_window = value

    def slider_value_changed(self, value):
        if self.img is None:
            return
        self.show_img(self.img, self.axis, value)

    def show_img(self, img, axis, value=0):
        self.img = img
        part_img = img.clip(self.min_window, self.max_window)
        if self.axis is None:
            self.axis = axis
        if axis == 0:
            value = int(value * part_img.shape[2] / 512)
            self.imageItem = pg.ImageItem(
                part_img[:, :, value].T,
                autoLevels=True,
                rect=QtCore.QRectF(
                    0, 0, self.gv.size().width(), self.gv.size().height()
                ),
            )
        elif axis == 1:
            value = int(value * part_img.shape[0] / 512)
            self.imageItem = pg.ImageItem(
                part_img[value, :, :].T,
                autoLevels=True,
                rect=QtCore.QRectF(
                    0, 0, self.gv.size().width(), self.gv.size().height()
                ),
            )
        elif axis == 2:
            value = int(value * part_img.shape[1] / 512)
            self.imageItem = pg.ImageItem(
                part_img[:, value, :].T,
                autoLevels=True,
                rect=QtCore.QRectF(
                    0, 0, self.gv.size().width(), self.gv.size().height()
                ),
            )
        self.gv.addItem(self.imageItem)


class ReconSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)


class ReconWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, params, parent_window):
        super().__init__()
        self.params = params
        self.parent_window = parent_window

    def run(self):
        p = self.params
        try:
            cb = ConeBeam(
                SOD=p["SOD"],
                TN=p["TN"],
                TM=p["TM"],
                SDD=p["SDD"],
                NX=p["NX"],
                NY=p["NY"],
                NZ=p["NZ"],
                dd_row=p["dd_row"],
                dd_column=p["dd_column"],
                voxel_size=p["voxel_size"],
                number_of_img=360,
                proj_path=p["proj_path"],
                detectorX=p["detector_x"],
                detectorY=p["detector_y"],
                eta=p["eta"],
                vc=p["vc"],
                vs=p["vs"],
                sx=p["sx"],
                sy=p["sy"],
                useHu=False,
                rescale_slope=p["rescale_slope"],
                rescale_intercept=p["rescale_intercept"],
                pixel_size_raw=p["pixel_size_raw"],
            )
            if p["use_scan"]:
                cb.load_from_dict(self.parent_window.scan_window.img_dict)
            else:
                self.progress.emit(5, "正在加载投影...")
                cb.load_img(
                    angle_from_filename=True,
                    progress_callback=lambda cur, tot, stage: self.progress.emit(
                        int(cur / tot * 60), f"加载投影 {cur}/{tot}"
                    ),
                )
            self.progress.emit(65, "构建几何...")
            self.progress.emit(80, "FDK 重建中...")
            rec = cb.reconstruct()
            self.progress.emit(95, "保存结果...")
            nii_img = nib.Nifti1Image(rec, np.eye(4))
            try:
                full_filename = os.path.join(
                    self.parent_window.project_path, "rec.nii.gz"
                )
                nib.save(nii_img, full_filename)
            except Exception as e:
                print("!!!!!", e)
            print("4444recon.shape", rec.shape, rec.dtype)
            self.progress.emit(100, "完成")
            self.finished.emit(rec)
        except Exception as e:
            self.error.emit(str(e))


class ReconstrcionDialog(QtWidgets.QDialog):
    ReconDone = QtCore.Signal(np.ndarray)

    def __init__(self, parent, parent_window=None) -> None:
        super().__init__(parent)
        loader = QUiLoader()
        self.parent_window = parent_window
        self.ui = loader.load(get_ui_path("reconstruction.ui"), self)
        self.angle_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "angleLineEdit")
        self.column_count_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "columnCountLineEdit"
        )
        self.row_count_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "rowCountLineEdit"
        )
        self.sdd_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "SDDLineEdit")
        self.sod_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "SODLineEdit")
        self.use_scan_check_box = self.ui.findChild(
            QtWidgets.QCheckBox, "useScanCheckBox"
        )
        self.use_ct_hu = self.ui.findChild(QtWidgets.QCheckBox, "useCTHUCheckBox")
        self.voxel_pixel_size_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "voxelPixelSizeLineEdit"
        )
        self.voxel_size_x_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "voxelSizeXLineEdit"
        )
        self.voxel_size_y_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "voxelSizeYLineEdit"
        )
        self.voxel_size_z_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "voxelSizeZLineEdit"
        )
        self.x_spacing_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "xSpacingLineEdit"
        )
        self.y_spacing_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "ySpacingLineEdit"
        )
        self.detector_x_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "detectorXLineEdit"
        )
        self.detector_y_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "detectorYLineEdit"
        )
        self.rotation_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "rotationLineEdit"
        )
        self.button_box = self.ui.findChild(QtWidgets.QDialogButtonBox, "buttonBox")

        self.button_box.accepted.connect(self.startReconstruction)

        self.recon_progress_bar = self.ui.findChild(
            QtWidgets.QProgressBar, "reconProgressBar"
        )
        self.recon_stage_label = self.ui.findChild(QtWidgets.QLabel, "reconStageLabel")
        self.recon_progress_bar.hide()
        self.recon_stage_label.hide()

        self.eta_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "etaLineEdit")
        self.vc_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "vcLineEdit")
        self.vs_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "vsLineEdit")
        self.sx_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "sxLineEdit")
        self.sy_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "syLineEdit")

        self.init_from_config()

    def init_from_config(self):
        config = Config.get("ReconParam", None)
        if not config:
            return
        self.angle_line_edit.setText(str(config["angle"]))
        self.column_count_line_edit.setText(str(config["columnCount"]))
        self.row_count_line_edit.setText(str(config["rowCount"]))
        self.sdd_line_edit.setText(str(config["SDD"]))
        self.sod_line_edit.setText(str(config["SOD"]))
        self.voxel_pixel_size_line_edit.setText(str(config["voxelPixelSize"]))
        self.voxel_size_x_line_edit.setText(str(config["voxelSizeX"]))
        self.voxel_size_y_line_edit.setText(str(config["voxelSizeY"]))
        self.voxel_size_z_line_edit.setText(str(config["voxelSizeZ"]))
        self.x_spacing_line_edit.setText(str(config["xSpacing"]))
        self.y_spacing_line_edit.setText(str(config["ySpacing"]))
        self.detector_x_line_edit.setText(str(config["detectorX"]))
        self.detector_y_line_edit.setText(str(config["detectorY"]))
        self.rotation_line_edit.setText(str(config["rotation"]))
        self.rescale_slope = float(config["rescale_slope"])
        self.rescale_intercept = float(config["rescale_intercept"])
        calib = Config.get("CalibResult", {})
        cal_config = Config.get("CalibrationParam", {})
        self.eta_line_edit.setText(str(calib.get("eta", "0.0")))
        self.vc_line_edit.setText(str(calib.get("vc_recon", "0.0")))
        self.vs_line_edit.setText(str(calib.get("vs_recon", "0.0")))
        self.sx_line_edit.setText(str(calib.get("sx", "0.5")))
        self.sy_line_edit.setText(str(calib.get("sy", "0.5")))

        if calib and cal_config:
            sx = float(calib.get("sx", 0.5))
            sy = float(calib.get("sy", 0.5))
            raw_w = int(cal_config.get("detectorWidth", 1536))
            raw_h = int(cal_config.get("detectorHeight", 1944))
            self.column_count_line_edit.setText(str(int(raw_w * sx)))
            self.row_count_line_edit.setText(str(int(raw_h * sy)))

    def save_config(self):
        config = Config.get("ReconParam", None)
        if not config:
            config = {}
        config["angle"] = self.angle_line_edit.text()
        config["columnCount"] = self.column_count_line_edit.text()
        config["rowCount"] = self.row_count_line_edit.text()
        config["SDD"] = self.sdd_line_edit.text()
        config["SOD"] = self.sod_line_edit.text()
        config["voxelPixelSize"] = self.voxel_pixel_size_line_edit.text()
        config["voxelSizeX"] = self.voxel_size_x_line_edit.text()
        config["voxelSizeY"] = self.voxel_size_y_line_edit.text()
        config["voxelSizeZ"] = self.voxel_size_z_line_edit.text()
        config["xSpacing"] = self.x_spacing_line_edit.text()
        config["ySpacing"] = self.y_spacing_line_edit.text()
        config["detectorX"] = self.detector_x_line_edit.text()
        config["detectorY"] = self.detector_y_line_edit.text()
        config["rotation"] = self.rotation_line_edit.text()
        Config["ReconParam"] = config
        calib = Config.get("CalibResult", {})
        calib["eta"] = float(self.eta_line_edit.text())
        calib["vc_recon"] = float(self.vc_line_edit.text())
        calib["vs_recon"] = float(self.vs_line_edit.text())
        calib["sx"] = float(self.sx_line_edit.text())
        calib["sy"] = float(self.sy_line_edit.text())
        Config["CalibResult"] = calib
        yaml.dump(Config, open(get_config_path(), "w"), Dumper=yaml.Dumper)

    def startReconstruction(self):
        params = {
            "NX": int(self.voxel_size_x_line_edit.text()),
            "NY": int(self.voxel_size_y_line_edit.text()),
            "NZ": int(self.voxel_size_z_line_edit.text()),
            "TN": int(self.column_count_line_edit.text()),
            "TM": int(self.row_count_line_edit.text()),
            "dd_column": float(self.x_spacing_line_edit.text()),
            "dd_row": float(self.y_spacing_line_edit.text()),
            "voxel_size": float(self.voxel_pixel_size_line_edit.text()),
            "SOD": float(self.sod_line_edit.text()),
            "SDD": float(self.sdd_line_edit.text()),
            "detector_x": float(self.detector_x_line_edit.text()),
            "detector_y": float(self.detector_y_line_edit.text()),
            "use_scan": self.use_scan_check_box.isChecked(),
            "use_ct_hu": self.use_ct_hu.isChecked(),
            "rescale_slope": self.rescale_slope,
            "rescale_intercept": self.rescale_intercept,
            "proj_path": self.parent_window.project_path,
            "eta": float(self.eta_line_edit.text()),
            "vc": float(self.vc_line_edit.text()),
            "vs": float(self.vs_line_edit.text()),
            "sx": float(self.sx_line_edit.text()),
            "sy": float(self.sy_line_edit.text()),
            "pixel_size_raw": float(
                Config.get("CalibrationParam", {}).get("detectorPixelSize", 0.0748)
            ),
        }

        self.parent_window.tab_widget.setEnabled(False)
        self.recon_progress_bar.show()
        self.recon_stage_label.show()
        self.recon_progress_bar.setValue(0)
        self.recon_stage_label.setText("准备中...")

        self.recon_worker = ReconWorker(params, self.parent_window)
        self.recon_worker.progress.connect(self.on_recon_progress)
        self.recon_worker.finished.connect(self.on_recon_finished)
        self.recon_worker.error.connect(self.on_recon_error)
        self.recon_worker.start()

    def on_recon_progress(self, percent, stage_name):
        self.recon_progress_bar.setValue(percent)
        self.recon_stage_label.setText(stage_name)

    def on_recon_finished(self, rec):
        self.recon_progress_bar.setStyleSheet(
            "QProgressBar::chunk { background-color: #98c379; }"
        )
        self.recon_progress_bar.setValue(100)
        self.ReconDone.emit(rec)
        self.save_config()
        QtCore.QTimer.singleShot(
            2000,
            lambda: (
                self.recon_progress_bar.hide(),
                self.recon_stage_label.hide(),
                self.recon_progress_bar.setStyleSheet(""),
            ),
        )

    def on_recon_error(self, msg):
        self.recon_stage_label.setText(f"重建失败: {msg}")
        self.parent_window.tab_widget.setEnabled(True)
        QtCore.QTimer.singleShot(
            5000,
            lambda: (
                self.recon_progress_bar.hide(),
                self.recon_stage_label.hide(),
            ),
        )
