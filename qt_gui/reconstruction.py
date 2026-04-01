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
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.gv = pg.GraphicsView(self)
        self.gv.setMinimumSize(200, 200)
        layout.addWidget(self.gv, 1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._on_slider_changed)

        self.axis = None
        self.img = None
        self.max_window = None
        self.min_window = None
        self.imageItem = None
        self.slider_value = 0

    def _get_slice(self, idx):
        if self.img is None:
            return None
        if self.axis == 0:
            s = self.img[:, :, idx]
        elif self.axis == 1:
            s = self.img[idx, :, :]
        elif self.axis == 2:
            s = self.img[:, idx, :]
        else:
            return None
        return s

    def _axis_size(self):
        if self.img is None:
            return 0
        if self.axis == 0:
            return self.img.shape[2]
        elif self.axis == 1:
            return self.img.shape[0]
        elif self.axis == 2:
            return self.img.shape[1]
        return 0

    def _update_display(self, slider_value):
        if self.img is None:
            return
        size = self._axis_size()
        if size == 0:
            return
        idx = max(0, min(slider_value, size - 1))
        self.slider_value = idx
        s = self._get_slice(idx)
        if s is None:
            return
        rect = QtCore.QRectF(0, 0, self.gv.size().width(), self.gv.size().height())
        use_global = self.min_window is not None and self.max_window is not None
        if self.imageItem is None:
            if use_global:
                self.imageItem = pg.ImageItem(
                    s.T,
                    autoLevels=False,
                    levels=(self.min_window, self.max_window),
                    rect=rect,
                )
            else:
                self.imageItem = pg.ImageItem(s.T, autoLevels=True, rect=rect)
            self.gv.addItem(self.imageItem)
        else:
            if use_global:
                self.imageItem.setImage(
                    s.T, autoLevels=False, levels=(self.min_window, self.max_window)
                )
            else:
                self.imageItem.setImage(s.T, autoLevels=True)
            self.imageItem.setRect(rect)

    def _on_slider_changed(self, value):
        self._update_display(value)

    def show_img(self, img, axis, value=0):
        self.img = img
        self.axis = axis
        size = self._axis_size()
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(size - 1, 0))
        self.slider.setValue(min(value, max(size - 1, 0)))
        self.slider.blockSignals(False)
        self._update_display(self.slider.value())

    def change_window(self):
        self._update_display(self.slider.value())

    def set_window_high(self, value):
        self.max_window = value

    def set_window_low(self, value):
        self.min_window = value


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
                vol_center_x=p["vol_center_x"],
                vol_center_y=p["vol_center_y"],
                vol_center_z=p["vol_center_z"],
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

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.ui)

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

        self.eta_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "etaLineEdit")
        self.vc_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "vcLineEdit")
        self.vs_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "vsLineEdit")
        self.sx_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "sxLineEdit")
        self.sy_line_edit = self.ui.findChild(QtWidgets.QLineEdit, "syLineEdit")
        self.roi_center_x_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "roiCenterXLineEdit"
        )
        self.roi_center_y_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "roiCenterYLineEdit"
        )
        self.roi_center_z_line_edit = self.ui.findChild(
            QtWidgets.QLineEdit, "roiCenterZLineEdit"
        )

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

        self.roi_center_x_line_edit.setText(str(config.get("roiCenterX", "0.0")))
        self.roi_center_y_line_edit.setText(str(config.get("roiCenterY", "0.0")))
        self.roi_center_z_line_edit.setText(str(config.get("roiCenterZ", "0.0")))

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
        config["roiCenterX"] = self.roi_center_x_line_edit.text()
        config["roiCenterY"] = self.roi_center_y_line_edit.text()
        config["roiCenterZ"] = self.roi_center_z_line_edit.text()
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
        self.save_config()
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
            "vol_center_x": float(self.roi_center_x_line_edit.text()),
            "vol_center_y": float(self.roi_center_y_line_edit.text()),
            "vol_center_z": float(self.roi_center_z_line_edit.text()),
        }
        self.params = params
        self.accept()
