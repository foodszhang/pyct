import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader
from threading import Thread
import subprocess
import pipe
import os
import numpy as np
import cv2
import pyqtgraph as pg
from algorithm.astra.conebeam import ConeBeam
from typing import Optional
import yaml
import numba as nb

loader = QUiLoader()
Config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)


class CTSliceView(QtWidgets.QWidget):
    def __init__(self, parent=None, window_sliders=None):
        super().__init__(parent)
        loader.registerCustomWidget(pg.GraphicsView)
        self.ui = loader.load("qt_gui/ctSlice.ui", self)
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


class ReconstrcionDialog(QtWidgets.QDialog):
    ReconDone = QtCore.Signal(np.ndarray)

    def __init__(self, parent, parent_window=None) -> None:
        super().__init__(parent)
        loader = QUiLoader()
        self.parent_window = parent_window
        self.ui = loader.load("qt_gui/reconstruction.ui", self)
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
        self.use_ct_hu = self.ui.findChild(
            QtWidgets.QCheckBox, "useCTHUCheckBox"
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
        yaml.dump(Config, open("config.yaml", "w"), Dumper=yaml.Dumper)

    def reconstruct_thread(self):
        NX = int(self.voxel_size_x_line_edit.text())
        NY = int(self.voxel_size_y_line_edit.text())
        NZ = int(self.voxel_size_z_line_edit.text())
        TM = int(self.column_count_line_edit.text())
        TN = int(self.row_count_line_edit.text())
        dd_column = float(self.x_spacing_line_edit.text())
        dd_row = float(self.y_spacing_line_edit.text())
        voxel_size = float(self.voxel_pixel_size_line_edit.text())
        number_of_img = float(self.angle_line_edit.text())
        rotation_angle = float(self.rotation_line_edit.text())
        number_of_img = int(360 // number_of_img)
        SOD = float(self.sod_line_edit.text())
        SDD = float(self.sdd_line_edit.text())
        proj_path = self.parent_window.project_path
        detector_x = float(self.detector_x_line_edit.text())
        detector_y = float(self.detector_y_line_edit.text())
        if self.use_scan_check_box.isChecked():
            cb = ConeBeam(
                SOD=SOD,
                TN=TN,
                TM=TM,
                SDD=SDD,
                NX=NX,
                NY=NY,
                NZ=NZ,
                dd_row=dd_row,
                dd_column=dd_column,
                voxel_size=voxel_size,
                number_of_img=360,
                proj_path=proj_path,
                detectorX=detector_x,
                detectorY=detector_y,
                rotation_angle=rotation_angle,
                useHu=self.use_ct_hu.isChecked(),
                rescale_slope=self.rescale_slope,
                rescale_intercept=self.rescale_intercept,
            )
            cb.load_from_dict(self.parent_window.scan_window.img_dict)
        else:
            cb = ConeBeam(
                SOD=SOD,
                TN=TN,
                TM=TM,
                SDD=SDD,
                NX=NX,
                NY=NY,
                NZ=NZ,
                dd_row=dd_row,
                dd_column=dd_column,
                voxel_size=voxel_size,
                number_of_img=360,
                proj_path=proj_path,
                detectorX=detector_x,
                detectorY=detector_y,
                rotation_angle=rotation_angle,
                useHu=self.use_ct_hu.isChecked(),
                rescale_slope=self.rescale_slope,
                rescale_intercept=self.rescale_intercept,
            )
            cb.load_img()
        rec = cb.reconstruct()
        rec.tofile("rec4.raw")
        try:
            full_filename = os.path.join(
                self.parent_window.project_path, 'rec.raw'
            )
            rec.tofile(full_filename)
        except Exception as e:
            print('!!!!!', e)
        print('4444recon.shape', rec.shape, rec.dtype)
        self.ReconDone.emit(rec)
        self.parent_window.tab_widget.setEnabled(True)
        self.save_config()

    def startReconstruction(self):
        self.parent_window.tab_widget.setEnabled(False)
        recon_thread = Thread(target=self.reconstruct_thread)
        recon_thread.start()
