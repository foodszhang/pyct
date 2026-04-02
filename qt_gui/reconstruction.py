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
        self.vb = pg.ViewBox(lockAspect=True)
        self.gv.setCentralItem(self.vb)
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
        use_global = (self.min_window is not None and self.max_window is not None)
        if self.imageItem is None:
            if use_global:
                self.imageItem = pg.ImageItem(
                    s.T, autoLevels=False,
                    levels=(self.min_window, self.max_window)
                )
            else:
                self.imageItem = pg.ImageItem(s.T, autoLevels=True)
            self.vb.addItem(self.imageItem)
        else:
            if use_global:
                self.imageItem.setImage(
                    s.T, autoLevels=False,
                    levels=(self.min_window, self.max_window)
                )
            else:
                self.imageItem.setImage(s.T, autoLevels=True)

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
    cuda_used = Signal(bool)

    def __init__(self, params, parent_window):
        super().__init__()
        self.params = params
        self.parent_window = parent_window

    def run(self):
        import traceback, pathlib, datetime

        log_path = pathlib.Path.home() / "pyct_crash.log"
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
            if p.get("ring_correction"):
                self.progress.emit(70, "环形伪影校正...")
            algo = p.get("algorithm", "FDK")
            if algo == "FDK":
                self.progress.emit(80, f"FDK 重建中（滤波器={p.get('filter_type','Ram-Lak')}）...")
            else:
                self.progress.emit(80, f"{algo} 迭代重建中（{p.get('iterations',50)} 次）...")
            rec, use_cuda = cb.reconstruct(
                filter_type=p.get("filter_type", "Ram-Lak"),
                algorithm=algo,
                iterations=p.get("iterations", 50),
                non_neg_constraint=p.get("non_neg_constraint", True),
                ring_correction=p.get("ring_correction", False),
                ring_kernel_size=p.get("ring_kernel_size", 9),
            )
            self.cuda_used.emit(use_cuda)
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
            msg = f"\n[{datetime.datetime.now()}] CRASH in ReconWorker:\n"
            msg += traceback.format_exc()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg)
            self.error.emit(f"重建崩溃，详情见 {log_path}:\n{e}")


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
        self.button_box.rejected.connect(self.reject)

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

        # 高级选项
        self.filter_type_combo = self.ui.findChild(QtWidgets.QComboBox, "filterTypeComboBox")
        self.ring_correction_check = self.ui.findChild(QtWidgets.QCheckBox, "ringCorrectionCheckBox")
        self.ring_kernel_spin = self.ui.findChild(QtWidgets.QSpinBox, "ringKernelSpinBox")
        self.algorithm_combo = self.ui.findChild(QtWidgets.QComboBox, "algorithmComboBox")
        self.iterations_spin = self.ui.findChild(QtWidgets.QSpinBox, "iterationsSpinBox")
        self.non_neg_check = self.ui.findChild(QtWidgets.QCheckBox, "nonNegConstraintCheckBox")

        # 联动逻辑
        self.ring_correction_check.toggled.connect(self.ring_kernel_spin.setEnabled)
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)

        self.init_from_config()

        # 自适应屏幕大小
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(520, int(avail.width() * 0.35))
            h = min(int(avail.height() * 0.85), 900)
            self.resize(w, h)

        self._setup_tooltips()

    def _on_algorithm_changed(self, text):
        self.iterations_spin.setEnabled(text in ("SIRT", "CGLS"))

    def _setup_tooltips(self):
        """为所有参数控件设置鼠标悬停提示"""

        # ── 几何参数 ──
        tip = "SOD (Source-Object Distance)\n光源（X射线管焦点）到旋转中心的距离。\n增大 SOD 会减小放大倍率，增加穿透力。"
        self.sod_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelSod").setToolTip(tip)

        tip = "SDD (Source-Detector Distance)\n光源到探测器平面的距离。\nSDD / SOD = 系统放大倍率 M。"
        self.sdd_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelSdd").setToolTip(tip)

        tip = "探测器图像在水平方向（u 轴）的像素数。\n经过 binning/缩放后的实际使用值。"
        self.column_count_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelCol").setToolTip(tip)

        tip = "探测器图像在垂直方向（v 轴）的像素数。\n经过 binning/缩放后的实际使用值。"
        self.row_count_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelRow").setToolTip(tip)

        tip = "探测器水平方向（u 轴）相邻像素中心的物理间距。\n单位 mm。影响重建的空间分辨率。"
        self.x_spacing_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelXSp").setToolTip(tip)

        tip = "探测器垂直方向（v 轴）相邻像素中心的物理间距。\n单位 mm。通常与水平间距相同。"
        self.y_spacing_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelYSp").setToolTip(tip)

        tip = "相邻投影之间的旋转角度间隔（度）。\n例如 1° 表示共 360 张投影覆盖完整 360°。"
        self.angle_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelAngle").setToolTip(tip)

        tip = "探测器中心相对于射线主轴的水平偏移（像素）。\n校准得出，补偿探测器安装的水平偏心。"
        self.detector_x_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelDetX").setToolTip(tip)

        tip = "探测器中心相对于射线主轴的垂直偏移（像素）。\n校准得出，补偿探测器安装的垂直偏心。"
        self.detector_y_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelDetY").setToolTip(tip)

        tip = "探测器绕射线主轴的面内旋转角度（°）。\n校准得出，补偿探测器安装时的倾斜。"
        self.rotation_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelRot").setToolTip(tip)

        # ── 重建体参数 ──
        tip = "重建体素的边长（mm）。\n值越小分辨率越高，但重建体积和计算量更大。\n典型值：0.1–0.5 mm。"
        self.voxel_pixel_size_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelVoxelPs").setToolTip(tip)

        tip = "重建体在 X 方向的体素个数。\n重建区域宽度 = 体素数 × 体素边长。"
        self.voxel_size_x_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelVoxX").setToolTip(tip)

        tip = "重建体在 Y 方向的体素个数。\n重建区域高度 = 体素数 × 体素边长。"
        self.voxel_size_y_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelVoxY").setToolTip(tip)

        tip = "重建体在 Z 方向（旋转轴方向）的体素个数。\n重建区域深度 = 体素数 × 体素边长。"
        self.voxel_size_z_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelVoxZ").setToolTip(tip)

        tip = "感兴趣区域（ROI）中心在 X 方向的偏移（mm）。\n0 表示以旋转轴为中心，非零值可偏移重建区域。"
        self.roi_center_x_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelRoiX").setToolTip(tip)

        tip = "感兴趣区域（ROI）中心在 Y 方向的偏移（mm）。\n用于调整重建区域的垂直位置。"
        self.roi_center_y_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelRoiY").setToolTip(tip)

        tip = "感兴趣区域（ROI）中心在 Z 方向的偏移（mm）。\n用于调整重建区域沿旋转轴的位置。"
        self.roi_center_z_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelRoiZ").setToolTip(tip)

        # ── 校准参数 ──
        tip = "eta — 锥束倾斜角（rad）。\n射线锥束中心线与探测器法线之间的微小偏转。\n由几何校准自动算出，通常接近 0。"
        self.eta_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelEta").setToolTip(tip)

        tip = "vc_recon — 探测器水平方向亚像素偏移校正值。\n校准得出，用于精确对齐投影中心。"
        self.vc_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelVc").setToolTip(tip)

        tip = "vs_recon — 探测器垂直方向亚像素偏移校正值。\n校准得出，用于精确对齐投影中心。"
        self.vs_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelVs").setToolTip(tip)

        tip = "sx — 水平方向缩放/binning 系数。\n0.5 表示 2×2 binning（宽度减半）。\n1.0 表示原始分辨率。"
        self.sx_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelSx").setToolTip(tip)

        tip = "sy — 垂直方向缩放/binning 系数。\n0.5 表示 2×2 binning（高度减半）。\n1.0 表示原始分辨率。"
        self.sy_line_edit.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelSy").setToolTip(tip)

        # ── 高级选项 ──
        tip = ("FDK 滤波器类型 — 控制频域滤波的窗函数：\n"
               "• Ram-Lak：标准斜坡滤波，分辨率最高但噪声也最大\n"
               "• Shepp-Logan：轻度平滑，略降分辨率，适合低噪场景\n"
               "• Cosine：中等平滑，噪声和分辨率的折中\n"
               "• Hamming：较强平滑，大幅抑制噪声，适合高噪数据\n"
               "• Hann：最强平滑，噪声抑制最佳但细节损失最多")
        self.filter_type_combo.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelFilter").setToolTip(tip)

        tip = "启用后对正弦图做中值滤波，去除由探测器坏像素\n导致的环形伪影（ring artifact）。"
        self.ring_correction_check.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelRingCorr").setToolTip(tip)

        tip = "环形伪影校正的中值滤波核大小（像素）。\n值越大去除越激进，但可能模糊边缘。\n建议 5–15 之间的奇数。"
        self.ring_kernel_spin.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelRingKernel").setToolTip(tip)

        tip = ("重建算法选择：\n"
               "• FDK：Feldkamp-Davis-Kress 解析算法\n"
               "  单次反投影，速度最快（秒级），适合数据充足时使用\n"
               "• SIRT：联合迭代重建\n"
               "  逐次逼近最小二乘解，对噪声和稀疏投影更鲁棒\n"
               "  但速度较慢，需设置迭代次数\n"
               "• CGLS：共轭梯度最小二乘\n"
               "  收敛比 SIRT 更快，适合投影数较少或高噪数据\n"
               "  但迭代过多可能过拟合噪声")
        self.algorithm_combo.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelAlgorithm").setToolTip(tip)

        tip = "迭代重建（SIRT/CGLS）的迭代次数。\nSIRT 建议 50–200 次，CGLS 建议 10–50 次。\n次数越多收敛越好，但耗时线性增加。\n仅在选择 SIRT 或 CGLS 时生效。"
        self.iterations_spin.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelIterations").setToolTip(tip)

        tip = "启用后将重建体中的负值裁剪为 0。\nCT 衰减系数物理上非负，开启可消除\n重建中因噪声/截断产生的负值伪影。"
        self.non_neg_check.setToolTip(tip)
        self.ui.findChild(QtWidgets.QLabel, "labelNonNeg").setToolTip(tip)

        tip = "勾选后使用当前扫描窗口中已采集的投影数据重建。\n不勾选则从项目目录下自动读取投影文件。"
        self.use_scan_check_box.setToolTip(tip)

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
            "filter_type": self.filter_type_combo.currentText(),
            "ring_correction": self.ring_correction_check.isChecked(),
            "ring_kernel_size": self.ring_kernel_spin.value(),
            "algorithm": self.algorithm_combo.currentText(),
            "iterations": self.iterations_spin.value(),
            "non_neg_constraint": self.non_neg_check.isChecked(),
        }
        self.params = params
        self.accept()
