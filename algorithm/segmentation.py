import os
import sys
import numpy as np
import nibabel as nib
from scipy import ndimage

def segment_surface(vol, thresh=None, min_component_size=500, surface_only=True, surface_thickness=1):
    """
    从重建体中分割物体，区分空气与实际物体。
    不依赖标定后的绝对衰减系数，使用自适应百分位阈值。

    参数:
        vol: ndarray float32, 重建体数据
        thresh: float or None, 手动阈值；None则自动计算
        min_component_size: int, 小于此体素数的连通域去除
        surface_only: bool, True只保留表面壳层
        surface_thickness: int, 表面厚度（腐蚀次数）

    返回:
        seg: ndarray uint8, 1=物体 0=背景
        thresh_used: float
    """
    if thresh is None:
        flat = vol.ravel()
        p_low = float(np.percentile(flat, 5))
        p_high = float(np.percentile(flat, 99.5))
        if p_high - p_low < 1e-9:
            print("[Segment] 警告：数据几乎无对比度，返回空分割")
            return np.zeros_like(vol, dtype=np.uint8), 0.0
        foreground = flat[(flat > p_low) & (flat < p_high)]
        if len(foreground) == 0:
            return np.zeros_like(vol, dtype=np.uint8), 0.0
        hist, bin_edges = np.histogram(foreground, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = hist.sum()
        w0 = np.cumsum(hist).astype(np.float64)
        w1 = total - w0
        sum0 = np.cumsum(hist * bin_centers)
        mu0 = sum0 / np.maximum(w0, 1)
        mu1 = (sum0[-1] - sum0) / np.maximum(w1, 1)
        variance = w0 * w1 * (mu0 - mu1) ** 2
        thresh = float(bin_centers[np.argmax(variance)])
    print("[Segment] 阈值: {:.6f}".format(thresh))
    binary = (vol > thresh).astype(np.uint8)
    print("[Segment] 二值化体素: {} / {}".format(binary.sum(), binary.size))
    labeled, num_features = ndimage.label(binary)
    if num_features > 0:
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        lut = np.zeros(num_features + 1, dtype=np.uint8)
        for i in range(len(sizes)):
            if sizes[i] >= min_component_size:
                lut[i + 1] = 1
        binary = lut[labeled]
        kept = int(lut.sum())
        print("[Segment] 连通域: {}, 保留 >= {}: {}".format(num_features, min_component_size, kept))
    binary_filled = ndimage.binary_fill_holes(binary).astype(np.uint8)
    if surface_only:
        eroded = ndimage.binary_erosion(binary_filled, iterations=surface_thickness).astype(np.uint8)
        seg = (binary_filled - eroded).astype(np.uint8)
        print("[Segment] 表面体素: {} (厚度={})".format(seg.sum(), surface_thickness))
    else:
        seg = binary_filled
        print("[Segment] 实心体素: {}".format(seg.sum()))
    return seg, thresh

def main():
    if len(sys.argv) < 2:
        print("用法: python -m algorithm.segmentation <project_path> [--solid] [--thresh 0.05] [--thickness 2]")
        sys.exit(1)
    project_path = sys.argv[1]
    rec_path = os.path.join(project_path, "rec.nii.gz")
    if not os.path.isfile(rec_path):
        print("[Error] 找不到 " + rec_path)
        sys.exit(1)
    surface_only = "--solid" not in sys.argv
    thresh = None
    if "--thresh" in sys.argv:
        idx = sys.argv.index("--thresh")
        thresh = float(sys.argv[idx + 1])
    thickness = 1
    if "--thickness" in sys.argv:
        idx = sys.argv.index("--thickness")
        thickness = int(sys.argv[idx + 1])
    print("[Segment] 加载 " + rec_path + " ...")
    nii = nib.load(rec_path)
    vol = nii.get_fdata().astype(np.float32)
    print("[Segment] shape: {}, 范围: [{:.6f}, {:.6f}]".format(vol.shape, vol.min(), vol.max()))
    seg, thresh_used = segment_surface(vol, thresh=thresh, surface_only=surface_only, surface_thickness=thickness)
    seg_path = os.path.join(project_path, "seg.nii.gz")
    seg_nii = nib.Nifti1Image(seg, affine=nii.affine, header=nii.header)
    seg_nii.header.set_data_dtype(np.uint8)
    nib.save(seg_nii, seg_path)
    print("[Segment] 已保存 -> " + seg_path)
    print("[Segment] ITK-SNAP: Open Image -> rec.nii.gz, Open Segmentation -> seg.nii.gz")

if __name__ == "__main__":
    main()