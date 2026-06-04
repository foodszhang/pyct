"""
路径工具模块 - 兼容 PyInstaller 打包和源码运行。

PyInstaller 打包后，运行时的工作目录 (CWD) 可能不是代码目录，
因此所有资源文件路径必须基于代码本身的位置来计算。

打包后资源（.ui、默认config）从 _MEIPASS 读取；
用户数据（运行时写入的 config.yaml）写到可执行文件同目录下。
"""

import os
import sys


def _recommended_default_config() -> dict:
    return {
        "BrightController": {"baudrate": 38400, "port": "COM3", "timeout": 4},
        "CalibrationParam": {
            "BBNumber": 6,
            "allowLineAngleOffsetFit": False,
            "allowLineRollFit": False,
            "applyLegacyThetaAsRoll": False,
            "beadLayout": "line_z",
            "beadSpacing": 10.0,
            "detectorHeight": 1944,
            "detectorPixelSize": 0.0748,
            "detectorWidth": 1536,
            "method": "vshift",
            "useSuspiciousBeads": False,
        },
        "ReconParam": {
            "SDD": "972.3",
            "SOD": "904.95",
            "algorithm": "FDK",
            "angle": "1",
            "columnCount": "1536",
            "detectorX": "918.49518",
            "detectorY": "148.46",
            "fillMissingDegrees": False,
            "filterType": "Hamming",
            "iterations": 50,
            "nonNegConstraint": True,
            "projectionMedianFilter": True,
            "projectionMedianKernel": 3,
            "rescale_intercept": -502.588,
            "rescale_slope": -7.99,
            "ringCorrection": True,
            "ringKernelSize": 15,
            "rotation": "0.0",
            "roiCenterX": "4.75",
            "roiCenterY": "20.375",
            "roiCenterZ": "-33.75",
            "rowCount": "1944",
            "voxelPixelSize": "0.1",
            "voxelSizeX": "192",
            "voxelSizeY": "256",
            "voxelSizeZ": "768",
            "xSpacing": "0.0748",
            "ySpacing": "0.0748",
        },
        "ZolixMcController": {"baudrate": 19200, "port": "COM4", "timeout": 400},
        "CalibResult": {
            "SOD": 904.95,
            "SDD": 972.3,
            "angle_offset_deg": 0.0,
            "detector_roll_deg": 0.0,
            "u0_cal": 916.12,
            "u0_est": 918.49518,
            "u0_raw": 918.5,
            "u0_used": 918.49518,
            "v0_raw": 148.46,
            "v0_used": 148.46,
            "theta_cal_deg": 0.17,
            "roll_source": "disabled_for_legacy",
            "eta": -3.0e-06,
            "vc_raw": 1.203457,
            "vs_raw": -0.158537,
            "rms_init": 1.0049,
            "rms_final": 0.7155,
            "sx": 1.0,
            "sy": 1.0,
            "u0_recon": 918.49518,
            "v0_recon": 148.46,
            "vc_recon": 1.203457,
            "vs_recon": -0.158537,
        },
    }


def _maybe_upgrade_legacy_default_config(cfg_path: str) -> None:
    import yaml

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader) or {}
    except (OSError, yaml.YAMLError):
        return

    recon = config.get("ReconParam", {})
    is_legacy_default = (
        str(recon.get("columnCount")) == "512"
        and str(recon.get("rowCount")) == "512"
        and str(recon.get("voxelPixelSize")) == "0.25"
        and str(recon.get("voxelSizeX")) == "512"
        and str(recon.get("voxelSizeY")) == "512"
        and str(recon.get("voxelSizeZ")) == "512"
        and "projectionMedianFilter" not in recon
    )
    if not is_legacy_default:
        return

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(_recommended_default_config(), f, Dumper=yaml.Dumper, allow_unicode=True)
    print(f"[Config] 检测到旧版默认重建配置，已升级为推荐默认配置: {cfg_path}")


def get_base_path() -> str:
    """
    返回项目根目录（代码所在目录）。

    - 打包后（frozen）：返回可执行文件所在目录（可写）
    - 源码运行：返回 __file__ 的父目录的父目录（即项目根）
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_resource_path(relative_path: str) -> str:
    """
    返回资源文件的绝对路径（用于读取 .ui、默认配置等）。

    - 打包后（frozen）：返回 _MEIPASS 下的资源路径
    - 源码运行：返回项目根目录 + relative_path
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(get_base_path(), relative_path)


def get_config_path() -> str:
    """
    返回 config.yaml 的读写路径。
    始终返回 get_base_path() 下的 config.yaml（即 exe 同目录或项目根）。
    """
    return os.path.join(get_base_path(), "config.yaml")


def get_ui_path(ui_filename: str) -> str:
    """
    返回 .ui 文件的绝对路径（用于读取）。
    .ui 文件放在 qt_gui/ 目录下。
    """
    return get_resource_path(os.path.join("qt_gui", ui_filename))


def ensure_config_exists() -> str:
    """
    确保 config.yaml 存在。

    - 如果 get_config_path() 已存在，直接返回
    - 如果不存在，尝试从资源路径复制（config/default_config.yaml）
    - 如果资源路径也不存在，创建一份最小默认配置

    返回 config.yaml 的路径。
    """
    cfg_path = get_config_path()
    if os.path.isfile(cfg_path):
        _maybe_upgrade_legacy_default_config(cfg_path)
        return cfg_path

    resource_default = get_resource_path(os.path.join("config", "default_config.yaml"))
    if os.path.isfile(resource_default):
        import shutil

        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        shutil.copy(resource_default, cfg_path)
        print(f"[Config] 从默认配置复制到 {cfg_path}")
        return cfg_path

    import yaml

    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(_recommended_default_config(), f, Dumper=yaml.Dumper, allow_unicode=True)
    print(f"[Config] 创建默认配置 {cfg_path}")
    return cfg_path


def get_logs_dir() -> str:
    """
    返回日志目录路径（get_base_path()/logs/）。
    日志文件名格式：pyct_yyyy-mm-dd.log
    """
    logs = os.path.join(get_base_path(), "logs")
    os.makedirs(logs, exist_ok=True)
    return logs


def ensure_cwd_for_develop():
    """
    开发阶段（源码运行）：将 CWD 切换到项目根目录，
    使现有的相对路径加载（如 open("config.yaml")）继续工作。

    打包后：不需要（CWD 应保持原样）。
    """
    if not getattr(sys, "frozen", False):
        base = get_base_path()
        if os.path.isdir(base):
            os.chdir(base)


def get_base_exe_dir() -> str:
    """
    返回可执行文件所在目录。
    frozen 模式返回 sys.executable 的目录，否则返回项目根目录。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_py34() -> str:
    """
    按优先级查找 Python 3.4 可执行文件路径：
    1. 捆绑在 exe 旁边的 detector_bridge/py34/python.exe
    2. 环境变量 py34

    返回路径字符串，找不到返回空字符串。
    """
    # 方式 1：exe 同级目录（PyInstaller frozen）
    exe_dir = get_base_exe_dir()
    bundled = os.path.join(exe_dir, "detector_bridge", "py34", "python.exe")
    if os.path.isfile(bundled):
        return bundled

    # 方式 2：环境变量
    env_py34 = os.environ.get("py34", "")
    if env_py34 and os.path.isfile(env_py34):
        return env_py34

    return ""


def get_detector_bridge_dir() -> str:
    """
    返回 detector_bridge 目录路径。
    detector.py 放在这里。
    """
    return os.path.join(get_base_exe_dir(), "detector_bridge")
