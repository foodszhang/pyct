# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

个人CBCT（锥形束CT）采集→校准→重建全流程软件，跨平台设计（主要Windows）。

## 构建/测试命令

### 包管理（uv）
```bash
uv pip install -r requirements.txt
uv pip install <包名>
```

### 运行测试
单个测试文件执行：
```bash
python ct_cli_test.py
python thread_test.py
python detector_test.py
python algorithm/astra/test_ast.py
```

### 运行应用程序
```bash
python main.py              # 主入口（调用qt_gui/gui.py）
python qt_gui/gui.py        # Qt GUI界面
python ct_cli.py            # CLI命令行界面
```

## 架构说明

### 核心流程
1. **采集**：`detector.py`（DexelaPy集成）→ `pipe.py`（Windows命名管道）
2. **校准**：`algorithm/calibration/cal.py`（计算SOD、SDD、u0、v0、theta）
3. **重建**：`algorithm/astra/conebeam.py`（FDK_CUDA算法）
4. **界面**：`qt_gui/gui.py`（PySide6界面）

### 目录结构
- `algorithm/` - 核心算法（校准、重建）
- `algorithm/calibration/` - 校准模块
- `algorithm/astra/` - ASTRA工具箱重建算法
- `qt_gui/` - PySide6 Qt GUI界面（.ui文件 + Python代码）
- `gui/` - 旧版dearpygui GUI（ct_control.py等）
- `utils/paths.py` - 路径工具，支持PyInstaller打包
- `config.yaml` - 配置文件（硬件参数、校准参数、重建参数）

### 关键依赖
- `numpy`、`opencv-python` - 数组/图像处理
- `numba` - 重建核的JIT编译加速
- `astra-toolbox` - GPU加速CT重建
- `PySide6` - Qt GUI框架
- `pyserial` - X射线控制器的串口通信
- `pyyaml` - 配置文件解析
- `DexelaPy` - Windows专用探测器硬件接口（.pyd文件）

### 硬件接口
- X射线控制器：`UltraBrightController`（串口，config.yaml中BrightController段）
- 旋转台：`ZolixMcController`（串口，config.yaml中ZolixMcController段）
- 探测器：DexelaPy（仅Windows）

### 配置说明
- 所有设置在 `config.yaml`
- 配置段：`BrightController`、`CalibrationParam`、`ReconParam`、`ZolixMcController`、`CalibResult`
- YAML加载：`yaml.load(open(get_config_path()), Loader=yaml.FullLoader)`

## 代码风格指南

### 导入规范
- 标准库导入放前面 → 第三方库（numpy为np，cv2，numba为nb）→ 本地模块
- 避免 `from xxx import *`

### 命名规范
- 类名：CamelCase（如 `ConeBeam`、`Detector`、`Calibration`）
- 函数/方法：snake_case（如 `load_img`、`calculate`、`seq_start`）
- 常量：UPPER_SNAKE_CASE 或模块级赋值

### 类型注解
- 函数参数和返回值使用类型注解
- numpy数组：`img: np.ndarray`，不用 `typing.Any`

### 跨平台注意事项
- Windows命名管道：`r"\\.\pipe\detectResult"`
- 串口端口：通过 `config.yaml` 配置（如COM3、COM4）
- 文件路径：始终使用 `os.path.join()` 构建路径
- 后台线程使用 `daemon=True`

### Numba JIT函数
- 性能关键循环使用 `@nb.jit(nopython=True, parallel=True)` 和 `nb.prange`

### 路径工具（utils/paths.py）
- `get_base_path()` - 项目根目录（兼容PyInstaller打包）
- `get_config_path()` - config.yaml路径
- `get_ui_path(ui_filename)` - .ui文件路径
- `ensure_cwd_for_develop()` - 开发阶段切换CWD到项目根
