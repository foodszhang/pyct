# PyCT 离线安装指南

## 系统要求
- Windows 10 x64（1809 或更高版本）
- NVIDIA GPU + 驱动（推荐，用于 CUDA 加速重建；无 GPU 会自动降级 CPU）
- VC++ 2015-2022 Redistributable x64（安装包已附带或需单独安装）

## 安装步骤

### 方式 1：安装包（推荐）
1. 双击 `PyCT_Setup_x.x.x.exe`
2. 按提示完成安装（默认安装到 `C:\Program Files\PyCT\`）
3. 桌面会出现 PyCT 快捷方式
4. 双击启动即可

### 方式 2：便携版
1. 解压 `PyCT_Portable_x.x.x.zip` 到任意目录（建议纯英文路径）
2. 双击 `PyCT.exe` 启动

## 设备扫描功能（可选）
如果需要连接 X 射线设备进行扫描：
1. 将 Python 3.4 便携运行时放到 `detector_bridge\py34\` 目录下
2. 确保设备 SDK 的 DLL 已安装或放在系统 PATH 中
3. 详见 `detector_bridge\README.txt`

**如果不需要扫描（只做校准/重建），无需配置此步骤。**

## 校准与重建（离线可用）
1. 启动 PyCT
2. 选择项目目录（包含投影 .tif 文件的文件夹）
3. 切换到"校准"选项卡，点击"校准"
4. 校准完成后点"保存校准结果"
5. 切换到"重建"选项卡，点击"开始重建"
6. 重建结果保存为项目目录下的 `rec.nii.gz`

## 投影文件要求
- 格式：`.tif`（16bit）
- 文件名：角度值（度数），例如 `0.tif`、`1.5.tif`、`359.0.tif`
- 支持缺角、非均匀角度分布

## 常见问题

### Q: 启动时提示"CUDA 不可用"
A: 程序会自动降级到 CPU 重建，功能不受影响，只是速度较慢。如需 GPU 加速，请安装 NVIDIA 驱动。

### Q: 启动时提示"硬件不可用，离线模式"
A: 正常现象。没有连接 X 射线设备时会显示此提示，校准和重建功能不受影响。

### Q: 缺少 VC++ 运行库
A: 请安装 Microsoft Visual C++ 2015-2022 Redistributable (x64)。
下载地址：https://aka.ms/vs/17/release/vc_redist.x64.exe
（需在联网机器下载后拷贝到目标机安装）

## 日志文件
运行日志保存在安装目录下的 `logs\` 文件夹，按日期命名：`pyct_yyyy-mm-dd.log`
