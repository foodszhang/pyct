# -*- mode: python ; coding: utf-8 -*-
# PyCT PyInstaller spec 文件
# 用法：pyinstaller scripts/pyct.spec

import os
import sys

block_cipher = None
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(SPECPATH)))

# 收集所有 .ui 文件
ui_files = []
qt_gui_dir = os.path.join(repo_root, 'qt_gui')
for f in os.listdir(qt_gui_dir):
    if f.endswith('.ui'):
        ui_files.append((os.path.join(qt_gui_dir, f), 'qt_gui'))

# 收集默认配置
data_files = ui_files + [
    (os.path.join(repo_root, 'config.yaml'), 'config'),
]

# 收集 .ini 文件（布局配置）
for f in ['custom_layout.ini', 'user_custom_layout.ini']:
    path = os.path.join(repo_root, f)
    if os.path.exists(path):
        data_files.append((path, '.'))

a = Analysis(
    [os.path.join(repo_root, 'pyct_app.py')],
    pathex=[repo_root],
    binaries=[],
    datas=data_files,
    hiddenimports=[
        'algorithm.astra.conebeam',
        'algorithm.calibration.cal',
        'qt_gui.gui',
        'qt_gui.reconstruction',
        'qt_gui.scan_window',
        'qt_gui.snap_window',
        'scipy.optimize',
        'scipy.optimize._lsq',
        'scipy.optimize._lsq.least_squares',
        'numba',
        'cv2',
        'nibabel',
        'pyqtgraph',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtUiTools',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',  # 只在 cal.py __main__ 诊断用，GUI 不需要
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PyCT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # 不用 UPX 压缩，避免误报杀软
    console=False,  # 无黑窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 如果有 icon 文件可以在这里指定
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PyCT',
)
