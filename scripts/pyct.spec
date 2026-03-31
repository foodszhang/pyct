# -*- mode: python ; coding: utf-8 -*-
# PyCT PyInstaller spec 文件
# 用法：pyinstaller scripts/pyct.spec

# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import importlib

block_cipher = None
repo_root = os.path.dirname(os.path.abspath(SPECPATH))

# ---- Collect .ui files ----
ui_files = []
qt_gui_dir = os.path.join(repo_root, 'qt_gui')
for f in os.listdir(qt_gui_dir):
    if f.endswith('.ui'):
        ui_files.append((os.path.join(qt_gui_dir, f), 'qt_gui'))

# ---- Collect config ----
data_files = ui_files + [
    (os.path.join(repo_root, 'config.yaml'), 'config'),
]

for f in ['custom_layout.ini', 'user_custom_layout.ini']:
    path = os.path.join(repo_root, f)
    if os.path.exists(path):
        data_files.append((path, '.'))

# ---- Collect ASTRA binaries ----
astra_binaries = []
astra_datas = []
try:
    import astra
    astra_dir = os.path.dirname(astra.__file__)
    # Collect all .pyd, .dll, .so files
    for root, dirs, files in os.walk(astra_dir):
        rel = os.path.relpath(root, os.path.dirname(astra_dir))
        for f in files:
            full = os.path.join(root, f)
            if f.endswith(('.pyd', '.dll', '.so')):
                astra_binaries.append((full, rel))
            elif f.endswith(('.py', '.cfg', '.txt', '.dat')):
                astra_datas.append((full, rel))
    print(f"[spec] Collected {len(astra_binaries)} astra binaries, {len(astra_datas)} astra data files")
except ImportError:
    print("[spec] WARNING: astra not found, CUDA reconstruction will not work in packaged app")

a = Analysis(
    [os.path.join(repo_root, 'pyct_app.py')],
    pathex=[repo_root],
    binaries=astra_binaries,
    datas=data_files + astra_datas,
    hiddenimports=[
        'astra',
        'astra.astra_c',
        'astra.data2d_c',
        'astra.data3d_c',
        'astra.projector_c',
        'astra.algorithm_c',
        'astra.creators',
        'astra.functions',
        'astra.extrautils',
        'astra.log',
        'algorithm.astra.conebeam',
        'algorithm.calibration.cal',
        'qt_gui.gui',
        'qt_gui.reconstruction',
        'qt_gui.scan_window',
        'qt_gui.snap_window',
        'scipy.optimize',
        'scipy.optimize._lsq',
        'scipy.optimize._lsq.least_squares',
        'scipy.ndimage',
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
        'matplotlib',
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
