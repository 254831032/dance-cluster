# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（单文件 exe，Windows）。

依赖的二进制都在构建前准备好：
  models/pose_landmarker_heavy.task   （CI 下载）
  .tools/bin/ffmpeg.exe, ffprobe.exe  （CI 下载）
"""
import os

import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

datas = []
binaries = []

# 姿态模型
if os.path.exists(os.path.join("models", "pose_landmarker_heavy.task")):
    datas.append(("models/pose_landmarker_heavy.task", "models"))

# ffmpeg / ffprobe
for exe in ("ffmpeg", "ffprobe"):
    for name in (exe, exe + ".exe"):
        p = os.path.join(".tools", "bin", name)
        if os.path.exists(p):
            datas.append((p, ".tools/bin"))
            break

# mediapipe 的原生库与数据
datas += collect_data_files("mediapipe")
binaries += collect_dynamic_libs("mediapipe")

# soundfile 自带 libsndfile
datas += collect_data_files("soundfile")
binaries += collect_dynamic_libs("soundfile")

# pywebview 的 JS 资源 + 平台后端
datas += collect_data_files("webview", includes=["js/*.js", "js/lib/*.js"])
hiddenimports_webview = ["webview", "bottle", "proxy_tools"]
if sys.platform == "win32":
    hiddenimports_webview += ["webview.platforms.winforms", "webview.platforms.edgechromium"]
    for pkg in ("pythonnet", "clr_loader", "pystray"):
        try:
            d, b, h = collect_all(pkg)
            datas += d
            binaries += b
            hiddenimports_webview += h
        except Exception:
            pass
elif sys.platform == "darwin":
    hiddenimports_webview += ["webview.platforms.cocoa"]

hiddenimports = [
    "mediapipe.tasks.python.vision",
    "mediapipe.tasks.python.core",
    "mediapipe.tasks.python.components",
    "mediapipe.tasks.python.components.containers",
    "soundfile",
    "librosa",
    "numba",
    "flask",
    "jinja2",
    "werkzeug",
] + hiddenimports_webview

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide6", "PySide6_Essentials"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="DanceCluster",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
