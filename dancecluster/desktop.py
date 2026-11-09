"""原生窗口模式（pywebview）+ Windows 托盘图标。

- Windows：关闭窗口 = 最小化到托盘，任务后台继续；托盘菜单「显示窗口 / 退出」。
- macOS：关闭窗口时若任务在跑，弹确认防误关（macOS 菜单栏图标与 pywebview 抢主线程，暂不做）。
"""

from __future__ import annotations

import os
import sys
import threading

import webview

from . import config, webui


def _setup_tray(window, quitting: dict):
    """Windows 托盘图标（pystray），返回 icon 对象。"""
    import pystray
    from PIL import Image, ImageDraw

    def _icon_image():
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((4, 4, 60, 60), fill=(91, 140, 255, 255))
        d.ellipse((22, 22, 42, 42), fill=(255, 255, 255, 255))
        return img

    def _show(icon, item):
        try:
            window.show()
        except Exception:
            pass

    def _quit(icon, item):
        quitting["v"] = True
        try:
            icon.stop()
        except Exception:
            pass
        try:
            window.destroy()
        except Exception:
            pass
        os._exit(0)

    icon = pystray.Icon(
        "DanceCluster",
        _icon_image(),
        "DanceCluster · 舞蹈视频自动分组",
        menu=pystray.Menu(
            pystray.MenuItem("显示窗口", _show, default=True),
            pystray.MenuItem("退出", _quit),
        ),
    )
    threading.Thread(target=icon.run, daemon=True).start()
    return icon


def main() -> int:
    if not config.acquire_instance_lock():
        print("DanceCluster 已在运行，请勿重复启动。", file=sys.stderr)
        return 0
    port = webui._free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"DanceCluster 桌面模式：{url}", flush=True)

    threading.Thread(
        target=lambda: webui.app.run(
            host="127.0.0.1", port=port, threaded=True, use_reloader=False
        ),
        daemon=True,
    ).start()

    webui._ui["desktop"] = True
    window = webview.create_window(
        "DanceCluster · 舞蹈视频自动分组",
        url,
        width=1120,
        height=820,
        min_size=(920, 640),
    )
    webui._ui["window"] = window

    quitting = {"v": False}
    tray = None
    if sys.platform == "win32":
        try:
            tray = _setup_tray(window, quitting)
        except Exception:
            tray = None

    def _on_closing():
        if quitting["v"]:
            return True
        if sys.platform == "win32" and tray is not None:
            # 最小化到托盘，后台继续
            try:
                window.hide()
            except Exception:
                pass
            return False
        # 无托盘的平台：任务进行中才提示，避免误关
        if webui._state.get("running"):
            try:
                return bool(
                    window.create_confirmation_dialog(
                        "退出确认",
                        "有任务正在运行，退出会中断任务。\n确定要退出吗？",
                    )
                )
            except Exception:
                return True
        return True

    window.events.closing += _on_closing
    webview.start()

    if tray is not None:
        try:
            tray.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
