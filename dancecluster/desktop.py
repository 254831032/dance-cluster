"""原生窗口模式（pywebview）。

本地起一个 Flask 服务，再用系统 WebView（macOS WKWebView / Windows WebView2）
把它装进一个独立窗口里——看起来就是一个桌面 App，同时保留 Web 的开发效率。

额外能力：窗口句柄交给 webui，前端就能调起系统的「选择文件夹」对话框。
"""

from __future__ import annotations

import threading

import webview

from . import webui


def main() -> int:
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
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
