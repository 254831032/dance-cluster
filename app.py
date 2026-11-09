"""打包入口：优先打开原生窗口（pywebview），失败则退回浏览器模式。"""

import sys


def main() -> int:
    try:
        from dancecluster.desktop import main as desktop_main

        return desktop_main()
    except Exception as exc:  # 没装 pywebview / WebView 不可用时退回浏览器
        print(f"桌面窗口不可用（{type(exc).__name__}: {exc}），改用浏览器模式", file=sys.stderr)
        from dancecluster.webui import main as web_main

        return web_main()


if __name__ == "__main__":
    raise SystemExit(main())
