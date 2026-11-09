"""打包入口：PyInstaller 从这里启动界面。"""

from dancecluster.webui import main

if __name__ == "__main__":
    raise SystemExit(main())
