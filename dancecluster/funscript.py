"""funscript 关联与解析。

一个视频可能对应多个 funscript（多轴），命名约定：
    xxx.mp4
    xxx.funscript            → 主轴（axis = "main"）
    xxx.row.funscript        → 附加轴（axis = "row"）
    xxx.twist.funscript      → 附加轴（axis = "twist"）

即：同目录下、文件名以「视频名 + '.'」开头、以「.funscript」结尾的文件。
轴数 = 关联到的 funscript 文件数量（0=无脚本，1=单轴，2+=多轴）。
"""

from __future__ import annotations

import json
from pathlib import Path

EXT = ".funscript"


def find_for(video_path: str | Path) -> list[tuple[str, Path]]:
    """找某视频关联的 funscript，返回 [(axis, path)]，main 排最前。"""
    video = Path(video_path)
    stem = video.stem
    folder = video.parent
    if not stem or not folder.is_dir():
        return []
    prefix = (stem + ".").lower()
    try:
        entries = list(folder.iterdir())
    except OSError:
        return []

    found: list[tuple[str, Path]] = []
    for p in entries:
        if not p.is_file():
            continue
        low = p.name.lower()
        if not low.endswith(EXT) or not low.startswith(prefix):
            continue
        mid = p.name[len(stem) + 1 : -len(EXT)]
        axis = mid if mid else "main"
        found.append((axis, p))

    found.sort(key=lambda x: (x[0] != "main", x[0].lower()))
    return found


def parse(path: str | Path, max_actions: int | None = None) -> dict:
    """解析 funscript，返回动作与概要统计。"""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"actions": [], "count": 0, "duration_ms": 0, "speed": 0.0}

    raw = data.get("actions") or []
    acts: list[list[int]] = []
    for a in raw:
        if isinstance(a, dict) and "at" in a and "pos" in a:
            try:
                acts.append([int(a["at"]), int(a["pos"])])
            except (TypeError, ValueError):
                continue

    count = len(acts)
    duration = acts[-1][0] - acts[0][0] if count >= 2 else 0
    speed = 0.0
    if count >= 2 and duration > 0:
        total = sum(abs(acts[i][1] - acts[i - 1][1]) for i in range(1, count))
        speed = total / (duration / 1000.0)

    out_actions = acts
    if max_actions and len(acts) > max_actions:
        step = len(acts) / max_actions
        out_actions = [acts[int(i * step)] for i in range(max_actions)]

    return {
        "actions": out_actions,
        "count": count,
        "duration_ms": duration,
        "speed": round(speed, 1),
    }


def scan(video_path: str | Path) -> dict:
    """扫描并返回概要：{count, axes:[...], files:[{axis,path,count,duration_ms,speed}]}。"""
    files = find_for(video_path)
    axes = [axis for axis, _ in files]
    detail = []
    for axis, p in files:
        info = parse(p)
        detail.append(
            {"axis": axis, "path": str(p), "count": info["count"],
             "duration_ms": info["duration_ms"], "speed": info["speed"]}
        )
    return {"count": len(files), "axes": axes, "files": detail}


def level(count: int) -> str:
    """把轴数归到 none / single / multi。"""
    if count <= 0:
        return "none"
    if count == 1:
        return "single"
    return "multi"
