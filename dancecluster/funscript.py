"""funscript 关联与解析。

一个视频可能对应多个 funscript（多轴）。有两种形态：
  1. 多文件：xxx.funscript（主轴）、xxx.r0.funscript（附加轴）… 文件名后缀 = 轴
  2. 单文件多轴：一个 .funscript 内同时含多个轴轨道，必须按内容判断

命名约定：
    xxx.funscript            → 主轴
    xxx.twist.funscript      → 附加轴

轴数按「文件内容」判断，不按文件数量：
  - 单文件里若检测到多个轴轨道 → 算多轴
  - 多文件 → 一定是多轴
"""

from __future__ import annotations

import json
from pathlib import Path

EXT = ".funscript"

# 键名 → 标准轴 映射表（尽量全，兼容多种工具/格式）
AXIS_ALIASES = {
    "actions": "L0",
    "l0": "L0", "stroke": "L0", "surge": "L0",
    "l1": "L1", "sway": "L1",
    "l2": "L2",
    "r0": "R0", "twist": "R0",
    "r1": "R1", "pitch": "R1",
    "r2": "R2", "roll": "R2",
    "v0": "V0",
    "v1": "V1",
}


def find_for(video_path: str | Path) -> list[tuple[str, Path]]:
    """找某视频关联的 funscript 文件，返回 [(axis, path)]，main 排最前。"""
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


def _is_action_list(v) -> bool:
    """判断某个值是否是「动作轨道」：非空数组、元素是 {at/pos} 或 {time/value} 或 {t/p}。"""
    if not isinstance(v, list) or not v:
        return False
    first = v[0]
    if not isinstance(first, dict):
        return False
    has_t = ("at" in first) or ("time" in first) or ("t" in first)
    has_p = ("pos" in first) or ("value" in first) or ("p" in first)
    return has_t and has_p


def _axis_name(key: str) -> str:
    """把键名映射成标准轴名；未知键名原样返回（仍算一个独立轴）。"""
    k = key.lower().strip()
    if k in AXIS_ALIASES:
        return AXIS_ALIASES[k]
    if k.startswith("actions_"):
        suffix = k[len("actions_"):]
        if suffix in AXIS_ALIASES:
            return AXIS_ALIASES[suffix]
        return suffix.upper()
    return key


def _norm_action(a) -> list[int] | None:
    if not isinstance(a, dict):
        return None
    at = a.get("at", a.get("time", a.get("t")))
    pos = a.get("pos", a.get("value", a.get("p")))
    if at is None or pos is None:
        return None
    try:
        return [int(at), int(pos)]
    except (TypeError, ValueError):
        return None


def detect_axes(data) -> list[str]:
    """从解析后的 JSON 判断里面有哪些轴（按内容，兼容多种格式）。"""
    axes: list[str] = []
    if isinstance(data, list):
        # 情况 F：顶层就是动作数组，每个动作带 axis
        for a in data:
            if isinstance(a, dict) and a.get("axis"):
                ax = str(a["axis"]).upper()
                if ax and ax not in axes:
                    axes.append(ax)
        return axes
    if not isinstance(data, dict):
        return []

    for key, val in data.items():
        if key == "metadata":
            continue
        if _is_action_list(val):
            ax = _axis_name(key)
            if ax and ax not in axes:
                axes.append(ax)

    # metadata.axes 做补充（仅在没有实际数组时兜底）
    if not axes:
        meta = data.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("axes"), list):
            for a in meta["axes"]:
                ax = str(a).upper()
                if ax and ax not in axes:
                    axes.append(ax)
    return axes


def _extract_actions(data, axis: str | None = None) -> list[list[int]]:
    """从 data 里取某个轴的动作列表（axis=None 取第一个动作轨道）。"""
    if isinstance(data, list):
        out = []
        for a in data:
            ax = str(a.get("axis", "")).upper() if isinstance(a, dict) else ""
            if axis is None or ax == axis:
                n = _norm_action(a)
                if n:
                    out.append(n)
        return out
    if not isinstance(data, dict):
        return []

    chosen = None
    for key, val in data.items():
        if _is_action_list(val):
            mapped = _axis_name(key).upper()
            if axis is None or mapped == axis:
                chosen = val
                break
    if chosen is None:
        return []
    out = []
    for a in chosen:
        n = _norm_action(a)
        if n:
            out.append(n)
    return out


def _stats(acts: list[list[int]]) -> dict:
    count = len(acts)
    duration = acts[-1][0] - acts[0][0] if count >= 2 else 0
    speed = 0.0
    if count >= 2 and duration > 0:
        total = sum(abs(acts[i][1] - acts[i - 1][1]) for i in range(1, count))
        speed = total / (duration / 1000.0)
    return {"count": count, "duration_ms": duration, "speed": round(speed, 1)}


def parse(path: str | Path, max_actions: int | None = None) -> dict:
    """解析一个 funscript 文件，返回其内部轴 + 主轴动作。

    返回字段：
      axes       检测到的轴列表（按内容）
      actions    主轴（优先 L0）的动作，供画曲线
      count      主轴动作数
      duration_ms / speed  主轴统计
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"axes": [], "actions": [], "count": 0, "duration_ms": 0, "speed": 0.0}

    axes = detect_axes(data)
    main_axis = "L0" if "L0" in axes else (axes[0] if axes else None)
    acts = _extract_actions(data, main_axis)

    stats = _stats(acts)
    out_actions = acts
    if max_actions and len(acts) > max_actions:
        step = len(acts) / max_actions
        out_actions = [acts[int(i * step)] for i in range(max_actions)]

    return {
        "axes": axes,
        "actions": out_actions,
        "count": stats["count"],
        "duration_ms": stats["duration_ms"],
        "speed": stats["speed"],
    }


def parse_tracks(path: str | Path) -> dict[str, list[list[int]]]:
    """解析文件，返回 {axis: [actions]} 全部轴轨道。"""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    tracks: dict[str, list[list[int]]] = {}
    for ax in detect_axes(data):
        acts = _extract_actions(data, ax)
        if acts:
            tracks[ax] = acts
    return tracks


def scan(video_path: str | Path) -> dict:
    """扫描并返回概要：轴数按「文件内容」统计（不按文件数量）。"""
    files = find_for(video_path)
    all_axes: list[str] = []
    detail = []
    for axis_from_name, p in files:
        info = parse(p)
        inner = info["axes"] or (
            [axis_from_name] if axis_from_name and axis_from_name != "main" else ["L0"]
        )
        for ax in inner:
            if ax not in all_axes:
                all_axes.append(ax)
        detail.append({
            "axis": axis_from_name,
            "path": str(p),
            "count": info["count"],
            "duration_ms": info["duration_ms"],
            "speed": info["speed"],
            "axes": inner,
        })
    return {"count": len(all_axes), "axes": all_axes, "files": detail}


def level(count: int) -> str:
    """把轴数归到 none / single / multi。"""
    if count <= 0:
        return "none"
    if count == 1:
        return "single"
    return "multi"
