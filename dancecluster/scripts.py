"""脚本管理：源脚本备份、关联、激活（原子替换）。

安全原则：
- 源脚本 master 放在「视频目录/源脚本备份/<视频名>/」，只读，永不写删。
- 视频同级的 xxx.funscript 系列永远是「工作副本」，可随时覆盖/删除。
- 激活 = 从 master 复制到临时文件 + 原子替换，中途失败不会损坏现有文件。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import funscript

BACKUP_DIR = "源脚本备份"


def backup_dir_for(video_path: str | Path) -> Path:
    """源脚本备份目录：<视频目录>/源脚本备份/<视频名>/"""
    v = Path(video_path)
    return v.parent / BACKUP_DIR / v.stem


def _axis_suffix(axis: str) -> str:
    """备份文件名里的轴 -> 同级文件后缀。main -> 无后缀，r0 -> .r0。"""
    return "" if axis == "main" else axis


def source_files(video_path: str | Path) -> list[tuple[str, Path]]:
    """源脚本备份目录里已有的轴文件 [(axis, path)]。"""
    d = backup_dir_for(video_path)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.funscript")):
        out.append((p.stem, p))
    return out


def has_source(video_path: str | Path) -> bool:
    return bool(source_files(video_path))


def ensure_source_backup(video_path: str | Path) -> bool:
    """若视频有脚本且尚未备份，则复制进备份目录（唯一一次写 master）。

    返回 True 表示本次创建了备份；已存在则返回 False（master 不可覆盖）。
    """
    v = Path(video_path)
    if has_source(v):
        return False
    files = funscript.find_for(v)
    if not files:
        return False
    d = backup_dir_for(v)
    d.mkdir(parents=True, exist_ok=True)
    for axis, src in files:
        dst = d / f"{axis}.funscript"
        if not dst.exists():
            shutil.copy2(src, dst)
    return True


def import_source(video_path: str | Path, script_path: str | Path) -> bool:
    """把外部 funscript 复制进备份目录作为源（对无脚本视频）。"""
    v = Path(video_path)
    src = Path(script_path)
    if not src.is_file():
        raise FileNotFoundError(f"脚本文件不存在：{src}")
    d = backup_dir_for(v)
    if has_source(v):
        raise FileExistsError("该视频已有源脚本，不能重复导入")
    d.mkdir(parents=True, exist_ok=True)
    # 单文件导入默认当作主轴
    shutil.copy2(src, d / "main.funscript")
    return True


def _sibling_targets(video_path: str | Path) -> list[tuple[str, Path]]:
    """返回该视频同级的所有脚本文件（匹配前缀），用于识别要清理的旧文件。"""
    v = Path(video_path)
    return funscript.find_for(v)


def activate(video_path: str | Path, source_video_path: str | Path) -> bool:
    """把 source 的源脚本复制到 video 同级（原子替换 + 清理旧轴文件）。

    source 的 master 只读；video 同级的旧脚本会被替换/清理（它们是工作副本）。
    """
    v = Path(video_path)
    src_dir = backup_dir_for(source_video_path)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"源脚本备份不存在：{src_dir}")

    stem = v.stem
    files = sorted(src_dir.glob("*.funscript"))
    if not files:
        raise FileNotFoundError(f"源脚本备份为空：{src_dir}")

    new_targets: set[Path] = set()
    temps: list[tuple[Path, Path]] = []
    try:
        for p in files:
            axis = p.stem
            suffix = _axis_suffix(axis)
            target = v.with_name(stem + (f".{suffix}" if suffix else "") + ".funscript")
            tmp = v.with_name(target.name + ".tmp")
            shutil.copy2(p, tmp)
            temps.append((tmp, target))
            new_targets.add(target)
        # 统一原子替换
        for tmp, target in temps:
            tmp.replace(target)
    except Exception:
        for tmp, _ in temps:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    # 清理本次未覆盖的旧轴文件（工作副本，可安全删）
    for _, old in _sibling_targets(v):
        if old not in new_targets:
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass
    return True
