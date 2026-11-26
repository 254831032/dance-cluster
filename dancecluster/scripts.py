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
RECYCLE_DIR = "待回收脚本"


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
    try:
        d.mkdir(parents=True, exist_ok=True)
        for axis, src in files:
            dst = d / f"{axis}.funscript"
            if not dst.exists():
                shutil.copy2(src, dst)
    except OSError:
        return False  # 源目录只读等情况：备份失败但不影响激活回退
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
        # 备份目录不存在（还没备份 / 源目录只读建不了），先尝试现场备份
        try:
            ensure_source_backup(source_video_path)
        except Exception:
            pass
    if src_dir.is_dir():
        src_axes = [(p.stem, p) for p in sorted(src_dir.glob("*.funscript"))]
    else:
        # 备份建不出来时，直接回退用源视频同级的脚本文件
        src_axes = funscript.find_for(source_video_path)
    if not src_axes:
        raise FileNotFoundError(f"源脚本不存在：{source_video_path}")

    stem = v.stem
    new_targets: set[Path] = set()
    temps: list[tuple[Path, Path]] = []
    try:
        for axis, p in src_axes:
            suffix = _axis_suffix(axis)
            target = v.with_name(stem + (f".{suffix}" if suffix else "") + ".funscript")
            tmp = v.with_name(target.name + ".tmp")
            shutil.copy2(p, tmp)
            temps.append((tmp, target))
            new_targets.add(target)
        # 统一原子替换
        for tmp, target in temps:
            tmp.replace(target)
    except PermissionError as e:
        for tmp, _ in temps:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        raise PermissionError(f"目标目录没有写入权限，无法激活脚本：{v.parent}") from e
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


def recycle_sibling_scripts(video_path: str | Path) -> list[Path]:
    """把视频同级的脚本移进 待回收脚本/<文件名>/ 下（非源视频删除时用）。

    返回实际移动到的目标路径列表。源目录只读时跳过并返回空。
    """
    v = Path(video_path)
    files = funscript.find_for(v)
    if not files:
        return []
    dest_dir = v.parent / RECYCLE_DIR / v.name
    moved: list[Path] = []
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    for _axis, src in files:
        dst = dest_dir / src.name
        if dst.exists():
            continue
        try:
            shutil.move(str(src), str(dst))
            moved.append(dst)
        except OSError:
            continue
    return moved
