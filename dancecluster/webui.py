"""本地 Web / 桌面界面：左侧菜单 + 三个页面 + 左下角设置。

页面：
  1. 原视频库 —— 添加/移除文件夹，扫描并分组，看视频清单
  2. 音频分组 —— 音乐组列表，组名可改（默认取组内一个视频的文件名）
  3. 动作分组 —— 同上
设置（左下角）：
  缓存目录（必填）。不设置时禁止其它操作。

后端是个 Flask 进程，桌面模式（pywebview）和浏览器模式共用同一套。
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import threading
import traceback
import webbrowser
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from . import config, ffmpeg, flow, funscript, pipeline, pose, scripts
from .config import AppConfig
from .db import Store

app = Flask(__name__)

_lock = threading.Lock()
_ui = {"desktop": False, "window": None}
_app_state: dict = {"cache_dir": ""}
_state: dict = {
    "running": False,
    "progress": 0,
    "total": 0,
    "stage": "",
    "name": "",
    "log": [],
    "error": "",
    "cancel": False,
}


class _Cancelled(Exception):
    pass


# ---------- 基础设施 ----------
def _current_cache_dir() -> str:
    cd = _app_state.get("cache_dir")
    if cd:
        return cd
    return config.load_settings().get("cache_dir", "") or ""


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.cache_dir = _current_cache_dir()
    s = config.load_settings()
    m = s.get("model", "heavy")
    cfg.pose.model = m if m in config.MODEL_FILES else "heavy"
    return cfg


def _writable_default_cache() -> str:
    """依次尝试几个候选位置，返回第一个真正可写的目录。

    这样「使用默认目录」按钮永远给出能用的路径：
      正常环境 → ~/DanceCluster/cache
      开发沙箱 → 项目目录/cache 或系统临时目录
    """
    import tempfile

    candidates = [
        Path.home() / "DanceCluster" / "cache",
        config.PROJECT_ROOT / "cache",
        Path(tempfile.gettempdir()) / "DanceCluster" / "cache",
    ]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return str(c)
        except Exception:
            continue
    return str(Path.home() / "DanceCluster" / "cache")


def _store() -> Store:
    cd = _current_cache_dir()
    if not cd:
        raise RuntimeError("尚未设置缓存目录")
    Path(cd).mkdir(parents=True, exist_ok=True)
    return Store(pipeline.db_path(_cfg()))


_ensure_locks: dict = {}
_ensure_locks_guard = threading.Lock()


def _ensure_cache(row) -> bool:
    """预览前确保该视频的中间文件存在；缺失则从源视频按需重建。

    带 per-video 锁，避免并发预览触发重复提取。源视频缺失时返回 False。
    """
    vid = int(row["id"])
    with _ensure_locks_guard:
        lock = _ensure_locks.setdefault(vid, threading.Lock())
    with lock:
        cdir = Path(row["cache_dir"])
        needed = (cdir / "clip.mp4", cdir / "pose.npz", cdir / "audio.m4a")
        if all(p.exists() for p in needed):
            return True
        try:
            pipeline.ensure_video_cache(row, _cfg())
        except Exception:
            return False
    return True


# ---------- 缩略图：原子生成 + 后台预热 ----------
def _thumb_dir() -> Path:
    return Path(_current_cache_dir()) / "thumbs"


_THUMB_FILES = {
    "thumb": "{id}.jpg",
    "gif": "{id}.gif",
    "thumb_mp4": "{id}_thumb.mp4",
    "skel": "{id}_skel.png",
    "skel_gif": "{id}_skel.gif",
    "preview": "{id}_preview.mp4",
}


def _thumb_path(video_id: int, kind: str) -> Path:
    return _thumb_dir() / _THUMB_FILES[kind].format(id=video_id)


def _get_row(video_id: int):
    try:
        store = _store()
    except RuntimeError:
        return None
    try:
        return store.get_by_id(video_id)
    finally:
        store.close()


def _atomic_generate(generate_fn, out: Path) -> bool:
    """生成到临时文件，成功后原子替换；失败清理，保证下次可重试。

    临时名保留原扩展名（.jpg/.gif/.mp4），否则 ffmpeg 无法按扩展名判断格式。
    """
    out = Path(out)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    try:
        generate_fn(tmp)
        if not tmp.exists():
            return False
        os.replace(tmp, out)
        return True
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _generate_thumb(video_id: int, kind: str, row: dict, out: Path) -> None:
    cdir = Path(row["cache_dir"])
    aspect = (row["width"] / row["height"]) if (row["width"] and row["height"]) else 0.75
    if kind == "thumb":
        ffmpeg.extract_thumbnail(row["path"], out, at=1.0)
    elif kind == "gif":
        dur = max(2.0, min(float(row["duration"] or 8.0), 6.0))
        ffmpeg.extract_gif(row["path"], out, at=0.0, duration=dur, fps=3, width=160)
    elif kind == "thumb_mp4":
        cfg = _cfg()
        clip = cdir / "clip.mp4"
        src = str(clip) if clip.exists() else row["path"]
        dur = max(2.0, min(float(row["duration"] or 20.0), float(cfg.clip.duration or 20.0)))
        ffmpeg.extract_mp4_thumb(src, out, at=0.0, duration=dur, fps=6, width=220)
    elif kind == "skel":
        pose_npz = cdir / "pose.npz"
        if pose_npz.exists():
            pose.render_skeleton_frame(pose_npz, out, width=320, aspect=aspect)
    elif kind == "skel_gif":
        pose_npz = cdir / "pose.npz"
        if pose_npz.exists():
            clip_sec = min(float(row["duration"] or 8.0), 20.0)
            pose.render_skeleton_gif(pose_npz, out, fps=6,
                                     max_seconds=max(2, int(min(clip_sec, 12.0))),
                                     width=320, aspect=aspect)
    elif kind == "preview":
        pose_npz = cdir / "pose.npz"
        audio = cdir / "audio.m4a"
        if not audio.exists():
            audio = cdir / "audio.wav"
        if pose_npz.exists() and audio.exists():
            pose.render_preview_video(pose_npz, audio, out, fps=6, width=320, aspect=aspect)


_thumb_locks: dict = {}
_thumb_locks_guard = threading.Lock()


def _thumb_lock(key) -> threading.Lock:
    with _thumb_locks_guard:
        return _thumb_locks.setdefault(key, threading.Lock())


def _generate_thumb_sync(video_id: int, kind: str) -> Path | None:
    """按需同步生成缩略图（带 per-key 锁，避免并发重复生成）。

    成功返回路径；失败返回 None（不落盘残缺文件，下次请求自动重试）。
    """
    out = _thumb_path(video_id, kind)
    if out.exists():
        return out
    row = _get_row(video_id)
    if row is None:
        return None
    key = (video_id, kind)
    with _thumb_lock(key):
        if out.exists():
            return out
        if _atomic_generate(lambda tmp: _generate_thumb(video_id, kind, row, tmp), out):
            return out
        return None


def pre_generate_thumbs(ids) -> None:
    """提取完成后后台预热缩略图 / GIF / 骨架视频（不阻塞，复用同一套同步生成）。"""
    def _worker():
        for vid in ids:
            for kind in ("thumb", "thumb_mp4"):
                try:
                    _generate_thumb_sync(vid, kind)
                except Exception:
                    pass
    threading.Thread(target=_worker, daemon=True).start()


def _snapshot() -> dict:
    with _lock:
        return {
            "running": _state["running"],
            "progress": _state["progress"],
            "total": _state["total"],
            "stage": _state["stage"],
            "name": _state["name"],
            "log": list(_state["log"]),
            "error": _state["error"],
        }


def _log(msg: str) -> None:
    with _lock:
        _state["log"].append(msg)


# ---------- 任务 ----------
def _start_job(params: dict) -> str | None:
    """启动后台任务（扫描/提取/分组）。已有任务在跑时返回错误信息，成功返回 None。"""
    with _lock:
        if _state["running"]:
            return "有任务正在运行，请稍候或先取消"
        _state.update(running=True, progress=0, total=0, stage="准备中", name="",
                      log=[], error="", cancel=False)
    threading.Thread(target=_run_job, args=(params,), daemon=True).start()
    return None


def _run_job(params: dict) -> None:
    def progress(i: int, total: int, name: str, stage: str) -> None:
        if _state["cancel"]:
            raise _Cancelled()
        with _lock:
            _state.update(progress=i, total=total, name=name, stage=stage)

    def _counts(store):
        rows = store.all_videos()
        return {
            "total": len(rows),
            "done": sum(1 for r in rows if r["status"] in ("ok", "no_pose")),
            "pending": sum(1 for r in rows if r["status"] in ("pending", "extracting")),
            "error": sum(1 for r in rows if r["status"] == "error"),
        }

    try:
        cfg = _cfg()
        cfg.clip.start = float(params.get("start", 0.0))
        cfg.clip.duration = float(params.get("duration", 20.0))
        cfg.match.music_threshold = float(params.get("music", 0.55))
        cfg.match.dance_threshold = float(params.get("dance", 0.35))
        cfg.match.mirror_invariant = bool(params.get("mirror", True))

        store = _store()
        store.reset_stale_extracting()
        store.close()

        # 可选阶段 0：扫描新文件夹/视频（登记为待识别），大目录走后台 + 进度条
        scan = params.get("scan")
        if scan:
            folder = scan.get("folder")
            paths = scan.get("paths")
            with _lock:
                _state["stage"] = "scanning"
            if folder:
                _log(f"扫描文件夹：{folder}")
                pipeline.register_folder(folder, cfg, progress=progress)
            elif paths:
                _log("扫描视频文件…")
                pipeline.register_videos(paths, cfg, progress=progress)

        before = _counts(_store())
        _log(f"开始提取（待处理 {before['pending']} 个，已完成 {before['done']} 个将跳过）…")
        records = pipeline.process_pending(cfg, progress=progress, only_ids=params.get("ids"))
        if _state["cancel"]:
            raise _Cancelled()
        _log("提取完成，正在分组…")
        pipeline.group(records, cfg)
        pre_generate_thumbs([r.id for r in records])
        after = _counts(_store())
        _log(
            f"完成：共 {after['total']} 个视频，"
            f"已完成 {after['done']} 个（本次新增 {max(after['done'] - before['done'], 0)} 个，"
            f"跳过 {before['done']} 个已完成），失败 {after['error']} 个，待处理 {after['pending']} 个"
        )
        with _lock:
            _state["stage"] = "完成"
    except _Cancelled:
        _log("已取消")
        with _lock:
            _state["stage"] = "已取消"
    except Exception:
        tb = traceback.format_exc()
        _log(tb)
        with _lock:
            _state["stage"] = "出错"
            _state["error"] = tb
    finally:
        with _lock:
            _state["running"] = False


# ---------- 页面 ----------
@app.get("/vendor/<path:name>")
def vendor_file(name):
    base = (Path(__file__).parent / "vendor").resolve()
    p = (base / name).resolve()
    try:
        p.relative_to(base)
    except Exception:
        return ("", 404)
    if not p.is_file():
        return ("", 404)
    return send_file(p)


@app.get("/")
def index():
    return INDEX_HTML


# ---------- 设置 ----------
@app.get("/api/config")
def get_config():
    s = config.load_settings()
    cache_dir = _current_cache_dir() or s.get("cache_dir", "")
    return jsonify({
        "cache_dir": cache_dir,
        "configured": bool(cache_dir),
        "suggested": _writable_default_cache(),
        "model": s.get("model", "heavy"),
        "models": [{"value": k, "label": v} for k, v in config.MODEL_LABELS.items()],
    })


@app.post("/api/config/set")
def set_config():
    body = request.get_json(force=True, silent=True) or {}
    cache_dir = str(body.get("cache_dir", "")).strip()
    if not cache_dir:
        return jsonify({"ok": False, "error": "缓存目录不能为空"}), 400
    p = Path(cache_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
        if not p.is_dir():
            return jsonify({"ok": False, "error": "该路径不是目录"}), 400
        probe = p / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"目录不可写：{resolved if 'resolved' in dir() else cache_dir}（{exc}）。请换一个有写入权限的目录。"}), 400

    if config.is_protected_source_dir(p):
        return jsonify({"ok": False, "error": "该目录包含程序源码/脚本（或位于项目目录之上），不能设为缓存目录，否则「清除所有数据」会误删源码；请选择独立文件夹。"}), 400

    resolved = str(p.resolve())
    s = config.load_settings()
    s["cache_dir"] = resolved
    m = str(body.get("model", s.get("model", "heavy")))
    if m in config.MODEL_FILES:
        s["model"] = m
    config.save_settings(s)
    _app_state["cache_dir"] = resolved
    # 立刻初始化数据库，让后续操作有表可用
    try:
        Store(pipeline.db_path(_cfg())).close()
    except Exception:
        pass
    return jsonify({"ok": True, "cache_dir": resolved})


# ---------- 原视频库 ----------
@app.get("/api/library")
def library():
    try:
        store = _store()
    except RuntimeError:
        return jsonify({"folders": [], "videos": [], "categories": [], "configured": False})
    try:
        folders = store.list_folders()
        cat_map = store.video_categories()
        music_names = store.group_names("music")
        dance_names = store.group_names("dance")
        videos = [
            {
                "id": r["id"],
                "name": r["name"],
                "duration": round(float(r["duration"] or 0), 1),
                "width": int(r["width"] or 0),
                "height": int(r["height"] or 0),
                "status": r["status"],
                "tempo": round(float(r["tempo"] or 0), 1),
                "music_group": r["music_group"],
                "dance_group": r["dance_group"],
                "confidence": round(float(r["confidence"]), 2) if r["confidence"] is not None else None,
                "cats": cat_map.get(int(r["id"]), []),
                "music_name": music_names.get(r["music_group"], f"M{r['music_group']}") if r["music_group"] is not None else None,
                "dance_name": dance_names.get(r["dance_group"], f"D{r['dance_group']}") if r["dance_group"] is not None else None,
                "funscript_count": int(r["funscript_count"] or 0),
                "funscript_axes": json.loads(r["funscript_axes"]) if r["funscript_axes"] else [],
            }
            for r in store.all_videos()
        ]
        return jsonify({
            "folders": folders,
            "videos": videos,
            "categories": store.list_categories(),
            "configured": True,
        })
    finally:
        store.close()


@app.post("/api/library/add")
def library_add():
    body = request.get_json(force=True, silent=True) or {}
    path = str(body.get("path", "")).strip()
    if not path or not Path(path).is_dir():
        return jsonify({"ok": False, "error": "目录不存在"}), 400
    if not _current_cache_dir():
        return jsonify({"ok": False, "error": "请先设置缓存目录"}), 400
    stats = pipeline.register_folder(path, _cfg())
    return jsonify({"ok": True, **stats})


@app.post("/api/library/add_videos")
def library_add_videos():
    body = request.get_json(force=True, silent=True) or {}
    paths = [str(x) for x in body.get("paths", [])]
    if not paths:
        return jsonify({"ok": False, "error": "未选择视频"}), 400
    if not _current_cache_dir():
        return jsonify({"ok": False, "error": "请先设置缓存目录"}), 400
    stats = pipeline.register_videos(paths, _cfg())
    return jsonify({"ok": True, **stats})


@app.post("/api/library/remove")
def library_remove():
    body = request.get_json(force=True, silent=True) or {}
    path = str(body.get("path", ""))
    try:
        store = _store()
    except RuntimeError:
        return jsonify({"ok": False, "error": "请先设置缓存目录"}), 400
    try:
        store.remove_folder(path)
        return jsonify({"ok": True, "folders": store.list_folders()})
    finally:
        store.close()


@app.post("/api/run")
def run():
    body = request.get_json(force=True, silent=True) or {}
    ids = [int(x) for x in body.get("ids", [])]
    err = _start_job({"ids": ids} if ids else {})
    if err:
        return jsonify({"ok": False, "error": err}), 409
    return jsonify({"ok": True})


@app.post("/api/import")
def import_videos():
    body = request.get_json(force=True, silent=True) or {}
    folder = str(body.get("folder", "")).strip()
    paths = [str(x) for x in body.get("paths", [])]
    if not folder and not paths:
        return jsonify({"ok": False, "error": "请选择文件夹或视频"}), 400
    scan = {"folder": folder} if folder else {"paths": paths}
    err = _start_job({"scan": scan})
    if err:
        return jsonify({"ok": False, "error": err}), 409
    return jsonify({"ok": True})


# ---------- 视频详情 ----------
@app.get("/api/video_detail/<int:video_id>")
def video_detail(video_id):
    row = _get_row(video_id)
    if row is None:
        return jsonify({"ok": False, "error": "视频不存在"}), 404
    try:
        store = _store()
    except RuntimeError:
        return jsonify({"ok": False, "error": "请先设置缓存目录"}), 400
    try:
        music_names = store.group_names("music")
        dance_names = store.group_names("dance")
        cats = store.list_categories()
        my_cat_ids = set(store.video_categories().get(int(video_id), []))
        my_cats = [c["id"] for c in cats if c["id"] in my_cat_ids]
    finally:
        store.close()

    # funscript 详情
    fs_axes = []
    try:
        scan_info = funscript.scan(row["path"])
        for f in scan_info["files"]:
            for ax in f["axes"]:
                fs_axes.append({
                    "axis": ax,
                    "path": f["path"],
                    "count": f["count"],
                    "duration_ms": f["duration_ms"],
                    "speed": f["speed"],
                })
    except Exception:
        pass

    mg = row["music_group"]
    dg = row["dance_group"]
    return jsonify({
        "ok": True,
        "id": int(row["id"]),
        "name": row["name"],
        "path": row["path"],
        "duration": round(float(row["duration"] or 0), 2),
        "width": int(row["width"] or 0),
        "height": int(row["height"] or 0),
        "fps": round(float(row["fps"] or 0), 2),
        "status": row["status"],
        "tempo": round(float(row["tempo"] or 0), 1),
        "confidence": round(float(row["confidence"]), 3) if row["confidence"] is not None else None,
        "present_ratio": round(float(row["present_ratio"] or 0), 3),
        "music_name": music_names.get(mg, f"M{mg}") if mg is not None else None,
        "dance_name": dance_names.get(dg, f"D{dg}") if dg is not None else None,
        "categories": [c["name"] for c in cats if c["id"] in my_cats],
        "funscript_count": len(fs_axes),
        "funscript_axes": fs_axes,
    })


# ---------- funscript ----------
@app.post("/api/funscripts/scan")
def funscripts_scan():
    body = request.get_json(force=True, silent=True) or {}
    ids = [int(x) for x in body.get("ids", [])]
    if not ids:
        try:
            store = _store()
        except RuntimeError:
            return jsonify({"ok": False, "error": "请先设置缓存目录"}), 400
        try:
            ids = [int(r["id"]) for r in store.all_videos()]
        finally:
            store.close()
    stats = pipeline.scan_funscripts(ids, _cfg())
    return jsonify({"ok": True, **stats})


@app.get("/api/funscript/<int:video_id>")
def funscript_detail(video_id):
    row = _get_row(video_id)
    if row is None:
        return jsonify({"ok": False, "error": "视频不存在"}), 404
    try:
        files = funscript.find_for(row["path"])
    except Exception:
        files = []
    axes = []
    for axis, path in files:
        tracks = funscript.parse_tracks(path)
        for ax, acts in tracks.items():
            st = funscript._stats(acts)
            axes.append({
                "axis": ax,
                "path": str(path),
                "count": st["count"],
                "duration_ms": st["duration_ms"],
                "speed": st["speed"],
                "actions": acts,
            })
    return jsonify({"ok": True, "count": len(axes), "axes": axes})


# ---------- 脚本管理 ----------
@app.get("/api/scripts/<int:video_id>")
def scripts_status(video_id):
    row = _get_row(video_id)
    if row is None:
        return jsonify({"ok": False, "error": "视频不存在"}), 404
    store = _store()
    try:
        linked = store.linked_sources(video_id)
        src_ids = store.source_videos()
        pool = []
        for svid in src_ids:
            srow = store.get_by_id(svid)
            if not srow:
                continue
            pool.append({
                "id": int(srow["id"]),
                "name": srow["name"],
                "music_group": int(srow["music_group"]) if srow["music_group"] is not None else None,
                "dance_group": int(srow["dance_group"]) if srow["dance_group"] is not None else None,
                "funscript_count": int(srow["funscript_count"] or 0),
                "funscript_axes": json.loads(srow["funscript_axes"]) if srow["funscript_axes"] else [],
                "status": srow["status"],
            })
    finally:
        store.close()
    active = next((s["source_video_id"] for s in linked if s["active"]), None)
    linked_out = []
    for s in linked:
        srow = _get_row(s["source_video_id"])
        linked_out.append({
            "source_video_id": s["source_video_id"],
            "active": s["active"],
            "is_own": s["is_own"],
            "name": srow["name"] if srow else f"#{s['source_video_id']}",
        })
    return jsonify({
        "ok": True,
        "active_source": active,
        "own_source": scripts.has_source(row["path"]),
        "linked": linked_out,
        "pool": pool,
        "music_group": int(row["music_group"]) if row["music_group"] is not None else None,
        "dance_group": int(row["dance_group"]) if row["dance_group"] is not None else None,
    })


@app.post("/api/scripts/activate")
def scripts_activate():
    body = request.get_json(force=True, silent=True) or {}
    vid = int(body.get("video_id", 0) or 0)
    src = int(body.get("source_video_id", 0) or 0)
    if not vid or not src:
        return jsonify({"ok": False, "error": "参数缺失"}), 400
    row = _get_row(vid)
    srow = _get_row(src)
    if row is None or srow is None:
        return jsonify({"ok": False, "error": "视频不存在"}), 404
    try:
        scripts.activate(row["path"], srow["path"])
    except Exception as e:
        return jsonify({"ok": False, "error": f"激活失败：{e}"}), 500
    store = _store()
    try:
        store.set_active_script(vid, src)
        info = funscript.scan(Path(row["path"]))
        store.set_funscript(vid, info["count"], info["axes"])
    finally:
        store.close()
    return jsonify({"ok": True})


@app.post("/api/scripts/import_source")
def scripts_import_source():
    body = request.get_json(force=True, silent=True) or {}
    vid = int(body.get("video_id", 0) or 0)
    path = (body.get("path") or "").strip()
    if not vid or not path:
        return jsonify({"ok": False, "error": "参数缺失"}), 400
    row = _get_row(vid)
    if row is None:
        return jsonify({"ok": False, "error": "视频不存在"}), 404
    try:
        scripts.import_source(row["path"], path)
        scripts.activate(row["path"], row["path"])
    except Exception as e:
        return jsonify({"ok": False, "error": f"导入失败：{e}"}), 500
    store = _store()
    try:
        store.set_active_script(vid, vid)
        info = funscript.scan(Path(row["path"]))
        store.set_funscript(vid, info["count"], info["axes"])
    finally:
        store.close()
    return jsonify({"ok": True})


@app.post("/api/scripts/rescan")
def scripts_rescan():
    store = _store()
    try:
        rows = store.all_videos()
    finally:
        store.close()
    stats = {"scanned": 0, "backed_up": 0}
    for r in rows:
        try:
            info = funscript.scan(Path(r["path"]))
            made = scripts.ensure_source_backup(r["path"])
        except Exception:
            continue
        try:
            store = _store()
            try:
                store.set_funscript(int(r["id"]), info["count"], info["axes"])
                if info["count"] > 0 and not store.linked_sources(int(r["id"])):
                    store.set_active_script(int(r["id"]), int(r["id"]))
            finally:
                store.close()
        except Exception:
            continue
        stats["scanned"] += 1
        if made:
            stats["backed_up"] += 1
    return jsonify({"ok": True, **stats})


# ---------- 分组 ----------
@app.get("/api/groups")
def groups():
    gtype = request.args.get("type", "music")
    if gtype not in ("music", "dance"):
        return jsonify({"ok": False, "error": "type 必须是 music 或 dance"}), 400
    try:
        store = _store()
    except RuntimeError:
        return jsonify({"groups": [], "configured": False})
    try:
        return jsonify({"groups": store.groups_overview(gtype), "configured": True})
    finally:
        store.close()


@app.post("/api/groups/rename")
def groups_rename():
    body = request.get_json(force=True, silent=True) or {}
    gtype = str(body.get("type", "music"))
    key = int(body.get("key", -1))
    name = str(body.get("name", "")).strip()
    if gtype not in ("music", "dance") or key < 0 or not name:
        return jsonify({"ok": False, "error": "参数不完整"}), 400
    store = _store()
    try:
        store.rename_group(gtype, key, name)
        return jsonify({"ok": True})
    finally:
        store.close()


@app.get("/api/thumb_gif/<int:video_id>")
def thumb_gif(video_id):
    out = _generate_thumb_sync(video_id, "gif")
    if out is None:
        return ("", 404)
    return send_file(out, mimetype="image/gif")


@app.get("/api/pose/<int:video_id>")
def pose_data(video_id):
    row = _get_row(video_id)
    if row is None:
        return jsonify({"ok": False, "error": "视频不存在"}), 404
    if not _ensure_cache(row):
        return jsonify({"ok": False, "error": "缓存缺失"}), 404
    npz = Path(row["cache_dir"]) / "pose.npz"
    if not npz.exists():
        return jsonify({"ok": False, "error": "无骨架数据"}), 404
    import numpy as np

    with np.load(npz) as d:
        l2d = d["landmarks2d"]
        vis = d["visibility"] if "visibility" in d else np.zeros(l2d.shape[:2], np.float32)
        fps = float(d["fps"]) if d["fps"] is not None and d["fps"].size else 15.0
    F = int(l2d.shape[0])
    step = max(1, F // 900)
    l2d = l2d[::step]
    vis = vis[::step]
    frames = [[[round(float(x), 3), round(float(y), 3)] for x, y in fr] for fr in l2d]
    vis_out = [[round(float(v), 2) for v in fr] for fr in vis]
    return jsonify({
        "ok": True,
        "fps": (fps / step) if step > 1 else fps,
        "frames": len(frames),
        "landmarks": frames,
        "visibility": vis_out,
        "aspect": (row["width"] / row["height"]) if (row["width"] and row["height"]) else 0.75,
    })


@app.get("/api/pose3d/<int:video_id>")
def pose3d_data(video_id):
    row = _get_row(video_id)
    if row is None:
        return jsonify({"ok": False, "error": "视频不存在"}), 404
    if not _ensure_cache(row):
        return jsonify({"ok": False, "error": "缓存缺失"}), 404
    npz = Path(row["cache_dir"]) / "pose.npz"
    if not npz.exists():
        return jsonify({"ok": False, "error": "无骨架数据"}), 404
    import numpy as np

    with np.load(npz) as d:
        w = d["world"]
        vis = d["visibility"] if "visibility" in d else np.zeros(w.shape[:2], np.float32)
        fps = float(d["fps"]) if d["fps"] is not None and d["fps"].size else 15.0
    F = int(w.shape[0])
    step = max(1, F // 900)
    w = w[::step]
    vis = vis[::step]
    frames = [[[round(float(x), 3), round(float(y), 3), round(float(z), 3)] for x, y, z in fr] for fr in w]
    vis_out = [[round(float(v), 2) for v in fr] for fr in vis]
    return jsonify({
        "ok": True,
        "fps": (fps / step) if step > 1 else fps,
        "frames": len(frames),
        "world": frames,
        "visibility": vis_out,
    })


@app.get("/api/flow/<int:video_id>")
def flow_data(video_id):
    row = _get_row(video_id)
    if row is None:
        return jsonify({"ok": False, "error": "视频不存在"}), 404
    src = row["path"]
    if not src or not Path(src).exists():
        return jsonify({"ok": False, "error": "源视频缺失"}), 404
    cdir = Path(row["cache_dir"])
    npz = cdir / "flow.npz"
    if not npz.exists():
        try:
            cfg = _cfg()
            seq = flow.extract_flow(src, cfg.pose.model_file(),
                                    max_seconds=float(row["duration"] or 0) or None)
            seq.save(npz)
        except Exception as e:
            return jsonify({"ok": False, "error": f"光流提取失败：{e}"}), 500
    import numpy as np
    with np.load(npz) as d:
        pos = d["pos"]
        fps = float(d["fps"]) if d["fps"] is not None and d["fps"].size else 15.0
    F = int(pos.shape[0])
    step = max(1, F // 900)
    pos = pos[::step]
    return jsonify({
        "ok": True,
        "fps": (fps / step) if step > 1 else fps,
        "frames": int(pos.shape[0]),
        "pos": [[round(float(x), 2), round(float(y), 2)] for x, y in pos],
    })


@app.get("/api/thumb_mp4/<int:video_id>")
def thumb_mp4(video_id):
    out = _generate_thumb_sync(video_id, "thumb_mp4")
    if out is None:
        return ("", 404)
    return send_file(out, conditional=True, mimetype="video/mp4")


@app.get("/api/preview/<int:video_id>")
def preview_video(video_id):
    row = _get_row(video_id)
    if row is None:
        return ("", 404)
    if not _ensure_cache(row):
        return ("", 404)
    out = _thumb_path(video_id, "preview")
    if not out.exists():
        if not _atomic_generate(lambda tmp: _generate_thumb(video_id, "preview", row, tmp), out):
            return ("", 404)
    return send_file(out, conditional=True, mimetype="video/mp4")


@app.get("/api/skeleton_gif/<int:video_id>")
def skeleton_gif(video_id):
    row = _get_row(video_id)
    if row is None:
        return ("", 404)
    if not _ensure_cache(row):
        return ("", 404)
    out = _thumb_path(video_id, "skel_gif")
    if not out.exists():
        if not _atomic_generate(lambda tmp: _generate_thumb(video_id, "skel_gif", row, tmp), out):
            return ("", 404)
    return send_file(out, mimetype="image/gif")


@app.get("/api/skeleton/<int:video_id>")
def skeleton_frame(video_id):
    row = _get_row(video_id)
    if row is None:
        return ("", 404)
    if not _ensure_cache(row):
        return ("", 404)
    out = _thumb_path(video_id, "skel")
    if not out.exists():
        if not _atomic_generate(lambda tmp: _generate_thumb(video_id, "skel", row, tmp), out):
            return ("", 404)
    return send_file(out, mimetype="image/png")


# ---------- 预览流式 ----------
def _stream(path: Path):
    if not path.exists():
        return ("", 404)
    return send_file(path, conditional=True)


@app.get("/api/video/<int:video_id>")
def video_stream(video_id):
    store = _store()
    try:
        row = store.get_by_id(video_id)
    finally:
        store.close()
    return _stream(Path(row["path"])) if row else ("", 404)


@app.get("/api/clip/<int:video_id>")
def clip_stream(video_id):
    store = _store()
    try:
        row = store.get_by_id(video_id)
    finally:
        store.close()
    if not row:
        return ("", 404)
    if not _ensure_cache(row):
        return ("", 404)
    return _stream(Path(row["cache_dir"]) / "clip.mp4")


@app.get("/api/audio/<int:video_id>")
def audio_stream(video_id):
    store = _store()
    try:
        row = store.get_by_id(video_id)
    finally:
        store.close()
    if not row:
        return ("", 404)
    if not _ensure_cache(row):
        return ("", 404)
    cdir = Path(row["cache_dir"])
    m4a = cdir / "audio.m4a"
    if m4a.exists():
        return _stream(m4a)
    return _stream(cdir / "audio.wav")


# ---------- 缩略图 ----------
@app.get("/api/thumb/<int:video_id>")
def thumb(video_id):
    out = _generate_thumb_sync(video_id, "thumb")
    if out is None:
        return ("", 404)
    return send_file(out, mimetype="image/jpeg")


# ---------- 视频操作（删除 / 重新提取 / 归类） ----------
@app.post("/api/videos/delete")
def videos_delete():
    body = request.get_json(force=True, silent=True) or {}
    ids = [int(x) for x in body.get("ids", [])]
    if not ids:
        return jsonify({"ok": False, "error": "未选择视频"}), 400
    import shutil

    store = _store()
    try:
        cache_dirs = [r["cache_dir"] for r in (store.get_by_id(i) for i in ids) if r]
        store.delete_videos(ids)
    finally:
        store.close()
    # 同步清理磁盘：只删缓存目录内的子目录，越界路径一律跳过（绝不碰缓存外的源码）
    cache_root = Path(_current_cache_dir()).resolve()
    for cd in cache_dirs:
        if not cd:
            continue
        try:
            Path(cd).resolve().relative_to(cache_root)  # 不在缓存目录内会抛 ValueError
            shutil.rmtree(cd, ignore_errors=True)
        except Exception:
            pass
    base = Path(_current_cache_dir())
    for i in ids:
        for f in (base / "thumbs").glob(f"{i}_*") if (base / "thumbs").exists() else []:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
    return jsonify({"ok": True})


@app.post("/api/videos/reextract")
def videos_reextract():
    body = request.get_json(force=True, silent=True) or {}
    ids = [int(x) for x in body.get("ids", [])]
    if not ids:
        return jsonify({"ok": False, "error": "未选择视频"}), 400
    store = _store()
    try:
        for vid in ids:
            store.reset_video(vid)
        return jsonify({"ok": True, "count": len(ids)})
    finally:
        store.close()


@app.post("/api/videos/assign")
def videos_assign():
    body = request.get_json(force=True, silent=True) or {}
    ids = [int(x) for x in body.get("ids", [])]
    cat_id = int(body.get("category_id", 0))
    if not ids or not cat_id:
        return jsonify({"ok": False, "error": "参数不完整"}), 400
    store = _store()
    try:
        cat_map = store.video_categories()
        for vid in ids:
            cur = set(cat_map.get(vid, []))
            cur.add(cat_id)
            store.set_video_categories(vid, sorted(cur))
        return jsonify({"ok": True})
    finally:
        store.close()


@app.post("/api/videos/unassign")
def videos_unassign():
    body = request.get_json(force=True, silent=True) or {}
    ids = [int(x) for x in body.get("ids", [])]
    cat_id = int(body.get("category_id", 0))
    if not ids or not cat_id:
        return jsonify({"ok": False, "error": "参数不完整"}), 400
    store = _store()
    try:
        cat_map = store.video_categories()
        for vid in ids:
            cur = set(cat_map.get(vid, []))
            cur.discard(cat_id)
            store.set_video_categories(vid, sorted(cur))
        return jsonify({"ok": True})
    finally:
        store.close()


# ---------- 分类 ----------
@app.get("/api/categories")
def categories_list():
    try:
        store = _store()
    except RuntimeError:
        return jsonify({"categories": []})
    try:
        return jsonify({"categories": store.list_categories()})
    finally:
        store.close()


@app.post("/api/categories/create")
def categories_create():
    body = request.get_json(force=True, silent=True) or {}
    name = str(body.get("name", "")).strip()
    store = _store()
    try:
        cid = store.create_category(name)
        if cid is None:
            return jsonify({"ok": False, "error": "名称不能为空"}), 400
        return jsonify({"ok": True, "id": cid})
    finally:
        store.close()


@app.post("/api/categories/rename")
def categories_rename():
    body = request.get_json(force=True, silent=True) or {}
    cid = int(body.get("id", 0))
    name = str(body.get("name", "")).strip()
    if not cid or not name:
        return jsonify({"ok": False, "error": "参数不完整"}), 400
    store = _store()
    try:
        store.rename_category(cid, name)
        return jsonify({"ok": True})
    finally:
        store.close()


@app.post("/api/categories/delete")
def categories_delete():
    body = request.get_json(force=True, silent=True) or {}
    cid = int(body.get("id", 0))
    if not cid:
        return jsonify({"ok": False, "error": "参数不完整"}), 400
    store = _store()
    try:
        store.delete_category(cid)
        return jsonify({"ok": True})
    finally:
        store.close()


@app.post("/api/clear")
def clear_all():
    cd = _current_cache_dir()
    if not cd:
        return jsonify({"ok": False, "error": "尚未设置缓存目录"}), 400
    with _lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "有任务正在运行，请先取消"}), 409
    cfg = _cfg()
    try:
        pipeline.clear_all(cfg)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True})


# ---------- 通用 ----------
@app.post("/api/pick_folder")
def pick_folder():
    win = _ui.get("window")
    if not win:
        return jsonify({"ok": False, "error": "浏览器模式不支持系统文件夹选择，请手动填写路径"}), 400
    try:
        import webview

        kind = getattr(webview, "FileDialog", None)
        dialog = kind.FOLDER if kind is not None else webview.FOLDER_DIALOG
        result = win.create_file_dialog(dialog)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"打开选择框失败：{exc}"}), 500
    if not result:
        return jsonify({"ok": False, "error": "已取消"}), 400
    path = result[0] if isinstance(result, (list, tuple)) else result
    return jsonify({"ok": True, "path": str(path)})


@app.post("/api/pick_files")
def pick_files():
    win = _ui.get("window")
    if not win:
        return jsonify({"ok": False, "error": "浏览器模式不支持系统文件选择"}), 400
    try:
        import webview

        kind = getattr(webview, "FileDialog", None)
        dialog = kind.OPEN if kind is not None else webview.OPEN_DIALOG
        result = win.create_file_dialog(dialog, allow_multiple=True)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"打开选择框失败：{exc}"}), 500
    if not result:
        return jsonify({"ok": False, "error": "已取消"}), 400
    paths = list(result) if isinstance(result, (list, tuple)) else [result]
    return jsonify({"ok": True, "paths": [str(x) for x in paths]})


@app.get("/api/mode")
def mode():
    return jsonify({"desktop": bool(_ui["desktop"])})


@app.get("/api/status")
def status():
    return jsonify(_snapshot())


@app.post("/api/cancel")
def cancel():
    with _lock:
        _state["cancel"] = True
    return jsonify({"ok": True})


@app.post("/api/open")
def open_folder():
    body = request.get_json(force=True, silent=True) or {}
    path = body.get("path") or _current_cache_dir()
    if path and Path(path).exists():
        if sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("win"):
            import os as _os
            _os.startfile(path)  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["xdg-open", path])
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "目录不存在"}), 404


@app.post("/api/quit")
def quit_app():
    import os as _os
    threading.Timer(0.3, lambda: _os._exit(0)).start()
    return jsonify({"ok": True})


# ---------- 启动 ----------
def _free_port(start: int = 8765) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def main(open_browser: bool = True) -> int:
    if not config.acquire_instance_lock():
        print("DanceCluster 已在运行，请勿重复启动。", file=sys.stderr)
        return 0
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"DanceCluster 已启动：{url}", flush=True)
    if open_browser and os.environ.get("DANCECLUSTER_NO_BROWSER") != "1":
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>DanceCluster · 舞蹈视频自动分组</title>
<style>
  :root { --bg:#0e1116; --panel:#161a22; --line:#262c38; --text:#e8ebf1; --dim:#8a93a6;
          --accent:#5b8cff; --warn:#ffb454; --bad:#ff6b6b; --ok:#4bd08a; }
  * { box-sizing: border-box; }
  html,body { margin:0; height:100%; }
  body { background:var(--bg); color:var(--text); font:14px/1.55 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  .app { display:flex; height:100vh; }
  aside { width:220px; background:var(--panel); border-right:1px solid var(--line); display:flex; flex-direction:column; flex-shrink:0; }
  .brand { padding:20px 18px 14px; font-size:16px; font-weight:600; }
  .brand small { display:block; color:var(--dim); font-weight:400; font-size:12px; margin-top:3px; }
  nav { flex:1; padding:6px 10px; }
  .nav { display:block; width:100%; text-align:left; background:none; border:0; color:var(--dim); padding:11px 12px; border-radius:9px; cursor:pointer; font-size:14px; margin-bottom:4px; }
  .nav:hover { color:var(--text); background:#1c222d; }
  .nav.active { color:#fff; background:var(--accent); }
  .nav:disabled { opacity:.4; cursor:not-allowed; }
  .nav .n { margin-right:9px; opacity:.75; }
  .foot { padding:14px 10px; border-top:1px solid var(--line); }
  #btn-settings { width:100%; text-align:left; background:none; border:0; color:var(--dim); padding:11px 12px; border-radius:9px; cursor:pointer; font-size:14px; }
  #btn-settings:hover { color:var(--text); background:#1c222d; }
  main { flex:1; overflow:auto; padding:24px 26px; }
  .view { display:none; }
  .view.active { display:block; }
  h2 { font-size:18px; margin:0 0 4px; }
  .sub { color:var(--dim); margin:0 0 18px; font-size:13px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin-bottom:16px; }
  .row { display:flex; gap:9px; align-items:center; flex-wrap:wrap; }
  button { background:var(--accent); border:0; color:#fff; padding:9px 14px; border-radius:8px; cursor:pointer; font-size:13px; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.ghost { background:transparent; border:1px solid var(--line); color:var(--text); }
  button.small { padding:5px 10px; font-size:12px; }
  button.toggle { background:#1c222d; border:1px solid var(--line); color:var(--dim); }
  button.toggle.on { color:#fff; background:var(--accent); border-color:var(--accent); }
  input[type=text] { background:#0c0f15; border:1px solid var(--line); color:var(--text); border-radius:8px; padding:8px 10px; font-size:13px; }
  input[type=text]:focus { outline:1px solid var(--accent); }
  input[type=checkbox] { accent-color: var(--accent); cursor:pointer; }
  .chip { display:inline-flex; align-items:center; gap:7px; background:#1c222d; border:1px solid var(--line); border-radius:8px; padding:5px 10px; font-size:13px; margin:0 8px 8px 0; cursor:pointer; user-select:none; }
  .chip.on { border-color:var(--accent); color:#fff; }
  .chip input[type=checkbox] { pointer-events:none; }
  .bar { height:8px; background:#0c0f15; border-radius:99px; overflow:hidden; margin:12px 0 6px; }
  .bar > i { display:block; height:100%; width:0; background:linear-gradient(90deg,#5b8cff,#7aa2ff); transition:width .3s; }
  .status { color:var(--dim); font-size:13px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:500; }
  tbody tr { cursor:pointer; }
  tbody tr.sel td { background:#1b2740; }
  tbody tr.np td { color:var(--bad); }
  .grid { display:grid; gap:14px; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); }
  .grid.size-s { grid-template-columns:repeat(auto-fill,minmax(110px,1fr)); }
  .grid.size-m { grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); }
  .grid.size-l { grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); }
  .card { position:relative; background:#10141c; border:1px solid var(--line); border-radius:10px; overflow:hidden; cursor:pointer; }
  .card.sel { border-color:var(--accent); box-shadow:0 0 0 2px rgba(91,140,255,.4); }
  .card img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:#0a0d12; }
  .card .ck { position:absolute; top:6px; right:6px; z-index:2; width:16px; height:16px; }
  .card .meta { padding:8px 9px; }
  .card .name { font-size:12px; word-break:break-all; }
  .card .badges { margin-top:4px; display:flex; gap:4px; flex-wrap:wrap; }
  .badge { font-size:11px; border-radius:5px; padding:1px 6px; background:#1c222d; color:var(--dim); }
  .badge.np { color:var(--bad); }
  .badge.fs-none { color:var(--dim); }
  .badge.fs-single { color:#7aa2ff; border:1px solid #2c4166; }
  .badge.fs-multi { color:var(--ok); border:1px solid #2a4a3a; }
  #pv-fs-canvas { width:100%; height:120px; background:#0c0f15; border-radius:8px; display:block; }
  .badge.done { color:var(--ok); }
  .badge.extracting { color:var(--warn); }
  .thumb-sm { width:64px; height:36px; object-fit:cover; border-radius:4px; vertical-align:middle; background:#0a0d12; }
  .group { border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin-bottom:12px; background:var(--panel); }
  .group-head { display:flex; align-items:center; gap:10px; }
  .group-head .tag { background:var(--accent); color:#fff; border-radius:6px; padding:2px 8px; font-size:12px; }
  .group-head input[type=text] { flex:0 1 320px; }
  .members { flex:1; min-width:0; display:flex; flex-wrap:wrap; gap:12px; align-content:flex-start; }
  .member { width:120px; }
  .member img { width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:6px; background:#0a0d12; }
  .member .mname { font-size:11px; color:var(--dim); word-break:break-all; }
  .member .np { color:var(--bad); }
  .group-body { display:flex; gap:16px; align-items:flex-start; margin-top:12px; }
  .group-preview { position:relative; flex:0 0 208px; width:208px; height:144px; border-radius:10px; overflow:hidden; cursor:pointer; background:#0a0d12; border:1px solid var(--line); }
  .group-preview img { width:100%; height:100%; object-fit:cover; display:block; }
  .group-preview:hover { border-color:var(--accent); }
  .group-preview .pv-tag { position:absolute; left:6px; top:6px; font-size:11px; color:#dfe6f5; background:rgba(0,0,0,.55); border-radius:5px; padding:1px 6px; }
  .group-preview .play { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:13px; color:#fff; background:rgba(0,0,0,.38); transition:background .15s; }
  .group-preview:hover .play { background:rgba(0,0,0,.6); }
  .empty { color:var(--dim); padding:30px 0; text-align:center; }
  .settings-box { max-width:560px; margin:30px auto 0; }
  .path-box { flex:1; background:#0c0f15; border:1px solid var(--line); border-radius:8px; padding:8px 10px; font-size:13px; color:var(--dim); word-break:break-all; min-height:38px; }
  .path-box.set { color:var(--text); border-color:#33405a; }
  .modal { position:fixed; inset:0; background:rgba(0,0,0,.62); display:flex; align-items:center; justify-content:center; z-index:100; }
  .modal-box { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; width:min(760px, 94vw); max-height:90vh; overflow:auto; }
  .modal-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
  .pv { margin-bottom:16px; }
  .pv-title { color:var(--dim); font-size:12px; margin-bottom:5px; }
  .pv video, .pv audio { width:100%; max-height:320px; background:#000; border-radius:8px; }
  pre.log { background:#0c0f15; border:1px solid var(--line); border-radius:8px; padding:10px; height:170px; overflow:auto; font-size:12px; color:#aab3c2; white-space:pre-wrap; }
  .muted { color:var(--dim); font-size:12px; }
  .actions { padding:8px 0; }
  .msel { position:relative; }
  .msel-panel { position:absolute; top:100%; left:0; margin-top:4px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:6px; display:none; z-index:60; max-height:260px; overflow:auto; min-width:160px; box-shadow:0 8px 24px rgba(0,0,0,.4); }
  .msel-panel.open { display:block; }
  .msel-panel label { display:flex; gap:7px; align-items:center; padding:4px 8px; cursor:pointer; font-size:13px; border-radius:6px; }
  .msel-panel label:hover { background:#1c222d; }
  .ss-item { display:flex; align-items:center; gap:10px; padding:8px; border:1px solid var(--line); border-radius:10px; cursor:pointer; }
  .ss-item:hover { border-color: var(--ok); }
  .ss-item img { width:96px; height:64px; object-fit:cover; border-radius:6px; background:#0a0d12; }
  .ss-item .ss-info { flex:1; min-width:0; }
  .ss-item .ss-name { font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .cmp-axis { display:flex; align-items:center; gap:8px; margin:4px 0; }
  .cmp-axis .lbl { width:28px; font-size:12px; color:#8a93a6; }
  .cmp-axis .bar { flex:1; height:14px; background:#141920; border-radius:7px; overflow:hidden; }
  .cmp-axis .fill { height:100%; width:50%; border-radius:7px; transition:width .1s; }

  #detail-modal { z-index: 110; }
  #preview-modal { z-index: 120; }
  #script-select-modal { z-index: 125; }
  #compare-modal { z-index: 126; }
  .detail-box { width:min(880px, 94vw); }
  .detail-body { display:flex; gap:16px; align-items:flex-start; }
  .detail-thumb { width:300px; max-height:220px; object-fit:contain; border-radius:10px; background:#000; flex-shrink:0; }
  .detail-info { flex:1; font-size:13px; min-width:0; }
  .di-row { display:flex; gap:10px; padding:5px 0; border-bottom:1px dashed #1e2430; }
  .di-row span { color:var(--dim); width:62px; flex-shrink:0; }
  .di-row b { font-weight:500; word-break:break-all; }
  .card .name { cursor:pointer; }
  .card .name:hover { color:#8fb0ff; text-decoration:underline; }
  #play-modal { z-index: 130; }
  .thumb-wrap { position:relative; }
  .thumb-wrap .thumb-media { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:#0a0d12; border-radius:6px; }
  .hover-btns { position:absolute; left:0; right:0; bottom:0; display:flex; gap:4px; padding:6px; background:linear-gradient(transparent, rgba(0,0,0,.75)); opacity:0; transition:opacity .15s; }
  .thumb-wrap:hover .hover-btns, .group-preview:hover .hover-btns { opacity:1; }
  .hover-btns button { padding:3px 9px; font-size:11px; border-radius:6px; background:rgba(30,36,48,.92); color:#fff; border:1px solid rgba(255,255,255,.16); cursor:pointer; line-height:1.4; }
  .hover-btns button:hover { background:var(--accent); border-color:var(--accent); }
  .skel-stage { position:relative; background:#000; border-radius:8px; overflow:hidden; margin:0 auto; }
  .test-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:900px) { .test-grid { grid-template-columns:1fr; } }
  .skel3d-stage { position:relative; height:420px; background:#0a0d12; border-radius:8px; overflow:hidden; cursor:grab; }
  .skel3d-stage:active { cursor:grabbing; }
  .skel3d-stage canvas { width:100%; height:100%; display:block; }
  .skel-ctrl { display:flex; align-items:center; gap:8px; padding:6px 2px; }
  .skel-stage video { width:100%; height:100%; object-fit:contain; display:block; max-height:none; }
  .skel-stage canvas { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:1; }
  .list-name { cursor:pointer; }
  .list-name:hover { color:#8fb0ff; text-decoration:underline; }
</style>
</head>
<body>
<div class="app">
  <aside>
    <div class="brand">DanceCluster<small>舞蹈视频自动分组</small></div>
    <nav>
      <button class="nav active" data-view="library"><span class="n">▤</span>原文件</button>
      <button class="nav" data-view="music"><span class="n">♪</span>音频分组</button>
      <button class="nav" data-view="dance"><span class="n">✣</span>动作分组</button>
      <button class="nav" data-view="test"><span class="n">🧪</span>测试</button>
    </nav>
    <div class="foot">
      <div style="padding:0 10px 8px"><button id="anim-btn" class="ghost small" style="width:100%">缩略图：动图</button></div>
      <button id="btn-settings">⚙ 设置</button>
    </div>
  </aside>

  <main>
    <!-- 设置 -->
    <div class="view" id="view-settings">
      <div class="settings-box panel">
        <h2>设置</h2>
        <p class="sub">缓存目录用于存放数据库和中间文件，<b>必须设置</b>。</p>
        <div class="row">
          <div id="cache-dir" class="path-box">未选择缓存目录</div>
          <button id="btn-pick-cache" class="ghost">选择…</button>
          <button id="btn-default-cache" class="ghost">使用默认目录</button>
          <button id="btn-save-cache">保存并启用</button>
        </div>
        <p class="muted" id="cache-hint"></p>
        <div class="row" style="margin-top:14px">
          <label for="pose-model" style="margin-right:10px; white-space:nowrap">MediaPipe 模型</label>
          <select id="pose-model" style="flex:1"></select>
        </div>
        <p class="muted">模型决定骨架精度与速度：Heavy 最准但慢、Full 均衡、Lite 最快但精度低。切换只影响之后新提取的视频，已完成的不会重新提取。</p>
        <div class="row" style="margin-top:18px; padding-top:14px; border-top:1px solid var(--line)">
          <button id="btn-clear" class="ghost" style="color:var(--bad)">清除全部数据（重新开始）</button>
        </div>
      </div>
    </div>

    <!-- 原文件 -->
    <div class="view active" id="view-library">
      <h2>原文件</h2>
      <p class="sub">导入文件夹或视频后自动识别，按背景音乐与舞蹈动作分组。</p>
      <div class="panel">
        <div class="row">
          <button id="btn-import-folder">导入文件夹</button>
          <button id="btn-import-videos" class="ghost">导入视频</button>
          <button id="btn-run" class="ghost">开始识别</button>
          <button id="btn-cancel" class="ghost" disabled>取消</button>
          <span style="flex:1"></span>
          <button id="btn-new-cat" class="ghost">＋ 新建分类</button>
        </div>
        <div class="bar"><i id="fill"></i></div>
        <div class="status" id="status">就绪</div>
      </div>
      <div class="panel">
        <div class="row">
          <button id="btn-view-list" class="toggle on">☰ 列表</button>
          <button id="btn-view-grid" class="toggle">▦ 图标</button>
          <span id="size-group" style="display:none">
            <button id="size-s" class="toggle small">小</button>
            <button id="size-m" class="toggle small on">中</button>
            <button id="size-l" class="toggle small">大</button>
          </span>
        </div>
        <div class="row" style="margin-top:10px">
          <input type="text" id="search" placeholder="搜索文件名…" style="flex:1;max-width:220px">
          <div class="msel"><button id="msel-music-btn" class="ghost small" data-label="音乐分组">音乐分组 ▾</button><div id="msel-music" class="msel-panel"></div></div>
          <div class="msel"><button id="msel-dance-btn" class="ghost small" data-label="动作分组">动作分组 ▾</button><div id="msel-dance" class="msel-panel"></div></div>
          <select id="status-filter" style="background:#1c222d;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 8px">
            <option value="all">全部状态</option>
            <option value="pending">待识别</option>
            <option value="extracting">识别中</option>
            <option value="done">识别完成</option>
            <option value="no_pose">无骨架</option>
            <option value="error">出错</option>
          </select>
          <select id="fs-filter" style="background:#1c222d;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 8px">
            <option value="all">全部脚本状态</option>
            <option value="none">无脚本</option>
            <option value="single">单轴</option>
            <option value="multi">多轴</option>
          </select>
          <button id="btn-reset-filter" class="ghost small">重置</button>
        </div>
        <div id="cat-bar" style="margin-top:8px"></div>
      </div>
      <div class="panel">
        <div class="row actions">
          <label class="muted" style="cursor:pointer"><input type="checkbox" id="check-all"> 全选</label>
          <span class="muted" id="sel-info">未选择</span>
          <span style="flex:1"></span>
          <select id="assign-cat" style="background:#1c222d;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 8px"></select>
          <button id="btn-assign" class="ghost small">归入分类</button>
          <button id="btn-preview" class="ghost small">预览</button>
          <button id="btn-scan-fs" class="ghost small">扫描脚本</button>
          <button id="btn-reextract" class="ghost small">重新提取</button>
          <button id="btn-delete" class="ghost small" style="color:var(--bad)">删除</button>
        </div>
        <div id="video-list"></div>
        <div class="empty" id="video-empty">还没有视频，点上方「导入文件夹」或「导入视频」开始。</div>
      </div>
    </div>

    <!-- 音频分组 -->
    <div class="view" id="view-music">
      <h2>音频分组</h2>
      <p class="sub">按背景音乐分组。组名默认取组内一个视频的文件名，可点击改名。</p>
      <div id="music-groups"></div>
    </div>

    <!-- 动作分组 -->
    <div class="view" id="view-dance">
      <h2>动作分组</h2>
      <p class="sub">在每组音乐内按舞蹈动作再分。组名可点击改名。</p>
      <div id="dance-groups"></div>
    </div>

    <!-- 测试 -->
        <div class="view" id="view-test">
      <h2>测试</h2>
      <p class="sub">视频骨架 + OSR6 运动同步对比，下方为脚本信号。</p>
      <div class="panel">
        <div class="row">
          <select id="test-video" style="background:#1c222d;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 8px;max-width:320px"></select>
          <button id="test-load" class="ghost small">加载</button>
          <button id="test-play" class="ghost small">▶ 播放</button>
          <input type="range" id="test-seek" min="0" max="100" value="0" style="flex:1;min-width:120px">
          <span id="test-time" class="muted" style="width:48px">0.0s</span>
          <label class="muted" style="white-space:nowrap"><input type="checkbox" id="test-3d-auto" checked> 3D自动旋转</label>
        </div>
      </div>
      <div class="test-grid">
        <div class="panel">
          <b>视频骨架（2D 叠加）</b>
          <div class="skel-stage" id="t2d-stage"><video id="t2d-video" playsinline preload="auto"></video><canvas id="t2d-canvas"></canvas></div>
        </div>
        <div class="panel">
          <b>OSR6 运动（真实模型）</b>
          <div id="osr6-emu" style="width:100%;height:340px;background:#0a0d12;border-radius:8px;overflow:hidden;margin-top:8px"></div>
          <p class="muted" id="osr6-emu-hint" style="margin-top:6px">真实 3D 模型（MIT 许可 · ayvajs）· 拖动旋转</p>
        </div>
      </div>
      <div class="panel">
        <b>手动控制</b>
        <label class="muted" style="display:inline-block;margin-bottom:6px"><input type="checkbox" id="manual-mode"> 手动模式（忽略信号，直接控制杯子）</label>
        <div id="manual-sliders" style="display:none">
          <div class="row" style="gap:8px;flex-wrap:wrap">
            <label class="muted">L0 <input type="range" id="m-l0" min="0" max="100" value="50" style="width:110px;vertical-align:middle"> <span id="m-l0-v">50</span></label>
            <label class="muted">L1 <input type="range" id="m-l1" min="0" max="100" value="50" style="width:110px;vertical-align:middle"> <span id="m-l1-v">50</span></label>
            <label class="muted">L2 <input type="range" id="m-l2" min="0" max="100" value="50" style="width:110px;vertical-align:middle"> <span id="m-l2-v">50</span></label>
            <label class="muted">R0 <input type="range" id="m-r0" min="0" max="100" value="50" style="width:110px;vertical-align:middle"> <span id="m-r0-v">50</span></label>
            <label class="muted">R1 <input type="range" id="m-r1" min="0" max="100" value="50" style="width:110px;vertical-align:middle"> <span id="m-r1-v">50</span></label>
            <label class="muted">R2 <input type="range" id="m-r2" min="0" max="100" value="50" style="width:110px;vertical-align:middle"> <span id="m-r2-v">50</span></label>
          </div>
        </div>
      </div>
      <div class="panel">
        <b>脚本信息</b>
        <p class="muted">六轴映射（以髋骨为准）：L0=上下 · L1=左右 · L2=前后 · R0=骨盆扭转 · R1=骨盆俯仰 · R2=骨盆侧倾（已做断点插值 + 平滑）</p>
        <div class="row" style="margin:6px 0 10px">
          <label class="muted">主行程幅度 <input type="range" id="sig-gain" min="30" max="200" value="100" style="width:110px;vertical-align:middle"></label>
        </div>
        <div class="row" style="margin:4px 0 8px;flex-wrap:wrap">
          <span class="muted">平移信号源：</span>
          <label class="muted"><input type="radio" name="src-mode" id="src-skel" checked> 骨架</label>
          <label class="muted"><input type="radio" name="src-mode" id="src-flow"> 光流(AI分割+Farneback)</label>
        </div>
        <div class="row" style="margin:4px 0 8px;flex-wrap:wrap">
          <span class="muted">启用：</span>
          <label class="muted">L0<input type="checkbox" id="en-l0" checked></label>
          <label class="muted">L1<input type="checkbox" id="en-l1" checked></label>
          <label class="muted">L2<input type="checkbox" id="en-l2" checked></label>
          <label class="muted">R0<input type="checkbox" id="en-r0" checked></label>
          <label class="muted">R1<input type="checkbox" id="en-r1" checked></label>
          <label class="muted">R2<input type="checkbox" id="en-r2" checked></label>
        </div>
        <div class="row" style="margin:4px 0 8px;flex-wrap:wrap">
          <span class="muted">反向：</span>
          <label class="muted">L0<input type="checkbox" id="inv-l0" checked></label>
          <label class="muted">L1<input type="checkbox" id="inv-l1"></label>
          <label class="muted">L2<input type="checkbox" id="inv-l2"></label>
          <label class="muted">R0<input type="checkbox" id="inv-r0"></label>
          <label class="muted">R1<input type="checkbox" id="inv-r1"></label>
          <label class="muted">R2<input type="checkbox" id="inv-r2"></label>
        </div>
        <canvas id="test-curve" height="150" style="width:100%;background:#0a0d12;border-radius:8px;display:block"></canvas>
      </div>
      <div class="panel">
        <b>3D 骨架（调试，可拖动旋转）</b>
        <div class="skel3d-stage"><canvas id="t3d-canvas"></canvas></div>
      </div>
    </div>

    </div>
  </main>
</div>

<div class="modal" id="preview-modal" style="display:none">
  <div class="modal-box">
    <div class="modal-head"><b id="preview-name"></b><button class="ghost small" id="preview-close">✕ 关闭</button></div>
    <div class="modal-body">
      <div class="pv pv-skel2d">
        <div class="pv-title">骨架动作预览（叠加在片段上，带声音）</div>
        <div class="skel-stage">
          <video id="skel-video" controls playsinline preload="auto"></video>
          <canvas id="skel-canvas"></canvas>
        </div>
      </div>
      <div class="pv pv-other"><div class="pv-title">原视频</div><video id="pv-video" controls playsinline></video></div>
      <div class="pv pv-other"><div class="pv-title">动作片段（前 20 秒）</div><video id="pv-clip" controls playsinline></video></div>
      <div class="pv pv-other"><div class="pv-title">提取的音频</div><audio id="pv-audio" controls></audio></div>
      <div class="pv pv-other" id="pv-fs-wrap" style="display:none">
        <div class="pv-title" id="pv-fs-title">Funscript 动作曲线</div>
        <canvas id="pv-fs-canvas" height="120"></canvas>
      </div>
    </div>
  </div>
</div>

<div class="modal" id="detail-modal" style="display:none">
  <div class="modal-box detail-box">
    <div class="modal-head"><b id="detail-name"></b><button class="ghost small" id="detail-close">✕ 关闭</button></div>
    <div class="detail-body">
      <video id="detail-video" class="detail-thumb" controls playsinline preload="metadata"></video>
      <div class="detail-info" id="detail-info"></div>
      <div class="panel" id="script-panel" style="margin-top:12px">
        <b>脚本管理</b>
        <div id="script-status" class="muted" style="margin-top:6px">加载中…</div>
        <div class="row" style="margin-top:8px;gap:8px">
          <button id="script-select-btn" class="ghost small">切换/关联脚本</button>
          <button id="script-import-btn" class="ghost small">导入源脚本</button>
        </div>
      </div>
    </div>
    <div class="row" style="margin-top:14px">
      <button id="detail-preview">预览</button>
      <button id="detail-reextract" class="ghost">重新提取</button>
      <button id="detail-delete" class="ghost" style="color:var(--bad)">删除</button>
    </div>
  </div>
</div>

<div class="modal" id="script-select-modal" style="display:none">
  <div class="modal-box" style="max-width:920px;max-height:90vh;display:flex;flex-direction:column">
    <div class="modal-head"><b>选择源脚本</b><button class="ghost small" id="script-select-close">✕ 关闭</button></div>
    <div class="row" style="gap:8px;flex-wrap:wrap;margin:10px 0">
      <input id="ss-name" placeholder="文件名搜索" style="flex:1;min-width:140px;background:#1c222d;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 8px">
      <select id="ss-music" style="background:#1c222d;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 8px"><option value="">全部音乐组</option></select>
      <select id="ss-dance" style="background:#1c222d;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 8px"><option value="">全部动作组</option></select>
    </div>
    <div id="ss-list" style="overflow:auto;flex:1;display:flex;flex-direction:column;gap:8px"></div>
  </div>
</div>

<div class="modal" id="compare-modal" style="display:none">
  <div class="modal-box" style="max-width:1120px;max-height:92vh;overflow:auto">
    <div class="modal-head"><b id="compare-name"></b><button class="ghost small" id="compare-close">✕ 关闭</button></div>
    <div class="row" style="gap:10px;align-items:stretch">
      <div class="panel" style="flex:1;min-width:0"><b>当前视频</b><video id="cmp-v1" controls playsinline style="width:100%;max-height:300px;background:#000"></video></div>
      <div class="panel" style="flex:1;min-width:0"><b>源视频</b><video id="cmp-v2" controls playsinline style="width:100%;max-height:300px;background:#000"></video></div>
    </div>
    <div class="panel" style="margin-top:10px"><b>OSR6 虚拟播放</b><div id="cmp-osr6" style="height:320px;background:#0a0d12;border-radius:8px;overflow:hidden;margin-top:6px"></div></div>
    <div class="panel" style="margin-top:10px"><div id="cmp-axes"></div></div>
    <div class="row" style="margin-top:12px;justify-content:flex-end;gap:8px">
      <button id="cmp-activate">激活此脚本</button>
    </div>
  </div>
</div>

<div class="modal" id="play-modal" style="display:none">
  <div class="modal-box">
    <div class="modal-head"><b id="play-name"></b><button class="ghost small" id="play-close">✕ 关闭</button></div>
    <video id="play-video" controls playsinline style="width:100%;max-height:72vh;background:#000;border-radius:8px"></video>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const st = { configured:false, view:"list", thumbSize:"m", anim:true, filterCat:null, timer:null, videos:[], cats:[], sel:new Set(), selFolders:new Set(), search:"", fMusic:new Set(), fDance:new Set(), fStatus:"all", fsLevel:"all", fsAxes:[], fsMaxAt:1, cacheDir:"", detailId:null, running:false };

async function api(path, opts) { const r = await fetch(path, opts); return r.json(); }
async function post(path, body) { return api(path, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})}); }

function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
const _thumbObs = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const v = e.target;
    if (e.isIntersecting) {
      if (!v.getAttribute("src")) v.src = v.dataset.src;
      v.play().catch(() => {});
    } else {
      v.pause();
    }
  }
}, { rootMargin: "300px" });
function observeThumb(v) { if (v) _thumbObs.observe(v); }
function statusBadge(s) {
  const m = {ok:["done","识别完成"], no_pose:["np","无骨架"], extracting:["extracting","识别中"], pending:["","待识别"], error:["np","出错"]}[s]||["",""];
  return `<span class="badge ${m[0]}">${m[1]||s}</span>`;
}
function confirm2(first, second) {
  if (!confirm(first)) return false;
  return confirm(second);
}
function funscriptBadge(v) {
  const n = v.funscript_count || 0;
  if (n <= 0) return `<span class="badge fs-none">无脚本</span>`;
  if (n === 1) return `<span class="badge fs-single">单轴</span>`;
  return `<span class="badge fs-multi">多轴×${n}</span>`;
}

// ---- 导航 ----
document.querySelectorAll(".nav").forEach(b => b.addEventListener("click", () => switchView(b.dataset.view)));
document.getElementById("btn-settings").addEventListener("click", () => switchView("settings"));
document.getElementById("anim-btn").addEventListener("click", () => {
  st.anim = !st.anim;
  $("#anim-btn").textContent = "缩略图：" + (st.anim ? "动图" : "静态");
  if (document.getElementById("view-library").classList.contains("active")) renderVideos();
  if (document.getElementById("view-music").classList.contains("active")) loadGroups("music");
  if (document.getElementById("view-dance").classList.contains("active")) loadGroups("dance");
});
async function switchView(name) {
  document.querySelectorAll(".nav").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.getElementById("view-"+name).classList.add("active");
  if (name==="music") loadGroups("music");
  if (name==="dance") loadGroups("dance");
  if (name==="library") refreshLibrary();
  if (name==="settings") refreshConfig();
  if (name==="test") {
    if (!st.videos.length) await refreshLibrary();
    testPopulate();
    if (!TEST.id && st.videos.length) testLoad();
    initOsr6Emu();
  }
}

// ---- 设置 ----
document.getElementById("btn-save-cache").addEventListener("click", async () => {
  if (!st.cacheDir) { $("#cache-hint").textContent = "请先选择缓存目录"; $("#cache-hint").style.color="var(--bad)"; return; }
  const r = await post("/api/config/set", {cache_dir: st.cacheDir, model: $("#pose-model").value});
  if (!r.ok) { $("#cache-hint").textContent = "✗ " + (r.error||"失败"); $("#cache-hint").style.color="var(--bad)"; return; }
  $("#cache-hint").textContent = "✓ 已启用：" + r.cache_dir; $("#cache-hint").style.color="var(--ok)";
  await refreshConfig(); switchView("library");
});
document.getElementById("btn-pick-cache").addEventListener("click", async () => {
  const r = await post("/api/pick_folder", {});
  if (r.ok) { st.cacheDir = r.path; $("#cache-dir").textContent = r.path; $("#cache-dir").classList.add("set"); }
  else if (r.error !== "已取消") alert(r.error);
});
document.getElementById("btn-default-cache").addEventListener("click", async () => {
  const c = await api("/api/config");
  if (c.suggested) { st.cacheDir = c.suggested; $("#cache-dir").textContent = c.suggested; $("#cache-dir").classList.add("set"); }
});
document.getElementById("btn-clear").addEventListener("click", async () => {
  if (!confirm("确定清除所有数据吗？将删除数据库、缓存、缩略图（不影响源视频和缓存目录设置）。")) return;
  const r = await post("/api/clear", {});
  if (r.ok) { alert("已清除"); st.sel.clear(); st.selFolders.clear(); switchView("library"); refreshLibrary(); }
  else alert(r.error || "清除失败");
});
async function refreshConfig() {
  const c = await api("/api/config");
  st.configured = c.configured;
  st.cacheDir = c.cache_dir || "";
  $("#cache-dir").textContent = st.cacheDir || "未选择缓存目录";
  $("#cache-dir").classList.toggle("set", !!st.cacheDir);
  const sel = $("#pose-model");
  sel.innerHTML = "";
  (c.models||[]).forEach(m => {
    const o = document.createElement("option");
    o.value = m.value; o.textContent = m.label;
    if (m.value === c.model) o.selected = true;
    sel.appendChild(o);
  });
  if (!c.configured && c.suggested) $("#cache-hint").textContent = "建议：" + c.suggested;
  applyGate();
}
function applyGate() {
  document.querySelectorAll(".nav").forEach(b => { if (b.dataset.view!=="settings") b.disabled = !st.configured; });
  applyRunUI(false);
  if (!st.configured) switchView("settings");
}

// ---- 导入 + 识别 ----
function applyRunUI(running) {
  const run = running || !st.configured;
  $("#btn-import-folder").disabled = run;
  $("#btn-import-videos").disabled = run;
  $("#btn-run").disabled = run;
  $("#btn-cancel").disabled = !running;
}
function beginPolling() {
  poll();
  if (st.timer) clearInterval(st.timer);
  st.timer = setInterval(poll, 700);
}
document.getElementById("btn-import-folder").addEventListener("click", async () => {
  const r = await post("/api/pick_folder", {});
  if (!r.ok) { if (r.error !== "已取消") alert(r.error); return; }
  const a = await post("/api/import", { folder: r.path });
  if (a.ok) { beginPolling(); }
  else alert(a.error || "启动失败");
});
document.getElementById("btn-import-videos").addEventListener("click", async () => {
  const r = await post("/api/pick_files", {});
  if (!r.ok) { if (r.error !== "已取消") alert(r.error); return; }
  const a = await post("/api/import", { paths: r.paths });
  if (a.ok) { beginPolling(); }
  else alert(a.error || "启动失败");
});
document.getElementById("btn-run").addEventListener("click", async () => {
  const r = await post("/api/run", {});
  if (!r.ok) { alert(r.error || "启动失败"); return; }
  beginPolling();
});
document.getElementById("btn-cancel").addEventListener("click", async () => {
  await post("/api/cancel");
  $("#btn-cancel").disabled = true;
});
async function poll() {
  const s = await api("/api/status");
  st.running = s.running;
  const pct = s.total ? Math.round(100 * s.progress / s.total) : 0;
  $("#fill").style.width = pct + "%";
  if (s.stage === "scanning") {
    $("#status").textContent = `扫描中 ${s.progress}${s.total ? "/" + s.total : ""} · ${s.name}`;
  } else if (s.stage === "extracting") {
    $("#status").textContent = `识别中 [${s.progress}/${s.total}] ${s.name}`;
  } else if (s.running) {
    $("#status").textContent = s.stage + (s.name ? " · " + s.name : "");
  } else {
    $("#status").textContent = s.stage || "就绪";
  }
  applyRunUI(s.running);
  if (!s.running) {
    if (st.timer) { clearInterval(st.timer); st.timer = null; }
    refreshLibrary();
  }
}

// ---- 原文件：视频浏览（多选）----
document.getElementById("btn-new-cat").addEventListener("click", async () => {
  const name = prompt("分类名称：");
  if (!name) return;
  const r = await post("/api/categories/create", {name});
  if (r.ok) refreshLibrary(); else alert(r.error);
});
document.getElementById("btn-view-list").addEventListener("click", () => { st.view="list"; syncViewBtns(); renderVideos(); });
document.getElementById("btn-view-grid").addEventListener("click", () => { st.view="grid"; syncViewBtns(); renderVideos(); });
document.getElementById("size-s").addEventListener("click", () => { st.thumbSize="s"; syncSizeBtns(); renderVideos(); });
document.getElementById("size-m").addEventListener("click", () => { st.thumbSize="m"; syncSizeBtns(); renderVideos(); });
document.getElementById("size-l").addEventListener("click", () => { st.thumbSize="l"; syncSizeBtns(); renderVideos(); });
function syncViewBtns() {
  $("#btn-view-list").classList.toggle("on", st.view==="list");
  $("#btn-view-grid").classList.toggle("on", st.view==="grid");
  $("#size-group").style.display = st.view==="grid" ? "" : "none";
  syncSizeBtns();
}
function syncSizeBtns() {
  ["s","m","l"].forEach(z => $("#size-"+z).classList.toggle("on", st.thumbSize===z));
}

document.getElementById("btn-reextract").addEventListener("click", async () => {
  if (!st.sel.size) { alert("请先勾选视频"); return; }
  if (st.running) { alert("识别进行中，请先取消再重新提取"); return; }
  const n = st.sel.size;
  if (!confirm2(`重新提取选中的 ${n} 个视频？（会清空它们的提取结果，需要重跑）`,
                "再次确认：确定重新提取？")) return;
  const ids = [...st.sel];
  const r = await post("/api/videos/reextract", {ids});
  if (!r.ok) { alert(r.error || "失败"); return; }
  st.sel.clear();
  const a = await post("/api/run", {ids});
  if (a.ok) { beginPolling(); } else alert(a.error || "启动失败");
});
document.getElementById("btn-delete").addEventListener("click", async () => {
  if (!st.sel.size) { alert("请先勾选视频"); return; }
  if (st.running) { alert("识别进行中，请先取消再删除"); return; }
  if (!confirm2(`从库里删除选中的 ${st.sel.size} 个视频？（不会删除源文件）`,
                "再次确认：确定删除？缓存和缩略图也会一并清理。")) return;
  const r = await post("/api/videos/delete", {ids:[...st.sel]});
  if (r.ok) { st.sel.clear(); refreshLibrary(); }
});
document.getElementById("btn-assign").addEventListener("click", async () => {
  if (!st.sel.size) { alert("请先勾选视频"); return; }
  const cat = parseInt($("#assign-cat").value);
  if (!cat) { alert("请先选择分类"); return; }
  const r = await post("/api/videos/assign", {ids:[...st.sel], category_id:cat});
  if (r.ok) { st.sel.clear(); refreshLibrary(); } else alert(r.error);
});
document.getElementById("btn-scan-fs").addEventListener("click", async () => {
  const ids = [...st.sel];
  const r = await post("/api/funscripts/scan", ids.length ? {ids} : {});
  if (!r.ok) { alert(r.error || "扫描失败"); return; }
  alert(`扫描 ${r.scanned} 个视频\n有脚本 ${r.with_script} 个（单轴 ${r.single}、多轴 ${r.multi}）\n无脚本 ${r.none} 个`);
  refreshLibrary();
});
document.getElementById("btn-preview").addEventListener("click", () => {
  if (st.sel.size !== 1) { alert("请先选中一个视频（单选）进行预览"); return; }
  openPreview([...st.sel][0]);
});
async function openPreview(id, name, mode) {
  const isSkel = (mode === true || mode === "skel");
  let title = name || (st.videos.find(x => x.id === id) || {}).name;
  if (!title) {
    const d = await api("/api/video_detail/" + id);
    title = d && d.ok ? d.name : ("#" + id);
  }
  $("#preview-name").textContent = title;
  document.querySelectorAll("#preview-modal .pv-other").forEach(el => {
    el.style.display = isSkel ? "none" : "";
  });
  $("#preview-modal").style.display = "flex";
  if (isSkel) {
    $("#pv-video").removeAttribute("src");
    $("#pv-clip").removeAttribute("src");
    $("#pv-audio").removeAttribute("src");
    $("#pv-fs-wrap").style.display = "none";
  } else {
    $("#pv-video").src = "/api/video/" + id;
    $("#pv-clip").src = "/api/clip/" + id;
    $("#pv-audio").src = "/api/audio/" + id;
    loadPreviewFunscript(id);
  }
  await loadSkelPlayer(id);
}

// ---- 单视频快速播放 ----
const PLAY_SRC = { orig: "/api/video/", skel: "/api/preview/", clip: "/api/clip/" };
const PLAY_LABEL = { orig: "原片", skel: "骨架视频", clip: "片段视频" };
function playOne(id, kind, name) {
  const base = PLAY_SRC[kind] || PLAY_SRC.orig;
  $("#play-name").textContent = (PLAY_LABEL[kind] || "") + " · " + (name || ("#" + id));
  $("#play-video").src = base + id;
  $("#play-modal").style.display = "flex";
}
function closePlay() {
  const v = $("#play-video");
  try { v.pause(); } catch (e) {}
  $("#play-modal").style.display = "none";
}
document.getElementById("play-close").addEventListener("click", closePlay);
document.getElementById("play-modal").addEventListener("click", e => {
  if (e.target.id === "play-modal") closePlay();
});

// ---- 骨架叠加播放器（视频 + 骨架 + 声音）----
const POSE_CONN = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
  [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
  [27,29],[28,30],[29,31],[30,32],[27,31],[28,32]];
const skel = { data: null, fps: 15, raf: null };

async function loadSkelPlayer(id) {
  stopSkel();
  const v = $("#skel-video");
  if (!v) return;
  skel.data = null;
  const d = await api("/api/pose/" + id);
  skel.data = (d.ok && d.landmarks && d.landmarks.length) ? d : null;
  skel.fps = d.fps || 15;
  // 舞台按视频宽高比自适应，避免画面被压成中间一条
  const stage = document.querySelector(".skel-stage");
  if (stage) {
    const asp = d.aspect || 0.75;
    const H = 420;
    stage.style.height = H + "px";
    stage.style.width = Math.round(H * asp) + "px";
  }
  v.poster = "/api/thumb/" + id;
  v.src = "/api/clip/" + id;
  v.load();
  v.onloadedmetadata = () => { try { v.currentTime = 0.05; } catch (e) {} };
  drawSkelAt(0);
  skel.raf = requestAnimationFrame(skelLoop);
}

function drawSkelAt(t) {
  const v = $("#skel-video");
  const cv = $("#skel-canvas");
  const d = skel.data;
  if (!v || !cv || !d) return;
  const W = v.clientWidth || 0;
  const H = v.clientHeight || 0;
  if (!W || !H) return;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  // object-fit: contain 时视频内容在元素内是 letterbox 的，骨架要映射到这个区域
  const vw = v.videoWidth || 0, vh = v.videoHeight || 0;
  let rx = 0, ry = 0, rw = W, rh = H;
  if (vw && vh) {
    const scale = Math.min(W / vw, H / vh);
    rw = vw * scale; rh = vh * scale;
    rx = (W - rw) / 2; ry = (H - rh) / 2;
  }
  const fps = skel.fps || 15;
  let idx = Math.floor((t || 0) * fps);
  if (idx < 0) idx = 0;
  if (idx >= d.frames) idx = d.frames - 1;
  const fr = d.landmarks[idx] || [];
  ctx.strokeStyle = "#5b8cff"; ctx.lineWidth = 2;
  for (const ab of POSE_CONN) {
    const a = fr[ab[0]], b = fr[ab[1]];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(rx + a[0]*rw, ry + a[1]*rh);
    ctx.lineTo(rx + b[0]*rw, ry + b[1]*rh);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffd166";
  for (const p of fr) {
    if (!p) continue;
    ctx.beginPath(); ctx.arc(rx + p[0]*rw, ry + p[1]*rh, 2.2, 0, Math.PI*2); ctx.fill();
  }
}

function skelLoop() {
  const v = $("#skel-video");
  if (!v) return;
  drawSkelAt(v.currentTime);
  skel.raf = requestAnimationFrame(skelLoop);
}

function stopSkel() {
  if (skel.raf) { cancelAnimationFrame(skel.raf); skel.raf = null; }
  const v = $("#skel-video");
  if (v) { try { v.pause(); } catch (e) {} }
}


// ================= 测试栏目 =================
const TEST = { id:null, p2:null, p3:null, fps:15, frame:0, playing:false, t0:0, raf:null,
  yaw:0.6, pitch:0.12, dragging:false, lx:0, ly:0, auto3d:true, signals:null };

function testVal(arr, f) { return (arr && arr[f] != null) ? arr[f] : 50; }

function testPopulate() {
  const sel = document.getElementById("test-video");
  if (!sel) return;
  const cur = TEST.id;
  sel.innerHTML = "";
  st.videos.forEach(v => {
    const o = document.createElement("option");
    o.value = v.id; o.textContent = v.name;
    if (v.id === cur) o.selected = true;
    sel.appendChild(o);
  });
}

async function testLoad(id) {
  id = id || parseInt(document.getElementById("test-video").value);
  if (!id) return;
  TEST.id = id;
  const [d2, d3] = await Promise.all([api("/api/pose/" + id), api("/api/pose3d/" + id)]);
  TEST.p2 = d2.ok ? d2 : null;
  TEST.p3 = d3.ok ? d3 : null;
  TEST.flow = null;
  if (signalSource() === "flow") {
    const df = await api("/api/flow/" + id);
    TEST.flow = df.ok ? df : null;
  }
  TEST.fps = (TEST.p2 && TEST.p2.fps) || (TEST.p3 && TEST.p3.fps) || 15;
  TEST.frame = 0;
  TEST.playing = false;
  document.getElementById("test-play").textContent = "▶ 播放";
  const v = document.getElementById("t2d-video");
  if (v) { v.src = "/api/clip/" + id; v.load(); }
  TEST.signals = testComputeSignals();
  emuReset();
  testDrawFrame(0);
}

function _slider(id, dflt) {
  const el = document.getElementById(id);
  return el ? parseFloat(el.value) : dflt;
}

function testComputeSignals() {
  const p2 = TEST.p2 ? TEST.p2.landmarks : [];
  const p3 = TEST.p3 ? TEST.p3.world : [];
  const n = Math.max(p2.length, p3.length);

  const fillNaNs = arr => {
    const out = arr.slice();
    let i = 0;
    while (i < out.length) {
      if (!isNaN(out[i])) { i++; continue; }
      let j = i;
      while (j < out.length && isNaN(out[j])) j++;
      const prev = i > 0 ? out[i-1] : (j < out.length ? out[j] : 0);
      const next = j < out.length ? out[j] : prev;
      for (let k = i; k < j; k++) {
        const t = (k - i + 1) / (j - i + 1);
        out[k] = prev + (next - prev) * t;
      }
      i = j;
    }
    return out;
  };
  const smooth = (arr, k) => {
    k = k || 3;
    const out = arr.slice();
    for (let i = 0; i < arr.length; i++) {
      let sum = 0, c = 0;
      for (let j = -k; j <= k; j++) { const t = i + j; if (t >= 0 && t < arr.length) { sum += arr[t]; c++; } }
      out[i] = c ? sum / c : arr[i];
    }
    return out;
  };
  const norm01 = arr => {
    const sorted = [...arr].sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.02)];
    const hi = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))];
    const span = (hi - lo) || 1;
    return arr.map(v => Math.min(1, Math.max(0, (v - lo) / span)));
  };
  const clampJumps01 = (arr, maxD) => {
    const out = arr.slice();
    for (let i = 1; i < out.length; i++) {
      const d = out[i] - out[i-1];
      if (Math.abs(d) > maxD) out[i] = out[i-1] + Math.sign(d) * maxD;
    }
    return out;
  };

  // 髋部 2D（x=左右, y=上下）+ 身体尺寸（前后）
  const wx = [], wy = [], wz = [];
  let sumBx = 0, sumBy = 0, cntB = 0;
  for (let f = 0; f < n; f++) {
    const fr = p2[f] || [];
    const h23 = fr[23], h24 = fr[24];
    if (h23 && h24) { wx.push((h23[0]+h24[0])/2); wy.push((h23[1]+h24[1])/2); }
    else { wx.push(NaN); wy.push(NaN); }
    const s11 = fr[11], s12 = fr[12];
    wz.push((s11 && s12) ? Math.hypot(s12[0]-s11[0], s12[1]-s11[1]) : NaN);
    // 身体轴（肩中心→髋中心，向上）：用于估计相机滚转
    if (s11 && s12 && h23 && h24) {
      const sx = (s11[0]+s12[0])/2, sy = (s11[1]+s12[1])/2;
      const hx = (h23[0]+h24[0])/2, hy = (h23[1]+h24[1])/2;
      sumBx += sx - hx; sumBy += sy - hy; cntB++;
    }
  }

  // ===== 参考坐标系对齐（相机滚转补偿）=====
  let rot = 0;
  if (cntB > 0) {
    const bx = sumBx / cntB, by = sumBy / cntB;
    const len = Math.hypot(bx, by);
    if (len > 1e-6) {
      const dotUp = -by / len;                 // 身体轴与图像「上」的夹角余弦
      if (dotUp > 0.7071) rot = Math.atan2(-bx, -by);   // 接近竖直才修正；坐/躺退回 0
    }
  }
  const cosR = Math.cos(rot), sinR = Math.sin(rot);
  const rot2 = (x, y) => [x * cosR - y * sinR, x * sinR + y * cosR];
  const rot3 = v => { const a = rot2(v[0], v[1]); return [a[0], a[1], v[2]]; };
  const rx = [], ry = [];
  for (let f = 0; f < n; f++) {
    const a = rot2(wx[f], wy[f]);
    rx.push(a[0]); ry.push(a[1]);
  }

  // 3D 旋转：全部以髋（骨盆）为准，先做滚转对齐
  const rawR0 = [], rawR1 = [], rawR2 = [];
  for (let f = 0; f < n; f++) {
    const fw = p3[f] || [];
    let yaw = NaN, pitch = NaN, roll = NaN;
    if (fw.length >= 33 && fw[23] && fw[24]) {
      const h23 = rot3(fw[23]), h24 = rot3(fw[24]);
      const hx = h24[0] - h23[0], hy = h24[1] - h23[1], hz = h24[2] - h23[2];
      if (Math.hypot(hx, hy, hz) > 1e-4) {
        yaw = Math.atan2(hz, hx);    // 骨盆扭转：左右髋前后错开
        roll = Math.atan2(hy, hx);   // 骨盆侧倾：左右髋高低差
        if (yaw > Math.PI/2) yaw -= Math.PI;
        else if (yaw < -Math.PI/2) yaw += Math.PI;
        if (roll > Math.PI/2) roll -= Math.PI;
        else if (roll < -Math.PI/2) roll += Math.PI;
      }
      if (fw[11] && fw[12]) {
        const s11 = rot3(fw[11]), s12 = rot3(fw[12]);
        const sm = [(s11[0]+s12[0])/2, (s11[1]+s12[1])/2, (s11[2]+s12[2])/2];
        const hm = [(h23[0]+h24[0])/2, (h23[1]+h24[1])/2, (h23[2]+h24[2])/2];
        const ux = sm[0]-hm[0], uy = sm[1]-hm[1], uz = sm[2]-hm[2];
        if (Math.hypot(ux, uy, uz) > 1e-4) pitch = Math.atan2(uz, -uy);
      }
    }
    rawR0.push(yaw); rawR1.push(pitch); rawR2.push(roll);
  }

  const smooth01 = arr => {
    let a = smooth(fillNaNs(arr), 4);
    a = norm01(a);
    a = clampJumps01(a, 0.05);
    return smooth(a, 2);
  };

  const gain = _slider("sig-gain", 100) / 100;
  const inv = id => {
    const el = document.getElementById("inv-" + id);
    return el && el.checked ? 1 : 0;
  };
  const flip = (arr, key) => inv(key) ? arr.map(v => 1 - v) : arr;
  const enabled = id => {
    const el = document.getElementById("en-" + id);
    return !el || el.checked;
  };
  const hold = (arr, key) => enabled(key) ? arr : arr.map(() => 0.5);

  let L0, L1;
  if (signalSource() === "flow" && TEST.flow && TEST.flow.pos && TEST.flow.pos.length) {
    const fx = TEST.flow.pos.map(q => q[0]);
    const fy = TEST.flow.pos.map(q => q[1]);
    L0 = hold(flip(smooth01(fy), "l0"), "l0");   // 上下（光流累积，向下为正）
    L1 = hold(flip(smooth01(fx), "l1"), "l1");   // 左右
  } else {
    L0 = hold(flip(smooth01(ry), "l0"), "l0");   // 上下（骨架，向下为正）
    L1 = hold(flip(smooth01(rx), "l1"), "l1");   // 左右
  }
  const L2 = hold(flip(smooth01(wz), "l2"), "l2");   // 前后
  const R0 = hold(flip(smooth01(rawR0), "r0"), "r0");
  const R1 = hold(flip(smooth01(rawR1), "r1"), "r1");
  const R2 = hold(flip(smooth01(rawR2), "r2"), "r2");

  return {
    L0: L0.map(v => 0.5 + (v - 0.5) * gain),
    L1, L2, R0, R1, R2,
  };
}

function testFrameCount() { return Math.max(TEST.p2 ? TEST.p2.frames : 0, TEST.p3 ? TEST.p3.frames : 0); }

function testDrawFrame(idx) {
  testDraw2d(idx);
  testDraw3d(idx);
  testDrawCurves(idx);
  testDrawOsr6(idx);
  const n = testFrameCount();
  const sk = document.getElementById("test-seek");
  if (sk) sk.value = n > 1 ? (idx / (n - 1)) * 100 : 0;
  const tm = document.getElementById("test-time");
  if (tm) tm.textContent = (idx / (TEST.fps || 15)).toFixed(1) + "s";
}

function testDraw2d(idx) {
  const v = document.getElementById("t2d-video");
  const cv = document.getElementById("t2d-canvas");
  const d = TEST.p2;
  if (!v || !cv || !d) return;
  const W = v.clientWidth || 0, H = v.clientHeight || 0;
  if (!W || !H) return;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const vw = v.videoWidth || 0, vh = v.videoHeight || 0;
  let rx = 0, ry = 0, rw = W, rh = H;
  if (vw && vh) { const sc = Math.min(W/vw, H/vh); rw = vw*sc; rh = vh*sc; rx = (W-rw)/2; ry = (H-rh)/2; }
  const fr = d.landmarks[idx] || [];
  ctx.strokeStyle = "#5b8cff"; ctx.lineWidth = 2;
  for (const ab of POSE_CONN) {
    const a = fr[ab[0]], b = fr[ab[1]];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(rx + a[0]*rw, ry + a[1]*rh);
    ctx.lineTo(rx + b[0]*rw, ry + b[1]*rh);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffd166";
  for (const p of fr) { if (!p) continue; ctx.beginPath(); ctx.arc(rx + p[0]*rw, ry + p[1]*rh, 2.2, 0, Math.PI*2); ctx.fill(); }
}

function testDraw3d(idx) {
  const cv = document.getElementById("t3d-canvas");
  const d = TEST.p3;
  if (!cv || !d) return;
  const W = cv.clientWidth || 480, H = cv.clientHeight || 420;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0a0d12"; ctx.fillRect(0, 0, W, H);
  const scale = Math.min(W, H) * 0.9;
  const fr = d.world[idx] || [];
  const cy = Math.cos(TEST.yaw), sy = Math.sin(TEST.yaw);
  const cp = Math.cos(TEST.pitch), sp = Math.sin(TEST.pitch);
  const proj = p => {
    const x = p[0], y = -p[1], z = -p[2];
    const x1 = x*cy - z*sy, z1 = x*sy + z*cy;
    const y1 = y*cp - z1*sp, z2 = y*sp + z1*cp;
    const persp = 1.8 / (1.8 + z2);
    return [W/2 + x1*scale*persp, H/2 - y1*scale*persp];
  };
  const pts = fr.map(proj);
  ctx.strokeStyle = "#5b8cff"; ctx.lineWidth = 2;
  for (const ab of POSE_CONN) {
    const a = pts[ab[0]], b = pts[ab[1]];
    if (!a || !b) continue;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  ctx.fillStyle = "#ffd166";
  for (const p of pts) { ctx.beginPath(); ctx.arc(p[0], p[1], 2.4, 0, Math.PI*2); ctx.fill(); }
}

function testDrawCurves(idx) {
  const cv = document.getElementById("test-curve");
  const s = TEST.signals;
  if (!cv || !s) return;
  const W = cv.clientWidth || 600, H = 150;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const axes = [["L0","#5b8cff"],["L1","#4bd08a"],["L2","#ffb454"],["R0","#ff6b6b"],["R1","#b48cff"],["R2","#e88cff"]];
  const n = s.L0.length;
  ctx.lineWidth = 1.4;
  for (const kv of axes) {
    const arr = s[kv[0]];
    if (!arr || !arr.length) continue;
    ctx.strokeStyle = kv[1];
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / Math.max(1, n-1)) * W;
      const y = H - arr[i] * (H - 6) - 3;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  if (n > 1) {
    const px = (idx / (n - 1)) * W;
    ctx.strokeStyle = "rgba(255,255,255,.8)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
  }
  ctx.font = "11px monospace";
  axes.forEach((kv, i) => { ctx.fillStyle = kv[1]; ctx.fillText(kv[0], 8 + i*40, 14); });
}

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w/2, h/2);
  ctx.beginPath();
  ctx.moveTo(x+r, y);
  ctx.arcTo(x+w, y, x+w, y+h, r);
  ctx.arcTo(x+w, y+h, x, y+h, r);
  ctx.arcTo(x, y+h, x, y, r);
  ctx.arcTo(x, y, x+w, y, r);
  ctx.closePath();
}

let _osr6emu = null;
let _osr6emuTried = false;

async function initOsr6Emu() {
  if (_osr6emu || _osr6emuTried) return;
  _osr6emuTried = true;
  const el = document.getElementById("osr6-emu");
  if (!el) return;
  try {
    const mod = await import("/vendor/osr-emu/osr-emu.dist.module.min.js");
    _osr6emu = new mod.OSREmulator(el, { model: "SR6" });
    const hint = document.getElementById("osr6-emu-hint");
    if (hint) hint.textContent = "模型已加载 · 拖动旋转";
    emuWrite(TEST.frame);
  } catch (e) {
    console.warn("osr-emu 加载失败", e);
    const hint = document.getElementById("osr6-emu-hint");
    if (hint) hint.textContent = "3D 模拟器加载失败：" + (e && e.message ? e.message : e);
  }
}

const manualVals = { L0: 50, L1: 50, L2: 50, R0: 50, R1: 50, R2: 50 };
function manualMode() {
  const el = document.getElementById("manual-mode");
  return !!el && el.checked;
}
function signalSource() {
  const el = document.getElementById("src-flow");
  return (el && el.checked) ? "flow" : "skel";
}

function emuWrite(idx) {
  const hint = document.getElementById("osr6-emu-hint");
  if (!_osr6emu) {
    if (hint) hint.textContent = "模拟器未加载（导入失败？）";
    return;
  }
  if (!manualMode() && !TEST.signals) {
    if (hint) hint.textContent = "信号未就绪";
    return;
  }
  const s = TEST.signals;
  const AXES = ["L0", "L1", "L2", "R0", "R1", "R2"];
  let v;
  if (manualMode()) {
    v = {
      L0: manualVals.L0 * 99.99,
      L1: manualVals.L1 * 99.99,
      L2: manualVals.L2 * 99.99,
      R0: manualVals.R0 * 99.99,
      R1: manualVals.R1 * 99.99,
      R2: manualVals.R2 * 99.99,
    };
  } else {
    v = {
      L0: testVal(s.L0, idx) * 9999,
      L1: testVal(s.L1, idx) * 9999,
      L2: testVal(s.L2, idx) * 9999,
      R0: testVal(s.R0, idx) * 9999,
      R1: testVal(s.R1, idx) * 9999,
      R2: testVal(s.R2, idx) * 9999,
    };
  }
  const cmd = AXES.map(a => {
    const n = Math.max(0, Math.min(9999, Math.round(v[a])));
    return a + String(n).padStart(4, "0") + "I50";
  }).join(" ") + "\n";
  try {
    _osr6emu.write(cmd);
    if (hint) hint.textContent = "L0=" + Math.round(v.L0/99.99) + " L1=" + Math.round(v.L1/99.99) + " L2=" + Math.round(v.L2/99.99) + " R0=" + Math.round(v.R0/99.99) + " R1=" + Math.round(v.R1/99.99) + " R2=" + Math.round(v.R2/99.99);
  } catch (e) {
    if (hint) hint.textContent = "写入错误：" + (e && e.message ? e.message : e);
  }
}

function testDrawOsr6(idx) { emuWrite(idx); }

function testPlayLoop() {
  if (!TEST.playing) return;
  const v = document.getElementById("t2d-video");
  const t = (v && v.duration) ? v.currentTime : (performance.now() - TEST.t0) / 1000;
  let idx = Math.floor(t * (TEST.fps || 15));
  const n = testFrameCount();
  if (idx >= n) { idx = Math.max(0, n - 1); TEST.playing = false; document.getElementById("test-play").textContent = "▶ 播放"; }
  TEST.frame = idx;
  testDrawFrame(idx);
  TEST.raf = requestAnimationFrame(testPlayLoop);
}

function emuReset() {
  if (!_osr6emu) return;
  try {
    _osr6emu.write("L005000I50 L105000I50 L205000I50 R005000I50 R105000I50 R205000I50\n");
  } catch (e) {}
}

function testTogglePlay() {
  const v = document.getElementById("t2d-video");
  if (!v || !TEST.id) return;
  if (TEST.playing) {
    TEST.playing = false; try { v.pause(); } catch(e){}
    document.getElementById("test-play").textContent = "▶ 播放";
    if (TEST.raf) { cancelAnimationFrame(TEST.raf); TEST.raf = null; }
  } else {
    emuReset();                       // 播放前先回正
    TEST.playing = true;
    document.getElementById("test-play").textContent = "⏸ 暂停";
    try { v.play(); } catch(e){}
    TEST.raf = requestAnimationFrame(testPlayLoop);
  }
}

document.getElementById("test-load").addEventListener("click", () => testLoad());
document.getElementById("test-play").addEventListener("click", testTogglePlay);
document.getElementById("test-seek").addEventListener("input", e => {
  const v = document.getElementById("t2d-video");
  const n = testFrameCount();
  const idx = Math.round((parseFloat(e.target.value) / 100) * (n - 1));
  TEST.frame = idx;
  if (v && v.duration) { try { v.currentTime = (idx / (TEST.fps || 15)); } catch(e){} }
  testDrawFrame(idx);
});
document.getElementById("test-3d-auto").addEventListener("change", e => { TEST.auto3d = e.target.checked; });
["l0","l1","l2","r0","r1","r2"].forEach(a => {
  const el = document.getElementById("m-" + a);
  if (el) el.addEventListener("input", () => {
    manualVals[a.toUpperCase()] = parseFloat(el.value);
    const lbl = document.getElementById("m-" + a + "-v");
    if (lbl) lbl.textContent = el.value;
    if (manualMode()) testDrawOsr6(TEST.frame);
  });
});
["src-skel", "src-flow"].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", async () => {
    if (signalSource() === "flow" && TEST.id && !TEST.flow) {
      const d = await api("/api/flow/" + TEST.id);
      TEST.flow = d.ok ? d : null;
    }
    TEST.signals = testComputeSignals();
    testDrawCurves(TEST.frame);
    testDrawOsr6(TEST.frame);
  });
});
document.getElementById("manual-mode").addEventListener("change", e => {
  document.getElementById("manual-sliders").style.display = e.target.checked ? "" : "none";
  testDrawOsr6(TEST.frame);
});
["l0","l1","l2","r0","r1","r2"].forEach(a => {
  const el = document.getElementById("en-" + a);
  if (el) el.addEventListener("change", () => {
    TEST.signals = testComputeSignals();
    testDrawCurves(TEST.frame);
    testDrawOsr6(TEST.frame);
  });
});
["l0","l1","l2","r0","r1","r2"].forEach(a => {
  const el = document.getElementById("inv-" + a);
  if (el) el.addEventListener("change", () => {
    TEST.signals = testComputeSignals();
    testDrawCurves(TEST.frame);
    testDrawOsr6(TEST.frame);
  });
});
["sig-gain"].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("input", () => {
    TEST.signals = testComputeSignals();
    testDrawCurves(TEST.frame);
    testDrawOsr6(TEST.frame);
  });
});

// 3D 拖动
const _t3d = document.getElementById("t3d-canvas");
_t3d.addEventListener("mousedown", e => { TEST.dragging = true; TEST.lx = e.clientX; TEST.ly = e.clientY; });
window.addEventListener("mousemove", e => {
  if (!TEST.dragging) return;
  TEST.yaw += (e.clientX - TEST.lx) * 0.01;
  TEST.pitch = Math.max(-1.2, Math.min(1.2, TEST.pitch + (e.clientY - TEST.ly) * 0.01));
  TEST.lx = e.clientX; TEST.ly = e.clientY;
  testDraw3d(TEST.frame);
});
window.addEventListener("mouseup", () => { TEST.dragging = false; });

// （OSR6 视角由内置模拟器的轨道控制处理）

// 3D 自动旋转
setInterval(() => {
  if (document.getElementById("view-test").classList.contains("active") && TEST.auto3d && !TEST.dragging) {
    TEST.yaw += 0.008;
    testDraw3d(TEST.frame);
  }
}, 40);



// ---- 源文件详情 ----
async function showDetail(id) {
  const d = await api("/api/video_detail/" + id);
  if (!d.ok) { alert(d.error || "获取详情失败"); return; }
  st.detailId = id;
  $("#detail-name").textContent = d.name;
  $("#detail-video").src = "/api/video/" + id;
  const fsLines = (d.funscript_axes || []).map(a =>
    `<div class="di-row"><span>${esc(a.axis)}</span><b>${a.count} 个动作 · ${(a.duration_ms/1000).toFixed(1)}s · 速度 ${a.speed}</b></div>`
  ).join("");
  $("#detail-info").innerHTML = `
    <div class="di-row"><span>路径</span><b>${esc(d.path)}</b></div>
    <div class="di-row"><span>分辨率</span><b>${d.width}×${d.height} @ ${d.fps}fps</b></div>
    <div class="di-row"><span>时长</span><b>${d.duration}s</b></div>
    <div class="di-row"><span>状态</span><b>${statusBadge(d.status)}</b></div>
    <div class="di-row"><span>音乐组</span><b>${esc(d.music_name || "-")}</b></div>
    <div class="di-row"><span>动作组</span><b>${esc(d.dance_name || "-")}</b></div>
    <div class="di-row"><span>BPM</span><b>${d.tempo || "-"}</b></div>
    <div class="di-row"><span>置信度</span><b>${d.confidence != null ? d.confidence : "-"}</b></div>
    <div class="di-row"><span>分类</span><b>${(d.categories||[]).map(esc).join("、") || "-"}</b></div>
    <div class="di-row"><span>脚本</span><b>${d.funscript_count ? d.funscript_count + " 轴" : "无"}</b></div>
    ${fsLines}`;
  loadScriptStatus(id);
  $("#detail-modal").style.display = "flex";
}
function closeDetail() {
  const v = $("#detail-video");
  try { v.pause(); } catch (e) {}
  $("#detail-modal").style.display = "none";
}
document.getElementById("detail-close").addEventListener("click", closeDetail);
document.getElementById("detail-modal").addEventListener("click", e => {
  if (e.target.id === "detail-modal") closeDetail();
});
document.getElementById("detail-preview").addEventListener("click", () => {
  if (st.detailId != null) {
    try { $("#detail-video").pause(); } catch (e) {}
    openPreview(st.detailId, null, true);
  }
});
document.getElementById("detail-reextract").addEventListener("click", async () => {
  if (st.detailId == null) return;
  if (st.running) { alert("识别进行中，请先取消再重新提取"); return; }
  if (!confirm2("重新提取这个视频？（会清空它的提取结果）", "再次确认：确定重新提取？")) return;
  const r = await post("/api/videos/reextract", {ids:[st.detailId]});
  if (!r.ok) { alert(r.error||"失败"); return; }
  $("#detail-modal").style.display = "none";
  const a = await post("/api/run", {ids:[st.detailId]});
  if (a.ok) { beginPolling(); } else alert(a.error || "启动失败");
});
document.getElementById("detail-delete").addEventListener("click", async () => {
  if (st.detailId == null) return;
  if (st.running) { alert("识别进行中，请先取消再删除"); return; }
  if (!confirm2("删除这个视频（含缓存/缩略图，不影响源文件）？", "再次确认：确定删除？")) return;
  const r = await post("/api/videos/delete", {ids:[st.detailId]});
  if (!r.ok) { alert(r.error||"失败"); return; }
  $("#detail-modal").style.display = "none";
  st.sel.delete(st.detailId);
  refreshLibrary();
});

async function loadPreviewFunscript(videoId) {
  const wrap = $("#pv-fs-wrap");
  const d = await api("/api/funscript/" + videoId);
  const axes = (d.axes || []).filter(a => a.actions && a.actions.length);
  st.fsAxes = axes;
  if (!axes.length) { wrap.style.display = "none"; return; }
  wrap.style.display = "";
  $("#pv-fs-title").textContent = "Funscript 动作曲线（" + axes.map(a => a.axis).join(" / ") + "）";
  st.fsMaxAt = Math.max(...axes.map(a => a.actions[a.actions.length - 1][0] || 1));
  drawFunscriptCurve(null);
}
function drawFunscriptCurve(t) {
  const cv = $("#pv-fs-canvas");
  if (!cv || !st.fsAxes || !st.fsAxes.length) return;
  const w = cv.clientWidth || 600;
  cv.width = w; cv.height = 120;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, 120);
  const colors = ["#5b8cff", "#4bd08a", "#ffb454", "#ff6b6b", "#b48cff"];
  const maxAt = st.fsMaxAt || 1;
  st.fsAxes.forEach((a, idx) => {
    ctx.strokeStyle = colors[idx % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    a.actions.forEach((pt, i) => {
      const x = (pt[0] / maxAt) * (w - 2) + 1;
      const y = 118 - (pt[1] / 100) * 116;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  if (t != null && isFinite(t)) {
    const x = Math.max(0, Math.min(w, (t / maxAt) * w));
    ctx.strokeStyle = "rgba(255,255,255,.8)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 120); ctx.stroke();
  }
}
$("#pv-video").addEventListener("timeupdate", e => {
  if (st.fsAxes && st.fsAxes.length) drawFunscriptCurve(e.target.currentTime * 1000);
});
document.getElementById("preview-close").addEventListener("click", () => {
  $("#preview-modal").style.display = "none";
  stopSkel();
  $("#pv-video").pause(); $("#pv-clip").pause(); $("#pv-audio").pause();
});
document.getElementById("preview-modal").addEventListener("click", e => {
  if (e.target.id === "preview-modal") {
    $("#preview-modal").style.display = "none";
    stopSkel();
  $("#pv-video").pause(); $("#pv-clip").pause(); $("#pv-audio").pause();
  }
});
document.getElementById("check-all").addEventListener("change", e => {
  const list = filteredVideos();
  if (e.target.checked) list.forEach(v=>st.sel.add(v.id)); else list.forEach(v=>st.sel.delete(v.id));
  renderVideos();
});

// ===== 脚本管理 =====
let cmpEmu = null, cmpAxes = null, cmpSrcId = null;

async function loadScriptStatus(id) {
  const el = $("#script-status");
  const d = await api("/api/scripts/" + id);
  if (!d.ok) { el.textContent = "脚本数据获取失败"; return; }
  st.scripts = d;
  let html = "";
  if (d.active_source != null) {
    const a = d.linked.find(x => x.source_video_id === d.active_source);
    const nm = a ? a.name : ("#" + d.active_source);
    html += `<div class="di-row"><span>当前激活</span><b>${esc(nm)}${a && a.is_own ? "（自己的源）" : ""}</b></div>`;
  } else {
    html += `<div class="di-row"><span>当前激活</span><b>无脚本</b></div>`;
  }
  if (d.linked.length) {
    html += `<div class="di-row"><span>关联源</span><b>${d.linked.map(x => esc(x.name) + (x.active ? " ★" : "")).join("、")}</b></div>`;
  }
  el.innerHTML = html;
  $("#script-select-btn").textContent = (d.active_source != null) ? "切换脚本" : "关联脚本";
  $("#script-import-btn").style.display = d.own_source ? "none" : "";
}

async function openScriptSelect() {
  const d = st.scripts;
  if (!d) return;
  $("#script-select-modal").style.display = "flex";
  const mus = [...new Set(d.pool.filter(p => p.music_group != null).map(p => p.music_group))].sort((a,b)=>a-b);
  const dan = [...new Set(d.pool.filter(p => p.dance_group != null).map(p => p.dance_group))].sort((a,b)=>a-b);
  $("#ss-music").innerHTML = `<option value="">全部音乐组</option>` + mus.map(g => `<option value="${g}">音乐组 ${g}</option>`).join("");
  $("#ss-dance").innerHTML = `<option value="">全部动作组</option>` + dan.map(g => `<option value="${g}">动作组 ${g}</option>`).join("");
  if (d.music_group != null) $("#ss-music").value = String(d.music_group);
  renderScriptList();
}

function renderScriptList() {
  const d = st.scripts;
  if (!d) return;
  const name = ($("#ss-name").value || "").toLowerCase();
  const music = $("#ss-music").value, dance = $("#ss-dance").value;
  const list = d.pool.filter(p => {
    if (name && !p.name.toLowerCase().includes(name)) return false;
    if (music && String(p.music_group) !== music) return false;
    if (dance && String(p.dance_group) !== dance) return false;
    return true;
  });
  const el = $("#ss-list");
  if (!list.length) { el.innerHTML = `<div class="muted" style="padding:14px">没有匹配的源视频</div>`; return; }
  el.innerHTML = list.map(p => `
    <div class="ss-item">
      <img src="/api/skeleton_gif/${p.id}" loading="lazy" onerror="this.style.visibility='hidden'">
      <div class="ss-info">
        <div class="ss-name">${esc(p.name)}</div>
        <div class="muted" style="font-size:12px">${p.funscript_count} 轴 ${(p.funscript_axes||[]).map(esc).join("/")} · 音乐组 ${p.music_group ?? "-"} · 动作组 ${p.dance_group ?? "-"}</div>
      </div>
      <button class="ghost small" data-cmp="${p.id}">对比</button>
      <button class="ghost small" data-act="${p.id}">激活</button>
    </div>`).join("");
  el.querySelectorAll("[data-act]").forEach(b => b.addEventListener("click", () => activateScript(+b.dataset.act)));
  el.querySelectorAll("[data-cmp]").forEach(b => b.addEventListener("click", () => openCompare(+b.dataset.cmp)));
}

async function activateScript(srcId) {
  const ok = await post("/api/scripts/activate", { video_id: st.detailId, source_video_id: srcId });
  if (!ok.ok) { alert(ok.error || "激活失败"); return; }
  $("#script-select-modal").style.display = "none";
  await loadScriptStatus(st.detailId);
  refreshLibrary();
}

async function importSource() {
  const r = await post("/api/pick_files");
  if (!r.ok) { if (r.error && r.error !== "已取消") alert(r.error); return; }
  const path = r.paths && r.paths[0];
  if (!path) return;
  const ok = await post("/api/scripts/import_source", { video_id: st.detailId, path });
  if (!ok.ok) { alert(ok.error || "导入失败"); return; }
  await loadScriptStatus(st.detailId);
  refreshLibrary();
}

async function openCompare(srcId) {
  cmpSrcId = srcId;
  const targetId = st.detailId;
  $("#compare-modal").style.display = "flex";
  $("#compare-name").textContent = "对比：当前视频 vs 源视频";
  const v1 = $("#cmp-v1"), v2 = $("#cmp-v2");
  v1.src = "/api/clip/" + targetId;
  v2.src = "/api/clip/" + srcId;
  const d = await api("/api/funscript/" + srcId);
  cmpAxes = (d.axes || []).filter(a => a.actions && a.actions.length);
  await initCmpEmu();
  const sync = (from, to) => { try { to.currentTime = from.currentTime; } catch (e) {} };
  v1.onplay = () => { sync(v1, v2); v2.play(); };
  v2.onplay = () => { sync(v2, v1); v1.play(); };
  v1.onpause = () => v2.pause();
  v2.onpause = () => v1.pause();
  v1.onseeked = () => sync(v1, v2);
  v2.onseeked = () => sync(v2, v1);
  function tick() {
    if ($("#compare-modal").style.display === "none") return;
    const t = (v1.currentTime || 0) * 1000;
    const vals = sampleCmpAxes(t);
    writeCmpEmu(vals);
    drawCmpAxes(vals);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function sampleCmpAxes(t) {
  const vals = { L0: 50, L1: 50, L2: 50, R0: 50, R1: 50, R2: 50 };
  for (const a of (cmpAxes || [])) {
    const ax = (a.axis || "").toUpperCase();
    if (!(ax in vals)) continue;
    const acts = a.actions;
    if (!acts || !acts.length) continue;
    let lo = 0, hi = acts.length - 1, idx = 0;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (acts[m][0] <= t) { idx = m; lo = m + 1; } else hi = m - 1; }
    vals[ax] = acts[idx][1];
  }
  return vals;
}

async function initCmpEmu() {
  const el = $("#cmp-osr6");
  if (cmpEmu) return;
  try {
    const mod = await import("/vendor/osr-emu/osr-emu.dist.module.min.js");
    cmpEmu = new mod.OSREmulator(el, { model: "SR6" });
  } catch (e) {
    el.innerHTML = `<div class="muted" style="padding:14px">模拟器加载失败</div>`;
  }
}

function writeCmpEmu(vals) {
  if (!cmpEmu) return;
  const AXES = ["L0", "L1", "L2", "R0", "R1", "R2"];
  const cmd = AXES.map(a => {
    const n = Math.max(0, Math.min(9999, Math.round((vals[a] || 50) * 99.99)));
    return a + String(n).padStart(4, "0") + "I50";
  }).join(" ") + "\n";
  try { cmpEmu.write(cmd); } catch (e) {}
}

function drawCmpAxes(vals) {
  const colors = { L0: "#5b8cff", L1: "#4bd08a", L2: "#ffb454", R0: "#ff6b6b", R1: "#b48cff", R2: "#e88cff" };
  $("#cmp-axes").innerHTML = Object.keys(colors).map(a => `
    <div class="cmp-axis">
      <div class="lbl">${a}</div>
      <div class="bar"><div class="fill" style="width:${vals[a]}%;background:${colors[a]}"></div></div>
      <div class="lbl" style="width:34px;text-align:right">${Math.round(vals[a])}</div>
    </div>`).join("");
}

function closeCompare() {
  const v1 = $("#cmp-v1"), v2 = $("#cmp-v2");
  try { v1.pause(); v2.pause(); } catch (e) {}
  $("#compare-modal").style.display = "none";
}

$("#script-select-btn").addEventListener("click", openScriptSelect);
$("#script-import-btn").addEventListener("click", importSource);
$("#script-select-close").addEventListener("click", () => { $("#script-select-modal").style.display = "none"; });
$("#ss-name").addEventListener("input", renderScriptList);
$("#ss-music").addEventListener("change", renderScriptList);
$("#ss-dance").addEventListener("change", renderScriptList);
$("#compare-close").addEventListener("click", closeCompare);
$("#cmp-activate").addEventListener("click", () => {
  if (cmpSrcId) { closeCompare(); activateScript(cmpSrcId); }
});

async function refreshLibrary() {
  const d = await api("/api/library");
  if (d.configured===false) { st.configured=false; applyGate(); return; }
  st.configured = true; st.videos = d.videos; st.cats = d.categories;
  renderCatBar(); renderAssignSelect(); renderGroupFilters(); renderVideos();
}
function renderCatBar() {
  const el = $("#cat-bar"); el.innerHTML="";
  const all=document.createElement("span"); all.className="chip"+(st.filterCat==null?" on":""); all.innerHTML="<span>全部</span>";
  all.onclick=()=>{st.filterCat=null; renderCatBar(); renderVideos();};
  el.appendChild(all);
  st.cats.forEach(c => {
    const s=document.createElement("span"); s.className="chip"+(st.filterCat===c.id?" on":"");
    s.innerHTML=`<span>${esc(c.name)} (${c.count})</span>`;
    s.onclick=()=>{st.filterCat=c.id; renderCatBar(); renderVideos();};
    el.appendChild(s);
  });
}
function renderAssignSelect() {
  const sel = $("#assign-cat"); sel.innerHTML = '<option value="0">选择分类…</option>';
  st.cats.forEach(c => { const o=document.createElement("option"); o.value=c.id; o.textContent=c.name; sel.appendChild(o); });
}

// ---- 搜索 / 分组过滤 ----
function groupOptions() {
  const m = new Map(); const d = new Map();
  st.videos.forEach(v => {
    if (v.music_group) m.set(v.music_group, v.music_name || ("M"+v.music_group));
    if (v.dance_group) d.set(v.dance_group, v.dance_name || ("D"+v.dance_group));
  });
  return { m: [...m.entries()].sort((a,b)=>a[0]-b[0]), d: [...d.entries()].sort((a,b)=>a[0]-b[0]) };
}
function updateMselLabel(btnId, selected) {
  const b = document.getElementById(btnId);
  b.textContent = selected.size ? b.dataset.label + " (" + selected.size + ") ▾" : b.dataset.label + " ▾";
}
function renderMsel(panelId, btnId, options, selected) {
  const panel = document.getElementById(panelId);
  panel.innerHTML = "";
  panel.onclick = e => e.stopPropagation();  // 点面板内部不触发「点外部关闭」
  if (!options.length) { panel.innerHTML = '<div class="muted" style="padding:4px 8px">暂无分组</div>'; return; }
  options.forEach(([key, name]) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = selected.has(key);
    cb.addEventListener("change", () => {
      cb.checked ? selected.add(key) : selected.delete(key);
      renderVideos();
      updateMselLabel(btnId, selected);
    });
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(name));
    panel.appendChild(lab);
  });
}
function renderGroupFilters() {
  const { m, d } = groupOptions();
  renderMsel("msel-music", "msel-music-btn", m, st.fMusic);
  renderMsel("msel-dance", "msel-dance-btn", d, st.fDance);
  updateMselLabel("msel-music-btn", st.fMusic);
  updateMselLabel("msel-dance-btn", st.fDance);
}
document.getElementById("search").addEventListener("input", e => { st.search = e.target.value.trim(); renderVideos(); });
document.getElementById("status-filter").addEventListener("change", e => { st.fStatus = e.target.value; renderVideos(); });
document.getElementById("fs-filter").addEventListener("change", e => { st.fsLevel = e.target.value; renderVideos(); });
document.getElementById("btn-reset-filter").addEventListener("click", () => {
  st.search = ""; $("#search").value = "";
  st.fMusic.clear(); st.fDance.clear(); st.filterCat = null;
  st.fStatus = "all"; $("#status-filter").value = "all";
  st.fsLevel = "all"; $("#fs-filter").value = "all";
  renderCatBar(); renderGroupFilters(); renderVideos();
});
document.getElementById("msel-music-btn").addEventListener("click", e => {
  e.stopPropagation();
  const p = document.getElementById("msel-music");
  document.getElementById("msel-dance").classList.remove("open");
  p.classList.toggle("open");
});
document.getElementById("msel-dance-btn").addEventListener("click", e => {
  e.stopPropagation();
  const p = document.getElementById("msel-dance");
  document.getElementById("msel-music").classList.remove("open");
  p.classList.toggle("open");
});
document.addEventListener("click", () => { document.querySelectorAll(".msel-panel").forEach(p => p.classList.remove("open")); });
function filteredVideos() {
  let list = st.videos;
  if (st.filterCat != null) list = list.filter(v => v.cats.includes(st.filterCat));
  if (st.search) list = list.filter(v => v.name.toLowerCase().includes(st.search.toLowerCase()));
  if (st.fStatus !== "all") {
    list = list.filter(v => st.fStatus === "done" ? v.status === "ok" : v.status === st.fStatus);
  }
  if (st.fMusic.size) list = list.filter(v => v.music_group != null && st.fMusic.has(v.music_group));
  if (st.fDance.size) list = list.filter(v => v.dance_group != null && st.fDance.has(v.dance_group));
  if (st.fsLevel !== "all") {
    list = list.filter(v => {
      const n = v.funscript_count || 0;
      if (st.fsLevel === "none") return n === 0;
      if (st.fsLevel === "single") return n === 1;
      return n >= 2;  // multi
    });
  }
  return list;
}
function renderVideos() {
  const list = filteredVideos();
  $("#video-empty").style.display = list.length ? "none" : "";
  const el = $("#video-list");
  if (st.view === "grid") { el.className="grid size-"+st.thumbSize; el.innerHTML=""; list.forEach(v=>el.appendChild(gridCard(v))); }
  else { el.className=""; el.innerHTML=""; el.appendChild(listTable(list)); }
  updateSelUI();
}
function updateSelUI() {
  const list = filteredVideos();
  $("#check-all").checked = list.length>0 && list.every(v=>st.sel.has(v.id));
  $("#sel-info").textContent = st.sel.size ? `已选 ${st.sel.size} 个` : "未选择";
}
function toggleSel(id) {
  st.sel.has(id) ? st.sel.delete(id) : st.sel.add(id);
}
function listTable(list) {
  const t = document.createElement("table");
  const thead = document.createElement("thead");
  thead.innerHTML = `<tr><th style="width:30px"></th><th style="width:76px"></th><th>文件</th><th>时长</th><th>音乐组</th><th>动作组</th><th>脚本</th><th>状态</th></tr>`;
  const tb = document.createElement("tbody");
  list.forEach(v => {
    const tr = document.createElement("tr");
    if (st.sel.has(v.id)) tr.className="sel";
    if (v.status==="no_pose") tr.classList.add("np");
    tr.dataset.id = v.id;
    const checked = st.sel.has(v.id) ? "checked":"";
    tr.innerHTML = `<td><input type="checkbox" data-cb ${checked}></td>
      <td><img class="thumb-sm" src="/api/thumb/${v.id}"></td>
      <td><span class="list-name">${esc(v.name)}</span></td><td>${v.duration}s</td>
      <td>${v.music_name||"-"}</td>
      <td>${v.dance_name||"-"}</td>
      <td>${funscriptBadge(v)}</td>
      <td>${statusBadge(v.status)}</td>`;
    tr.addEventListener("click", e => {
      if (e.target.dataset.cb === "") {
        toggleSel(v.id);
        tr.classList.toggle("sel", st.sel.has(v.id));
        updateSelUI();
        return;
      }
      if (e.target.tagName === "IMG" || e.target.classList.contains("list-name")) {
        showDetail(v.id);
        return;
      }
      toggleSel(v.id);
      tr.classList.toggle("sel", st.sel.has(v.id));
      const cb = tr.querySelector("input[data-cb]");
      if (cb) cb.checked = st.sel.has(v.id);
      updateSelUI();
    });
    tb.appendChild(tr);
  });
  t.appendChild(thead); t.appendChild(tb);
  return t;
}
function gridCard(v) {
  const d = document.createElement("div");
  d.className = "card" + (st.sel.has(v.id) ? " sel":"");
  d.dataset.id = v.id;
  const checked = st.sel.has(v.id) ? "checked":"";
  const media = st.anim
    ? `<video class="thumb-media" loop muted playsinline preload="none" poster="/api/thumb/${v.id}" data-src="/api/thumb_mp4/${v.id}"></video>`
    : `<img class="thumb-media" src="/api/thumb/${v.id}" loading="lazy">`;
  d.innerHTML = `<input type="checkbox" class="ck" ${checked}>
    <div class="thumb-wrap">
      ${media}
      <div class="hover-btns"><button data-kind="orig">原片</button><button data-kind="skel">骨架</button><button data-kind="clip">片段</button></div>
    </div>
    <div class="meta"><div class="name">${esc(v.name)}</div>
    <div class="badges">${statusBadge(v.status)}${funscriptBadge(v)}${v.music_name?`<span class="badge">${esc(v.music_name)}</span>`:""}${v.dance_name?`<span class="badge">${esc(v.dance_name)}</span>`:""}</div></div>`;
  d.querySelector(".hover-btns").addEventListener("click", e => {
    e.stopPropagation();
    const btn = e.target.closest("button");
    if (!btn) return;
    if (btn.dataset.kind === "skel") { openPreview(v.id, v.name, true); return; }
    playOne(v.id, btn.dataset.kind, v.name);
  });
  observeThumb(d.querySelector("video.thumb-media"));
  d.addEventListener("click", e => {
    if (e.target.classList.contains("ck")) {
      toggleSel(v.id);
      d.classList.toggle("sel", st.sel.has(v.id));
      e.target.checked = st.sel.has(v.id);
      updateSelUI();
      return;
    }
    if (e.target.tagName === "IMG" || e.target.tagName === "VIDEO" || e.target.classList.contains("name")) {
      showDetail(v.id);
      return;
    }
    toggleSel(v.id);
    d.classList.toggle("sel", st.sel.has(v.id));
    const cb = d.querySelector(".ck");
    if (cb) cb.checked = st.sel.has(v.id);
    updateSelUI();
  });
  return d;
}

// ---- 分组 ----
async function loadGroups(type) {
  const d = await api("/api/groups?type="+type);
  const el = document.getElementById(type+"-groups"); el.innerHTML="";
  if (!d.configured) { el.innerHTML='<div class="empty">请先设置缓存目录。</div>'; return; }
  if (!d.groups.length) { el.innerHTML='<div class="empty">还没有分组结果，去「原文件」导入并识别。</div>'; return; }
  d.groups.forEach(g => {
    const div=document.createElement("div"); div.className="group";
    const head=document.createElement("div"); head.className="group-head";
    const inp=document.createElement("input"); inp.type="text"; inp.value=g.name;
    inp.addEventListener("change", async ()=>{ await post("/api/groups/rename", {type, key:g.key, name:inp.value}); });
    head.appendChild(inp);
    const cnt=document.createElement("span"); cnt.className="muted"; cnt.textContent=g.members.length+" 个"; head.appendChild(cnt);
    div.appendChild(head);
    const body = document.createElement("div"); body.className = "group-body";
    const rep = g.members[0];
    if (rep) {
      const pv = document.createElement("div"); pv.className = "group-preview";
      pv.title = "点击预览：" + rep.name;
      pv.innerHTML = `<img src="/api/thumb/${rep.id}" loading="lazy">
        <div class="hover-btns"><button data-kind="orig">原片</button><button data-kind="skel">骨架</button><button data-kind="clip">片段</button></div>`;
      pv.addEventListener("click", () => openPreview(rep.id, rep.name));
      pv.querySelector(".hover-btns").addEventListener("click", e => {
        e.stopPropagation();
        const btn = e.target.closest("button");
        if (!btn) return;
        if (btn.dataset.kind === "skel") { openPreview(rep.id, rep.name, true); return; }
        playOne(rep.id, btn.dataset.kind, rep.name);
      });
      body.appendChild(pv);
    }
    const mem=document.createElement("div"); mem.className="members";
    g.members.forEach(m=>{
      const mm=document.createElement("div"); mm.className="member";
      const mmedia = st.anim
        ? `<video class="thumb-media" loop muted playsinline preload="none" poster="/api/thumb/${m.id}" data-src="/api/thumb_mp4/${m.id}"></video>`
        : `<img class="thumb-media" src="/api/thumb/${m.id}" loading="lazy">`;
      mm.innerHTML=`<div class="thumb-wrap">${mmedia}<div class="hover-btns"><button data-kind="orig">原片</button><button data-kind="skel">骨架</button><button data-kind="clip">片段</button></div></div><div class="mname ${m.status==="no_pose"?"np":""}">${esc(m.name)}</div><div style="margin-top:3px">${funscriptBadge(m)}</div>`;
      mm.querySelector(".hover-btns").addEventListener("click", e => {
        e.stopPropagation();
        const btn = e.target.closest("button");
        if (!btn) return;
        if (btn.dataset.kind === "skel") { openPreview(m.id, m.name, true); return; }
        playOne(m.id, btn.dataset.kind, m.name);
      });
      observeThumb(mm.querySelector("video.thumb-media"));
      mem.appendChild(mm);
    });
    body.appendChild(mem);
    div.appendChild(body); el.appendChild(div);
  });
}

// ---- 启动 ----
(async function init(){ await refreshConfig(); if (st.configured) { switchView("library"); refreshLibrary(); } })();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
