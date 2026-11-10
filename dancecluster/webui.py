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

from . import config, ffmpeg, funscript, pipeline, pose
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
            for kind in ("thumb", "gif", "preview"):
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
        store = _store()
        store.reset_stale_extracting()
        before = _counts(store)
        store.close()

        cfg = _cfg()
        cfg.clip.start = float(params.get("start", 0.0))
        cfg.clip.duration = float(params.get("duration", 20.0))
        cfg.match.music_threshold = float(params.get("music", 0.55))
        cfg.match.dance_threshold = float(params.get("dance", 0.35))
        cfg.match.mirror_invariant = bool(params.get("mirror", True))

        _log(f"开始提取（待处理 {before['pending']} 个，已完成 {before['done']} 个将跳过）…")
        records = pipeline.process_pending(cfg, progress=progress, only_ids=params.get("ids"))
        if _state["cancel"]:
            raise _Cancelled()
        _log(f"提取完成，正在分组…")
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
    if not ids:
        return jsonify({"ok": False, "error": "请先勾选要提取的视频"}), 400
    with _lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "任务正在进行"}), 409
        _state.update(running=True, progress=0, total=0, stage="准备中", name="",
                      log=[], error="", cancel=False)
    threading.Thread(target=_run_job, args=(body,), daemon=True).start()
    return jsonify({"ok": True})


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
        info = funscript.parse(path)
        axes.append({
            "axis": axis,
            "path": str(path),
            "count": info["count"],
            "duration_ms": info["duration_ms"],
            "speed": info["speed"],
            "actions": info["actions"],
        })
    return jsonify({"ok": True, "count": len(axes), "axes": axes})


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
    # 同步清理磁盘：缓存目录 + 缩略图/骨架预览
    for cd in cache_dirs:
        if cd:
            try:
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
    pipeline.clear_all(cfg)
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
  .card .ck { position:absolute; top:6px; left:6px; }
  .card .meta { padding:8px 9px; }
  .card .name { font-size:12px; word-break:break-all; }
  .card .badges { margin-top:4px; display:flex; gap:4px; flex-wrap:wrap; }
  .badge { font-size:11px; border-radius:5px; padding:1px 6px; background:#1c222d; color:var(--dim); }
  .badge.np { color:var(--bad); }
  .badge.fs-none { color:var(--dim); }
  .badge.fs-single { color:#7aa2ff; border:1px solid #2c4166; }
  .badge.fs-multi { color:var(--ok); border:1px solid #2a4a3a; }
  .pv canvas { width:100%; height:120px; background:#0c0f15; border-radius:8px; display:block; }
  .badge.done { color:var(--ok); }
  .badge.extracting { color:var(--warn); }
  .thumb-sm { width:64px; height:36px; object-fit:cover; border-radius:4px; vertical-align:middle; background:#0a0d12; }
  .group { border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin-bottom:12px; background:var(--panel); }
  .group-head { display:flex; align-items:center; gap:10px; }
  .group-head .tag { background:var(--accent); color:#fff; border-radius:6px; padding:2px 8px; font-size:12px; }
  .group-head input[type=text] { flex:0 1 320px; }
  .members { margin-top:12px; display:flex; flex-wrap:wrap; gap:12px; }
  .member { width:120px; }
  .member img { width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:6px; background:#0a0d12; }
  .member .mname { font-size:11px; color:var(--dim); word-break:break-all; }
  .member .np { color:var(--bad); }
  .group-preview { margin:10px 0; }
  .group-preview .skel { width:230px; max-height:330px; object-fit:contain; border-radius:8px; background:#000; display:block; }
  .empty { color:var(--dim); padding:30px 0; text-align:center; }
  .settings-box { max-width:560px; margin:30px auto 0; }
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
</style>
</head>
<body>
<div class="app">
  <aside>
    <div class="brand">DanceCluster<small>舞蹈视频自动分组</small></div>
    <nav>
      <button class="nav active" data-view="task"><span class="n">▶</span>识别任务</button>
      <button class="nav" data-view="library"><span class="n">▤</span>原文件</button>
      <button class="nav" data-view="music"><span class="n">♪</span>音频分组</button>
      <button class="nav" data-view="dance"><span class="n">✣</span>动作分组</button>
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
          <input id="cache-dir" type="text" placeholder="例如 D:\\DanceCluster\\cache" style="flex:1">
          <button id="btn-pick-cache" class="ghost">选择…</button>
          <button id="btn-default-cache" class="ghost">使用默认目录</button>
          <button id="btn-save-cache">保存并启用</button>
        </div>
        <p class="muted" id="cache-hint"></p>
        <div class="row" style="margin-top:14px">
          <label for="pose-model" style="margin-right:10px; white-space:nowrap">MediaPipe 模型</label>
          <select id="pose-model" style="flex:1"></select>
        </div>
        <p class="muted">模型决定骨架精度与速度：Heavy 最准但慢、Full 均衡、Lite 最快但精度低。切换后，已完成的视频会在下次运行时自动重新提取。</p>
        <div class="row" style="margin-top:18px; padding-top:14px; border-top:1px solid var(--line)">
          <button id="btn-clear" class="ghost" style="color:var(--bad)">清除全部数据（重新开始）</button>
        </div>
      </div>
    </div>

    <!-- 识别任务 -->
    <div class="view active" id="view-task">
      <h2>识别任务</h2>
      <p class="sub">第 1 步：添加文件夹或单个视频 → 第 2 步：开始提取。提取结果会出现在「原文件」。</p>
      <div class="panel">
        <div class="row">
          <button id="btn-add-folder">导入任务</button>
          <button id="btn-add-videos" class="ghost">导入视频</button>
        </div>
        <div id="task-msg" class="muted" style="margin-top:10px"></div>
      </div>
      <div class="panel">
        <div class="row" style="justify-content:space-between">
          <b>待提取视频</b>
          <span><button id="btn-remove-task" class="ghost small">移除选中</button> <span class="muted" id="task-sel-info">未选择</span></span>
        </div>
        <div id="task-list"></div>
        <div class="empty" id="task-empty">暂无待提取视频，请先添加文件夹或视频。</div>
      </div>
      <div class="panel">
        <div class="row">
          <button id="btn-run">开始提取</button>
          <button id="btn-cancel" class="ghost" disabled>取消</button>
          <span class="muted">已提取过的会自动跳过。</span>
        </div>
        <div class="bar"><i id="fill"></i></div>
        <div class="status" id="status">就绪</div>
      </div>
      <div class="panel">
        <b>日志</b>
        <pre class="log" id="log"></pre>
      </div>
    </div>

    <!-- 原文件 -->
    <div class="view" id="view-library">
      <h2>原文件</h2>
      <p class="sub">提取结果。可多选视频进行：归入分类、重新提取、删除。</p>
      <div class="panel">
        <div class="row">
          <button id="btn-new-cat" class="ghost">＋ 新建分类</button>
          <span style="flex:1"></span>
          <button id="btn-view-list" class="toggle on">☰ 列表</button>
          <button id="btn-view-grid" class="toggle">▦ 图标</button>
          <span id="size-group" style="display:none">
            <button id="size-s" class="toggle small">小</button>
            <button id="size-m" class="toggle small on">中</button>
            <button id="size-l" class="toggle small">大</button>
          </span>
        </div>
        <div class="row" style="margin-top:10px">
          <input type="text" id="search" placeholder="搜索文件名…" style="flex:1;max-width:260px">
          <div class="msel"><button id="msel-music-btn" class="ghost small" data-label="音乐分组">音乐分组 ▾</button><div id="msel-music" class="msel-panel"></div></div>
          <div class="msel"><button id="msel-dance-btn" class="ghost small" data-label="动作分组">动作分组 ▾</button><div id="msel-dance" class="msel-panel"></div></div>
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
        <div class="empty" id="video-empty">还没有视频，请先去「识别任务」添加文件夹并提取。</div>
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
  </main>
</div>

<div class="modal" id="preview-modal" style="display:none">
  <div class="modal-box">
    <div class="modal-head"><b id="preview-name"></b><button class="ghost small" id="preview-close">✕ 关闭</button></div>
    <div class="modal-body">
      <div class="pv"><div class="pv-title">原视频</div><video id="pv-video" controls playsinline></video></div>
      <div class="pv"><div class="pv-title">动作片段（前 20 秒）</div><video id="pv-clip" controls playsinline></video></div>
      <div class="pv"><div class="pv-title">提取的音频</div><audio id="pv-audio" controls></audio></div>
      <div class="pv" id="pv-fs-wrap" style="display:none">
        <div class="pv-title" id="pv-fs-title">Funscript 动作曲线</div>
        <canvas id="pv-fs-canvas" height="120"></canvas>
      </div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const st = { configured:false, view:"list", thumbSize:"m", anim:true, filterCat:null, timer:null, videos:[], cats:[], sel:new Set(), selFolders:new Set(), selTask:new Set(), search:"", fMusic:new Set(), fDance:new Set(), fsLevel:"all", fsAxes:[], fsMaxAt:1 };

async function api(path, opts) { const r = await fetch(path, opts); return r.json(); }
async function post(path, body) { return api(path, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})}); }

function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function statusBadge(s) {
  const m = {ok:["done","已完成"], no_pose:["np","无骨架"], extracting:["extracting","提取中"], pending:["","待处理"], error:["np","出错"]}[s]||["",""];
  return `<span class="badge ${m[0]}">${m[1]||s}</span>`;
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
function switchView(name) {
  document.querySelectorAll(".nav").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.getElementById("view-"+name).classList.add("active");
  if (name==="music") loadGroups("music");
  if (name==="dance") loadGroups("dance");
  if (name==="library") refreshLibrary();
  if (name==="task") refreshTask();
  if (name==="settings") refreshConfig();
}

// ---- 设置 ----
document.getElementById("btn-save-cache").addEventListener("click", async () => {
  const r = await post("/api/config/set", {cache_dir: $("#cache-dir").value, model: $("#pose-model").value});
  if (!r.ok) { $("#cache-hint").textContent = "✗ " + (r.error||"失败"); $("#cache-hint").style.color="var(--bad)"; return; }
  $("#cache-hint").textContent = "✓ 已启用：" + r.cache_dir; $("#cache-hint").style.color="var(--ok)";
  await refreshConfig(); switchView("task");
});
document.getElementById("btn-pick-cache").addEventListener("click", async () => {
  const r = await post("/api/pick_folder", {});
  if (r.ok) $("#cache-dir").value = r.path; else if (r.error!=="已取消") alert(r.error);
});
document.getElementById("btn-default-cache").addEventListener("click", async () => {
  const c = await api("/api/config"); if (c.suggested) $("#cache-dir").value = c.suggested;
});
document.getElementById("btn-clear").addEventListener("click", async () => {
  if (!confirm("确定清除所有数据吗？将删除数据库、缓存、缩略图（不影响源视频和缓存目录设置）。")) return;
  const r = await post("/api/clear", {});
  if (r.ok) { alert("已清除"); st.sel.clear(); st.selFolders.clear(); st.selTask.clear(); switchView("task"); refreshTask(); }
  else alert(r.error || "清除失败");
});
async function refreshConfig() {
  const c = await api("/api/config");
  st.configured = c.configured; $("#cache-dir").value = c.cache_dir||"";
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
  $("#btn-run").disabled = !st.configured;
  $("#btn-add-folder").disabled = !st.configured;
  if (!st.configured) switchView("settings");
}

// ---- 识别任务：文件夹 ----
document.getElementById("btn-add-folder").addEventListener("click", async () => {
  const r = await post("/api/pick_folder", {});
  if (!r.ok) { if (r.error!=="已取消") alert(r.error); return; }
  const a = await post("/api/library/add", {path:r.path});
  if (a.ok) { showTaskMsg(`扫描 ${a.scanned} 个视频，成功添加任务 ${a.added} 个，已完成提取 ${a.already} 个`); refreshTask(); }
  else alert(a.error);
});
document.getElementById("btn-add-videos").addEventListener("click", async () => {
  const r = await post("/api/pick_files", {});
  if (!r.ok) { if (r.error!=="已取消") alert(r.error); return; }
  const a = await post("/api/library/add_videos", {paths:r.paths});
  if (a.ok) { showTaskMsg(`扫描 ${a.scanned} 个视频，成功添加任务 ${a.added} 个，已完成提取 ${a.already} 个`); refreshTask(); }
  else alert(a.error);
});
document.getElementById("btn-remove-task").addEventListener("click", async () => {
  if (!st.selTask.size) { alert("请先勾选视频"); return; }
  if (!confirm(`移除选中的 ${st.selTask.size} 个视频？`)) return;
  await post("/api/videos/delete", {ids:[...st.selTask]});
  st.selTask.clear(); refreshTask();
});
async function refreshTask() {
  const d = await api("/api/library");
  if (d.configured===false) { st.configured=false; applyGate(); return; }
  st.configured = true;
  renderTaskList(d.videos.filter(v => v.status==="pending" || v.status==="extracting"));
}
function showTaskMsg(t) { const el = $("#task-msg"); el.textContent = t; el.style.color = "var(--ok)"; }
function renderTaskList(videos) {
  const el = $("#task-list");
  // 只保留仍在「待提取」里的选择，避免提取完成后误删已完成的视频
  const valid = new Set(videos.map(v => v.id));
  for (const id of [...st.selTask]) if (!valid.has(id)) st.selTask.delete(id);
  $("#task-empty").style.display = videos.length ? "none" : "";
  $("#task-sel-info").textContent = st.selTask.size ? `已选 ${st.selTask.size} 个` : "未选择";
  el.innerHTML = "";
  if (!videos.length) return;
  const t = document.createElement("table");
  const thead = document.createElement("thead");
  thead.innerHTML = `<tr><th style="width:30px"><input type="checkbox" id="task-check-all" title="全选"></th><th style="width:76px"></th><th>文件</th><th>分辨率</th><th>时长</th><th>状态</th></tr>`;
  const tb = document.createElement("tbody");
  videos.forEach(v => {
    const tr = document.createElement("tr");
    tr.dataset.id = v.id;
    if (st.selTask.has(v.id)) tr.className="sel";
    const ck = st.selTask.has(v.id) ? "checked":"";
    tr.innerHTML = `<td><input type="checkbox" ${ck}></td><td><img class="thumb-sm" src="/api/thumb/${v.id}"></td><td>${esc(v.name)}</td><td>${v.width}×${v.height}</td><td>${v.duration}s</td><td>${statusBadge(v.status)}</td>`;
    const cb = tr.querySelector("input[type=checkbox]");
    cb.addEventListener("click", e => {
      e.stopPropagation();
      st.selTask.has(v.id) ? st.selTask.delete(v.id) : st.selTask.add(v.id);
      renderTaskList(videos);
    });
    tr.addEventListener("click", e => {
      if (e.target.type === "checkbox") return;
      st.selTask.has(v.id) ? st.selTask.delete(v.id) : st.selTask.add(v.id);
      renderTaskList(videos);
    });
    tb.appendChild(tr);
  });
  t.appendChild(thead); t.appendChild(tb);
  const all = t.querySelector("#task-check-all");
  all.checked = videos.length > 0 && videos.every(v => st.selTask.has(v.id));
  all.addEventListener("click", e => {
    e.stopPropagation();
    if (all.checked) videos.forEach(v => st.selTask.add(v.id)); else videos.forEach(v => st.selTask.delete(v.id));
    renderTaskList(videos);
  });
  el.appendChild(t);
}
// ---- 识别任务：运行 ----
document.getElementById("btn-run").addEventListener("click", async () => {
  if (!st.selTask.size) { alert("请先勾选要提取的视频"); return; }
  const r = await post("/api/run", {ids:[...st.selTask]});
  if (!r.ok) { alert(r.error||"启动失败"); return; }
  $("#btn-run").disabled=true; $("#btn-cancel").disabled=false;
  poll(); if (st.timer) clearInterval(st.timer); st.timer=setInterval(poll, 800);
});
document.getElementById("btn-cancel").addEventListener("click", ()=>{ post("/api/cancel"); $("#btn-cancel").disabled=true; });
async function poll() {
  const s = await api("/api/status");
  const pct = s.total ? Math.round(100*s.progress/s.total) : 0;
  $("#fill").style.width = pct+"%";
  $("#status").textContent = s.running ? `[${s.progress}/${s.total}] ${s.name} · ${s.stage}` : s.stage;
  if (s.log && s.log.length) { $("#log").textContent = s.log.join("\n"); $("#log").scrollTop=$("#log").scrollHeight; }
  if (!s.running) {
    $("#btn-run").disabled=false; $("#btn-cancel").disabled=true;
    if (st.timer){clearInterval(st.timer); st.timer=null;}
    st.selTask.clear();
    refreshLibrary();
    refreshTask();
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
  const r = await post("/api/videos/reextract", {ids:[...st.sel]});
  if (r.ok) { alert("已标记重新提取，请到「识别任务」点开始提取"); st.sel.clear(); refreshLibrary(); }
});
document.getElementById("btn-delete").addEventListener("click", async () => {
  if (!st.sel.size) { alert("请先勾选视频"); return; }
  if (!confirm(`从库里删除选中的 ${st.sel.size} 个视频？（不会删除源文件）`)) return;
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
  const id = [...st.sel][0];
  const v = st.videos.find(x => x.id === id);
  if (!v) return;
  $("#preview-name").textContent = v.name;
  $("#pv-video").src = "/api/video/" + id;
  $("#pv-clip").src = "/api/clip/" + id;
  $("#pv-audio").src = "/api/audio/" + id;
  $("#preview-modal").style.display = "flex";
  loadPreviewFunscript(id);
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
  $("#pv-video").pause(); $("#pv-clip").pause(); $("#pv-audio").pause();
});
document.getElementById("preview-modal").addEventListener("click", e => {
  if (e.target.id === "preview-modal") {
    $("#preview-modal").style.display = "none";
    $("#pv-video").pause(); $("#pv-clip").pause(); $("#pv-audio").pause();
  }
});
document.getElementById("check-all").addEventListener("change", e => {
  const list = filteredVideos();
  if (e.target.checked) list.forEach(v=>st.sel.add(v.id)); else list.forEach(v=>st.sel.delete(v.id));
  renderVideos();
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
document.getElementById("fs-filter").addEventListener("change", e => { st.fsLevel = e.target.value; renderVideos(); });
document.getElementById("btn-reset-filter").addEventListener("click", () => {
  st.search = ""; $("#search").value = "";
  st.fMusic.clear(); st.fDance.clear(); st.filterCat = null;
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
      <td>${esc(v.name)}</td><td>${v.duration}s</td>
      <td>${v.music_name||"-"}</td>
      <td>${v.dance_name||"-"}</td>
      <td>${funscriptBadge(v)}</td>
      <td>${statusBadge(v.status)}</td>`;
    tr.addEventListener("click", e => {
      if (e.target.dataset.cb === "") { toggleSel(v.id); }
      else { toggleSel(v.id); }
      renderVideos();
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
  const src = st.anim ? "/api/thumb_gif/"+v.id : "/api/thumb/"+v.id;
  d.innerHTML = `<input type="checkbox" class="ck" ${checked}><img src="${src}" loading="lazy">
    <div class="meta"><div class="name">${esc(v.name)}</div>
    <div class="badges">${statusBadge(v.status)}${funscriptBadge(v)}${v.music_name?`<span class="badge">${esc(v.music_name)}</span>`:""}${v.dance_name?`<span class="badge">${esc(v.dance_name)}</span>`:""}</div></div>`;
  d.addEventListener("click", e => {
    if (e.target.type === "checkbox") { toggleSel(v.id); }
    else { toggleSel(v.id); }
    renderVideos();
  });
  return d;
}

// ---- 分组 ----
async function loadGroups(type) {
  const d = await api("/api/groups?type="+type);
  const el = document.getElementById(type+"-groups"); el.innerHTML="";
  if (!d.configured) { el.innerHTML='<div class="empty">请先设置缓存目录。</div>'; return; }
  if (!d.groups.length) { el.innerHTML='<div class="empty">还没有分组结果，去「识别任务」提取。</div>'; return; }
  d.groups.forEach(g => {
    const div=document.createElement("div"); div.className="group";
    const head=document.createElement("div"); head.className="group-head";
    const inp=document.createElement("input"); inp.type="text"; inp.value=g.name;
    inp.addEventListener("change", async ()=>{ await post("/api/groups/rename", {type, key:g.key, name:inp.value}); });
    head.appendChild(inp);
    const cnt=document.createElement("span"); cnt.className="muted"; cnt.textContent=g.members.length+" 个"; head.appendChild(cnt);
    div.appendChild(head);
    const rep = g.members[0];
    if (rep) {
      const pv = document.createElement("div"); pv.className = "group-preview";
      pv.innerHTML = `<video class="skel" controls preload="metadata" src="/api/preview/${rep.id}"></video>`;
      div.appendChild(pv);
    }
    const mem=document.createElement("div"); mem.className="members";
    g.members.forEach(m=>{
      const mm=document.createElement("div"); mm.className="member";
      const msrc = st.anim ? "/api/thumb_gif/" + m.id : "/api/thumb/" + m.id;
      mm.innerHTML=`<img src="${msrc}" loading="lazy"><div class="mname ${m.status==="no_pose"?"np":""}">${esc(m.name)}</div><div style="margin-top:3px">${funscriptBadge(m)}</div>`;
      mem.appendChild(mm);
    });
    div.appendChild(mem); el.appendChild(div);
  });
}

// ---- 启动 ----
(async function init(){ await refreshConfig(); if (st.configured) { switchView("task"); refreshTask(); } })();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
