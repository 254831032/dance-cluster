"""端到端流水线：索引 → 音乐分组 → 动作分组 → 报告/导出。

数据全部落在一个 SQLite 文件里（<cache>/dancecluster.db）：
  * 断点续跑：已提取的视频直接复用
  * 分组结果写回库，方便以后做「手动合并/拆分、重新导出」
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio, clustering, dtw, features, ffmpeg, funscript, pose
from .config import AppConfig
from .db import Store

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# 特征算法版本：改动特征计算后 +1，老缓存会自动被标记重提取
FEATURE_VERSION = 2
# 模型是特征的一部分：切换模型 → 版本号变化 → 自动强制重新提取
_MODEL_CODES = {"heavy": 0, "full": 1, "lite": 2}


def _version_for(model: str) -> int:
    return FEATURE_VERSION * 10 + _MODEL_CODES.get(model, 0)


@dataclass
class Record:
    path: str
    name: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    status: str = "pending"      # ok / no_pose / no_audio / error
    error: str = ""
    cache: str = ""
    id: int = 0


@dataclass
class Result:
    records: list[Record]
    music_sim: np.ndarray
    music_labels: np.ndarray
    dance_dist: np.ndarray
    dance_labels: np.ndarray
    tempo: np.ndarray
    confidence: np.ndarray
    db_path: str = ""


def list_videos(root: str | Path) -> list[Path]:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    return sorted(files, key=lambda p: p.name.lower())


def db_path(cfg: AppConfig) -> Path:
    return Path(cfg.cache_dir) / "dancecluster.db"


def _cache_dir(cfg: AppConfig, name: str) -> Path:
    d = Path(cfg.cache_dir) / _safe(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return keep[:80] or "video"


def _record_from_row(row: dict) -> Record:
    return Record(
        id=int(row["id"]),
        path=row["path"],
        name=row["name"],
        duration=float(row["duration"] or 0),
        width=int(row["width"] or 0),
        height=int(row["height"] or 0),
        fps=float(row["fps"] or 0),
        has_audio=bool(row["has_audio"]),
        status=row["status"] or "pending",
        error=row["error"] or "",
        cache=row["cache_dir"] or "",
    )


def index_video(
    path: Path,
    cfg: AppConfig,
    store: Store,
    landmarker=None,
    ts_offset_ms: float = 0.0,
) -> Record:
    cdir = _cache_dir(cfg, path.stem)
    info = ffmpeg.probe(path)
    vid = store.upsert_video(
        path=str(path), name=path.name, duration=info["duration"],
        width=info["width"], height=info["height"], fps=info["fps"],
        has_audio=int(bool(info["has_audio"])), cache_dir=str(cdir), status="extracting",
    )
    rec = Record(
        id=vid, path=str(path), name=path.name, duration=info["duration"],
        width=info["width"], height=info["height"], fps=info["fps"],
        has_audio=bool(info["has_audio"]), cache=str(cdir),
    )

    try:
        # 1) 切片段（留档，便于人工核对与后续调参）
        clip = cdir / "clip.mp4"
        if not clip.exists():
            ffmpeg.cut_clip(path, clip, cfg.clip.start, cfg.clip.duration, cfg.clip.reencode, max_height=cfg.clip.max_height)

        # 2) 音频（从原视频抽，长度独立于骨架片段）
        wav = cdir / "audio.wav"
        if not wav.exists():
            adur = cfg.clip.audio_duration if cfg.clip.audio_duration > 0 else None
            ffmpeg.extract_audio(path, wav, start=0.0, duration=adur, sample_rate=cfg.audio.sample_rate)
        afeat = audio.feature_from_audio(
            str(wav), sr=cfg.audio.sample_rate, hop_length=cfg.audio.hop_length,
            match_fps=cfg.audio.match_fps, max_lag_sec=cfg.audio.max_lag_sec,
        )
        fp = audio.fingerprint(str(wav))
        # 音频特征已入库，wav 只留给预览 → 压成 AAC 省空间
        m4a = cdir / "audio.m4a"
        if not m4a.exists():
            try:
                ffmpeg.to_aac(wav, m4a, bitrate="96k")
                wav.unlink(missing_ok=True)
            except Exception:
                pass
        store.set_audio(
            vid,
            {
                "chroma": afeat["chroma"],
                "chroma_full": afeat["chroma_full"],
                "tempo": np.array(afeat["tempo"]),
                "duration": np.array(afeat["duration"]),
                "fps": np.array(afeat["fps"]),
                "max_lag": np.array(afeat["max_lag"]),
                "fingerprint": fp if fp is not None else np.zeros((0, 32), dtype=np.uint8),
            },
            tempo=afeat["tempo"],
        )

        # 3) 骨架特征
        seq = pose.extract_pose(
            str(clip), cfg.pose.model_file(),
            num_poses=cfg.pose.num_poses,
            analysis_fps=max(cfg.pose.target_fps, 10.0),
            min_detection_confidence=cfg.pose.min_detection_confidence,
            min_presence_confidence=cfg.pose.min_presence_confidence,
            min_tracking_confidence=cfg.pose.min_tracking_confidence,
            smooth_window=cfg.pose.smooth_window,
            landmarker=landmarker,
            ts_offset_ms=ts_offset_ms,
        )
        seq.save(cdir / "pose.npz")
        feat, feat_mirror = features.prepare_features(
            seq.world, seq.visibility, seq.present, seq.fps,
            target_fps=cfg.pose.target_fps, min_visibility=cfg.pose.min_visibility,
        )
        present_ratio = float(seq.present.mean())
        # 要求「大部分时间都有人」，否则宁可不比，避免把零星误检当成动作
        if present_ratio < 0.35 or feat.shape[0] < 30:
            rec.status = "no_pose"
            rec.error = f"人体检测不稳定（present={present_ratio:.0%}, frames={feat.shape[0]}）"
        else:
            store.set_motion(
                vid,
                {
                    "feat": feat,
                    "feat_mirror": feat_mirror,
                    "fps": np.array(cfg.pose.target_fps),
                    "present_ratio": np.array(present_ratio),
                    "shots": np.array(seq.shots, dtype=np.float32).reshape(-1, 2),
                },
                present_ratio=present_ratio,
            )
            rec.status = "ok"
    except Exception as exc:  # 单个失败不拖垮整批
        rec.status = "error"
        rec.error = f"{type(exc).__name__}: {exc}"

    store.set_status(vid, rec.status, rec.error, feature_version=_version_for(cfg.pose.model))
    return rec


def ensure_video_cache(row: dict, cfg: AppConfig) -> Path:
    """按需补建某个视频的中间文件（clip / pose / audio），供预览与骨架渲染用。

    不写数据库特征、不改变分组结果，只补磁盘文件；缺失才重建，存在则跳过。
    源视频不存在时抛 FileNotFoundError。
    """
    src = Path(row["path"])
    if not src.exists():
        raise FileNotFoundError(f"源视频不存在：{src}")
    cdir = Path(row["cache_dir"])
    cdir.mkdir(parents=True, exist_ok=True)

    clip = cdir / "clip.mp4"
    if not clip.exists():
        ffmpeg.cut_clip(
            src, clip, cfg.clip.start, cfg.clip.duration,
            cfg.clip.reencode, max_height=cfg.clip.max_height,
        )

    m4a = cdir / "audio.m4a"
    wav = cdir / "audio.wav"
    if not m4a.exists() and not wav.exists():
        adur = cfg.clip.audio_duration if cfg.clip.audio_duration > 0 else None
        ffmpeg.extract_audio(src, wav, start=0.0, duration=adur, sample_rate=cfg.audio.sample_rate)
    if not m4a.exists() and wav.exists():
        try:
            ffmpeg.to_aac(wav, m4a, bitrate="96k")
        except Exception:
            pass

    pose_npz = cdir / "pose.npz"
    if not pose_npz.exists():
        # 显式创建 landmarker 并后台关闭，避免 extract_pose 内部 close() 阻塞 ~60s 遥测
        lm = pose.create_landmarker(
            cfg.pose.model_file(),
            num_poses=cfg.pose.num_poses,
            min_detection_confidence=cfg.pose.min_detection_confidence,
            min_presence_confidence=cfg.pose.min_presence_confidence,
            min_tracking_confidence=cfg.pose.min_tracking_confidence,
        )
        try:
            seq = pose.extract_pose(
                str(clip), cfg.pose.model_file(),
                num_poses=cfg.pose.num_poses,
                analysis_fps=max(cfg.pose.target_fps, 10.0),
                min_detection_confidence=cfg.pose.min_detection_confidence,
                min_presence_confidence=cfg.pose.min_presence_confidence,
                min_tracking_confidence=cfg.pose.min_tracking_confidence,
                smooth_window=cfg.pose.smooth_window,
                landmarker=lm,
            )
            seq.save(pose_npz)
        finally:
            _close_async(lm)
    return cdir


def _close_async(landmarker) -> None:
    """MediaPipe 的 close() 会等遥测上传（约 60s），放后台线程里，不阻塞流程。"""
    import threading

    def _run():
        try:
            landmarker.close()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def _funscript_fields(video_path) -> dict:
    """扫描视频同目录的 funscript，返回要写库的字段。"""
    try:
        info = funscript.scan(video_path)
    except Exception:
        info = {"count": 0, "axes": []}
    axes = info.get("axes") or []
    return {
        "funscript_count": int(info.get("count", 0)),
        "funscript_axes": json.dumps(axes, ensure_ascii=False) if axes else None,
    }


def scan_funscripts(ids: list[int], cfg: AppConfig) -> dict:
    """手动批量扫描：为指定视频重新关联 funscript。"""
    store = Store(db_path(cfg))
    stats = {"scanned": 0, "with_script": 0, "single": 0, "multi": 0, "none": 0}
    try:
        for vid in ids:
            row = store.get_by_id(int(vid))
            if not row:
                continue
            try:
                info = funscript.scan(Path(row["path"]))
            except Exception:
                info = {"count": 0, "axes": []}
            store.set_funscript(int(vid), info["count"], info["axes"])
            stats["scanned"] += 1
            if info["count"] <= 0:
                stats["none"] += 1
            elif info["count"] == 1:
                stats["single"] += 1
                stats["with_script"] += 1
            else:
                stats["multi"] += 1
                stats["with_script"] += 1
        return stats
    finally:
        store.close()


def _discover(folders: list, cfg: AppConfig, store: Store) -> dict:
    """把文件夹里的视频登记为 pending（只登记，不提取）。返回统计。"""
    stats = {"scanned": 0, "added": 0, "already": 0}
    for folder in folders:
        for p in list_videos(folder):
            stats["scanned"] += 1
            row = store.get_by_path(str(p))
            if row is None:
                info = ffmpeg.probe(p)
                store.upsert_video(
                    path=str(p), name=p.name, duration=info["duration"],
                    width=info["width"], height=info["height"], fps=info["fps"],
                    has_audio=int(bool(info["has_audio"])),
                    cache_dir=str(_cache_dir(cfg, p.stem)), status="pending",
                    **_funscript_fields(p),
                )
                stats["added"] += 1
            elif row["status"] in ("ok", "no_pose"):
                stats["already"] += 1
    return stats


def register_folder(folder: str | Path, cfg: AppConfig) -> dict:
    """扫描文件夹、把视频登记为待提取任务（不保存文件夹本身）。"""
    store = Store(db_path(cfg))
    try:
        return _discover([str(Path(folder).resolve())], cfg, store)
    finally:
        store.close()


def register_videos(paths: list, cfg: AppConfig) -> dict:
    store = Store(db_path(cfg))
    stats = {"scanned": 0, "added": 0, "already": 0}
    try:
        for p in paths:
            p = Path(p)
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            stats["scanned"] += 1
            row = store.get_by_path(str(p))
            if row is None:
                info = ffmpeg.probe(p)
                store.upsert_video(
                    path=str(p), name=p.name, duration=info["duration"],
                    width=info["width"], height=info["height"], fps=info["fps"],
                    has_audio=int(bool(info["has_audio"])),
                    cache_dir=str(_cache_dir(cfg, p.stem)), status="pending",
                    **_funscript_fields(p),
                )
                stats["added"] += 1
            elif row["status"] in ("ok", "no_pose"):
                stats["already"] += 1
        return stats
    finally:
        store.close()


def _worker_count(n: int) -> int:
    """并行度：CPU 一半，最多 4，且不超过待处理数。"""
    try:
        cpu = os.cpu_count() or 4
    except Exception:
        cpu = 4
    return max(1, min(4, cpu // 2, n))


def process_pending(cfg: AppConfig, progress=None, only_ids: list | None = None) -> list[Record]:
    """处理待提取（pending/error）的视频；only_ids 指定只处理选中的。

    多个视频时用线程池并行（每个 worker 独立 SQLite 连接 + 独立 PoseLandmarker），
    实测 3 个视频在 10 核上约 1.8x 加速。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    store = Store(db_path(cfg))
    try:
        store.reset_stale_extracting()
        pending = [
            r for r in store.all_videos()
            if r["status"] in ("pending", "error")
            or (r["status"] in ("ok", "no_pose") and (r["feature_version"] or 0) != _version_for(cfg.pose.model))
        ]
        if only_ids is not None:
            idset = set(int(x) for x in only_ids)
            pending = [r for r in pending if int(r["id"]) in idset]
    finally:
        store.close()

    total = len(pending)
    records: list[Record] = []
    if total == 0:
        store = Store(db_path(cfg))
        try:
            return [_record_from_row(r) for r in store.all_videos()
                    if r["status"] in ("ok", "no_pose")]
        finally:
            store.close()

    lock = threading.Lock()
    stop = threading.Event()
    done = {"n": 0}

    def _run_chunk(chunk):
        s = Store(db_path(cfg))
        lm = None
        ts = 0.0
        try:
            lm = pose.create_landmarker(
                cfg.pose.model_file(),
                num_poses=cfg.pose.num_poses,
                min_detection_confidence=cfg.pose.min_detection_confidence,
                min_presence_confidence=cfg.pose.min_presence_confidence,
                min_tracking_confidence=cfg.pose.min_tracking_confidence,
            )
            for row in chunk:
                if stop.is_set():
                    break
                video = Path(row["path"])
                rec = index_video(video, cfg, s, landmarker=lm, ts_offset_ms=ts)
                ts += (rec.duration + 1.0) * 1000.0
                with lock:
                    records.append(rec)
                    done["n"] += 1
                    n = done["n"]
                try:
                    if progress:
                        progress(n, total, video.name, "extracting")
                except BaseException:
                    stop.set()
                    break
        except Exception:
            pass  # 单个 worker 失败不影响整批；剩余视频保持 pending，下次重试
        finally:
            if lm is not None:
                _close_async(lm)
            s.close()

    workers = _worker_count(total)
    if workers <= 1:
        _run_chunk(pending)
    else:
        chunks = [pending[i::workers] for i in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_run_chunk, [c for c in chunks if c]))

    store = Store(db_path(cfg))
    try:
        done_ids = {r.id for r in records}
        for r in store.all_videos():
            if r["id"] not in done_ids and r["status"] in ("ok", "no_pose"):
                records.append(_record_from_row(r))
    finally:
        store.close()
    records.sort(key=lambda r: r.name.lower())
    return records


def index_all(data_dir: str | Path | list, cfg: AppConfig, progress=None) -> list[Record]:
    """命令行用：扫描文件夹 → 登记 → 处理待提取。"""
    folders = [data_dir] if isinstance(data_dir, (str, Path)) else list(data_dir)
    store = Store(db_path(cfg))
    try:
        store.reset_stale_extracting()
        _discover(folders, cfg, store)
    finally:
        store.close()
    return process_pending(cfg, progress)


def clear_all(cfg: AppConfig) -> None:
    """清空缓存目录下所有数据（数据库 + 每个视频的中间文件 + 缩略图）。保留缓存目录本身。"""
    import shutil

    base = Path(cfg.cache_dir)
    if base.exists():
        for item in base.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception:
                pass


def all_records(cfg: AppConfig) -> list[Record]:
    """从库里取出全部已索引视频（不限于某个文件夹）。"""
    store = Store(db_path(cfg))
    try:
        return [_record_from_row(r) for r in store.all_videos()]
    finally:
        store.close()


def fold_tempo(t: float) -> float:
    """把 BPM 折到 [70, 140)，消除倍频（60/120）误判。"""
    if t <= 0:
        return 0.0
    while t < 70:
        t *= 2
    while t >= 140:
        t /= 2
    return t


def _tempo_conflict(ta: float, tb: float, tol: float = 0.06) -> bool:
    """BPM 相差超过 tol 视为「加速/减速版」，不参与同组。"""
    fa, fb = fold_tempo(ta), fold_tempo(tb)
    if fa <= 0 or fb <= 0:
        return False
    return abs(fa - fb) / ((fa + fb) / 2) > tol


def group(records: list[Record], cfg: AppConfig) -> Result:
    store = Store(db_path(cfg))
    try:
        rows = [store.get_by_path(r.path) for r in records]
        audios = [store.audio_features(row) if row and row.get("audio_npz") else None for row in rows]
        motions = [store.motion_features(row) if row and row.get("motion_npz") else {} for row in rows]

        # 老缓存没有指纹 → 从 audio.wav 现场补算
        for i, r in enumerate(records):
            a = audios[i]
            if a is None:
                continue
            fp = a.get("fingerprint")
            if fp is None or (hasattr(fp, "shape") and fp.shape[0] == 0):
                wav = Path(r.cache) / "audio.wav"
                if wav.exists():
                    a["fingerprint"] = audio.fingerprint(str(wav))
        tempos = np.array(
            [float(a["tempo"]) if a else 0.0 for a in audios], dtype=np.float32
        )

        ids = [r.id for r in records]
        music_sim = _music_matrix(audios, ids, store)
        music_labels = clustering.cluster_from_distances(
            1.0 - music_sim, threshold=1.0 - cfg.match.music_threshold
        )
        dance_dist = _dance_matrix(motions, tempos, music_labels, ids, cfg, store)
        dance_labels = _cluster_per_group(dance_dist, music_labels, cfg.match.dance_threshold)
        conf = _confidence(dance_dist, dance_labels)

        # 没有动作特征（未检测到骨架）的视频不参与动作分组
        has_motion = np.array(["feat" in m for m in motions], dtype=bool)
        dance_labels = np.where(has_motion, dance_labels, 0)
        conf = np.where(has_motion, conf, 0.0)

        store.set_groups([
            (records[i].id, int(music_labels[i]),
             int(dance_labels[i]) if dance_labels[i] > 0 else None,
             float(conf[i]) if dance_labels[i] > 0 else None)
            for i in range(len(records))
        ])
        store.ensure_groups("music", set(int(x) for x in music_labels))
        store.ensure_groups("dance", set(int(x) for x in dance_labels if x > 0))
        return Result(
            records=records,
            music_sim=music_sim,
            music_labels=music_labels,
            dance_dist=dance_dist,
            dance_labels=dance_labels,
            tempo=tempos,
            confidence=conf,
            db_path=str(db_path(cfg)),
        )
    finally:
        store.close()


def _music_matrix(audios: list[dict | None], ids: list[int], store: Store, n_jobs: int | None = None) -> np.ndarray:
    """音乐相似度矩阵：带 pair_cache 缓存 + joblib 线程并行。"""
    from joblib import Parallel, delayed

    n = len(audios)
    sim = np.eye(n, dtype=np.float32)
    pairs = []
    for i in range(n):
        if audios[i] is None:
            continue
        for j in range(i + 1, n):
            if audios[j] is None:
                continue
            v = store.get_pair(ids[i], ids[j], "music")
            if v is not None:
                sim[i, j] = sim[j, i] = v
            else:
                pairs.append((i, j))

    if pairs:
        def _c(ij):
            i, j = ij
            return i, j, audio.music_similarity(audios[i], audios[j])

        results = Parallel(n_jobs=n_jobs or min(4, os.cpu_count() or 4), prefer="threads")(
            delayed(_c)(p) for p in pairs
        )
        for i, j, v in results:
            sim[i, j] = sim[j, i] = v
            store.set_pair(ids[i], ids[j], "music", v)
    return sim


def _dance_matrix(
    motions: list[dict],
    tempos: np.ndarray,
    music_labels: np.ndarray,
    ids: list[int],
    cfg: AppConfig,
    store: Store,
    n_jobs: int | None = None,
) -> np.ndarray:
    """动作 DTW 距离矩阵：带 pair_cache 缓存 + joblib 线程并行。"""
    from joblib import Parallel, delayed

    n = len(motions)
    D = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    pairs = []
    for i in range(n):
        if "feat" not in motions[i]:
            continue
        for j in range(i + 1, n):
            if music_labels[i] != music_labels[j]:
                continue  # 只在同一首 BGM 内比动作
            if "feat" not in motions[j]:
                continue
            if _tempo_conflict(tempos[i], tempos[j]):
                continue
            v = store.get_pair(ids[i], ids[j], "dance")
            if v is not None:
                D[i, j] = D[j, i] = v
            else:
                pairs.append((i, j))

    if pairs:
        def _c(ij):
            i, j = ij
            d = dtw.distance(
                motions[i]["feat"], motions[j]["feat"], motions[j]["feat_mirror"],
                window_ratio=cfg.match.dtw_window_ratio,
                mirror_invariant=cfg.match.mirror_invariant,
            )
            return i, j, d

        results = Parallel(n_jobs=n_jobs or min(4, os.cpu_count() or 4), prefer="threads")(
            delayed(_c)(p) for p in pairs
        )
        for i, j, d in results:
            D[i, j] = D[j, i] = d
            store.set_pair(ids[i], ids[j], "dance", float(d))
    return D


def _cluster_per_group(D: np.ndarray, music_labels: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.zeros(len(D), dtype=int)
    next_id = 1
    for grp in clustering.order_groups(music_labels):
        idx = np.array(grp)
        sub = D[np.ix_(idx, idx)]
        # 缺失（inf）的对先当作“最远”，让层次聚类能跑
        sub = np.where(np.isfinite(sub), sub, 1e3)
        local = clustering.cluster_from_distances(sub, threshold=threshold)
        labels[idx] = local + next_id - 1
        next_id += int(local.max())
    return labels


def _confidence(D: np.ndarray, dance_labels: np.ndarray) -> np.ndarray:
    """组内距离越近、离最近的外组越远，置信度越高。"""
    Df = np.where(np.isfinite(D), D, 1e3)
    return clustering.confidence(Df, dance_labels)


def export(result: Result, out_dir: str | Path, copy_files: bool = False) -> Path:
    """输出 CSV + 按分组建目录（默认硬链接，跨盘自动退化为复制）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = ["文件,音乐组,动作组,置信度,BPM,状态"]
    for i, r in enumerate(result.records):
        lines.append(
            f'"{r.name}",M{result.music_labels[i]},D{result.dance_labels[i]},'
            f'{result.confidence[i]:.2f},{result.tempo[i]:.1f},{r.status}'
        )
    (out / "分组结果.csv").write_text("\n".join(lines), encoding="utf-8")

    # 把 SQLite 数据库也拷一份到输出目录，数据和结果一起走
    if result.db_path and Path(result.db_path).exists():
        shutil.copy2(result.db_path, out / "分组数据.db")

    for i, r in enumerate(result.records):
        target = out / f"M{result.music_labels[i]}" / f"D{result.dance_labels[i]}"
        target.mkdir(parents=True, exist_ok=True)
        src, dst = Path(r.path), target / r.name
        if dst.exists():
            continue
        if copy_files:
            shutil.copy2(src, dst)
        else:
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
    return out
