"""端到端流水线：索引 → 音乐分组 → 动作分组 → 报告/导出。

分两阶段（准确度优先）：
  阶段一 index：逐视频切片段、提音频特征、提骨架特征（可断点续跑，结果落盘）
  阶段二 group：统一算音乐相似度 → 音乐分组 → 组内算动作距离 → 动作聚类
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import audio, clustering, dtw, features, ffmpeg, pose
from .config import AppConfig

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


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


@dataclass
class Result:
    records: list[Record]
    music_sim: np.ndarray
    music_labels: np.ndarray
    dance_dist: np.ndarray
    dance_labels: np.ndarray
    tempo: np.ndarray
    confidence: np.ndarray


def list_videos(root: str | Path) -> list[Path]:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    return sorted(files, key=lambda p: p.name.lower())


def _cache_dir(cfg: AppConfig, name: str) -> Path:
    d = Path(cfg.cache_dir) / _safe(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return keep[:80] or "video"


def index_video(path: Path, cfg: AppConfig, landmarker=None, ts_offset_ms: float = 0.0) -> Record:
    cdir = _cache_dir(cfg, path.stem)
    rec = Record(path=str(path), name=path.name, cache=str(cdir))
    info = ffmpeg.probe(path)
    rec.duration = info["duration"]
    rec.width, rec.height, rec.fps = info["width"], info["height"], info["fps"]
    rec.has_audio = info["has_audio"]

    try:
        # 1) 切片段（留档，便于人工核对与后续调参）
        clip = cdir / "clip.mp4"
        if not clip.exists():
            ffmpeg.cut_clip(path, clip, cfg.clip.start, cfg.clip.duration, cfg.clip.reencode)

        # 2) 音频特征
        wav = cdir / "audio.wav"
        if not wav.exists():
            ffmpeg.extract_audio(clip, wav, sample_rate=cfg.audio.sample_rate)
        afeat = audio.feature_from_audio(
            str(wav), sr=cfg.audio.sample_rate, hop_length=cfg.audio.hop_length,
            match_fps=cfg.audio.match_fps, max_lag_sec=cfg.audio.max_lag_sec,
        )
        np.savez_compressed(
            cdir / "audio.npz",
            chroma=afeat["chroma"], chroma_full=afeat["chroma_full"],
            tempo=np.array(afeat["tempo"]), duration=np.array(afeat["duration"]),
            fps=np.array(afeat["fps"]), max_lag=np.array(afeat["max_lag"]),
        )

        # 3) 骨架特征
        seq = pose.extract_pose(
            str(clip), cfg.pose.model_path,
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
            np.savez_compressed(cdir / "motion.npz", feat=feat, feat_mirror=feat_mirror,
                                fps=np.array(cfg.pose.target_fps),
                                present_ratio=np.array(present_ratio),
                                shots=np.array(seq.shots, dtype=np.float32).reshape(-1, 2))
            rec.status = "ok"
    except Exception as exc:  # 单个失败不拖垮整批
        rec.status = "error"
        rec.error = f"{type(exc).__name__}: {exc}"
    return rec


def load_records(cfg: AppConfig) -> dict[str, Record]:
    p = Path(cfg.cache_dir) / "records.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: Record(**v) for k, v in raw.items()}


def save_records(cfg: AppConfig, records: list[Record]) -> None:
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    data = {r.path: asdict(r) for r in records}
    (Path(cfg.cache_dir) / "records.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _close_async(landmarker) -> None:
    """MediaPipe 的 close() 会等遥测上传（约 60s），放后台线程里，不阻塞流程。"""
    import threading

    def _run():
        try:
            landmarker.close()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def index_all(data_dir: str | Path, cfg: AppConfig, progress=None) -> list[Record]:
    videos = list_videos(data_dir)
    cache = load_records(cfg)
    records: list[Record] = []
    landmarker = None
    ts_offset = 0.0
    try:
        for i, video in enumerate(videos, 1):
            old = cache.get(str(video))
            if old and old.status in ("ok", "no_pose") and (Path(old.cache) / "pose.npz").exists():
                records.append(old)
                if progress:
                    progress(i, len(videos), video.name, "cached")
                continue
            if progress:
                progress(i, len(videos), video.name, "extracting")
            if landmarker is None:
                landmarker = pose.create_landmarker(
                    cfg.pose.model_path,
                    num_poses=cfg.pose.num_poses,
                    min_detection_confidence=cfg.pose.min_detection_confidence,
                    min_presence_confidence=cfg.pose.min_presence_confidence,
                    min_tracking_confidence=cfg.pose.min_tracking_confidence,
                )
            rec = index_video(video, cfg, landmarker=landmarker, ts_offset_ms=ts_offset)
            records.append(rec)
            ts_offset += (rec.duration + 1.0) * 1000.0
            save_records(cfg, records)
    finally:
        if landmarker is not None:
            _close_async(landmarker)
    save_records(cfg, records)
    return records


def _load_audio(rec: Record) -> dict | None:
    p = Path(rec.cache) / "audio.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def _load_motion(rec: Record) -> dict:
    p = Path(rec.cache) / "motion.npz"
    if not p.exists():
        return {}
    d = np.load(p)
    return {k: d[k] for k in d.files}


def build_music_matrix(records: list[Record], cfg: AppConfig) -> np.ndarray:
    feats = [_load_audio(r) for r in records]
    n = len(feats)
    sim = np.eye(n, dtype=np.float32)
    for i in range(n):
        if feats[i] is None:
            continue
        for j in range(i + 1, n):
            if feats[j] is None:
                continue
            s = audio.similarity(feats[i]["chroma"], feats[j]["chroma"],
                                 max_lag=int(feats[i]["max_lag"]))
            sim[i, j] = sim[j, i] = s
    return sim


def fold_tempo(t: float) -> float:
    """把 BPM 折到 [70, 140)，消除倍频（60/120）误判。"""
    if t <= 0:
        return 0.0
    while t < 70:
        t *= 2
    while t >= 140:
        t /= 2
    return t


def build_dance_matrix(
    records: list[Record],
    music_labels: np.ndarray,
    cfg: AppConfig,
) -> np.ndarray:
    motions = [_load_motion(r) for r in records]
    tempos = np.array(
        [(_load_audio(r) or {}).get("tempo", 0.0) for r in records], dtype=np.float32
    )
    n = len(records)
    D = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(D, 0.0)

    for i in range(n):
        if "feat" not in motions[i]:
            continue
        for j in range(i + 1, n):
            if music_labels[i] != music_labels[j]:
                continue  # 只在同一首 BGM 内比动作
            if "feat" not in motions[j]:
                continue
            if _tempo_conflict(tempos[i], tempos[j], cfg):
                continue
            d = dtw.distance(
                motions[i]["feat"], motions[j]["feat"], motions[j]["feat_mirror"],
                window_ratio=cfg.match.dtw_window_ratio,
                mirror_invariant=cfg.match.mirror_invariant,
            )
            D[i, j] = D[j, i] = d
    return D


def _tempo_conflict(ta: float, tb: float, cfg: AppConfig, tol: float = 0.06) -> bool:
    """BPM 相差超过 tol 视为「加速/减速版」，不参与同组。"""
    fa, fb = fold_tempo(ta), fold_tempo(tb)
    if fa <= 0 or fb <= 0:
        return False
    return abs(fa - fb) / ((fa + fb) / 2) > tol


def group(records: list[Record], cfg: AppConfig) -> Result:
    music_sim = build_music_matrix(records, cfg)
    music_dist = 1.0 - music_sim
    music_labels = clustering.cluster_from_distances(
        music_dist, threshold=1.0 - cfg.match.music_threshold
    )

    dance_dist = build_dance_matrix(records, music_labels, cfg)
    dance_labels = _cluster_per_group(dance_dist, music_labels, cfg.match.dance_threshold)
    conf = _confidence(dance_dist, dance_labels)
    return Result(
        records=records,
        music_sim=music_sim,
        music_labels=music_labels,
        dance_dist=dance_dist,
        dance_labels=dance_labels,
        tempo=np.array(
            [(_load_audio(r) or {}).get("tempo", 0.0) for r in records], dtype=np.float32
        ),
        confidence=conf,
    )


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
