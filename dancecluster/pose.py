"""姿态提取：MediaPipe Tasks PoseLandmarker（33 点，world 坐标）。

针对你数据的几个关键处理：
- 2~3 人画面：自动选「主舞者」，并用 IoU 做时序跟踪防跳人；
- 镜头切换：检测切点，供下游只取主镜头段；
- 低置信点：标记缺失，下游按时间插值，避免抖动被放大成假差异。
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# matplotlib 会在媒体模块里被间接导入；给个可写的缓存目录，避免沙箱报警
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache-mpl"))

import mediapipe as mp  # noqa: E402

NUM_LANDMARKS = 33
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# MediaPipe Pose 骨架连线（用于叠加渲染）
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (15, 17), (16, 18), (17, 19),
    (18, 20), (27, 29), (28, 30), (29, 31), (30, 32),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
]

# MediaPipe 的左右对称点对（用于镜像）
MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6),          # 眼
    (7, 8),                          # 耳
    (9, 10),                         # 嘴
    (11, 12),                        # 肩
    (13, 14),                        # 肘
    (15, 16),                        # 腕
    (17, 18),                        # 小指
    (19, 20),                        # 食指
    (21, 22),                        # 拇指
    (23, 24),                        # 髋
    (25, 26),                        # 膝
    (27, 28),                        # 踝
    (29, 30),                        # 脚跟
    (31, 32),                        # 脚尖
]


@dataclass
class PoseSequence:
    world: np.ndarray          # (F, 33, 3) 世界坐标（米）
    visibility: np.ndarray     # (F, 33)
    present: np.ndarray        # (F,) 该帧是否检测到人
    fps: float                 # 实际采样帧率
    shots: list[tuple[float, float]]  # 镜头段（秒）
    person_score: np.ndarray   # (F,) 被选主舞者的得分，便于排查
    landmarks2d: np.ndarray = None  # (F, 33, 2) 归一化 2D 关键点（叠加渲染用）

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            world=self.world.astype(np.float32),
            visibility=self.visibility.astype(np.float32),
            present=self.present.astype(bool),
            fps=np.array(self.fps, dtype=np.float32),
            shots=np.array(self.shots, dtype=np.float32).reshape(-1, 2),
            person_score=self.person_score.astype(np.float32),
            landmarks2d=self.landmarks2d.astype(np.float32) if self.landmarks2d is not None else np.zeros((0, 33, 2), dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        d = np.load(Path(path))
        shots = [tuple(map(float, s)) for s in d["shots"]] if d["shots"].size else []
        return cls(
            world=d["world"],
            visibility=d["visibility"],
            present=d["present"],
            fps=float(d["fps"]),
            shots=shots,
            person_score=d["person_score"],
            landmarks2d=d["landmarks2d"] if d["landmarks2d"].size else None,
        )


def _bbox_2d(pts: np.ndarray) -> tuple[float, float, float, float]:
    xs, ys = pts[:, 0], pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = max((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter, 1e-9)
    return inter / union


_CORE = (11, 12, 23, 24)          # 肩 2 + 髋 2：必须大部分可见才算「一个完整的人」
_MIN_VALID = 12                   # 至少 12 个点可见，排除只露一条胳膊的残肢
_VIS_THRESHOLD = 0.5


def _hip_center(pts: np.ndarray, vis: np.ndarray) -> np.ndarray | None:
    if vis[23] > _VIS_THRESHOLD and vis[24] > _VIS_THRESHOLD:
        return (pts[23, :2] + pts[24, :2]) / 2.0
    return None


def _pick_person(
    norm_pts: list[np.ndarray],
    world_pts: list[np.ndarray],
    vis: list[np.ndarray],
    prev_bbox: tuple[float, float, float, float] | None,
    prev_hip: np.ndarray | None = None,
) -> int:
    """在多个候选里挑主舞者。

    过滤掉「残肢/不完整的人」：核心点（双肩+双髋）至少 3 个可见、总可见点 >= 12。
    bbox 只用可见点计算，避免低置信点把框撑大。
    时序上用 IoU + 髋中心距离双重约束，防止跳到画面里的其他人。
    """
    best_idx, best_score = -1, -1e9
    for i, pts in enumerate(norm_pts):
        v = vis[i]
        valid = v > _VIS_THRESHOLD
        n_valid = int(valid.sum())
        if n_valid < _MIN_VALID:
            continue
        if sum(1 for k in _CORE if v[k] > _VIS_THRESHOLD) < 3:
            continue

        p2 = pts[valid, :2]
        bbox = (float(p2[:, 0].min()), float(p2[:, 1].min()),
                float(p2[:, 0].max()), float(p2[:, 1].max()))
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        area = max(0.0, w) * max(0.0, h)
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        centered = 1.0 - min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5 / 0.7071)
        score = 2.0 * area + 0.8 * centered + 1.0 * float(v[valid].mean())

        if prev_bbox is not None:
            score += 1.5 * _iou(bbox, prev_bbox)
            hip = _hip_center(pts, v)
            if prev_hip is not None and hip is not None:
                d = float(np.linalg.norm(hip - prev_hip))
                score -= 2.5 * min(d, 1.0)     # 距离惩罚：跳太远基本可以排除

        if score > best_score:
            best_score, best_idx = score, i
    return best_idx


def _enhance_low_light(bgr: np.ndarray, mean_threshold: float = 95.0) -> np.ndarray:
    """低照度提亮：CLAHE + 自适应 gamma。几何不变，因此 2D 坐标可直接叠加回原视频。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean >= mean_threshold:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    out = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    m = float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean())
    target = 118.0
    if 1.0 < m < target:
        # out = (i/255)^exp，exp<1 变亮；由「目标均值」反解 exp
        exp = float(np.log(target / 255.0) / np.log(m / 255.0))
        exp = min(1.0, max(0.35, exp))
        lut = np.array(
            [min(255, max(0, int(((i / 255.0) ** exp) * 255.0))) for i in range(256)],
            dtype=np.uint8,
        )
        out = cv2.LUT(out, lut)
    return out


def _is_shot_cut(prev_gray: np.ndarray | None, gray: np.ndarray, threshold: float = 0.22) -> bool:
    if prev_gray is None:
        return False
    small_prev = cv2.resize(prev_gray, (64, 36)).astype(np.float32)
    small = cv2.resize(gray, (64, 36)).astype(np.float32)
    return float(np.mean(np.abs(small_prev - small))) / 255.0 > threshold


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """沿时间轴滑动平均；只对存在的帧做，避免把缺失帧抹成 0。"""
    if window <= 1 or x.shape[0] < 3:
        return x
    k = int(window) | 1
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    out = np.empty_like(x)
    for f in range(x.shape[0]):
        out[f] = np.tensordot(kernel, xp[f:f + k], axes=(0, 0))
    return out


def create_landmarker(
    model_path: str | Path,
    num_poses: int = 3,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """创建并复用 PoseLandmarker。批量处理时只创建一次，避免每个视频都加载模型。"""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"姿态模型不存在：{model_path}")
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=int(num_poses),
        min_pose_detection_confidence=float(min_detection_confidence),
        min_pose_presence_confidence=float(min_presence_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_pose(
    video: str | Path,
    model_path: str | Path,
    num_poses: int = 3,
    analysis_fps: float = 15.0,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    smooth_window: int = 5,
    max_seconds: float | None = None,
    progress: callable | None = None,
    landmarker=None,
    ts_offset_ms: float = 0.0,
) -> PoseSequence:
    from contextlib import ExitStack

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"姿态模型不存在：{model_path}")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=int(num_poses),
        min_pose_detection_confidence=float(min_detection_confidence),
        min_pose_presence_confidence=float(min_presence_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / analysis_fps)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    worlds, viss, presents, scores, l2ds = [], [], [], [], []
    shots: list[tuple[float, float]] = []
    shot_start = 0.0
    prev_gray = None
    prev_bbox = None
    prev_hip = None
    frame_idx = 0
    out_idx = 0

    with ExitStack() as stack:
        if landmarker is None:
            landmarker = stack.enter_context(vision.PoseLandmarker.create_from_options(options))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / src_fps
            if max_seconds is not None and t > max_seconds:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if _is_shot_cut(prev_gray, gray):
                shots.append((shot_start, t))
                shot_start = t
                prev_bbox = None
                prev_hip = None
            prev_gray = gray

            rgb = cv2.cvtColor(_enhance_low_light(frame), cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(round(t * 1000 + ts_offset_ms)))

            norm_pts, world_pts, vis_list = [], [], []
            for pose_norm, pose_world in zip(
                result.pose_landmarks or [], result.pose_world_landmarks or []
            ):
                norm_pts.append(np.array([[l.x, l.y, l.z] for l in pose_norm], dtype=np.float32))
                world_pts.append(np.array([[l.x, l.y, l.z] for l in pose_world], dtype=np.float32))
                vis_list.append(np.array([getattr(l, "visibility", 1.0) for l in pose_norm], dtype=np.float32))

            idx = _pick_person(norm_pts, world_pts, vis_list, prev_bbox, prev_hip) if world_pts else -1
            if idx >= 0:
                bbox = _bbox_2d(norm_pts[idx])
                prev_bbox = bbox
                prev_hip = _hip_center(norm_pts[idx], vis_list[idx])
                worlds.append(world_pts[idx])
                viss.append(vis_list[idx])
                presents.append(True)
                scores.append(1.0)
                l2ds.append(norm_pts[idx][:, :2])
            else:
                worlds.append(np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))
                viss.append(np.zeros(NUM_LANDMARKS, dtype=np.float32))
                presents.append(False)
                scores.append(0.0)
                l2ds.append(np.zeros((NUM_LANDMARKS, 2), dtype=np.float32))

            out_idx += 1
            frame_idx += 1
            if progress and out_idx % 30 == 0:
                progress(out_idx, total // step)

    cap.release()
    shots.append((shot_start, frame_idx / src_fps))

    world = np.stack(worlds) if worlds else np.zeros((0, NUM_LANDMARKS, 3), np.float32)
    vis = np.stack(viss) if viss else np.zeros((0, NUM_LANDMARKS), np.float32)
    present = np.array(presents, dtype=bool)
    score = np.array(scores, dtype=np.float32)
    l2d = np.stack(l2ds) if l2ds else np.zeros((0, NUM_LANDMARKS, 2), np.float32)

    if world.shape[0] >= 3:
        smooth = _smooth(world, smooth_window)
        world = np.where(present[:, None, None], smooth, world)

    return PoseSequence(
        world=world.astype(np.float32),
        visibility=vis.astype(np.float32),
        present=present,
        fps=float(src_fps / step),
        shots=shots,
        person_score=score,
        landmarks2d=l2d.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# 骨架叠加渲染：在原画面上画出识别到的关键点 + 连线，输出 GIF / 静态图
# ---------------------------------------------------------------------------
def _draw_skeleton(frame, pts, vis, w, h):
    px = (np.asarray(pts)[:, 0] * w).astype(np.int32)
    py = (np.asarray(pts)[:, 1] * h).astype(np.int32)
    ok = np.asarray(vis) > 0.3
    for a, b in POSE_CONNECTIONS:
        if ok[a] and ok[b]:
            cv2.line(frame, (px[a], py[a]), (px[b], py[b]), (0, 255, 0), 2)
    for i in range(NUM_LANDMARKS):
        if ok[i]:
            cv2.circle(frame, (px[i], py[i]), 2, (0, 255, 0), -1)
    return frame


def render_skeleton_gif(pose_path, out_gif, fps=6, max_seconds=8, width=320, aspect=0.75):
    """把骨架（仅关键点数据）渲染成黑底循环 GIF。不依赖视频片段。"""
    from PIL import Image

    seq = PoseSequence.load(pose_path)
    l2d = seq.landmarks2d
    if l2d is None or len(l2d) < 2:
        return None
    h = max(1, int(round(width / aspect))) if aspect > 0 else int(width * 0.75)
    step = max(1, int(round(seq.fps / fps)))
    frames = []
    for k in range(0, len(l2d), step):
        vis = seq.visibility[k] if k < len(seq.visibility) else np.ones(NUM_LANDMARKS)
        canvas = np.zeros((h, width, 3), np.uint8)
        _draw_skeleton(canvas, l2d[k], vis, width, h)
        frames.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        if len(frames) >= max_seconds * fps:
            break
    if len(frames) < 2:
        return None
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    return out_gif


def render_skeleton_frame(pose_path, out_img, width=320, aspect=0.75):
    """渲染单帧骨架图（黑底静态）。不依赖视频片段。"""
    from PIL import Image

    seq = PoseSequence.load(pose_path)
    l2d = seq.landmarks2d
    if l2d is None or len(l2d) < 1:
        return None
    h = max(1, int(round(width / aspect))) if aspect > 0 else int(width * 0.75)
    k = len(l2d) // 2
    vis = seq.visibility[k] if k < len(seq.visibility) else np.ones(NUM_LANDMARKS)
    canvas = np.zeros((h, width, 3), np.uint8)
    _draw_skeleton(canvas, l2d[k], vis, width, h)
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    out_img = Path(out_img)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img)
    return out_img


def render_preview_video(pose_path, audio_path, out_mp4, fps=6, width=320, aspect=0.75, max_seconds=12):
    """骨架动作 + 音频，合成一条带声音的预览视频（mp4）。失败返回 None。"""
    import shutil
    import subprocess
    import tempfile

    from . import ffmpeg as _ffmpeg

    seq = PoseSequence.load(pose_path)
    l2d = seq.landmarks2d
    if l2d is None or len(l2d) < 2:
        return None
    h = max(2, int(round(width / aspect))) if aspect > 0 else int(width * 0.75)
    h -= h % 2  # libx264 要求偶数尺寸
    step = max(1, int(round(seq.fps / fps)))
    tmpdir = Path(tempfile.mkdtemp())
    try:
        idx = 0
        max_frames = int(max_seconds * fps)
        for k in range(0, len(l2d), step):
            vis = seq.visibility[k] if k < len(seq.visibility) else np.ones(NUM_LANDMARKS)
            canvas = np.zeros((h, width, 3), np.uint8)
            _draw_skeleton(canvas, l2d[k], vis, width, h)
            cv2.imwrite(str(tmpdir / f"f{idx:04d}.png"), canvas)
            idx += 1
            if idx >= max_frames:
                break
        if idx < 2:
            return None
        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            _ffmpeg.ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(tmpdir / "f%04d.png"),
            "-i", str(audio_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "96k", "-shortest", str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True, creationflags=_NO_WINDOW)
        return out_mp4
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
