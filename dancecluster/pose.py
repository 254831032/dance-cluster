"""姿态提取：MediaPipe Tasks PoseLandmarker（33 点，world 坐标）。

针对你数据的几个关键处理：
- 2~3 人画面：自动选「主舞者」，并用 IoU 做时序跟踪防跳人；
- 镜头切换：检测切点，供下游只取主镜头段；
- 低置信点：标记缺失，下游按时间插值，避免抖动被放大成假差异。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# matplotlib 会在媒体模块里被间接导入；给个可写的缓存目录，避免沙箱报警
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache-mpl"))

import mediapipe as mp  # noqa: E402

NUM_LANDMARKS = 33

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


def _pick_person(
    norm_pts: list[np.ndarray],
    world_pts: list[np.ndarray],
    vis: list[np.ndarray],
    prev_bbox: tuple[float, float, float, float] | None,
) -> int:
    """在多个候选里挑主舞者：面积 + 居中 + 可见度 + 时序连续性。"""
    best_idx, best_score = -1, -1e9
    for i, pts in enumerate(norm_pts):
        bbox = _bbox_2d(pts)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        area = max(0.0, w) * max(0.0, h)
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        centered = 1.0 - min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5 / 0.7071)
        score = 2.0 * area + 0.8 * centered + 1.0 * float(np.mean(vis[i]))
        if prev_bbox is not None:
            score += 1.5 * _iou(bbox, prev_bbox)
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx


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

    worlds, viss, presents, scores = [], [], [], []
    shots: list[tuple[float, float]] = []
    shot_start = 0.0
    prev_gray = None
    prev_bbox = None
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
            prev_gray = gray

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(round(t * 1000 + ts_offset_ms)))

            norm_pts, world_pts, vis_list = [], [], []
            for pose_norm, pose_world in zip(
                result.pose_landmarks or [], result.pose_world_landmarks or []
            ):
                norm_pts.append(np.array([[l.x, l.y, l.z] for l in pose_norm], dtype=np.float32))
                world_pts.append(np.array([[l.x, l.y, l.z] for l in pose_world], dtype=np.float32))
                vis_list.append(np.array([getattr(l, "visibility", 1.0) for l in pose_norm], dtype=np.float32))

            if world_pts:
                idx = _pick_person(norm_pts, world_pts, vis_list, prev_bbox)
                bbox = _bbox_2d(norm_pts[idx])
                prev_bbox = bbox
                worlds.append(world_pts[idx])
                viss.append(vis_list[idx])
                presents.append(True)
                scores.append(1.0)
            else:
                worlds.append(np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))
                viss.append(np.zeros(NUM_LANDMARKS, dtype=np.float32))
                presents.append(False)
                scores.append(0.0)

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
    )
