"""髋部区域光流：AI 分割定位 + 传统光流（Farneback）。

与骨架（pose.py）独立的第二套运动信号：
- 定位：MediaPipe PoseLandmarker 的 segmentation mask（AI 分割，抗残肢/特写/暗光）
- 运动：Farneback 稠密光流，只在「髋部区域」内取平均
- 相机补偿：减去背景（mask 外）的中位光流
- 输出：每帧速度 vel + 累积位置 pos（去线性漂移）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache-mpl"))

import mediapipe as mp  # noqa: E402


@dataclass
class FlowSequence:
    pos: np.ndarray        # (F, 2) 累积位置（像素，去漂移后）
    vel: np.ndarray        # (F, 2) 每帧速度（像素/帧，背景补偿后）
    present: np.ndarray    # (F,) 该帧是否找到人区域
    fps: float

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            pos=self.pos.astype(np.float32),
            vel=self.vel.astype(np.float32),
            present=self.present.astype(bool),
            fps=np.array(self.fps, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FlowSequence":
        d = np.load(Path(path))
        return cls(
            pos=d["pos"],
            vel=d["vel"],
            present=d["present"],
            fps=float(d["fps"]),
        )


def _person_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask > 0.5)
    if len(xs) < 50:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _hip_region(bbox, W: int, H: int) -> tuple[int, int, int, int]:
    """在人 bbox 里取「骨盆区域」：下半身、水平居中 50% 宽。"""
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    hx0 = max(0, int(cx - (x1 - x0) * 0.25))
    hx1 = min(W, int(cx + (x1 - x0) * 0.25))
    hy0 = max(0, int(y0 + (y1 - y0) * 0.5))
    hy1 = min(H, int(y1))
    return hx0, hy0, hx1, hy1


def extract_flow(
    video: str | Path,
    model_path: str | Path,
    num_poses: int = 1,
    analysis_fps: float = 15.0,
    max_seconds: float | None = None,
    progress: callable | None = None,
) -> FlowSequence:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=int(num_poses),
        output_segmentation_masks=True,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(f"打不开视频：{video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / analysis_fps)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    vels: list[np.ndarray] = []
    poss: list[np.ndarray] = []
    presents: list[bool] = []
    prev_gray: np.ndarray | None = None
    prev_bbox = None
    pos = np.array([0.0, 0.0], dtype=np.float32)
    frame_idx = 0
    out_idx = 0

    try:
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

            H, W = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, int(round(t * 1000)))

            # AI 定位：分割 mask → 人 bbox（失败用上一帧 bbox）
            bbox = None
            mask = None
            if result.segmentation_masks:
                m = result.segmentation_masks[0].numpy_view()
                mask = (m[:, :, 0] if m.ndim == 3 else m).astype(np.float32)
                bbox = _person_bbox(mask)
            if bbox is None:
                bbox = prev_bbox

            if bbox is not None and prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                hx0, hy0, hx1, hy1 = _hip_region(bbox, W, H)
                if hx1 > hx0 and hy1 > hy0:
                    fg = flow[hy0:hy1, hx0:hx1].reshape(-1, 2)
                    vel = fg.mean(axis=0)
                    # 相机补偿：背景（mask 外）中位光流
                    if mask is not None:
                        bg = flow[mask < 0.5].reshape(-1, 2)
                        if len(bg) > 100:
                            cam = np.median(bg, axis=0)
                            vel = vel - cam
                    pos = pos + vel
                    vels.append(vel.astype(np.float32))
                    poss.append(pos.copy())
                    presents.append(True)
                else:
                    vels.append(np.zeros(2, np.float32))
                    poss.append(pos.copy())
                    presents.append(False)
            else:
                vels.append(np.zeros(2, np.float32))
                poss.append(pos.copy())
                presents.append(False)

            prev_gray = gray
            prev_bbox = bbox
            out_idx += 1
            frame_idx += 1
            if progress and out_idx % 30 == 0:
                progress(out_idx, total // step)
    finally:
        cap.release()
        landmarker.close()

    vel = np.stack(vels) if vels else np.zeros((0, 2), np.float32)
    pos = np.stack(poss) if poss else np.zeros((0, 2), np.float32)

    # 去线性漂移（积分会累积慢速偏移；对振荡信号无影响）
    if len(pos) >= 3:
        x = np.arange(len(pos), dtype=np.float32)
        for c in range(2):
            k, b = np.polyfit(x, pos[:, c], 1)
            pos[:, c] = pos[:, c] - (k * x + b)

    return FlowSequence(
        pos=pos.astype(np.float32),
        vel=vel.astype(np.float32),
        present=np.array(presents, dtype=bool),
        fps=float(src_fps / step),
    )
