"""动作特征 + DTW 自测：合成骨架序列，验证「同/镜像/不同动作」的区分。

运行：.venv/bin/python dance-cluster/tests/quick_motion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecluster import dtw, features  # noqa: E402

FPS = 15.0


def base_skeleton() -> np.ndarray:
    x = np.zeros((33, 3), dtype=np.float32)
    x[23] = [-0.10, 0.00, 0.0]
    x[24] = [0.10, 0.00, 0.0]
    x[11] = [-0.20, 0.50, 0.0]
    x[12] = [0.20, 0.50, 0.0]
    x[0] = [0.0, 0.72, 0.0]
    x[13] = [-0.38, 0.28, 0.0]
    x[14] = [0.38, 0.28, 0.0]
    x[15] = [-0.42, 0.05, 0.0]
    x[16] = [0.42, 0.05, 0.0]
    x[25] = [-0.10, -0.45, 0.0]
    x[26] = [0.10, -0.45, 0.0]
    x[27] = [-0.10, -0.90, 0.0]
    x[28] = [0.10, -0.90, 0.0]
    x[1], x[4] = [0.03, 0.75, 0.0], [-0.03, 0.75, 0.0]
    x[2], x[5] = [0.06, 0.75, 0.0], [-0.06, 0.75, 0.0]
    x[3], x[6] = [0.09, 0.75, 0.0], [-0.09, 0.75, 0.0]
    x[7], x[8] = [0.10, 0.70, 0.0], [-0.10, 0.70, 0.0]
    x[9], x[10] = [0.04, 0.66, 0.0], [-0.04, 0.66, 0.0]
    x[29], x[30] = [-0.12, -0.92, 0.0], [0.12, -0.92, 0.0]
    x[31], x[32] = [-0.14, -0.88, 0.0], [0.14, -0.88, 0.0]
    return x


def sequence(freq: float = 1.0, phase: float = 0.0, noise: float = 0.0,
             frames: int = 300, mirrored: bool = False, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    base = base_skeleton()
    out = np.tile(base[None], (frames, 1, 1))
    t = np.arange(frames) / FPS

    # 双臂前后摆 + 髋部左右摇 + 膝盖微屈，幅度不同
    swing = 0.35 * np.sin(2 * np.pi * freq * t + phase)
    swing_r = 0.35 * np.sin(2 * np.pi * freq * t + phase + np.pi / 3)
    hip = 0.04 * np.sin(2 * np.pi * freq * 0.5 * t)

    out[:, 13, 1] += swing
    out[:, 15, 1] += 1.4 * swing
    out[:, 14, 1] += swing_r
    out[:, 16, 1] += 1.4 * swing_r
    out[:, 13, 2] -= 0.5 * swing
    out[:, 15, 2] -= 0.7 * swing
    out[:, 14, 2] -= 0.5 * swing_r
    out[:, 16, 2] -= 0.7 * swing_r
    out[:, 23, 0] += hip
    out[:, 11, 0] += 0.5 * hip
    out[:, 25, 0] += 0.8 * hip
    out[:, 27, 0] += 0.8 * hip

    if mirrored:
        out = out[:, features.MIRROR_MAP]
        out[..., 0] *= -1
    if noise:
        out = out + rng.normal(0, noise, out.shape)
    return out.astype(np.float32)


def feats(x: np.ndarray):
    vis = np.ones(x.shape[:2], np.float32)
    pres = np.ones(x.shape[0], bool)
    return features.prepare_features(x, vis, pres, FPS)


def main() -> None:
    rng = np.random.default_rng(42)
    a = sequence(1.0, rng=rng)
    b = sequence(1.0, noise=0.008, rng=rng)          # 同一支舞 + 轻微差异
    c = sequence(1.0, mirrored=True, rng=rng)         # 镜像
    d = sequence(1.6, phase=1.2, rng=rng)             # 不同动作
    e = sequence(1.0, phase=0.15, rng=rng)            # 起手错开

    fa = feats(a)
    rows = {
        "同一支舞+噪声": feats(b),
        "镜像": feats(c),
        "不同动作": feats(d),
        "起手错开": feats(e),
    }
    for name, fb in rows.items():
        dist = dtw.distance(fa[0], fb[0], fb[1], window_ratio=0.08)
        sim = dtw.similarity_from_distance(dist)
        print(f"{name:10s} d={dist:.4f}  sim={sim:.3f}")


if __name__ == "__main__":
    main()
