"""DTW 距离：带速度约束（Sakoe-Chiba 窗口）+ 子序列匹配 + 镜像不变。"""

from __future__ import annotations

import numpy as np

from . import fastdtw


def dtw_distance(
    fa: np.ndarray,
    fb: np.ndarray,
    window_ratio: float = 0.08,
    subsequence: bool = False,
) -> float:
    """按帧数归一化的 DTW 距离（越小越像）。fa/fb: (F,D)。

    subsequence=True 时允许 a 匹配 b 的任意一段（起手不同步的场景）。
    """
    if len(fa) < 2 or len(fb) < 2:
        return float("inf")
    window = max(2, int(round(window_ratio * max(len(fa), len(fb)))))
    if subsequence:
        d = fastdtw.subsequence_distance(fa, fb, window)
    else:
        d = fastdtw.distance(fa, fb, window)
    return float(d) / float(max(len(fa), len(fb)))


def distance(
    fa: np.ndarray,
    fb: np.ndarray,
    fb_mirror: np.ndarray | None = None,
    window_ratio: float = 0.08,
    mirror_invariant: bool = True,
    subsequence: bool = False,
) -> float:
    """两个视频的特征距离；镜像时取「原/镜像」较小值。"""
    d = dtw_distance(fa, fb, window_ratio, subsequence)
    if mirror_invariant and fb_mirror is not None and len(fb_mirror) >= 2:
        d = min(d, dtw_distance(fa, fb_mirror, window_ratio, subsequence))
    return d


def distance_multi(
    feats_a: list[np.ndarray],
    feats_b: list[np.ndarray],
    window_ratio: float = 0.08,
) -> float:
    """多段采样：两两取最小距离（不同段落起手也能对上）。"""
    best = float("inf")
    for a in feats_a:
        for b in feats_b:
            best = min(best, dtw_distance(a, b, window_ratio, subsequence=True))
    return best


def similarity_from_distance(d: float, scale: float = 0.5) -> float:
    """距离 -> (0,1] 相似度。scale 越大，对距离越宽容。"""
    if not np.isfinite(d):
        return 0.0
    return float(np.exp(-max(d, 0.0) / max(scale, 1e-6)))
