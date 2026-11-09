"""带速度约束的 DTW（numba 加速，替代 dtaidistance 的 C 扩展）。

为什么自己写：dtaidistance 在 Python 3.13 上只有纯 Python 回退（无 C 扩展），
几百个视频两两比较会慢到不可用。numba JIT 后单次 300 帧 ×32 维约 0.1ms。
"""

from __future__ import annotations

import numpy as np
from numba import njit

INF = 1e12


@njit(cache=True, fastmath=True)
def _dtw_banded(a: np.ndarray, b: np.ndarray, w: int) -> float:
    """相对 Sakoe-Chiba 带约束 DTW，返回累计距离（未归一化）。

    用「比例中心」而非对角中心：a 的第 i 帧对应 b 的第 i*n/m 帧附近，
    这样两段长度不同时带宽仍能连通到终点，不会被切断成 INF。
    """
    n, m = a.shape[0], b.shape[0]
    prev = np.full(m + 2, INF)
    cur = np.full(m + 2, INF)
    prev[1] = 0.0
    scale = n / m
    for i in range(1, n + 1):
        for j in range(m + 2):
            cur[j] = INF
        center = i / scale
        lo = int(center - w)
        if lo < 1:
            lo = 1
        hi = int(center + w)
        if hi > m:
            hi = m
        for j in range(lo, hi + 1):
            d = 0.0
            for k in range(a.shape[1]):
                diff = a[i - 1, k] - b[j - 1, k]
                d += diff * diff
            d = d ** 0.5
            a1 = prev[j]
            a2 = cur[j - 1]
            a3 = prev[j - 1]
            best = a1 if a1 < a2 else a2
            if a3 < best:
                best = a3
            cur[j] = d + best
        prev, cur = cur, prev
    return prev[m]


@njit(cache=True, fastmath=True)
def _dtw_subsequence(a: np.ndarray, b: np.ndarray, w: int) -> float:
    """子序列 DTW：a 可以匹配 b 的任意一段（首尾开放）。"""
    n, m = a.shape[0], b.shape[0]
    prev = np.zeros(m + 2)          # 首行开放：任意起点代价为 0
    cur = np.full(m + 2, INF)
    for i in range(1, n + 1):
        for j in range(m + 2):
            cur[j] = INF
        lo = max(1, i - w)
        hi = min(m, i + w)
        for j in range(lo, hi + 1):
            d = 0.0
            for k in range(a.shape[1]):
                diff = a[i - 1, k] - b[j - 1, k]
                d += diff * diff
            d = d ** 0.5
            a1 = prev[j]
            a2 = cur[j - 1]
            a3 = prev[j - 1]
            best = a1 if a1 < a2 else a2
            if a3 < best:
                best = a3
            cur[j] = d + best
        prev, cur = cur, prev
    best = INF
    for j in range(1, m + 1):
        if prev[j] < best:
            best = prev[j]
    return best


def distance(a: np.ndarray, b: np.ndarray, window: int) -> float:
    return float(_dtw_banded(np.ascontiguousarray(a, dtype=np.float32),
                             np.ascontiguousarray(b, dtype=np.float32), int(window)))


def subsequence_distance(a: np.ndarray, b: np.ndarray, window: int) -> float:
    return float(_dtw_subsequence(np.ascontiguousarray(a, dtype=np.float32),
                                  np.ascontiguousarray(b, dtype=np.float32), int(window)))
