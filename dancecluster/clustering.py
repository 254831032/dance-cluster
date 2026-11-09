"""聚类：基于预计算距离矩阵的层次聚类 + 置信度。

为什么用层次聚类：阈值直接对应「多相似才算一组」，可解释、可手调，
比 KMeans 更适合「不知道分几组」的场景。
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def cluster_from_distances(D: np.ndarray, threshold: float, method: str = "average") -> np.ndarray:
    """D: (N,N) 距离矩阵；threshold: 合并的最大距离。返回 1..K 标签。"""
    n = D.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.array([1], dtype=int)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method=method)
    return fcluster(Z, t=threshold, criterion="distance")


def confidence(D: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """每个样本的置信度（0~1）：离本组越近、离别组越远越高。"""
    n = len(labels)
    conf = np.zeros(n, dtype=np.float32)
    for i in range(n):
        same = np.where(labels == labels[i])[0]
        other = np.where(labels != labels[i])[0]
        same = same[same != i]
        intra = float(np.mean(D[i, same])) if same.size else 0.0
        inter = float(np.min(D[i, other])) if other.size else float("inf")
        if np.isfinite(inter):
            conf[i] = inter / (inter + intra + 1e-9)
        else:
            conf[i] = 1.0
    return conf


def order_groups(labels: np.ndarray) -> list[list[int]]:
    """按组大小降序返回每组的样本索引。"""
    groups: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(int(idx))
    return sorted(groups.values(), key=len, reverse=True)
