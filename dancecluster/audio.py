"""音频层：chroma 特征 + 调性不变的相似度。

设计要点：
- 不用 chromaprint（静态 ffmpeg 没编进去，且它怕变调）；
- chroma 天然按「音高类别」表示，变调 = 12 个 bin 的循环移位，
  所以「12 个旋转取最大 + 时间轴对齐」就同时解决了变调和不同步。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

DEFAULT_SR = 22050
EPS = 1e-9


def load_audio(path: str, sr: int = DEFAULT_SR) -> np.ndarray:
    """读成单声道 float32。"""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def chroma(y: np.ndarray, sr: int = DEFAULT_SR, hop_length: int = 1024) -> np.ndarray:
    """(12, T) 的 CQT chroma，按帧做 L2 归一化。"""
    import librosa

    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=36)
    c = np.asarray(c, dtype=np.float32)
    norms = np.linalg.norm(c, axis=0, keepdims=True)
    return c / np.maximum(norms, EPS)


def estimate_tempo(y: np.ndarray, sr: int = DEFAULT_SR) -> float:
    import librosa

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        arr = np.atleast_1d(tempo).astype(float)
        return float(arr[0]) if arr.size else 0.0
    except Exception:
        return 0.0


def pool_time(c: np.ndarray, fps: float, target_fps: float = 5.0) -> np.ndarray:
    """按时间平均池化到 target_fps，降低相似度计算量。"""
    if fps <= target_fps or c.shape[1] == 0:
        return c
    step = max(1, int(round(fps / target_fps)))
    n = (c.shape[1] // step) * step
    if n == 0:
        return c
    return c[:, :n].reshape(c.shape[0], -1, step).mean(axis=2)


def _frame_energy(c: np.ndarray) -> np.ndarray:
    return (c ** 2).sum(axis=0)


def similarity(
    ca: np.ndarray,
    cb: np.ndarray,
    max_lag: int = 15,
    transposition_invariant: bool = True,
) -> float:
    """两段 chroma 的最佳归一化互相关（0~1）。

    ca/cb: (12, T)，帧已 L2 归一化。
    在 12 个调性旋转 × 时间滞后 [-max_lag, max_lag] 上取最大。
    """
    ta, tb = ca.shape[1], cb.shape[1]
    if ta < 2 or tb < 2:
        return 0.0
    n = 1 << int(np.ceil(np.log2(ta + tb + 1)))
    fa = np.fft.rfft(ca, n=n, axis=1)
    fb = np.fft.rfft(cb, n=n, axis=1)

    ea = np.concatenate([[0.0], np.cumsum(_frame_energy(ca))])
    eb = np.concatenate([[0.0], np.cumsum(_frame_energy(cb))])

    shifts = range(12) if transposition_invariant else (0,)
    best = 0.0
    for s in shifts:
        # 把 b 的 12 个 bin 循环移位 s，等价于「变调后再比」
        fb_shift = fb[np.roll(np.arange(12), s)]
        corr = np.fft.irfft((fa * np.conj(fb_shift)).sum(axis=0), n=n)

        for lag in range(-max_lag, max_lag + 1):
            t0 = max(0, lag)
            t1 = min(ta, tb + lag)
            if t1 - t0 < 2:
                continue
            num = float(corr[lag % n])
            den = np.sqrt(max(ea[t1] - ea[t0], EPS) * max(eb[t1 - lag] - eb[t0 - lag], EPS))
            val = num / den
            if val > best:
                best = val
    return float(np.clip(best, 0.0, 1.0))


def feature_from_audio(
    path: str,
    sr: int = DEFAULT_SR,
    hop_length: int = 1024,
    match_fps: float = 5.0,
    max_lag_sec: float = 3.0,
) -> dict:
    """从 wav 抽一组可比对的特征。"""
    y = load_audio(path, sr=sr)
    c = chroma(y, sr=sr, hop_length=hop_length)
    frame_fps = sr / hop_length
    c_pooled = pool_time(c, frame_fps, match_fps)
    max_lag = max(1, int(round(max_lag_sec * match_fps)))
    return {
        "chroma": c_pooled,           # (12, T) @ match_fps
        "chroma_full": c,             # (12, T) @ 原始帧率，配 DTW 用
        "tempo": estimate_tempo(y, sr=sr),
        "duration": float(len(y) / sr),
        "fps": float(match_fps),
        "max_lag": max_lag,
    }


def match_features(fa: dict, fb: dict) -> float:
    return similarity(fa["chroma"], fb["chroma"], max_lag=fa["max_lag"])

# ---------------------------------------------------------------------------
# chromaprint 指纹：比 chroma 判别力强得多（不同歌曲指纹完全不同）。
# fpcalc 从 .tools/bin 或 PATH 找；没有时自动退化为 chroma。
# ---------------------------------------------------------------------------
def _fpcalc_path() -> str | None:
    env = os.environ.get("DANCECLUSTER_FPCALC")
    if env and Path(env).exists():
        return env
    local = Path(__file__).resolve().parents[1] / ".tools" / "bin" / "fpcalc"
    if local.exists():
        return str(local)
    return shutil.which("fpcalc")


def fingerprint(path: str) -> np.ndarray | None:
    """chromaprint 指纹，返回 (T, 32) 的 0/1 位矩阵。不可用返回 None。"""
    exe = _fpcalc_path()
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "-raw", str(path)], capture_output=True, text=True, timeout=180
        ).stdout
    except Exception:
        return None
    fp = None
    for line in out.splitlines():
        if line.startswith("FINGERPRINT="):
            fp = line.split("=", 1)[1]
    if not fp:
        return None
    try:
        ints = np.array([int(x) for x in fp.split(",") if x.strip()], dtype=np.uint32)
    except Exception:
        return None
    if ints.size < 2:
        return None
    return np.unpackbits(ints.view(np.uint8)).reshape(-1, 32).astype(np.uint8)


def fingerprint_similarity(fa: np.ndarray, fb: np.ndarray, max_offset: int = 20) -> float:
    """1 - 最小比特错误率（BER）。fa/fb: (T,32)。"""
    na, nb = len(fa), len(fb)
    if na < 2 or nb < 2:
        return 0.0
    short, long_ = (fa, fb) if na <= nb else (fb, fa)
    ns, nl = len(short), len(long_)
    max_off = min(max_offset, nl - ns)
    best_ber = 1.0
    for off in range(0, max_off + 1):
        ber = float(np.mean(short != long_[off:off + ns]))
        if ber < best_ber:
            best_ber = ber
    return 1.0 - best_ber


def _fold_tempo(t: float) -> float:
    if t <= 0:
        return 0.0
    while t < 70:
        t *= 2
    while t >= 140:
        t /= 2
    return t


def _tempo_conflict(ta: float, tb: float, tol: float = 0.08) -> bool:
    fa, fb = _fold_tempo(ta), _fold_tempo(tb)
    if fa <= 0 or fb <= 0:
        return False
    return abs(fa - fb) / ((fa + fb) / 2) > tol


def music_similarity(fa: dict, fb: dict) -> float:
    """音频是否同一首的综合判定（0~1），强判别：
    1) tempo 差太大 → 0（不同/变速版）
    2) 指纹 BER <= 0.35 → 同一首（返回指纹相似度）
    3) chroma 变调不变相似度 >= 0.92 → 同一首（兜底，覆盖变调）
    否则 0。"""

    if _tempo_conflict(float(fa.get("tempo") or 0), float(fb.get("tempo") or 0)):
        return 0.0

    fpa = fa.get("fingerprint")
    fpb = fb.get("fingerprint")
    if fpa is not None and fpb is not None:
        s = fingerprint_similarity(fpa, fpb)
        if s >= 0.65:
            return s

    cs = similarity(
        fa["chroma"], fb["chroma"],
        max_lag=int(fa.get("max_lag", 15)),
        transposition_invariant=True,
    )
    return cs if cs >= 0.92 else 0.0
