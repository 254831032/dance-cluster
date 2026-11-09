"""音频相似度自测：合成几段信号，验证「变调/平移/不同旋律」的区分度。

运行：.venv/bin/python dance-cluster/tests/quick_audio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecluster import audio  # noqa: E402

SR = 22050


def midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def melody(notes: list[int], dur: float = 0.4, reps: int = 1, sr: int = SR) -> np.ndarray:
    """用带谐波的正弦合成一段「旋律」，比纯正弦更接近真实音乐。"""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(reps):
        for m in notes:
            t = np.arange(int(dur * sr)) / sr
            f = midi_to_hz(m)
            wave = (
                0.6 * np.sin(2 * np.pi * f * t)
                + 0.25 * np.sin(2 * np.pi * 2 * f * t)
                + 0.15 * np.sin(2 * np.pi * 3 * f * t)
            )
            env = np.exp(-3.0 * t)  # 简单包络，避免拼接爆音
            out.append(env * wave)
    y = np.concatenate(out)
    return (0.5 * y / (np.abs(y).max() + 1e-9)).astype(np.float32) + rng.normal(0, 1e-3, len(y)).astype(np.float32)


def save(y: np.ndarray, name: str) -> Path:
    p = Path(__file__).parent / "_tmp" / name
    p.parent.mkdir(exist_ok=True)
    sf.write(p, y, SR)
    return p


def main() -> None:
    c_major = [60, 64, 67, 72, 67, 64, 60, 64] * 3   # C E G C ...
    a_minor = [57, 60, 64, 69, 64, 60, 57, 60] * 3   # A C E A ...

    base = melody(c_major)
    transposed = melody([m + 2 for m in c_major])     # 升高 2 个半音
    shifted = np.concatenate([np.zeros(int(1.5 * SR), np.float32), base])  # 前移 1.5s
    other = melody(a_minor)
    noise = (np.random.default_rng(1).normal(0, 0.1, len(base))).astype(np.float32)

    files = {
        "base": save(base, "base.wav"),
        "transposed": save(transposed, "transposed.wav"),
        "shifted": save(shifted, "shifted.wav"),
        "other": save(other, "other.wav"),
        "noise": save(noise, "noise.wav"),
    }

    feats = {k: audio.feature_from_audio(str(v)) for k, v in files.items()}
    b = feats["base"]
    for name in ("transposed", "shifted", "other", "noise"):
        print(f"base vs {name:11s} = {audio.match_features(b, feats[name]):.3f}")
    print(f"tempo estimate (base) = {b['tempo']:.1f} bpm, duration={b['duration']:.1f}s")


if __name__ == "__main__":
    main()
