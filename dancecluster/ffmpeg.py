"""FFmpeg / ffprobe 封装：探测、切片、抽音频。

优先级：环境变量 DANCECLUSTER_FFMPEG > 项目 .tools/bin > PATH。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tool(name: str) -> str:
    env = os.environ.get(f"DANCECLUSTER_{name.upper()}")
    if env and Path(env).exists():
        return env
    base = PROJECT_ROOT / ".tools" / "bin"
    for cand in (base / name, base / (name + ".exe")):
        if cand.exists():
            return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"找不到 {name}。请设置 DANCECLUSTER_{name.upper()}，或放到 {PROJECT_ROOT / '.tools/bin'}"
    )


def ffmpeg() -> str:
    return _tool("ffmpeg")


def ffprobe() -> str:
    return _tool("ffprobe")


def _run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
        raise RuntimeError(f"命令失败({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc


def probe(video: str | Path) -> dict:
    """返回 duration / width / height / fps / has_audio。"""
    proc = _run(
        [
            ffprobe(), "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,width,height,r_frame_rate,codec_name",
            "-of", "json", str(video),
        ]
    )
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    fps = 0.0
    rate = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        pass
    return {
        "duration": float(info.get("format", {}).get("duration") or 0.0),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": fps,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "video_codec": video_stream.get("codec_name"),
    }


def cut_clip(
    video: str | Path,
    out: str | Path,
    start: float = 0.0,
    duration: float = 20.0,
    reencode: bool = False,
) -> Path:
    """切出 [start, start+duration) 片段。默认流拷贝，速度快。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if reencode:
        cmd = [
            ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", str(out),
        ]
    else:
        cmd = [
            ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero", str(out),
        ]
    _run(cmd)
    return out


def extract_audio(
    video: str | Path,
    out: str | Path,
    start: float | None = None,
    duration: float | None = None,
    sample_rate: int = 22050,
) -> Path:
    """抽单声道 wav。直接对原视频做 -ss/-t，避免多切一次。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", str(out)]
    _run(cmd)
    return out


def pitch_shift(
    src: str | Path,
    out: str | Path,
    semitones: float,
    sample_rate: int = 22050,
) -> Path:
    """变调不变速：asetrate 改音高+速度，atempo 把速度拉回。

    本项目的 chroma 方案用「调性旋转」就能覆盖变调，这里保留给需要
    实际生成变调音频的场景（例如以后接入 chromaprint 指纹时）。
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    factor = 2.0 ** (semitones / 12.0)
    af = f"asetrate={sample_rate}*{factor:.6f},aresample={sample_rate},atempo={1.0 / factor:.6f}"
    _run(
        [
            ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src), "-af", af, "-ac", "1", "-ar", str(sample_rate), str(out),
        ]
    )
    return out
