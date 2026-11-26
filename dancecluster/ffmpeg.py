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

# Windows 下避免每个 ffmpeg/ffprobe 调用都弹出命令行窗口
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# 限制视频编码线程数，避免并行识别时多个 ffmpeg 各自用满所有核把 CPU 拉满。
# 可用环境变量 DANCECLUSTER_FFMPEG_THREADS 覆盖（如 "1"）。
_MAX_THREADS = os.environ.get("DANCECLUSTER_FFMPEG_THREADS", "2")

_nvenc_cache: bool | None = None


def _nvenc_available() -> bool:
    """检测当前 ffmpeg 是否支持 NVIDIA NVENC 硬件编码（结果缓存）。

    有 nvidia-smi（NVIDIA 驱动）+ ffmpeg 带 h264_nvenc 编码器才返回 True，
    否则回退 CPU 软编。任何异常都按「不可用」处理，保证不崩。
    """
    global _nvenc_cache
    if _nvenc_cache is not None:
        return _nvenc_cache
    if shutil.which("nvidia-smi") is None:
        _nvenc_cache = False
        return False
    try:
        proc = subprocess.run(
            [ffmpeg(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=20, creationflags=_NO_WINDOW,
        )
        _nvenc_cache = "h264_nvenc" in (proc.stdout or "")
    except Exception:
        _nvenc_cache = False
    return _nvenc_cache


_nvenc_warmup_done = False


def _hwaccel() -> list[str]:
    """硬件解码参数（按平台）。4K 解码是代理/预览生成的主要瓶颈，GPU 解码可提速数倍。"""
    import platform

    sysname = platform.system()
    if sysname == "Darwin":
        return ["-hwaccel", "videotoolbox"]
    if sysname == "Windows":
        return ["-hwaccel", "cuda"] if _nvenc_available() else ["-hwaccel", "d3d11va"]
    return ["-hwaccel", "cuda"] if _nvenc_available() else []


def nvenc_warmup() -> None:
    """一次性预热 NVENC：在并行 worker 启动前跑一次极小编码，触发驱动级 CUDA 初始化。

    多个 ffmpeg 进程同时冷启动 CUDA 可能偶发「打开编码器失败」，先串行预热一次
    可显著降低这种瞬时竞争。失败忽略，不影响后续（仍有 libx264 回退兜底）。
    """
    global _nvenc_warmup_done
    if _nvenc_warmup_done:
        return
    _nvenc_warmup_done = True
    if not _nvenc_available():
        return
    try:
        subprocess.run(
            [
                ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:size=128x128:rate=5:duration=0.2",
                "-c:v", "h264_nvenc", "-preset", "p4", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW,
        )
    except Exception:
        pass


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


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW)
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
        ],
        timeout=60,
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
    max_height: int = 0,
) -> Path:
    """切出 [start, start+duration) 片段。

    默认流拷贝（快）。若 max_height>0 且源视频更高，则重编码并缩到该高度
    （4K→720p 省大量空间，骨架识别用不到 4K）。
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    need_scale = False
    if max_height > 0:
        info = probe(video)
        need_scale = info["height"] > max_height

    if reencode or need_scale:
        hw = _hwaccel()
        base = [
            ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            *hw,
            "-threads", _MAX_THREADS,
            "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
        ]
        if need_scale:
            base += ["-vf", f"scale=-2:{max_height}"]
        tail = ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)]
        # 先试硬件解码（GPU）+ NVENC；失败自动回退
        try:
            if _nvenc_available():
                _run(base + ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"] + tail)
            else:
                _run(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", _MAX_THREADS] + tail)
            return out
        except RuntimeError:
            pass
        # 硬件解码失败（驱动/编码器不支持等）→ 回退纯软解软编
        base_sw = [x for x in base if x not in set(hw)]
        _run(base_sw + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", _MAX_THREADS] + tail)
    else:
        _run([
            ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", str(out),
        ])
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


def extract_thumbnail(
    video: str | Path,
    out: str | Path,
    at: float = 1.0,
    width: int = 320,
) -> Path:
    """抽一帧当缩略图。at 秒处取帧，等比缩到 width 宽，存 JPEG。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{at:.2f}", "-i", str(video),
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(out),
    ]
    _run(cmd)
    return out


def extract_gif(
    video: str | Path,
    out: str | Path,
    at: float = 1.0,
    duration: float = 2.0,
    fps: int = 5,
    width: int = 240,
) -> Path:
    """抽一小段做成循环 GIF 动图缩略图（低帧率低画质，省空间）。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={int(fps)},scale={int(width)}:-2:flags=lanczos,"
        "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    )
    cmd = [
        ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{at:.2f}", "-t", f"{duration:.2f}", "-i", str(video),
        "-vf", vf, "-loop", "0", str(out),
    ]
    _run(cmd)
    return out

def to_aac(src: str | Path, dst: str | Path, bitrate: str = "96k") -> Path:
    """把音频压成 AAC（预览用），比 WAV 小约 5 倍。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vn", "-c:a", "aac", "-b:a", bitrate, str(dst),
    ])
    return dst


def extract_mp4_thumb(
    video: str | Path,
    out: str | Path,
    at: float = 1.0,
    duration: float = 3.0,
    fps: int = 8,
    width: int = 240,
) -> Path:
    """抽一小段做成无声循环 mp4 动图缩略图（比 GIF 小很多、加载更快）。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = f"fps={int(fps)},scale={int(width)}:-2:flags=lanczos"
    cmd = [
        ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-threads", _MAX_THREADS,   # 限制解码线程
        "-ss", f"{at:.2f}", "-t", f"{duration:.2f}", "-i", str(video),
        "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-threads", _MAX_THREADS,   # 限制编码线程
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]
    _run(cmd)
    return out
