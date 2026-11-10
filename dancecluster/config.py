"""集中配置。所有阈值、路径都从这里来，方便按真实数据标定。"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def app_data_dir() -> Path:
    """应用级配置目录（跨重启记住 cache 目录等设置）。

    正常双击运行：Windows → %APPDATA%/DanceCluster，macOS → ~/Library/Application Support/DanceCluster。
    开发/测试可用环境变量 DANCECLUSTER_CONFIG_DIR 指到别处。
    """
    env = os.environ.get("DANCECLUSTER_CONFIG_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "DanceCluster"


def settings_path() -> Path:
    return app_data_dir() / "config.json"


_instance_lock_handle = None


def acquire_instance_lock() -> bool:
    """单实例锁：成功返回 True 并保持文件句柄，已被占用返回 False。

    用操作系统级文件锁（Windows msvcrt / POSIX fcntl），进程退出自动释放，
    不会因异常退出留下死锁。拿不到锁文件时（如只读环境）放行，避免误伤。
    """
    global _instance_lock_handle
    try:
        path = app_data_dir() / "instance.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, "a+")
    except OSError:
        return True
    try:
        if sys.platform == "win32":
            import msvcrt

            if os.path.getsize(path) == 0:
                f.write("1")
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return False
    _instance_lock_handle = f
    return True


def load_settings() -> dict:
    p = settings_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_settings(data: dict) -> Path:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


@dataclass
class ClipConfig:
    """视频切片参数。默认取开头 20 秒，可被单文件覆盖。"""

    start: float = 0.0
    duration: float = 20.0
    # 音频片段长度（秒）。音频比骨架便宜，可取更长；0 = 整段
    audio_duration: float = 60.0
    # 截取片段的最大高度（像素）。超过就重编码缩到该高度，省空间（4K→720p）
    max_height: int = 720
    # False = 流拷贝（秒切，但有 keyframe 误差）；True = 重编码（精确，慢）
    reencode: bool = False


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    hop_length: int = 1024
    # 相似度计算时把 chroma 池化到的帧率（帧/秒），降低计算量
    match_fps: float = 5.0
    # 时间对齐搜索窗口（秒）
    max_lag_sec: float = 3.0


MODEL_FILES = {
    "heavy": "pose_landmarker_heavy.task",
    "full": "pose_landmarker_full.task",
    "lite": "pose_landmarker_lite.task",
}
MODEL_LABELS = {
    "heavy": "Heavy（最准，慢）",
    "full": "Full（均衡）",
    "lite": "Lite（最快，精度低）",
}


@dataclass
class PoseConfig:
    model: str = "heavy"  # heavy / full / lite
    num_poses: int = 3
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    target_fps: float = 15.0          # 下游统一重采样到的帧率
    smooth_window: int = 5            # 平滑窗口（帧）
    min_visibility: float = 0.35      # 低于该可见度的点视为缺失

    def model_file(self) -> Path:
        name = MODEL_FILES.get(self.model, MODEL_FILES["heavy"])
        return PROJECT_ROOT / "models" / name


@dataclass
class MatchConfig:
    # DTW 速度约束：允许的时间弯曲比例（0.08 = ±8%）
    dtw_window_ratio: float = 0.08
    # 音乐相似度阈值（0~1，越高越严格）
    music_threshold: float = 0.60
    # 动作距离阈值（DTW 归一化距离，特征已逐帧 L2 归一化，越小越严格）
    dance_threshold: float = 0.35
    mirror_invariant: bool = True     # 镜像视为同一支舞
    multi_segment: bool = True        # 多段采样取最佳匹配


@dataclass
class AppConfig:
    clip: ClipConfig = field(default_factory=ClipConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    cache_dir: str = str(PROJECT_ROOT / "cache")
    output_dir: str = str(PROJECT_ROOT / "output")

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> "AppConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls()
        for section in ("clip", "audio", "pose", "match"):
            if section in raw:
                sub = getattr(cfg, section)
                for k, v in raw[section].items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
        for k in ("cache_dir", "output_dir"):
            if k in raw:
                setattr(cfg, k, raw[k])
        return cfg
