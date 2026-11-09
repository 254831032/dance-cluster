"""集中配置。所有阈值、路径都从这里来，方便按真实数据标定。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ClipConfig:
    """视频切片参数。默认取开头 20 秒，可被单文件覆盖。"""

    start: float = 0.0
    duration: float = 20.0
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


@dataclass
class PoseConfig:
    model_path: str = str(PROJECT_ROOT / "models" / "pose_landmarker_heavy.task")
    num_poses: int = 3
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    target_fps: float = 15.0          # 下游统一重采样到的帧率
    smooth_window: int = 5            # 平滑窗口（帧）
    min_visibility: float = 0.35      # 低于该可见度的点视为缺失


@dataclass
class MatchConfig:
    # DTW 速度约束：允许的时间弯曲比例（0.08 = ±8%）
    dtw_window_ratio: float = 0.08
    # 音乐相似度阈值（0~1，越高越严格）
    music_threshold: float = 0.55
    # 动作相似度阈值（0~1，越高越严格）
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
