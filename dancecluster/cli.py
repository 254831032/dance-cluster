"""命令行入口。

用法：
  python -m dancecluster.cli index  --data ./data
  python -m dancecluster.cli group  --data ./data
  python -m dancecluster.cli run    --data ./data --out ./output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pipeline
from .config import AppConfig


def _config(args) -> AppConfig:
    cfg = AppConfig()
    if getattr(args, "config", None):
        cfg = AppConfig.from_json(args.config)
    if getattr(args, "cache", None):
        cfg.cache_dir = str(Path(args.cache).resolve())
    if getattr(args, "out", None):
        cfg.output_dir = str(Path(args.out).resolve())
    return cfg


def _progress(i: int, total: int, name: str, stage: str) -> None:
    print(f"[{i}/{total}] {stage:11s} {name}", flush=True)


def cmd_index(args) -> int:
    cfg = _config(args)
    records = pipeline.index_all(args.data, cfg, progress=_progress)
    ok = sum(r.status == "ok" for r in records)
    no_pose = sum(r.status == "no_pose" for r in records)
    err = [r for r in records if r.status == "error"]
    print(f"\n完成：{ok} 个可比对，{no_pose} 个未检测到人体，{len(err)} 个失败")
    for r in err[:10]:
        print(f"  ✗ {r.name}: {r.error}")
    return 0 if ok else 1


def cmd_group(args) -> int:
    cfg = _config(args)
    recs = pipeline.index_all(args.data, cfg)
    if not recs:
        print("数据目录里没找到视频")
        return 1
    result = pipeline.group(recs, cfg)
    print(f"\n音乐组 {len(set(result.music_labels))} 个，动作组 {len(set(result.dance_labels))} 个")
    for i, r in enumerate(result.records):
        print(f"  M{result.music_labels[i]} / D{result.dance_labels[i]}  "
              f"conf={result.confidence[i]:.2f}  {r.name}")
    return 0


def cmd_run(args) -> int:
    cfg = _config(args)
    recs = pipeline.index_all(args.data, cfg, progress=_progress)
    if not recs:
        print("数据目录里没找到视频")
        return 1
    result = pipeline.group(recs, cfg)
    out = pipeline.export(result, cfg.output_dir)
    print(f"\n输出：{out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dancecluster", description="舞蹈视频自动分组")
    p.add_argument("--config", help="配置文件 JSON（可选）")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, func, help_text in (
        ("index", cmd_index, "只做特征提取（可断点续跑）"),
        ("group", cmd_group, "用已有缓存做分组"),
        ("run", cmd_run, "提取 + 分组 + 导出"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--data", required=True, help="视频目录")
        sp.add_argument("--cache", help="缓存目录")
        sp.add_argument("--out", help="输出目录")
        sp.set_defaults(func=func)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # MediaPipe 退出时会等遥测上传（约 60s）；CLI 里直接硬退出，跳过析构
    import os

    os._exit(int(code or 0))
