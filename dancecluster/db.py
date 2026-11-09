"""SQLite 存储：视频元数据、特征、分组结果，全在一个文件里。

为什么用 SQLite：
- 单文件、零配置、随程序走，不需要用户装数据库；
- 特征数据（chroma / 骨架序列）直接存 BLOB，配合元数据可查询；
- 以后要做「手动合并/拆分分组、改主舞者、重新导出」都能直接在库里改。

表结构：
  videos  每个视频一行：元数据 + 状态 + 分组结果 + 两个特征 BLOB
  runs    每次运行记录（参数、输出目录），用于追溯
"""

from __future__ import annotations

import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    duration      REAL    DEFAULT 0,
    width         INTEGER DEFAULT 0,
    height        INTEGER DEFAULT 0,
    fps           REAL    DEFAULT 0,
    has_audio     INTEGER DEFAULT 0,
    status        TEXT    DEFAULT 'pending',
    error         TEXT    DEFAULT '',
    cache_dir     TEXT    DEFAULT '',
    present_ratio REAL    DEFAULT 0,
    tempo         REAL    DEFAULT 0,
    music_group   INTEGER,
    dance_group   INTEGER,
    confidence    REAL,
    audio_npz     BLOB,
    motion_npz    BLOB,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT,
    finished_at TEXT,
    data_dir    TEXT,
    out_dir     TEXT,
    params_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_groups ON videos(music_group, dance_group);
"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def dumps_npz(data: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **data)
    return buf.getvalue()


def loads_npz(blob: bytes | None) -> dict:
    if not blob:
        return {}
    with np.load(io.BytesIO(blob)) as f:
        return {k: f[k] for k in f.files}


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ---------- 视频 ----------
    def upsert_video(self, **fields) -> int:
        fields.setdefault("updated_at", _now())
        cur = self.conn.execute(
            "SELECT id FROM videos WHERE path = ?", (fields["path"],)
        )
        row = cur.fetchone()
        if row:
            cols = ", ".join(f"{k} = ?" for k in fields)
            self.conn.execute(
                f"UPDATE videos SET {cols} WHERE id = ?", (*fields.values(), row["id"])
            )
            vid = int(row["id"])
        else:
            cols = ", ".join(fields)
            marks = ", ".join("?" for _ in fields)
            cur = self.conn.execute(
                f"INSERT INTO videos ({cols}) VALUES ({marks})", tuple(fields.values())
            )
            vid = int(cur.lastrowid)
        self.conn.commit()
        return vid

    def set_status(self, video_id: int, status: str, error: str = "") -> None:
        self.conn.execute(
            "UPDATE videos SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, error, _now(), video_id),
        )
        self.conn.commit()

    def set_audio(self, video_id: int, audio: dict, tempo: float) -> None:
        self.conn.execute(
            "UPDATE videos SET audio_npz = ?, tempo = ?, updated_at = ? WHERE id = ?",
            (dumps_npz(audio), float(tempo), _now(), video_id),
        )
        self.conn.commit()

    def set_motion(self, video_id: int, motion: dict, present_ratio: float) -> None:
        self.conn.execute(
            "UPDATE videos SET motion_npz = ?, present_ratio = ?, updated_at = ? WHERE id = ?",
            (dumps_npz(motion), float(present_ratio), _now(), video_id),
        )
        self.conn.commit()

    def get_by_path(self, path: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM videos WHERE path = ?", (path,)).fetchone()
        return dict(row) if row else None

    def all_videos(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM videos ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def audio_features(self, row: dict) -> dict:
        return loads_npz(row.get("audio_npz"))

    def motion_features(self, row: dict) -> dict:
        return loads_npz(row.get("motion_npz"))

    def set_groups(self, assignments: list[tuple[int, int, int, float]]) -> None:
        """assignments: [(video_id, music_group, dance_group, confidence)]"""
        self.conn.executemany(
            "UPDATE videos SET music_group = ?, dance_group = ?, confidence = ?, updated_at = ? "
            "WHERE id = ?",
            [(m, d, c, _now(), vid) for vid, m, d, c in assignments],
        )
        self.conn.commit()

    def clear_groups(self) -> None:
        self.conn.execute(
            "UPDATE videos SET music_group = NULL, dance_group = NULL, confidence = NULL"
        )
        self.conn.commit()

    # ---------- 运行记录 ----------
    def create_run(self, data_dir: str, out_dir: str, params: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (created_at, data_dir, out_dir, params_json) VALUES (?, ?, ?, ?)",
            (_now(), data_dir, out_dir, json.dumps(params, ensure_ascii=False)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ? WHERE id = ?", (_now(), run_id)
        )
        self.conn.commit()

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) n, SUM(status='ok') ok, SUM(status='no_pose') nop, "
            "SUM(status='error') err, COUNT(DISTINCT music_group) mg, "
            "COUNT(DISTINCT dance_group) dg FROM videos"
        ).fetchone()
        return dict(row)
