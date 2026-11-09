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

CREATE TABLE IF NOT EXISTS library_folders (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    path     TEXT UNIQUE NOT NULL,
    added_at TEXT
);

CREATE TABLE IF NOT EXISTS groups (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    group_type TEXT    NOT NULL,
    group_key  INTEGER NOT NULL,
    name       TEXT,
    UNIQUE(group_type, group_key)
);

CREATE TABLE IF NOT EXISTS categories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS video_categories (
    video_id    INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    PRIMARY KEY (video_id, category_id)
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

    def get_by_id(self, video_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return dict(row) if row else None

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

    # ---------- 视频库文件夹 ----------
    def list_folders(self) -> list[str]:
        rows = self.conn.execute("SELECT path FROM library_folders ORDER BY added_at").fetchall()
        return [r["path"] for r in rows]

    def add_folder(self, path: str) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO library_folders (path, added_at) VALUES (?, ?)",
            (path, _now()),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def remove_folder(self, path: str) -> None:
        self.conn.execute("DELETE FROM library_folders WHERE path = ?", (path,))
        self.conn.commit()

    # ---------- 分组命名 ----------
    def _member_stem(self, group_type: str, key: int) -> str:
        col = "music_group" if group_type == "music" else "dance_group"
        row = self.conn.execute(
            f"SELECT name FROM videos WHERE {col} = ? ORDER BY name LIMIT 1", (key,)
        ).fetchone()
        if row:
            return Path(row["name"]).stem
        return f"{'音乐' if group_type == 'music' else '动作'}组 {key}"

    def ensure_groups(self, group_type: str, keys) -> None:
        """聚类后补建组名（默认用组内一个视频的文件名）。"""
        for key in keys:
            self.conn.execute(
                "INSERT OR IGNORE INTO groups (group_type, group_key, name) VALUES (?, ?, ?)",
                (group_type, int(key), self._member_stem(group_type, int(key))),
            )
        self.conn.commit()

    def group_names(self, group_type: str) -> dict[int, str]:
        rows = self.conn.execute(
            "SELECT group_key, name FROM groups WHERE group_type = ?", (group_type,)
        ).fetchall()
        return {int(r["group_key"]): r["name"] for r in rows if r["name"]}

    def rename_group(self, group_type: str, key: int, name: str) -> None:
        name = (name or "").strip()
        self.conn.execute(
            "UPDATE groups SET name = ? WHERE group_type = ? AND group_key = ?",
            (name, group_type, int(key)),
        )
        self.conn.commit()

    def groups_overview(self, group_type: str) -> list[dict]:
        """返回 [{key, name, members:[{id,name,status}]}]，name 未命名时用默认。"""
        col = "music_group" if group_type == "music" else "dance_group"
        rows = self.conn.execute(
            f"SELECT id, name, status, {col} AS gkey FROM videos "
            f"WHERE {col} IS NOT NULL ORDER BY {col}, name"
        ).fetchall()
        named = {
            r["group_key"]: r["name"]
            for r in self.conn.execute(
                "SELECT group_key, name FROM groups WHERE group_type = ?", (group_type,)
            )
        }
        grouped: dict[int, dict] = {}
        for r in rows:
            k = int(r["gkey"])
            grouped.setdefault(k, {"key": k, "members": []})
            grouped[k]["members"].append({"id": int(r["id"]), "name": r["name"], "status": r["status"]})
        out = []
        for k in sorted(grouped):
            members = grouped[k]["members"]
            out.append({
                "key": k,
                "name": named.get(k) or Path(members[0]["name"]).stem,
                "members": members,
            })
        return out

    # ---------- 状态 / 去重 / 重提取 / 删除 ----------
    def reset_stale_extracting(self) -> None:
        """把上次异常中断残留的 extracting 状态清回 pending。"""
        self.conn.execute("UPDATE videos SET status='pending' WHERE status='extracting'")
        self.conn.commit()

    def reset_video(self, video_id: int) -> None:
        """标记为待处理并清空特征，用于「重新提取」。"""
        self.conn.execute(
            "UPDATE videos SET status='pending', error='', music_group=NULL, dance_group=NULL, "
            "confidence=NULL, audio_npz=NULL, motion_npz=NULL, updated_at=? WHERE id=?",
            (_now(), video_id),
        )
        self.conn.commit()

    def delete_videos(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        self.conn.execute(f"DELETE FROM video_categories WHERE video_id IN ({marks})", ids)
        self.conn.execute(f"DELETE FROM videos WHERE id IN ({marks})", ids)
        self.conn.commit()

    def video_categories(self) -> dict[int, list[int]]:
        rows = self.conn.execute("SELECT video_id, category_id FROM video_categories").fetchall()
        out: dict[int, list[int]] = {}
        for r in rows:
            out.setdefault(int(r["video_id"]), []).append(int(r["category_id"]))
        return out

    def set_video_categories(self, video_id: int, category_ids: list[int]) -> None:
        self.conn.execute("DELETE FROM video_categories WHERE video_id = ?", (video_id,))
        self.conn.executemany(
            "INSERT OR IGNORE INTO video_categories (video_id, category_id) VALUES (?, ?)",
            [(video_id, c) for c in category_ids],
        )
        self.conn.commit()

    # ---------- 分类 ----------
    def list_categories(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT c.id, c.name, COUNT(vc.video_id) n "
            "FROM categories c LEFT JOIN video_categories vc ON vc.category_id = c.id "
            "GROUP BY c.id ORDER BY c.name"
        ).fetchall()
        return [{"id": r["id"], "name": r["name"], "count": r["n"]} for r in rows]

    def create_category(self, name: str) -> int | None:
        name = (name or "").strip()
        if not name:
            return None
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO categories (name, created_at) VALUES (?, ?)", (name, _now())
        )
        self.conn.commit()
        if cur.rowcount:
            return int(cur.lastrowid)
        row = self.conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
        return int(row["id"]) if row else None

    def rename_category(self, cat_id: int, name: str) -> None:
        self.conn.execute("UPDATE categories SET name = ? WHERE id = ?", ((name or "").strip(), cat_id))
        self.conn.commit()

    def delete_category(self, cat_id: int) -> None:
        self.conn.execute("DELETE FROM video_categories WHERE category_id = ?", (cat_id,))
        self.conn.execute("DELETE FROM categories WHERE id = ?", (cat_id,))
        self.conn.commit()
