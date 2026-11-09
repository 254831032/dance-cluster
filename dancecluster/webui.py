"""本地 Web 界面：启动后自动打开浏览器，后端跑任务，前端轮询进度。

为什么不用 Qt：Qt 轮子太大（200MB+），弱网环境装不动；Flask 方案依赖极小，
打包体积反而更接近「引擎本身」的大小，启动也快。
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import traceback
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request

from . import pipeline
from .config import AppConfig

app = Flask(__name__)

_lock = threading.Lock()
_ui = {"desktop": False, "window": None}
_state: dict = {
    "running": False,
    "progress": 0,
    "total": 0,
    "stage": "",
    "name": "",
    "log": [],
    "rows": [],
    "out_dir": "",
    "error": "",
    "cancel": False,
}


class _Cancelled(Exception):
    pass


def _snapshot() -> dict:
    with _lock:
        return {
            "running": _state["running"],
            "progress": _state["progress"],
            "total": _state["total"],
            "stage": _state["stage"],
            "name": _state["name"],
            "log": list(_state["log"]),
            "rows": list(_state["rows"]),
            "out_dir": _state["out_dir"],
            "error": _state["error"],
        }


def _run_job(cfg: AppConfig, data_dir: str, out_dir: str) -> None:
    def progress(i: int, total: int, name: str, stage: str) -> None:
        if _state["cancel"]:
            raise _Cancelled()
        with _lock:
            _state.update(progress=i, total=total, name=name, stage=stage)

    try:
        records = pipeline.index_all(data_dir, cfg, progress=progress)
        if _state["cancel"]:
            raise _Cancelled()
        with _lock:
            _state["log"].append(f"提取完成（{len(records)} 个视频），计算音乐相似度…")
        result = pipeline.group(records, cfg)
        if _state["cancel"]:
            raise _Cancelled()
        out = pipeline.export(result, out_dir)
        rows = [
            {
                "file": r.name,
                "m": f"M{result.music_labels[i]}",
                "d": f"D{result.dance_labels[i]}",
                "conf": round(float(result.confidence[i]), 2),
                "bpm": round(float(result.tempo[i]), 1),
                "status": r.status,
            }
            for i, r in enumerate(result.records)
        ]
        with _lock:
            _state.update(rows=rows, out_dir=str(out), stage="完成",
                          log=_state["log"] + [f"已导出到 {out}"])
    except _Cancelled:
        with _lock:
            _state.update(stage="已取消", log=_state["log"] + ["已取消"])
    except Exception:
        tb = traceback.format_exc()
        with _lock:
            _state.update(stage="出错", error=tb, log=_state["log"] + [tb])
    finally:
        with _lock:
            _state["running"] = False


@app.get("/")
def index():
    return INDEX_HTML


@app.post("/api/start")
def start():
    with _lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "任务正在进行"}), 409
    body = request.get_json(force=True)
    data_dir = str(body.get("data", "")).strip()
    out_dir = str(body.get("out", "")).strip()
    if not data_dir or not Path(data_dir).is_dir():
        return jsonify({"ok": False, "error": "视频目录不存在"}), 400
    if not out_dir:
        return jsonify({"ok": False, "error": "请填写输出目录"}), 400

    cfg = AppConfig()
    cfg.clip.start = float(body.get("start", 0.0))
    cfg.clip.duration = float(body.get("duration", 20.0))
    cfg.match.music_threshold = float(body.get("music", 0.55))
    cfg.match.dance_threshold = float(body.get("dance", 0.35))
    cfg.match.mirror_invariant = bool(body.get("mirror", True))
    cfg.cache_dir = str(Path(out_dir) / "_cache")

    with _lock:
        _state.update(running=True, progress=0, total=0, stage="准备中", name="",
                      log=[], rows=[], out_dir="", error="", cancel=False)
    threading.Thread(target=_run_job, args=(cfg, data_dir, out_dir), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/status")
def status():
    return jsonify(_snapshot())


@app.post("/api/cancel")
def cancel():
    with _lock:
        _state["cancel"] = True
    return jsonify({"ok": True})


@app.post("/api/open")
def open_folder():
    body = request.get_json(force=True, silent=True) or {}
    path = body.get("path") or _state["out_dir"]
    if path and Path(path).exists():
        if sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("win"):
            import os
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["xdg-open", path])
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "目录不存在"}), 404


@app.post("/api/pick_folder")
def pick_folder():
    win = _ui.get("window")
    if not win:
        return jsonify({"ok": False, "error": "浏览器模式不支持系统文件夹选择，请手动填写路径"}), 400
    try:
        import webview

        kind = getattr(webview, "FileDialog", None)
        dialog = kind.FOLDER if kind is not None else webview.FOLDER_DIALOG
        result = win.create_file_dialog(dialog)
    except Exception as exc:  # pragma: no cover - 依赖桌面环境
        return jsonify({"ok": False, "error": f"打开选择框失败：{exc}"}), 500
    if not result:
        return jsonify({"ok": False, "error": "已取消"}), 400
    path = result[0] if isinstance(result, (list, tuple)) else result
    return jsonify({"ok": True, "path": str(path)})


@app.get("/api/mode")
def mode():
    return jsonify({"desktop": bool(_ui["desktop"])})


@app.post("/api/quit")
def quit_app():
    import os
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return jsonify({"ok": True})


def _free_port(start: int = 8765) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def main(open_browser: bool = True) -> int:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"DanceCluster 已启动：{url}", flush=True)
    if open_browser and os.environ.get("DANCECLUSTER_NO_BROWSER") != "1":
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
    return 0


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>DanceCluster · 舞蹈视频自动分组</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --text:#e7e9ee; --dim:#9aa3b2; --accent:#5b8cff; --warn:#ffb454; --bad:#ff6b6b; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--dim); margin-bottom: 20px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px; margin-bottom: 16px; }
  label { display:block; color: var(--dim); font-size: 12px; margin: 10px 0 4px; }
  input[type=text], input[type=number] { width:100%; background:#0c0e12; border:1px solid var(--line); color:var(--text); border-radius:8px; padding:9px 10px; font-size:13px; }
  .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .grid4 { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  button { background: var(--accent); border:0; color:white; padding:10px 18px; border-radius:8px; font-size:14px; cursor:pointer; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.ghost { background: transparent; border:1px solid var(--line); color: var(--text); }
  .row { display:flex; gap:10px; align-items:center; margin-top: 14px; }
  .bar { height: 8px; background:#0c0e12; border-radius: 99px; overflow:hidden; margin-top: 12px; }
  .bar > i { display:block; height:100%; width:0%; background: linear-gradient(90deg,#5b8cff,#7aa2ff); transition: width .3s; }
  .status { color: var(--dim); margin-top: 8px; font-size: 13px; }
  table { width:100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align:left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  th { color: var(--dim); font-weight: 500; }
  tr.lowconf td:nth-child(3), tr.lowconf td:nth-child(4) { background: rgba(255,180,84,.12); }
  tr.nopose td { color: var(--bad); }
  pre#log { background:#0c0e12; border:1px solid var(--line); border-radius:8px; padding:12px; height: 180px; overflow:auto; font-size:12px; color:#aab3c2; white-space: pre-wrap; }
  .muted { color: var(--dim); font-size: 12px; }
  .inputrow { display: flex; gap: 8px; }
  .inputrow input { flex: 1; }
  .inputrow button { padding: 8px 12px; white-space: nowrap; }
</style>
</head>
<body>
<div class="wrap">
  <h1>DanceCluster</h1>
  <div class="sub">按「同一首 BGM」和「同一支舞」自动分组 · 本地处理，视频不出机器</div>

  <div class="panel">
    <div class="grid2">
      <div>
        <label>视频目录</label>
        <div class="inputrow"><input id="data" type="text" placeholder="例如 D:\\videos 或 /Users/you/videos"><button class="ghost pick" data-target="data">选择…</button></div>
      </div>
      <div>
        <label>输出目录</label>
        <div class="inputrow"><input id="out" type="text" placeholder="例如 D:\\output"><button class="ghost pick" data-target="out">选择…</button></div>
      </div>
    </div>
    <div class="grid4" style="margin-top:14px">
      <div><label>起始（秒）</label><input id="start" type="number" value="0" step="1"></div>
      <div><label>时长（秒）</label><input id="duration" type="number" value="20" step="1"></div>
      <div><label>音乐阈值</label><input id="music" type="number" value="0.55" step="0.01" min="0.3" max="0.99"></div>
      <div><label>动作阈值</label><input id="dance" type="number" value="0.35" step="0.01" min="0.05" max="0.99"></div>
    </div>
    <div class="row">
      <label style="margin:0"><input id="mirror" type="checkbox" checked> 镜像视为同一支舞</label>
      <span style="flex:1"></span>
      <button id="run">开始处理</button>
      <button id="cancel" class="ghost" disabled>取消</button>
      <button id="open" class="ghost" disabled>打开输出目录</button>
      <button id="quit" class="ghost" style="display:none">退出程序</button>
    </div>
    <div class="bar"><i id="fill"></i></div>
    <div class="status" id="status">就绪</div>
  </div>

  <div class="panel">
    <table id="table">
      <thead><tr><th>文件</th><th>音乐组</th><th>动作组</th><th>置信度</th><th>BPM</th><th>状态</th></tr></thead>
      <tbody></tbody>
    </table>
    <div class="muted" id="empty">暂无结果</div>
  </div>

  <div class="panel">
    <label style="margin-top:0">日志</label>
    <pre id="log">等待开始…</pre>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
const inputs = ["data","out","start","duration","music","dance"];
for (const k of inputs) {
  const el = $("." + "noop") || null;
}
// 记住上次路径
for (const k of ["data","out"]) {
  const saved = localStorage.getItem("dc_" + k);
  if (saved) $("#" + k).value = saved;
  $("#" + k).addEventListener("change", () => localStorage.setItem("dc_" + k, $("#" + k).value));
}
$("#mirror").checked = localStorage.getItem("dc_mirror") !== "0";
$("#mirror").addEventListener("change", () => localStorage.setItem("dc_mirror", $("#mirror").checked ? "1" : "0"));

let timer = null, lastLog = 0;

async function start() {
  const body = {
    data: $("#data").value, out: $("#out").value,
    start: parseFloat($("#start").value) || 0,
    duration: parseFloat($("#duration").value) || 20,
    music: parseFloat($("#music").value) || 0.55,
    dance: parseFloat($("#dance").value) || 0.35,
    mirror: $("#mirror").checked,
  };
  const r = await fetch("/api/start", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
  const j = await r.json();
  if (!j.ok) { alert(j.error || "启动失败"); return; }
  $("#run").disabled = true; $("#cancel").disabled = false; $("#open").disabled = true;
  $("#empty").style.display = "none";
  poll();
  if (timer) clearInterval(timer);
  timer = setInterval(poll, 800);
}

async function cancel() {
  await fetch("/api/cancel", {method:"POST"});
  $("#cancel").disabled = true;
}

async function openFolder() {
  const r = await fetch("/api/open", {method:"POST", headers:{"Content-Type":"application/json"}, body: "{}"});
  const j = await r.json();
  if (!j.ok) alert(j.error || "打不开");
}

async function poll() {
  const r = await fetch("/api/status");
  const s = await r.json();
  const pct = s.total ? Math.round(100 * s.progress / s.total) : 0;
  $("#fill").style.width = pct + "%";
  $("#status").textContent = s.running
    ? `[${s.progress}/${s.total}] ${s.name} · ${s.stage}`
    : (s.error ? "处理失败，详见日志" : `就绪 ${s.stage ? "· " + s.stage : ""}`);
  if (s.log && s.log.length !== lastLog) {
    lastLog = s.log.length;
    $("#log").textContent = s.log.join("\\n");
    $("#log").scrollTop = $("#log").scrollHeight;
  }
  if (s.rows && s.rows.length) {
    renderRows(s.rows);
    $("#open").disabled = false;
  }
  if (!s.running) {
    $("#run").disabled = false;
    $("#cancel").disabled = true;
    if (timer) { clearInterval(timer); timer = null; }
  }
  if (s.error) { $("#run").disabled = false; $("#cancel").disabled = true; }
}

function renderRows(rows) {
  const tb = $("#table tbody");
  tb.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    if (r.conf < 0.5) tr.className = "lowconf";
    if (r.status === "no_pose") tr.className = "nopose";
    for (const key of ["file","m","d","conf","bpm","status"]) {
      const td = document.createElement("td");
      td.textContent = r[key];
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  $("#empty").style.display = "none";
}

document.querySelectorAll(".pick").forEach(btn => btn.addEventListener("click", async () => {
  const r = await fetch("/api/pick_folder", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({which: btn.dataset.target})});
  const j = await r.json();
  if (j.ok) { const el = $("#" + btn.dataset.target); el.value = j.path; el.dispatchEvent(new Event("change")); }
  else if (j.error && j.error !== "已取消") alert(j.error);
}));

fetch("/api/mode").then(r => r.json()).then(m => {
  if (!m.desktop) {
    document.querySelectorAll(".pick").forEach(b => b.style.display = "none");
  } else {
    $("#quit").style.display = "";
  }
});

$("#run").addEventListener("click", start);
$("#cancel").addEventListener("click", cancel);
$("#open").addEventListener("click", openFolder);
$("#quit").addEventListener("click", () => fetch("/api/quit", {method:"POST"}));
poll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
