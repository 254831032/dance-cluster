# DanceCluster 项目知识库

> 本文件是这个项目的「大脑外置」——记录架构、数据模型、关键算法、踩过的坑、构建发布流程。
> 任何新会话 / 新开发者读完这份就能快速接续。

---

## 1. 项目是什么

**DanceCluster**：把一批**舞蹈视频**（用户场景：Virt-A-Mate 渲染的 3D 角色跳舞视频）按两个维度自动分组：

1. **音乐组**（同一首背景音乐）
2. **动作组**（同一支舞 / 相同舞蹈动作，忽略体型、外貌、镜像、机位）

本质是**两层无监督聚类**：先按 BGM 聚类，再在每个音乐组内按动作聚类。几百个视频量级，跑在本机（Windows 交付 / macOS 开发）。

---

## 2. 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 语言 | Python 3.12（Windows 打包）/ 3.13（Mac 开发） | 全部 Python 生态 |
| 骨架提取 | **MediaPipe Tasks**（PoseLandmarker heavy 模型，33 点 world + 2D） | `mp.solutions` 已废弃，用 Tasks API |
| 音频指纹 | **chromaprint**（`fpcalc` 二进制） | 判「是否同一首」，比 chroma 判别力强几个量级 |
| 音频特征 | librosa（chroma CQT）+ fpcalc 指纹 | chroma 只做变调兜底 |
| 动作距离 | **DTW**（numba 自实现，相对 Sakoe-Chiba 带） | dtaidistance 在 3.13 无 C 扩展，弃用 |
| 聚类 | scipy `linkage` + `fcluster`（层次聚类） | sklearn 大多数聚类不吃距离矩阵 |
| 后端 | Flask（本地 HTTP 服务，只监听 127.0.0.1） | 前端是单页 HTML 内嵌字符串 |
| 桌面壳 | **pywebview**（macOS WKWebView / Windows WebView2） | 比 Qt 轻 100 倍 |
| 托盘 | pystray（仅 Windows） | macOS 托盘与 pywebview 抢主线程，未做 |
| 存储 | **SQLite**（单文件，随输出走） | 特征存 BLOB |
| 打包 | PyInstaller（单文件 exe） | CI 在 Windows runner 上打 |

---

## 3. 目录结构

```
dance-cluster/
├── app.py                    # 打包入口：优先桌面窗口，失败退回浏览器
├── dance-cluster.spec        # PyInstaller 配置（单文件）
├── requirements.txt
├── README.md
├── docs/KNOWLEDGE_BASE.md    # 本文件
├── .github/workflows/build-windows.yml
├── dancecluster/
│   ├── config.py             # 所有配置 + 应用级 settings.json 持久化
│   ├── db.py                 # SQLite 封装（Store 类）
│   ├── ffmpeg.py             # ffmpeg/ffprobe 封装 + 缩略图/GIF/AAC
│   ├── audio.py              # chroma 特征 + chromaprint 指纹 + music_similarity
│   ├── pose.py               # MediaPipe 提取 + 骨架渲染（黑底 GIF/视频）
│   ├── features.py           # 归一化 + 关节角度 + 镜像 + L2 逐帧归一化
│   ├── fastdtw.py            # numba 带约束 DTW（相对带宽）
│   ├── dtw.py                # DTW 封装（镜像/子序列/多段）
│   ├── clustering.py         # 层次聚类 + 置信度
│   ├── pipeline.py           # 端到端：索引 → 分组 → 导出；FEATURE_VERSION
│   ├── webui.py              # Flask 后端 + 内嵌 HTML 单页（所有前端）
│   ├── desktop.py            # pywebview 窗口 + Windows 托盘
│   └── cli.py                # 命令行入口
├── .tools/bin/               # ffmpeg/ffprobe/fpcalc 二进制（不入库）
└── models/pose_landmarker_heavy.task  # 骨架模型（不入库，CI 下载）
```

---

## 4. 数据流

```
识别任务页
  ├─ 导入任务（选文件夹）→ 扫描 → 登记为 pending（只登记不提取）
  ├─ 导入视频（选单个/多个文件）→ 登记为 pending
  └─ 待提取列表（缩略图+分辨率+时长+状态）→ 勾选 → 开始提取
        │  process_pending(only_ids=勾选)
        ▼
  index_video（每个视频）：
    1. ffprobe 探测
    2. cut_clip → clip.mp4（720p 封顶，4K→720p）
    3. extract_audio（60s）→ 算 chroma+指纹 → 压成 audio.m4a，删 wav
    4. MediaPipe 提骨架 → pose.npz（world+2D+visibility+shots）
    5. prepare_features → feat/feat_mirror → 存 DB motion_npz
        ▼
  group()（全部 ok/no_pose 视频）：
    - music_similarity → 音乐距离 → 层次聚类 → music_group
    - 组内 DTW → 动作距离 → 组内聚类 → dance_group
    - 写回 DB（groups 表默认名=组内一个视频文件名）
        ▼
原文件页：浏览/搜索/过滤/多选/分类/删除/重新提取/预览
音频分组页 / 动作分组页：组名可改 + 骨架+声音预览
```

---

## 5. SQLite 数据模型

数据库在 `缓存目录/dancecluster.db`，导出时复制一份 `分组数据.db`。

```sql
videos(id, path UNIQUE, name, duration, width, height, fps, has_audio,
       status, error, cache_dir, present_ratio, tempo,
       music_group, dance_group, confidence,
       audio_npz BLOB, motion_npz BLOB, feature_version, updated_at)

runs(id, created_at, finished_at, data_dir, out_dir, params_json)

library_folders(id, path UNIQUE, added_at)      -- 早期设计，UI 已弃用（去文件夹化）

groups(id, group_type, group_key, name, UNIQUE(group_type,group_key))
   -- 分组命名，name 默认=组内一个视频文件名，可改

categories(id, name UNIQUE, created_at)          -- 自定义分类表
video_categories(video_id, category_id)          -- 视频↔分类 多对多
```

**关键约定：**
- `status`：`pending / extracting / ok / no_pose / error`
- `audio_npz`：chroma + chroma_full + tempo + duration + fps + max_lag + fingerprint（位矩阵）
- `motion_npz`：feat + feat_mirror + fps + present_ratio + shots
- `feature_version`：算法版本号，不匹配就强制重新提取（当前 = 2）
- **动作判断只用 `motion_npz`**；clip.mp4/audio/pose.npz 只服务于预览，删了不影响分组

---

## 6. 关键算法

### 6.1 音乐相似度（`audio.music_similarity`）

强判别、宁漏不错，三层：

1. **BPM 门**：八度折叠后差 >8% → 直接判不同（加速版不合并）
2. **chromaprint 指纹**：fpcalc 输出 32 位整数序列 → 展开成位矩阵 → 时间滑动取最小 BER，**BER ≤ 0.35 → 同一首**
3. **chroma 兜底**：变调不变相似度 ≥ 0.92 → 同一首（覆盖调高/调低，指纹对变调敏感）

> 早期只用 chroma + 阈值 0.55，把不同歌全并成一组（chroma 只有 12 维，不同歌也能到 0.65-0.82）。这是重大教训。

### 6.2 动作特征（`features.py`）

1. 补帧（低可见度点按时间线性插值）
2. 归一化：髋中心平移 + 躯干长度缩放 + 人体坐标系旋转（右=髋左右、上=髋→肩、前=cross）
3. 特征 = **8 个关节角 + 8 个关键点人体坐标**（32 维）
4. **逐帧 L2 归一化**（只比姿态方向，忽略幅度）—— 提交 831c7c6 加的，修动作误分组
5. 镜像不变：同时比原序列和镜像序列取最小距离

### 6.3 DTW（`fastdtw.py`）

- numba JIT，**相对 Sakoe-Chiba 带**：`center = i / (n/m)`，带内 `[center-w, center+w]`
- 为什么相对：早期用绝对对角带，**长度不一致时把路径切断、返回 INF**（所有不同时长视频都判成不同组）
- 归一化：`距离 / max(帧数)`

### 6.4 聚类（`clustering.py`）

- 音乐：`1 - 相似度` 距离矩阵 → 层次聚类 `fcluster(distance, threshold=1-music_threshold)`
- 动作：只在**同一音乐组内**聚类（`_cluster_per_group`），组编号全局唯一
- 无骨架视频（no_pose）**不参与动作分组**（dance_group=NULL）

---

## 7. 踩过的坑（务必别重犯）

| 坑 | 现象 | 修复 |
|---|---|---|
| MediaPipe `close()` 卡 60 秒 | 等遥测上传 play.googleapis.com | 异步 close + CLI 里 `os._exit(0)` |
| DTW 长度不一致返回 INF | 不同时长视频全被分成不同组 | 相对带宽（见 6.3） |
| chroma 相似度误并歌 | 不同歌归成 M1 | 换 chromaprint 指纹（见 6.1） |
| 4K 片段占空间 | clip.mp4 上百 MB | cut_clip 加 `max_height=720` 重编码 |
| audio.wav 未压缩 | 每个 2.6MB | 算完特征后压 AAC、删 wav |
| GIF 太大 | 12s 竖屏 GIF 4MB | 3fps/160宽/6s 封顶 |
| 下拉多选点一下面板就关 | 事件冒泡到 document 的关闭逻辑 | panel `onclick` stopPropagation + **把 modal 移到 script 前**（dfb239c） |
| Windows 命令行黑窗弹窗 | 子进程 ffmpeg/fpcalc 弹窗 | `CREATE_NO_WINDOW` |
| database is locked | 后台任务与页面轮询并发 | `PRAGMA journal_mode=WAL` + `busy_timeout=5000` |
| 重复启动 | 多实例 | config.acquire_instance_lock() 单实例锁 |
| GitHub 直连被墙 | github.com:443 超时 | 走 api.github.com（Git Data API 推送） |
| 旧缓存特征不兼容 | 算法改了旧结果还在 | `FEATURE_VERSION` 强制重提取 |

---

## 8. 构建发布（Windows exe）

1. 本地改完 → 推 GitHub（master）
2. GitHub Actions `build-windows.yml` 自动触发：
   - Python 3.12 + `pip install -r requirements.txt` + pyinstaller
   - 下载 ffmpeg.exe（gyan.dev essentials）+ fpcalc（chromaprint release）+ pose_landmarker_heavy.task
   - `pyinstaller dance-cluster.spec` → 单文件 `DanceCluster.exe`（~298MB）
3. Artifacts 下载，双击运行

**打包注意（spec 里）**：
- `collect_data_files/dynamic_libs("mediapipe")`、`collect_all("pythonnet","clr_loader","pystray")`
- 隐藏导入 `webview.platforms.winforms/edgechromium`（win）、`webview.platforms.cocoa`（mac）
- `console=False`（无黑窗），首次启动解压 5~10 秒

**GitHub 直连不通时的推送办法**（重要）：
用 Git Data API 推送（走 api.github.com）——建 blob → 建 tree → 建 commit → PATCH refs/heads/master。脚本逻辑见会话历史，`/tmp/dc-test/api_push.py` 是可用模板。

---

## 9. 本地开发

```bash
cd dance-cluster
../.venv/bin/python -m dancecluster.desktop   # 桌面窗口
# 或 python -m dancecluster.webui             # 浏览器模式（备用）
# 或 python -m dancecluster.cli run --data ... # 命令行
```

**环境变量**：
- `DANCECLUSTER_CONFIG_DIR`：应用配置目录（默认 ~/Library/Application Support/DanceCluster 或 %APPDATA%）
- `DANCECLUSTER_NO_BROWSER=1`：不自动开浏览器（测试用）
- `MPLCONFIGDIR`：matplotlib 缓存目录（沙箱下需要指向可写路径）

**配置持久化**：`config.json`（应用配置目录下）存 `cache_dir`。缓存目录**必填**，未设置时界面只允许进设置页。

**中间文件**（缓存目录/<视频名>/）：`clip.mp4`（720p 片段）、`audio.m4a`（AAC 音频）、`pose.npz`（骨架）。删掉缓存会自动重提取。

---

## 10. 配置项（`config.py`）

```python
ClipConfig:
  start=0.0            # 切片起始
  duration=20.0        # 骨架片段长度
  audio_duration=60.0  # 音频片段长度（0=整段）
  max_height=720       # 片段分辨率封顶（省空间）
  reencode=False       # 默认流拷贝；超 max_height 时自动重编码

AudioConfig: sample_rate=22050, hop_length=1024, match_fps=5.0, max_lag_sec=3.0
PoseConfig: model_path, num_poses=3, target_fps=15, smooth_window=5, min_visibility=0.35
MatchConfig:
  dtw_window_ratio=0.08
  music_threshold=0.60
  dance_threshold=0.35
  mirror_invariant=True
```

---

## 11. 已知待办 / 未来方向

- **骨架叠加目前是黑底**（不叠原视频）——早期叠在 clip 上，为省空间改成了黑底；若要叠原视频需保留 clip 并在渲染时读帧
- **导出功能**：pipeline 有 `export()`（CSV + M/D 目录硬链接），但新 UI 未接入口
- **macOS 托盘**：与 pywebview 抢主线程，未做（Windows 托盘已做）
- **音频多段采样**：若发现同首歌被拆成多组（不同起点），可加开头/中段/后段各取一段比对
- **分类与分组的联动**：分类是手动的（categories 表），分组是自动的（groups 表），尚未打通
