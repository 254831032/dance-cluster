# DanceCluster · 舞蹈视频自动分组

把一批舞蹈视频按「**同一首 BGM**」和「**同一支舞**」自动分组，导出成文件夹，方便人工复核。

## 它做什么

```
视频 → ① 切前 N 秒 → ② 抽音频特征（chroma，变调不变）
                  → ③ 提骨架（MediaPipe 33 点，自动选主舞者）
      → ④ 音乐相似度聚类 → ⑤ 组内动作 DTW 距离 → ⑥ 动作聚类
      → 输出：CSV + M1/D1 目录结构
```

几个刻意设计的点：

- **变调不变**：chroma 按音高类别表示，变调 = 12 个 bin 的循环移位，所以「调高一两个半音」能正确归到同一组
- **镜像不变**：同时比较原序列和左右镜像序列，取更小距离
- **不看体型/外貌**：髋中心平移 + 躯干长度缩放 + 人体坐标系旋转 + 关节角度特征
- **加速版不合并**：BPM 差超过阈值直接判不同组（DTW 本身对速度不敏感，必须额外约束）
- **断点续跑**：每个视频的特征落在 `_cache/`，中断后重跑不会重算

## 本地运行（macOS / Linux / Windows）

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 准备 ffmpeg / ffprobe：放到 .tools/bin/ 或放进 PATH
# 准备模型：models/pose_landmarker_heavy.task

python -m dancecluster.gui       # 图形界面
```

命令行：

```bash
python -m dancecluster.cli run --data ./data --cache ./_cache --out ./output
python -m dancecluster.cli index --data ./data      # 只提特征
python -m dancecluster.cli group --data ./data      # 用缓存分组
```

## 输出

```
output/
├── 分组结果.csv          # 文件、音乐组、动作组、置信度、BPM、状态
├── M1/
│   ├── D1/ *.mp4
│   └── D2/ *.mp4
└── M2/ ...
```

## Windows 打包（GitHub Actions）

仓库推到 GitHub 后：

1. Actions 里选 **build-windows** → Run workflow（或直接 push 到 main 自动触发）
2. 构建完成后在 Artifacts 下载 `DanceCluster-windows`
3. 解压得到 `DanceCluster.exe`，双击即用（无需安装 Python / ffmpeg）

打包内容：Python 解释器、mediapipe、姿态模型、ffmpeg、PySide6 全部塞进单个 exe。

## 目录

```
dancecluster/
├── config.py      # 所有参数
├── ffmpeg.py      # 切片、抽音频、变调
├── audio.py       # chroma 特征 + 变调不变相似度
├── pose.py        # MediaPipe 骨架提取、主舞者选择、镜头切分
├── features.py    # 归一化、角度特征、镜像、重采样
├── fastdtw.py     # numba 带约束 DTW
├── dtw.py         # DTW 封装（镜像/子序列/多段）
├── clustering.py  # 层次聚类 + 置信度
├── pipeline.py    # 端到端：索引 → 匹配 → 导出
├── cli.py         # 命令行
└── gui.py         # PySide6 界面
```
