"""PySide6 桌面界面：选目录 → 后台处理 → 看分组结果。

设计原则：
- 所有重活在 QThread 里跑，界面不卡；
- 进度、日志实时回传；
- 结果表可直接看出每组的置信度，低置信度的行标色，方便人工复核。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import pipeline
from .config import AppConfig


class _Cancelled(Exception):
    pass


class Worker(QThread):
    progress = Signal(int, int, str, str)
    log = Signal(str)
    done = Signal(object, str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, data_dir: str, out_dir: str, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.cfg = cfg
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _on_progress(self, i: int, total: int, name: str, stage: str) -> None:
        if self._cancel:
            raise _Cancelled()
        self.progress.emit(i, total, name, stage)

    def run(self) -> None:
        try:
            records = pipeline.index_all(self.data_dir, self.cfg, progress=self._on_progress)
            if self._cancel:
                self.cancelled.emit()
                return
            self.log.emit(f"提取完成（{len(records)} 个视频），开始计算音乐相似度…")
            result = pipeline.group(records, self.cfg)
            if self._cancel:
                self.cancelled.emit()
                return
            self.log.emit("分组完成，正在导出…")
            out = pipeline.export(result, self.out_dir)
            self.done.emit(result, str(out))
        except _Cancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DanceCluster · 舞蹈视频自动分组")
        self.resize(1080, 720)
        self.worker: Worker | None = None
        self.settings = QSettings("DanceCluster", "DanceCluster")
        self._build_ui()
        self._restore_paths()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        # 路径
        paths = QGroupBox("目录")
        form = QFormLayout(paths)
        self.data_edit = QLineEdit()
        self.data_btn = QPushButton("选择…")
        self.data_btn.clicked.connect(lambda: self._pick_dir(self.data_edit))
        form.addRow("视频目录", self._row(self.data_edit, self.data_btn))

        self.out_edit = QLineEdit()
        self.out_btn = QPushButton("选择…")
        self.out_btn.clicked.connect(lambda: self._pick_dir(self.out_edit))
        form.addRow("输出目录", self._row(self.out_edit, self.out_btn))
        root.addWidget(paths)

        # 参数
        params = QGroupBox("参数")
        pf = QFormLayout(params)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 3600)
        self.start_spin.setSuffix(" 秒")
        self.start_spin.setValue(0.0)
        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(2, 600)
        self.dur_spin.setSuffix(" 秒")
        self.dur_spin.setValue(20.0)
        self.music_spin = QDoubleSpinBox()
        self.music_spin.setRange(0.30, 0.99)
        self.music_spin.setSingleStep(0.01)
        self.music_spin.setValue(0.55)
        self.dance_spin = QDoubleSpinBox()
        self.dance_spin.setRange(0.05, 0.99)
        self.dance_spin.setSingleStep(0.01)
        self.dance_spin.setValue(0.35)
        self.mirror_check = QCheckBox("镜像视为同一支舞")
        self.mirror_check.setChecked(True)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("起始"))
        row1.addWidget(self.start_spin)
        row1.addWidget(QLabel("时长"))
        row1.addWidget(self.dur_spin)
        row1.addStretch(1)
        pf.addRow(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("音乐阈值"))
        row2.addWidget(self.music_spin)
        row2.addWidget(QLabel("动作阈值"))
        row2.addWidget(self.dance_spin)
        row2.addWidget(self.mirror_check)
        row2.addStretch(1)
        pf.addRow(row2)
        root.addWidget(params)

        # 操作
        actions = QHBoxLayout()
        self.run_btn = QPushButton("开始处理")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.open_btn = QPushButton("打开输出目录")
        self.open_btn.clicked.connect(self._open_output)
        self.open_btn.setEnabled(False)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.open_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        # 进度
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.status = QLabel("就绪")
        root.addWidget(self.progress)
        root.addWidget(self.status)

        # 结果 + 日志
        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["文件", "音乐组", "动作组", "置信度", "BPM", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 6):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        split.addWidget(self.table)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        split.addWidget(self.log_view)
        split.setSizes([420, 180])
        root.addWidget(split, stretch=1)

        self.setCentralWidget(central)

    @staticmethod
    def _row(*widgets):
        lay = QHBoxLayout()
        for w in widgets:
            lay.addWidget(w)
        lay.setContentsMargins(0, 0, 0, 0)
        return lay

    # ---------- 行为 ----------
    def _pick_dir(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择目录", edit.text() or str(Path.home()))
        if d:
            edit.setText(d)

    def _restore_paths(self) -> None:
        self.data_edit.setText(self.settings.value("data", ""))
        self.out_edit.setText(self.settings.value("out", ""))
        self.start_spin.setValue(float(self.settings.value("start", 0.0)))
        self.dur_spin.setValue(float(self.settings.value("dur", 20.0)))
        self.music_spin.setValue(float(self.settings.value("music", 0.55)))
        self.dance_spin.setValue(float(self.settings.value("dance", 0.35)))
        self.mirror_check.setChecked(self.settings.value("mirror", True, type=bool))

    def _save_paths(self) -> None:
        self.settings.setValue("data", self.data_edit.text())
        self.settings.setValue("out", self.out_edit.text())
        self.settings.setValue("start", self.start_spin.value())
        self.settings.setValue("dur", self.dur_spin.value())
        self.settings.setValue("music", self.music_spin.value())
        self.settings.setValue("dance", self.dance_spin.value())
        self.settings.setValue("mirror", self.mirror_check.isChecked())

    def _cfg(self) -> AppConfig:
        cfg = AppConfig()
        cfg.clip.start = self.start_spin.value()
        cfg.clip.duration = self.dur_spin.value()
        cfg.match.music_threshold = self.music_spin.value()
        cfg.match.dance_threshold = self.dance_spin.value()
        cfg.match.mirror_invariant = self.mirror_check.isChecked()
        cfg.cache_dir = str(Path(self.out_edit.text()) / "_cache")
        return cfg

    def _start(self) -> None:
        data_dir = self.data_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        if not data_dir or not Path(data_dir).is_dir():
            QMessageBox.warning(self, "提示", "请先选择一个包含视频的目录")
            return
        if not out_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return
        self._save_paths()
        self.table.setRowCount(0)
        self.log_view.clear()
        self.progress.setValue(0)
        self.status.setText("准备中…")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)

        self.worker = Worker(data_dir, out_dir, self._cfg(), self)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status.setText("正在取消（当前视频跑完就停）…")

    def _on_progress(self, i: int, total: int, name: str, stage: str) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(i)
        self.status.setText(f"[{i}/{total}] {name} · {stage}")

    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _on_done(self, result, out_dir: str) -> None:
        self._fill_table(result)
        self.progress.setValue(self.progress.maximum())
        self.status.setText(f"完成：{len(result.records)} 个视频，输出在 {out_dir}")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.open_btn.setEnabled(True)
        self._last_out = out_dir

    def _fill_table(self, result) -> None:
        self.table.setRowCount(len(result.records))
        for i, r in enumerate(result.records):
            low_conf = result.confidence[i] < 0.5
            values = [
                r.name,
                f"M{result.music_labels[i]}",
                f"D{result.dance_labels[i]}",
                f"{result.confidence[i]:.2f}",
                f"{result.tempo[i]:.1f}",
                r.status,
            ]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if low_conf and c in (2, 3):
                    item.setBackground(QColor(255, 235, 200))
                if r.status == "no_pose":
                    item.setForeground(QColor(180, 60, 60))
                self.table.setItem(i, c, item)

    def _on_failed(self, tb: str) -> None:
        self._log(tb)
        self.status.setText("处理失败，详见日志")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QMessageBox.critical(self, "出错了", tb.strip().splitlines()[-1] if tb else "未知错误")

    def _on_cancelled(self) -> None:
        self.status.setText("已取消")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _open_output(self) -> None:
        out = getattr(self, "_last_out", self.out_edit.text())
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("DanceCluster")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
