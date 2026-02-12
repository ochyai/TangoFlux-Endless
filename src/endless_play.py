#!/usr/bin/env python3
"""
TangoFlux Endless Player v7 - subprocess分離 + アーカイブフォールバック
- 音声生成を完全に別プロセス（subprocess.Popen）で実行 → GIL競合ゼロ
- .npyファイル経由でプロセス間通信
- バッファ切れ時はアーカイブからエンベディング類似度でフォールバック再生
- Nレイヤー重ね（ずらし投入）
"""

import sys
import os
import time
import json
import tempfile
import shutil
import subprocess
import threading
import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QSpinBox, QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

SAMPLE_RATE = 44100
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(SCRIPT_DIR, "clip_archive")


# ── ユーティリティ ──

def safe_listdir(path):
    """os.listdirの安全ラッパー"""
    if not path:
        return []
    try:
        return os.listdir(path)
    except (OSError, FileNotFoundError, TypeError):
        return []


def find_similar_clip(current_embedding, recently_used, top_k=3):
    """アーカイブからエンベディング類似度の高いクリップを取得"""
    index_path = os.path.join(ARCHIVE_DIR, "index.json")
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path, "r") as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not index:
        return None

    scored = []
    for entry in index:
        if entry["id"] in recently_used:
            continue
        emb = np.array(entry["embedding"], dtype=np.float32)
        sim = float(np.dot(current_embedding, emb))
        scored.append((sim, entry))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = scored[:top_k]
    pick = candidates[np.random.randint(len(candidates))]
    entry = pick[1]

    clip_path = os.path.join(ARCHIVE_DIR, entry["filename"])
    if not os.path.exists(clip_path):
        return None
    try:
        audio = np.load(clip_path)
        return audio, entry["id"], entry["prompt"], pick[0]
    except Exception:
        return None


# ── Layer ──

class Layer:
    """再生中の1クリップ。エンベロープ事前適用済み"""
    def __init__(self, audio, fade_in_samples, fade_out_samples, clip_id):
        self.length = len(audio)
        self.fade_out = min(fade_out_samples, self.length // 2)
        fade_in = min(fade_in_samples, self.length // 2)
        self.pos = 0
        self.clip_id = clip_id
        self.done = False

        envelope = np.ones(self.length, dtype=np.float32)
        if fade_in > 0:
            t = np.linspace(0, np.pi / 2, fade_in, dtype=np.float32)
            envelope[:fade_in] = np.sin(t) ** 2
        if self.fade_out > 0:
            t = np.linspace(0, np.pi / 2, self.fade_out, dtype=np.float32)
            envelope[-self.fade_out:] = np.cos(t) ** 2
        self.audio = (audio * envelope).astype(np.float32)

    def read(self, n):
        if self.done:
            return None
        start = self.pos
        end = min(start + n, self.length)
        chunk = self.audio[start:end]
        self.pos = end
        if end >= self.length:
            self.done = True
        return chunk

    @property
    def progress(self):
        return self.pos / self.length if self.length > 0 else 1.0


# ── GUI ──

class EndlessPlayer(QWidget):
    play_signal = pyqtSignal(str)
    gen_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stream = None
        self.layers = []
        self.layers_lock = threading.Lock()
        self.running = False
        self.clip_count = 0
        self.layer_info_str = "---"
        self.worker_proc = None
        self.temp_dir = None
        self._feeder_stopped = threading.Event()

        self.play_signal.connect(lambda t: self.play_label.setText(f"Player: {t}"))
        self.gen_signal.connect(lambda t: self.gen_label.setText(f"Generator: {t}"))
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TangoFlux Endless Player")
        self.setMinimumSize(650, 450)
        self.setStyleSheet("""
            QWidget { background-color: #1a1a2e; color: #e0e0e0; }
            QGroupBox { border: 1px solid #444; border-radius: 6px;
                        margin-top: 10px; padding-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; }
            QPushButton { background-color: #e94560; color: white; border: none;
                          border-radius: 6px; padding: 10px 24px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #ff6b81; }
            QPushButton:disabled { background-color: #555; color: #999; }
            QPushButton#stopBtn { background-color: #555; }
            QPushButton#stopBtn:hover { background-color: #777; }
            QTextEdit { background-color: #16213e; border: 1px solid #444; border-radius: 4px;
                        color: #e0e0e0; padding: 6px; }
            QSpinBox { background-color: #16213e; border: 1px solid #444;
                       border-radius: 4px; color: #e0e0e0; padding: 4px; }
            QLabel#title { font-size: 20px; font-weight: bold; color: #e94560; }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)

        title = QLabel("TangoFlux Endless Player")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        grp = QGroupBox("Prompt")
        gl = QVBoxLayout()
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText("A gentle rain falling on leaves with distant thunder")
        self.prompt_edit.setMaximumHeight(60)
        self.prompt_edit.setFont(QFont("Menlo", 13))
        gl.addWidget(self.prompt_edit)
        grp.setLayout(gl)
        layout.addWidget(grp)

        pgrp = QGroupBox("Parameters")
        pl = QHBoxLayout()
        pl.addWidget(QLabel("Duration:"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 30)
        self.duration_spin.setValue(18)
        self.duration_spin.setSuffix("s")
        pl.addWidget(self.duration_spin)
        pl.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 100)
        self.steps_spin.setValue(25)
        pl.addWidget(self.steps_spin)
        pl.addWidget(QLabel("Crossfade:"))
        self.fade_spin = QSpinBox()
        self.fade_spin.setRange(3, 15)
        self.fade_spin.setValue(6)
        self.fade_spin.setSuffix("s")
        pl.addWidget(self.fade_spin)
        pl.addWidget(QLabel("Layers:"))
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(2, 5)
        self.overlap_spin.setValue(3)
        pl.addWidget(self.overlap_spin)
        pgrp.setLayout(pl)
        layout.addWidget(pgrp)

        cl = QHBoxLayout()
        self.start_btn = QPushButton("Start Endless Play")
        self.start_btn.clicked.connect(self.start_endless)
        cl.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_endless)
        cl.addWidget(self.stop_btn)
        self.update_btn = QPushButton("Update Prompt")
        self.update_btn.setEnabled(False)
        self.update_btn.clicked.connect(self.update_prompt)
        cl.addWidget(self.update_btn)
        layout.addLayout(cl)

        sgrp = QGroupBox("Status")
        sl = QVBoxLayout()
        mono = QFont("Menlo", 12)
        self.gen_label = QLabel("Generator: idle")
        self.gen_label.setFont(mono)
        sl.addWidget(self.gen_label)
        self.play_label = QLabel("Player: idle")
        self.play_label.setFont(mono)
        sl.addWidget(self.play_label)
        self.layers_label = QLabel("Layers: ---")
        self.layers_label.setFont(mono)
        sl.addWidget(self.layers_label)
        sgrp.setLayout(sl)
        layout.addWidget(sgrp)

        self.setLayout(layout)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._update_ui)
        self.ui_timer.start(200)

    def _update_ui(self):
        self.layers_label.setText(f"Layers: {self.layer_info_str}")

    # ── Audio callback ──
    def _audio_callback(self, outdata, frames, time_info, status):
        mixed = np.zeros(frames, dtype=np.float32)
        with self.layers_lock:
            for layer in self.layers:
                chunk = layer.read(frames)
                if chunk is not None:
                    if len(chunk) < frames:
                        mixed[:len(chunk)] += chunk
                    else:
                        mixed += chunk
            self.layers = [l for l in self.layers if not l.done]
        # レイヤー数に応じたゲイン調整でクリッピング防止
        overlap = getattr(self, '_overlap', 3)
        if overlap > 1:
            mixed /= overlap
        outdata[:, 0] = mixed

    # ── Worker監視スレッド ──
    def _monitor_worker(self):
        proc = self.worker_proc
        while self.running and proc and proc.poll() is None:
            try:
                line = proc.stdout.readline()
            except (ValueError, OSError):
                break
            if line:
                msg = line.strip()
                if msg.startswith("WORKER:"):
                    self.gen_signal.emit(msg[7:].strip())
            else:
                time.sleep(0.1)
        if self.running:
            self.gen_signal.emit("Worker process ended")

    # ── Feeder thread ──
    def _feeder_loop(self):
        fade_samples = int(self._fade_sec * SAMPLE_RATE)
        overlap = self._overlap
        recently_used_archive = set()
        temp_dir = self.temp_dir  # ローカル変数にキャプチャ

        archive_available = os.path.exists(os.path.join(ARCHIVE_DIR, "index.json"))
        if archive_available:
            self.play_signal.emit("Buffering... (archive available)")
        else:
            self.play_signal.emit(f"Buffering... waiting for {overlap} clips")

        preloaded = []
        already_triggered = set()
        playback_started = False

        try:
            while self.running:
                time.sleep(0.02)

                # temp_dirから新クリップを先読み
                for fname in sorted(safe_listdir(temp_dir)):
                    if fname.startswith("clip_") and fname.endswith(".npy") and not fname.endswith(".tmp"):
                        fpath = os.path.join(temp_dir, fname)
                        try:
                            audio = np.load(fpath)
                            os.remove(fpath)
                            preloaded.append(audio)
                            self.play_signal.emit(f"Buffered {len(preloaded)} clips")
                        except (OSError, ValueError):
                            pass

                # バッファが減ったらアーカイブから自動補充（常にoverlap個キープ）
                if playback_started and len(preloaded) < overlap:
                    shortage = overlap - len(preloaded)
                    self._fill_from_archive(preloaded, shortage,
                                            recently_used_archive, temp_dir)

                # 初期ずらし投入
                if not playback_started:
                    if len(preloaded) < overlap:
                        self._fill_from_archive(preloaded, overlap - len(preloaded),
                                                recently_used_archive, temp_dir)
                    if len(preloaded) >= overlap:
                        playback_started = True
                        clip_len = len(preloaded[0])
                        interval = clip_len // overlap

                        with self.layers_lock:
                            for i in range(overlap):
                                audio = preloaded.pop(0)
                                self.clip_count += 1
                                layer = Layer(audio, fade_samples, fade_samples, self.clip_count)
                                skip = (overlap - 1 - i) * interval
                                layer.pos = min(skip, layer.length - 1)
                                already_triggered.add(self.clip_count)
                                self.layers.append(layer)
                        self.play_signal.emit(f"Playing {overlap} staggered layers")
                    continue

                # レイヤー状態取得
                with self.layers_lock:
                    active = [l for l in self.layers if not l.done]

                # UI更新
                buf_info = f" [buf:{len(preloaded)}]"
                parts = [f"#{l.clip_id}: {l.progress*100:.0f}%" for l in active]
                self.layer_info_str = (" | ".join(parts) if parts else "---") + buf_info

                # トリガー判定
                need_next = False
                for l in active:
                    trigger_pos = l.length // overlap
                    if l.pos >= trigger_pos and l.clip_id not in already_triggered:
                        need_next = True
                        already_triggered.add(l.clip_id)
                        break
                if not active:
                    need_next = True

                if need_next:
                    audio = None
                    source = "gen"
                    if preloaded:
                        audio = preloaded.pop(0)
                    else:
                        result = self._get_archive_clip(recently_used_archive, temp_dir)
                        if result:
                            audio, arc_id, _, sim = result
                            recently_used_archive.add(arc_id)
                            if len(recently_used_archive) > 10:
                                recently_used_archive.clear()
                            source = f"archive(sim:{sim:.2f})"

                    if audio is not None:
                        self.clip_count += 1
                        with self.layers_lock:
                            self.layers.append(Layer(audio, fade_samples, fade_samples, self.clip_count))
                        n_active = sum(1 for l in self.layers if not l.done)
                        self.play_signal.emit(f"Playing #{self.clip_count} ({n_active} layers) [{source}]")
        finally:
            self._feeder_stopped.set()

    def _get_archive_clip(self, recently_used, temp_dir):
        """アーカイブから類似クリップを取得"""
        if not temp_dir:
            return None
        emb_path = os.path.join(temp_dir, "current_embedding.npy")
        try:
            if os.path.exists(emb_path):
                current_emb = np.load(emb_path)
                return find_similar_clip(current_emb, recently_used)
        except (OSError, ValueError):
            pass
        return None

    def _fill_from_archive(self, preloaded, count, recently_used, temp_dir):
        """アーカイブからcount個のクリップをpreloadedに追加"""
        if not temp_dir:
            return
        emb_path = os.path.join(temp_dir, "current_embedding.npy")
        try:
            if not os.path.exists(emb_path):
                return
            current_emb = np.load(emb_path)
        except (OSError, ValueError):
            return
        for _ in range(count):
            result = find_similar_clip(current_emb, recently_used)
            if result:
                audio, arc_id, _, sim = result
                preloaded.append(audio)
                recently_used.add(arc_id)
                self.play_signal.emit(f"Archive fallback: sim={sim:.2f}")

    # ── Controls ──
    def start_endless(self):
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            return

        self._fade_sec = self.fade_spin.value()
        self._overlap = self.overlap_spin.value()
        duration = self.duration_spin.value()
        steps = self.steps_spin.value()
        self.running = True
        self.clip_count = 0
        self._feeder_stopped.clear()

        with self.layers_lock:
            self.layers.clear()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.update_btn.setEnabled(True)

        self.temp_dir = tempfile.mkdtemp(prefix="tangoflux_")
        self.gen_signal.emit("Starting generator process...")

        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32',
            callback=self._audio_callback, blocksize=2048, latency='high',
        )
        self.stream.start()

        venv_python = os.path.join(SCRIPT_DIR, "..", "venv", "bin", "python")  # adjust if venv is elsewhere
        if not os.path.exists(venv_python):
            venv_python = sys.executable  # fallback to current interpreter
        worker_script = os.path.join(SCRIPT_DIR, "generator_worker.py")
        self.worker_proc = subprocess.Popen(
            [venv_python, worker_script, self.temp_dir, prompt, str(duration), str(steps)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        threading.Thread(target=self._monitor_worker, daemon=True).start()
        threading.Thread(target=self._feeder_loop, daemon=True).start()

    def stop_endless(self):
        self.running = False

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        with self.layers_lock:
            self.layers.clear()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_btn.setEnabled(False)
        self.play_signal.emit("Stopped")
        self.gen_signal.emit("idle")
        self.layer_info_str = "---"

        # クリーンアップは別スレッド（feeder停止を待ってからtemp dir削除）
        threading.Thread(target=self._cleanup_worker, daemon=True).start()

    def _cleanup_worker(self):
        temp_dir = self.temp_dir

        # feederスレッドが確実に停止するのを待つ（最大2秒）
        self._feeder_stopped.wait(timeout=2)

        # ワーカーに停止命令
        if temp_dir:
            cmd_file = os.path.join(temp_dir, "command.json")
            try:
                if os.path.isdir(temp_dir):
                    with open(cmd_file, "w") as f:
                        json.dump({"action": "stop"}, f)
            except OSError:
                pass

        # ワーカー終了待ち
        if self.worker_proc:
            try:
                self.worker_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker_proc.kill()
            self.worker_proc = None

        # temp dir削除（feeder + worker 両方停止済み）
        if temp_dir:
            try:
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                pass
        self.temp_dir = None

    def update_prompt(self):
        new = self.prompt_edit.toPlainText().strip()
        temp_dir = self.temp_dir
        if new and temp_dir:
            cmd_file = os.path.join(temp_dir, "command.json")
            try:
                if os.path.isdir(temp_dir):
                    with open(cmd_file, "w") as f:
                        json.dump({
                            "action": "update",
                            "prompt": new,
                            "duration": self.duration_spin.value(),
                            "steps": self.steps_spin.value(),
                        }, f)
                    self.gen_signal.emit("Prompt updated (next gen)")
            except OSError:
                pass

    def closeEvent(self, event):
        self.stop_endless()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    player = EndlessPlayer()
    player.show()
    sys.exit(app.exec())
