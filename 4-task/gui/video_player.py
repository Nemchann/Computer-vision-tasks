import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PyQt5.QtCore import Qt, QTimer
from core.utils import numpy_to_qimage
from PyQt5.QtGui import QPixmap


class VideoPlayer(QWidget):
    """Виджет для воспроизведения видео."""

    def __init__(self, title: str = "Видео", parent=None):
        super().__init__(parent)
        self.title = title
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label_title = QLabel(self.title)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet(
            "font-weight: bold; font-size: 14px; padding: 5px; "
            "background-color: #f0f0f0; border: 1px solid #ccc;"
        )

        self.label_video = QLabel()
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setMinimumSize(400, 300)
        self.label_video.setStyleSheet(
            "background-color: #2b2b2b; border: 1px solid #555; "
            "color: #888; font-size: 14px;"
        )
        self.label_video.setText("Нет видео")

        # Контролы воспроизведения
        controls_layout = QHBoxLayout()

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._toggle_play)
        controls_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        controls_layout.addWidget(self.btn_stop)

        self.slider_position = QSlider(Qt.Horizontal)
        self.slider_position.setRange(0, 100)
        self.slider_position.valueChanged.connect(self._seek)
        controls_layout.addWidget(self.slider_position)

        layout.addWidget(self.label_title)
        layout.addWidget(self.label_video, stretch=1)
        layout.addLayout(controls_layout)

    def load_video(self, filepath: str):
        """Загружает видеофайл."""
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {filepath}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0

        self.slider_position.setRange(0, self.total_frames - 1)
        self._show_frame()

    def _show_frame(self):
        """Показывает текущий кадр."""
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        if ret:
            qimage = numpy_to_qimage(frame)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(
                self.label_video.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label_video.setPixmap(scaled_pixmap)
            self.slider_position.setValue(self.current_frame)

    def _next_frame(self):
        """Переход к следующему кадру."""
        if self.cap is None:
            return

        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0

        self._show_frame()

    def _toggle_play(self):
        """Play/Pause."""
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            interval = int(1000 / self.fps)
            self.timer.start(interval)
            self.btn_play.setText("Pause")

    def _stop(self):
        """Остановка воспроизведения."""
        self.timer.stop()
        self.btn_play.setText("Play")
        self.current_frame = 0
        self._show_frame()

    def _seek(self, position: int):
        """Перемотка."""
        self.current_frame = position
        self._show_frame()

    def closeEvent(self, event):
        """Освобождение ресурсов."""
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)