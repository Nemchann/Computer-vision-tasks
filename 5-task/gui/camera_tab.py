import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QLabel, QComboBox, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from core.camera_worker import CameraWorker


class CameraTab(QWidget):
    def __init__(self, face_db, parent=None):
        super().__init__(parent)
        self.face_db = face_db
        self.camera_worker = None
        self._setup_ui()

    def _setup_ui(self):
        """Настраивает интерфейс вкладки камеры."""
        layout = QVBoxLayout(self)

        # Область отображения видео
        self.label_video = QLabel("Камера не активна")
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setMinimumSize(640, 480)
        self.label_video.setStyleSheet(
            "background-color: #2b2b2b; border: 2px solid #555; "
            "color: #888; font-size: 18px;"
        )
        layout.addWidget(self.label_video)

        # Панель управления
        control_group = QGroupBox("Управление камерой")
        control_layout = QVBoxLayout(control_group)

        # Выбор камеры
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Камера:"))
        self.combo_camera = QComboBox()
        for i in range(3):
            self.combo_camera.addItem(f"Камера {i}", i)
        camera_layout.addWidget(self.combo_camera)
        control_layout.addLayout(camera_layout)

        # Слайдер порога уверенности
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Порог уверенности:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(50, 200)
        self.slider_threshold.setValue(120)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.slider_threshold)
        self.label_threshold = QLabel("120")
        threshold_layout.addWidget(self.label_threshold)
        control_layout.addLayout(threshold_layout)

        # Кнопки управления
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("▶ Запустить камеру")
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; color: #888888; }"
        )
        self.btn_start.clicked.connect(self._on_start_camera)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹ Остановить камеру")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #da190b; }"
            "QPushButton:disabled { background-color: #cccccc; color: #888888; }"
        )
        self.btn_stop.clicked.connect(self._on_stop_camera)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        control_layout.addLayout(btn_layout)

        # Статус и FPS
        info_layout = QHBoxLayout()
        self.label_status = QLabel("Статус: камера не активна")
        self.label_status.setStyleSheet("font-size: 13px; padding: 5px;")
        info_layout.addWidget(self.label_status)

        self.label_fps = QLabel("FPS: —")
        self.label_fps.setStyleSheet("font-size: 13px; padding: 5px; font-weight: bold;")
        info_layout.addWidget(self.label_fps)

        control_layout.addLayout(info_layout)

        layout.addWidget(control_group)

    def _on_start_camera(self):
        """Запуск камеры и распознавания."""
        # Проверяем базу
        profiles = self.face_db.get_all_profiles()
        if len(profiles) < 2:
            QMessageBox.warning(
                self, "Внимание",
                "В базе лиц должно быть минимум 2 фото для обучения распознавателя.\n"
                "Добавьте людей во вкладке 'База лиц'."
            )
            return

        # Создаём и запускаем worker
        camera_index = self.combo_camera.currentData()
        self.camera_worker = CameraWorker(
            face_db=self.face_db,
            camera_index=camera_index,
            detection_scale=0.5,
            confidence_threshold=float(self.slider_threshold.value())
        )

        # Подключаем сигналы
        self.camera_worker.frame_ready.connect(self._on_frame_ready)
        self.camera_worker.status_updated.connect(self._on_status_updated)
        self.camera_worker.fps_updated.connect(self._on_fps_updated)

        self.camera_worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_camera.setEnabled(False)
        self.label_status.setText("Статус: камера активна")

    def _on_stop_camera(self):
        """Остановка камеры."""
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo_camera.setEnabled(True)
        self.label_video.setText("Камера остановлена")
        self.label_video.setStyleSheet(
            "background-color: #2b2b2b; border: 2px solid #555; "
            "color: #888; font-size: 18px;"
        )
        self.label_status.setText("Статус: камера не активна")
        self.label_fps.setText("FPS: —")

    def _on_frame_ready(self, qimage):
        """Слот для получения нового кадра из потока."""
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self.label_video.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_video.setPixmap(scaled)
        self.label_video.setStyleSheet("background-color: #000; border: 2px solid #555;")

    def _on_status_updated(self, status: str):
        """Слот для обновления статуса."""
        self.label_status.setText(f"Статус: {status}")

    def _on_fps_updated(self, fps: float):
        """Слот для обновления FPS."""
        self.label_fps.setText(f"FPS: {fps:.1f}")

    def _on_threshold_changed(self, value: int):
        """Обработчик изменения порога уверенности."""
        self.label_threshold.setText(str(value))
        if self.camera_worker is not None:
            self.camera_worker.set_confidence_threshold(float(value))

    def closeEvent(self, event):
        """Останавливаем камеру при закрытии вкладки."""
        if self.camera_worker is not None:
            self.camera_worker.stop()
        super().closeEvent(event)