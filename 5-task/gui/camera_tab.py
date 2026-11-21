from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QLabel, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt


class CameraTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.combo_camera.addItem("Камера 0 (по умолчанию)", 0)
        self.combo_camera.addItem("Камера 1", 1)
        self.combo_camera.addItem("Камера 2", 2)
        camera_layout.addWidget(self.combo_camera)
        control_layout.addLayout(camera_layout)

        # Кнопки управления
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("Запустить камеру")
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.btn_start.clicked.connect(self._on_start_camera)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Остановить камеру")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        self.btn_stop.clicked.connect(self._on_stop_camera)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        control_layout.addLayout(btn_layout)

        # Статус
        self.label_status = QLabel("Статус: камера не активна")
        self.label_status.setAlignment(Qt.AlignCenter)
        self.label_status.setStyleSheet(
            "font-size: 14px; padding: 5px; background-color: #f5f5f5;"
        )
        control_layout.addWidget(self.label_status)

        layout.addWidget(control_group)

    def _on_start_camera(self):
        """Обработчик запуска камеры."""
        QMessageBox.information(self, "Камера", "Функция будет реализована на следующем шаге.")

    def _on_stop_camera(self):
        """Обработчик остановки камеры."""
        QMessageBox.information(self, "Камера", "Функция будет реализована на следующем шаге.")