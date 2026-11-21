from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from gui.image_viewer import ImageViewer


class RecognitionTab(QWidget):
    def __init__(self, face_db, parent=None):
        super().__init__(parent)
        self.face_db = face_db
        self._setup_ui()

    def _setup_ui(self):
        """Настраивает интерфейс вкладки распознавания."""
        layout = QVBoxLayout(self)

        # Верхняя часть - изображение для распознавания
        top_layout = QHBoxLayout()

        # Левая панель - исходное изображение
        self.viewer_original = ImageViewer(title="Изображение для распознавания")
        top_layout.addWidget(self.viewer_original)

        # Правая панель - результат
        self.viewer_result = ImageViewer(title="Результат распознавания")
        top_layout.addWidget(self.viewer_result)

        layout.addLayout(top_layout)

        # Нижняя часть - управление
        control_group = QGroupBox("Распознавание")
        control_layout = QVBoxLayout(control_group)

        # Кнопка загрузки
        btn_load = QPushButton("Загрузить изображение для распознавания...")
        btn_load.clicked.connect(self._on_load_image)
        control_layout.addWidget(btn_load)

        # Кнопка распознавания
        btn_recognize = QPushButton("Распознать лицо")
        btn_recognize.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 10px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        btn_recognize.clicked.connect(self._on_recognize)
        control_layout.addWidget(btn_recognize)

        # Метка результата
        self.label_result = QLabel("Результат: не распознано")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 10px; "
            "background-color: #f0f0f0; border: 1px solid #ccc;"
        )
        control_layout.addWidget(self.label_result)

        layout.addWidget(control_group)

    def _on_load_image(self):
        """Обработчик загрузки изображения для распознавания."""
        QMessageBox.information(self, "Загрузка", "Функция будет реализована на следующем шаге.")

    def _on_recognize(self):
        """Обработчик распознавания лица."""
        QMessageBox.information(self, "Распознавание", "Функция будет реализована на следующем шаге.")