import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QLabel, QFileDialog, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt
from gui.image_viewer import ImageViewer
from core.face_detector import detect_faces, draw_face_boxes, validate_single_face
from core.face_encoder import extract_face_encoding
from core.face_matcher import FaceMatcher


class RecognitionTab(QWidget):
    def __init__(self, face_db, parent=None):
        super().__init__(parent)
        self.face_db = face_db
        self.matcher = FaceMatcher()
        self.current_image = None
        self._setup_ui()

    def _setup_ui(self):
        """Настраивает интерфейс вкладки распознавания."""
        layout = QVBoxLayout(self)

        # Верхняя часть - два изображения
        top_layout = QHBoxLayout()

        # Левая панель - исходное изображение
        self.viewer_original = ImageViewer(title="Изображение для распознавания")
        top_layout.addWidget(self.viewer_original)

        # Правая панель - результат с рамкой
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

        # Слайдер порога уверенности
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Порог уверенности:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(20, 120)
        self.slider_threshold.setValue(60)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.slider_threshold)
        self.label_threshold = QLabel("60")
        threshold_layout.addWidget(self.label_threshold)
        control_layout.addLayout(threshold_layout)

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
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", "Изображения (*.jpg *.jpeg *.png *.bmp)"
        )
        if not filepath:
            return

        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Не удалось прочитать изображение.")

            self.current_image = image
            self.viewer_original.set_image(image)

            # Проверяем, есть ли лицо
            is_valid, message, face_location = validate_single_face(image)
            if not is_valid:
                QMessageBox.warning(self, "Внимание", message)
                self.current_image = None

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{e}")

    def _on_threshold_changed(self, value):
        """Обработчик изменения порога уверенности."""
        self.label_threshold.setText(str(value))
        self.matcher.set_threshold(float(value))

    def _on_recognize(self):
        """Обработчик распознавания лица."""
        if self.current_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        # Проверяем, есть ли лица в базе
        profiles = self.face_db.get_all_profiles()
        if len(profiles) < 2:
            QMessageBox.warning(
                self, "Внимание",
                "Для распознавания необходимо минимум 2 профиля в базе лиц.\n"
                "Добавьте ещё хотя бы одного человека."
            )
            return

        try:
            # Проверяем, что на изображении ровно одно лицо
            is_valid, message, face_location = validate_single_face(self.current_image)
            if not is_valid:
                QMessageBox.warning(self, "Ошибка валидации", message)
                return

            # Обучаем распознаватель на текущей базе
            success = self.matcher.train(profiles)
            if not success:
                QMessageBox.warning(self, "Ошибка", "Не удалось обучить распознаватель.")
                return

            # Распознаём
            name, confidence = self.matcher.predict(self.current_image)

            # Всегда показываем лучшее совпадение, но с пометкой
            if confidence < 50:
                status_text = f"✅ Высокая уверенность: {name} (confidence: {confidence:.1f})"
                color_style = "background-color: #d4edda; border: 1px solid #28a745; color: #155724;"
            elif confidence < 80:
                status_text = f"⚠️ Средняя уверенность: {name} (confidence: {confidence:.1f})"
                color_style = "background-color: #fff3cd; border: 1px solid #ffc107; color: #856404;"
            else:
                status_text = f"❓ Низкая уверенность: возможно {name} (confidence: {confidence:.1f})"
                color_style = "background-color: #f8d7da; border: 1px solid #dc3545; color: #721c24;"

            print(status_text)
            self.label_result.setText(status_text)
            self.label_result.setStyleSheet(
                f"font-size: 16px; font-weight: bold; padding: 10px; {color_style}"
            )

            # Рисуем результат
            result_image = draw_face_boxes(
                self.current_image,
                [face_location],
                color=(0, 255, 0) if name != "Неизвестно" else (0, 0, 255),
                thickness=3
            )

            # Добавляем текст с именем
            if face_location is not None:
                top, right, bottom, left = face_location
                text = f"{name} ({confidence:.1f})"
                cv2.putText(
                    result_image, text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if name != "Неизвестно" else (0, 0, 255),
                    2
                )

            self.viewer_result.set_image(result_image)

            # Обновляем метку результата
            if name != "Неизвестно":
                self.label_result.setText(f"Распознано: {name} (уверенность: {confidence:.1f})")
                self.label_result.setStyleSheet(
                    "font-size: 16px; font-weight: bold; padding: 10px; "
                    "background-color: #d4edda; border: 1px solid #28a745; color: #155724;"
                )
            else:
                self.label_result.setText(f"Неизвестное лицо (уверенность: {confidence:.1f})")
                self.label_result.setStyleSheet(
                    "font-size: 16px; font-weight: bold; padding: 10px; "
                    "background-color: #f8d7da; border: 1px solid #dc3545; color: #721c24;"
                )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка распознавания:\n{e}")