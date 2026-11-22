import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QSplitter, QInputDialog
)
from PyQt5.QtCore import Qt
from gui.image_viewer import ImageViewer
from core.face_detector import validate_single_face, get_face_crop, draw_face_boxes
from core.face_encoder import extract_single_face_encoding


class DatabaseTab(QWidget):
    def __init__(self, face_db, parent=None):
        super().__init__(parent)
        self.face_db = face_db
        self._setup_ui()
        self._refresh_profile_list()

    def _setup_ui(self):
        """Настраивает интерфейс вкладки базы лиц."""
        layout = QVBoxLayout(self)

        # Разделитель
        splitter = QSplitter(Qt.Horizontal)

        # Левая панель - список профилей
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        db_group = QGroupBox("База лиц")
        db_layout = QVBoxLayout(db_group)

        self.list_profiles = QListWidget()
        self.list_profiles.setMinimumWidth(250)
        self.list_profiles.currentItemChanged.connect(self._on_profile_selected)
        db_layout.addWidget(self.list_profiles)

        # Кнопки управления
        btn_layout = QHBoxLayout()

        btn_add = QPushButton("Добавить лицо")
        btn_add.clicked.connect(self._on_add_face)
        btn_layout.addWidget(btn_add)

        btn_remove = QPushButton("Удалить")
        btn_remove.clicked.connect(self._on_remove_face)
        btn_layout.addWidget(btn_remove)

        db_layout.addLayout(btn_layout)
        left_layout.addWidget(db_group)

        splitter.addWidget(left_panel)

        # Правая панель - отображение выбранного лица
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.viewer = ImageViewer(title="Фото из базы")
        right_layout.addWidget(self.viewer)

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 500])

        layout.addWidget(splitter)

    def _refresh_profile_list(self):
        """Обновляет список профилей в QListWidget."""
        self.list_profiles.clear()
        profiles = self.face_db.get_all_profiles()
        for p in profiles:
            item = QListWidgetItem(p["name"])
            item.setData(Qt.UserRole, p["image_path"])  # Сохраняем путь в элементе
            self.list_profiles.addItem(item)

    def _on_profile_selected(self, current, previous):
        """Отображает фото при выборе профиля в списке."""
        if current is None:
            self.viewer.clear()
            return

        image_path = current.data(Qt.UserRole)
        try:
            image = cv2.imread(image_path)
            if image is not None:
                self.viewer.set_image(image)
            else:
                self.viewer.clear()
        except Exception:
            self.viewer.clear()

    def _on_add_face(self):
        """Обработчик добавления лица в базу (с поддержкой нескольких фото)."""
        # ВАЖНО: getOpenFileNames (с 's' на конце) возвращает список файлов
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите одно или несколько фото лица", "",
            "Изображения (*.jpg *.jpeg *.png *.bmp)"
        )
        if not filepaths:
            return

        # Запрашиваем имя один раз для всех выбранных фото
        name, ok = QInputDialog.getText(self, "Новый профиль", "Введите имя человека:")
        if not ok or not name.strip():
            return

        name = name.strip()
        added_count = 0
        skipped_count = 0

        for filepath in filepaths:
            try:
                image = cv2.imread(filepath)
                if image is None:
                    skipped_count += 1
                    continue

                is_valid, message, face_location = validate_single_face(image)
                if not is_valid:
                    skipped_count += 1
                    continue

                face_crop = get_face_crop(image, face_location, padding=30)
                success, msg = self.face_db.add_person(name, face_crop)

                if success:
                    added_count += 1

            except Exception as e:
                print(f"Ошибка обработки {filepath}: {e}")
                skipped_count += 1
                continue

        if added_count > 0:
            QMessageBox.information(
                self, "Успех",
                f"Добавлено фото: {added_count}\n"
                f"Пропущено: {skipped_count}\n\n"
                f"Чем больше фото, тем точнее распознавание!"
            )
            self._refresh_profile_list()
        else:
            QMessageBox.warning(
                self, "Ошибка",
                "Не удалось добавить ни одного фото.\n"
                "Проверьте, что на изображениях ровно одно лицо."
            )


    def _on_remove_face(self):
        """Обработчик удаления лица из базы."""
        current_item = self.list_profiles.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "Внимание", "Выберите профиль для удаления.")
            return

        name = current_item.text()
        reply = QMessageBox.question(
            self, "Подтверждение",
            f"Удалить профиль '{name}' из базы?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            success, msg = self.face_db.remove_person(name)
            if success:
                QMessageBox.information(self, "Успех", msg)
                self._refresh_profile_list()
                self.viewer.clear()
            else:
                QMessageBox.warning(self, "Ошибка", msg)