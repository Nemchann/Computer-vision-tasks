import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QAction, QFileDialog, QMessageBox,
    QStatusBar, QSplitter, QPushButton, QGroupBox,
    QSpinBox, QLabel, QComboBox
)
from PyQt5.QtCore import Qt

from core.segmentation import segment_kmeans
from core.edge_detectors import apply_roberts, apply_canny
from gui.image_viewer import ImageViewer
from core.keypoint_detectors import detect_sift, detect_surf, detect_fast
from core.utils import load_image_original, load_image_as_grayscale, imwrite_unicode
from core.license_plate import detect_license_plate


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Состояние приложения
        self.original_image = None  # Цветной оригинал (BGR)
        self.grayscale_image = None  # Полутоновая версия для детекторов границ

        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()

    def _setup_ui(self):
        """Настраивает основной интерфейс: три панели (исходное / результат / операции)."""
        self.setWindowTitle("Контуры, ключевые точки и сегментация")
        self.setMinimumSize(1400, 700)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Разделитель между тремя панелями
        splitter = QSplitter(Qt.Horizontal)

        # Левая панель — исходное изображение
        self.viewer_original = ImageViewer(title="Исходное изображение")
        splitter.addWidget(self.viewer_original)

        # Средняя панель — результат
        self.viewer_result = ImageViewer(title="Результат")
        splitter.addWidget(self.viewer_result)

        # Настраиваем пропорции: 40% / 40% / 20%
        splitter.setSizes([500, 500, 300])

        main_layout.addWidget(splitter)

        # Правая панель — операции
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(5, 5, 5, 5)

        # Кнопка сброса
        btn_reset = QPushButton("Вернуться к исходному")
        btn_reset.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        btn_reset.clicked.connect(self._reset_to_original)
        controls_layout.addWidget(btn_reset)

        controls_layout.addSpacing(10)

        # === Группа 1: Детекторы границ ===
        edges_group = QGroupBox("Детекторы границ")
        edges_layout = QVBoxLayout(edges_group)

        # Canny с двумя порогами
        canny_layout = QVBoxLayout()
        canny_layout.addWidget(QLabel("Детектор Canny:"))

        thresh1_layout = QHBoxLayout()
        thresh1_layout.addWidget(QLabel("Порог 1 (min):"))
        self.spin_canny_t1 = QSpinBox()
        self.spin_canny_t1.setRange(0, 255)
        self.spin_canny_t1.setValue(50)
        thresh1_layout.addWidget(self.spin_canny_t1)
        canny_layout.addLayout(thresh1_layout)

        thresh2_layout = QHBoxLayout()
        thresh2_layout.addWidget(QLabel("Порог 2 (max):"))
        self.spin_canny_t2 = QSpinBox()
        self.spin_canny_t2.setRange(0, 255)
        self.spin_canny_t2.setValue(150)
        thresh2_layout.addWidget(self.spin_canny_t2)
        canny_layout.addLayout(thresh2_layout)

        btn_canny = QPushButton("Применить Canny")
        btn_canny.clicked.connect(self._apply_canny)
        canny_layout.addWidget(btn_canny)

        edges_layout.addLayout(canny_layout)
        edges_layout.addSpacing(5)

        # Робертс
        btn_roberts = QPushButton("Оператор Робертса")
        btn_roberts.clicked.connect(self._apply_roberts)
        edges_layout.addWidget(btn_roberts)

        controls_layout.addWidget(edges_group)

        # === Группа 2: Сегментация ===
        seg_group = QGroupBox("Сегментация (K-means)")
        seg_layout = QVBoxLayout(seg_group)

        n_clusters_layout = QHBoxLayout()
        n_clusters_layout.addWidget(QLabel("Количество кластеров (N):"))
        self.spin_n_clusters = QSpinBox()
        self.spin_n_clusters.setRange(2, 20)
        self.spin_n_clusters.setValue(5)
        n_clusters_layout.addWidget(self.spin_n_clusters)
        seg_layout.addLayout(n_clusters_layout)

        btn_kmeans = QPushButton("Сегментировать")
        btn_kmeans.clicked.connect(self._apply_kmeans)
        seg_layout.addWidget(btn_kmeans)

        controls_layout.addWidget(seg_group)

        # === Группа 3: Ключевые точки ===
        kp_group = QGroupBox("Ключевые точки")
        kp_layout = QVBoxLayout(kp_group)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Метод:"))
        self.combo_keypoint_method = QComboBox()
        self.combo_keypoint_method.addItems(["SIFT", "SURF", "FAST"])
        method_layout.addWidget(self.combo_keypoint_method)
        kp_layout.addLayout(method_layout)

        btn_keypoints = QPushButton("Выделить ключевые точки")
        btn_keypoints.clicked.connect(self._apply_keypoints)
        kp_layout.addWidget(btn_keypoints)

        controls_layout.addWidget(kp_group)

        # === Группа 4: Автомобильный номер ===
        plate_group = QGroupBox("Автомобильный номер")
        plate_layout = QVBoxLayout(plate_group)

        # Выбор режима
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Режим:"))
        self.combo_plate_mode = QComboBox()
        self.combo_plate_mode.addItem("Нарисовать рамку", "rect")
        self.combo_plate_mode.addItem("Обрезать номер", "crop")
        mode_layout.addWidget(self.combo_plate_mode)
        plate_layout.addLayout(mode_layout)

        btn_plate = QPushButton("Выделить номер")
        btn_plate.clicked.connect(self._apply_license_plate)
        plate_layout.addWidget(btn_plate)

        controls_layout.addWidget(plate_group)

        # Растягивающий spacer, чтобы группы были прижаты к верху
        controls_layout.addStretch(1)

        # Добавляем панель управления в главный layout
        main_layout.addWidget(controls_panel)

    def _setup_menu(self):
        """Создаёт меню приложения."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Файл")

        action_load = QAction("Загрузить изображение...", self)
        action_load.setShortcut("Ctrl+O")
        action_load.triggered.connect(self._on_load_image)
        file_menu.addAction(action_load)

        action_save = QAction("Сохранить результат...", self)
        action_save.setShortcut("Ctrl+S")
        action_save.triggered.connect(self._on_save_result)
        file_menu.addAction(action_save)

        file_menu.addSeparator()

        action_exit = QAction("Выход", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    def _setup_statusbar(self):
        """Настраивает статус-бар."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Готово к работе. Загрузите изображение.")

    def _on_load_image(self):
        """Обработчик загрузки изображения."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Все файлы (*)"
        )

        if not filepath:
            return

        try:
            # Загружаем цветной оригинал
            self.original_image = load_image_original(filepath)

            # Конвертируем в grayscale для детекторов границ
            self.grayscale_image = load_image_as_grayscale(filepath)

            # Отображаем цветной оригинал слева
            self.viewer_original.set_image(self.original_image)

            # Справа тоже показываем цветной оригинал (как стартовую точку)
            self.viewer_result.set_image(self.original_image)

            h, w = self.original_image.shape[:2]
            channels = self.original_image.shape[2] if len(self.original_image.shape) == 3 else 1
            self.statusbar.showMessage(
                f"Загружено: {filepath} | Размер: {w}×{h} | Каналов: {channels}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{e}")

    def _on_save_result(self):
        """Обработчик сохранения результата."""
        result_image = self.viewer_result.get_image()

        if result_image is None:
            QMessageBox.warning(self, "Внимание", "Нет изображения для сохранения.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            "",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;Все файлы (*)"
        )

        if not filepath:
            return

        try:
            success = imwrite_unicode(filepath, result_image)
            if success:
                self.statusbar.showMessage(f"Сохранено: {filepath}")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении:\n{e}")

    def _reset_to_original(self):
        """Возвращает результат к исходному цветному изображению."""
        if self.original_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        self.viewer_result.set_image(self.original_image)
        self.statusbar.showMessage("Возврат к исходному изображению")

    # === Заглушки для операций (будем наполнять на следующих шагах) ===

    def _apply_canny(self):
        """Применяет детектор границ Кэнни."""
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        t1 = self.spin_canny_t1.value()
        t2 = self.spin_canny_t2.value()

        if t1 >= t2:
            QMessageBox.warning(self, "Внимание", "Порог 1 должен быть меньше порога 2.")
            return

        try:
            result = apply_canny(self.grayscale_image, t1, t2)
            self.viewer_result.set_image(result)
            self.statusbar.showMessage(f"Применён детектор Canny (пороги: {t1}, {t2})")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить Canny:\n{e}")

    def _apply_roberts(self):
        """Применяет оператор Робертса."""
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        try:
            result = apply_roberts(self.grayscale_image)
            self.viewer_result.set_image(result)
            self.statusbar.showMessage("Применён оператор Робертса")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить Робертса:\n{e}")

    def _apply_kmeans(self):
        """Применяет сегментацию изображения с помощью K-means."""
        if self.original_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        n = self.spin_n_clusters.value()

        try:
            # Показываем курсор загрузки, так как K-means может работать пару секунд
            self.setCursor(Qt.WaitCursor)

            # Применяем сегментацию к ЦВЕТНОМУ изображению
            result = segment_kmeans(self.original_image, n)

            self.viewer_result.set_image(result)
            self.statusbar.showMessage(f"Сегментация K-means завершена (кластеров: {n})")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сегментации:\n{e}")
        finally:
            # Возвращаем обычный курсор
            self.unsetCursor()

    def _apply_keypoints(self):
        """Выделяет ключевые точки выбранным методом."""
        if self.original_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        method = self.combo_keypoint_method.currentText()

        try:
            # Ключевые точки детектируются на grayscale
            gray = self.grayscale_image

            if method == "SIFT":
                result, keypoints, _ = detect_sift(gray)
                count = len(keypoints) if keypoints else 0
                self.statusbar.showMessage(f"Метод SIFT: найдено {count} ключевых точек")

            elif method == "SURF":
                result, keypoints, _ = detect_surf(gray)
                count = len(keypoints) if keypoints else 0
                self.statusbar.showMessage(f"Метод SURF: найдено {count} ключевых точек")

            elif method == "FAST":
                result, keypoints, _ = detect_fast(gray, threshold=20)
                count = len(keypoints) if keypoints else 0
                self.statusbar.showMessage(f"Метод FAST: найдено {count} ключевых точек")

            # Рисуем ключевые точки поверх ЦВЕТНОГО изображения для наглядности
            if self.original_image is not None and len(self.original_image.shape) == 3:
                # Конвертируем grayscale-результат в BGR для корректного отображения
                # (drawKeypoints уже возвращает цветное изображение, но на всякий случай)
                if len(result.shape) == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            self.viewer_result.set_image(result)

        except cv2.error as e:
            # Специальная обработка для SURF (патентные ограничения)
            if "SURF" in str(e) or "non-free" in str(e).lower():
                QMessageBox.critical(
                    self, "Ошибка SURF",
                    "Метод SURF недоступен в вашей версии OpenCV.\n\n"
                    "SURF запатентован, и для его использования OpenCV должен быть собран "
                    "с флагом OPENCV_ENABLE_NONFREE.\n\n"
                    "Попробуйте использовать SIFT или FAST."
                )
            else:
                QMessageBox.critical(self, "Ошибка OpenCV", f"Ошибка детектора:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось выделить ключевые точки:\n{e}")

    def _apply_license_plate(self):
        """Выделяет автомобильный номер на изображении."""
        if self.original_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        # Определяем режим: рамка или обрезка
        mode = self.combo_plate_mode.currentData()
        draw_rect = (mode == "rect")

        try:
            result_image, plate_found = detect_license_plate(self.original_image, draw_rect)

            if plate_found:
                self.viewer_result.set_image(result_image)
                mode_text = "рамка" if draw_rect else "обрезка"
                self.statusbar.showMessage(f"Автомобильный номер найден (режим: {mode_text})")
            else:
                QMessageBox.warning(
                    self, "Номер не найден",
                    "На изображении не удалось обнаружить объект, похожий на автомобильный номер.\n\n"
                    "Попробуйте:\n"
                    "• Загрузить изображение с чётко видимым номером\n"
                    "• Убедиться, что номер расположен горизонтально\n"
                    "• Использовать изображение с хорошим освещением"
                )
                self.viewer_result.set_image(result_image)
                self.statusbar.showMessage("Автомобильный номер не найден")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось выделить номер:\n{e}")