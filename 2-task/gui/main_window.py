import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QAction, QFileDialog, QMessageBox,
    QStatusBar, QSplitter, QPushButton, QGroupBox, QSpinBox, QLabel, QDialog
)
from PyQt5.QtCore import Qt
from gui.image_viewer import ImageViewer
from gui.kernel_editor import KernelEditor
from core.utils import load_image_original, load_image_as_grayscale, imwrite_unicode
from gui.custom_filter_dialog import CustomFilterDialog
from core.filters import (
    get_sharpen_kernel, get_motion_blur_kernel, get_emboss_kernel,
    apply_median_filter, apply_custom_filter
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Состояние приложения
        self.original_image = None  # Цветной оригинал (BGR)
        self.grayscale_image = None  # Полутоновая версия для морфологии

        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()

    def _setup_ui(self):
        """Настраивает основной интерфейс."""
        self.setWindowTitle("Морфологические операции с изображениями")
        self.setMinimumSize(1000, 600)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Разделитель между панелями (можно тянуть мышкой)
        splitter = QSplitter(Qt.Horizontal)

        # Левая панель — исходное изображение
        self.viewer_original = ImageViewer(title="Исходное изображение")
        splitter.addWidget(self.viewer_original)

        # Правая панель — результат
        self.viewer_result = ImageViewer(title="Результат")
        splitter.addWidget(self.viewer_result)

        # Настраиваем пропорции разделителя (40% / 40% / 20% для панели управления)
        splitter.setSizes([450, 450, 300])

        main_layout.addWidget(splitter)

        # Панель управления (справа)
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)

        # Редактор ядра
        self.kernel_editor = KernelEditor()
        self.kernel_editor.kernel_changed.connect(self._on_kernel_changed)
        controls_layout.addWidget(self.kernel_editor)

        # Группа "Морфологические операции"
        morph_group = QGroupBox("Морфологические операции")
        morph_layout = QVBoxLayout(morph_group)

        # Кнопка сброса к исходному изображению
        btn_reset = QPushButton("Вернуться к исходному")
        btn_reset.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        btn_reset.clicked.connect(self._reset_to_original)
        morph_layout.addWidget(btn_reset)

        # Разделитель
        morph_layout.addSpacing(10)

        # Количество итераций
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Итерации:"))
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(1, 10)
        self.spin_iterations.setValue(1)
        iter_layout.addWidget(self.spin_iterations)
        morph_layout.addLayout(iter_layout)

        # Кнопки операций
        btn_erosion = QPushButton("Эрозия")
        btn_erosion.clicked.connect(lambda: self._apply_morphology('erosion'))
        morph_layout.addWidget(btn_erosion)

        btn_dilation = QPushButton("Дилатация")
        btn_dilation.clicked.connect(lambda: self._apply_morphology('dilation'))
        morph_layout.addWidget(btn_dilation)

        btn_opening = QPushButton("Открытие")
        btn_opening.clicked.connect(lambda: self._apply_morphology('opening'))
        morph_layout.addWidget(btn_opening)

        btn_closing = QPushButton("Закрытие")
        btn_closing.clicked.connect(lambda: self._apply_morphology('closing'))
        morph_layout.addWidget(btn_closing)

        btn_gradient = QPushButton("Градиент")
        btn_gradient.clicked.connect(lambda: self._apply_morphology('gradient'))
        morph_layout.addWidget(btn_gradient)

        btn_tophat = QPushButton("Цилиндр (Top-Hat)")
        btn_tophat.clicked.connect(lambda: self._apply_morphology('tophat'))
        morph_layout.addWidget(btn_tophat)

        btn_blackhat = QPushButton("Чёрная шляпа")
        btn_blackhat.clicked.connect(lambda: self._apply_morphology('blackhat'))
        morph_layout.addWidget(btn_blackhat)

        controls_layout.addWidget(morph_group)

        # Группа "Фильтры"
        filters_group = QGroupBox("Фильтры")
        filters_layout = QVBoxLayout(filters_group)

        # Повышение резкости
        btn_sharpen = QPushButton("Повышение резкости")
        btn_sharpen.clicked.connect(self._apply_sharpen)
        filters_layout.addWidget(btn_sharpen)

        # Размытие в движении
        motion_layout = QHBoxLayout()
        motion_layout.addWidget(QLabel("Размер:"))
        self.spin_motion_size = QSpinBox()
        self.spin_motion_size.setRange(3, 21)
        self.spin_motion_size.setValue(9)
        self.spin_motion_size.setSingleStep(2)  # Только нечётные
        motion_layout.addWidget(self.spin_motion_size)

        motion_layout.addWidget(QLabel("Угол:"))
        self.spin_motion_angle = QSpinBox()
        self.spin_motion_angle.setRange(0, 180)
        self.spin_motion_angle.setValue(0)
        motion_layout.addWidget(self.spin_motion_angle)

        btn_motion_blur = QPushButton("Размытие")
        btn_motion_blur.clicked.connect(self._apply_motion_blur)
        filters_layout.addWidget(btn_motion_blur)

        # Тиснение
        btn_emboss = QPushButton("Тиснение")
        btn_emboss.clicked.connect(self._apply_emboss)
        filters_layout.addWidget(btn_emboss)

        # Медианный фильтр
        median_layout = QHBoxLayout()
        median_layout.addWidget(QLabel("Размер:"))
        self.spin_median_size = QSpinBox()
        self.spin_median_size.setRange(3, 21)
        self.spin_median_size.setValue(3)
        self.spin_median_size.setSingleStep(2)  # Только нечётные
        median_layout.addWidget(self.spin_median_size)

        btn_median = QPushButton("Медианный")
        btn_median.clicked.connect(self._apply_median)
        filters_layout.addWidget(btn_median)

        # Пользовательский фильтр
        btn_custom = QPushButton("Пользовательский фильтр...")
        btn_custom.clicked.connect(self._apply_custom_filter)
        filters_layout.addWidget(btn_custom)

        controls_layout.addWidget(filters_group)

        # Добавляем панель управления в главный layout
        main_layout.addWidget(controls_panel)

    def _setup_menu(self):
        """Создаёт меню приложения."""
        menubar = self.menuBar()

        # Меню "Файл"
        file_menu = menubar.addMenu("Файл")

        # Действие: Загрузить
        action_load = QAction("Загрузить изображение...", self)
        action_load.setShortcut("Ctrl+O")
        action_load.triggered.connect(self._on_load_image)
        file_menu.addAction(action_load)

        # Действие: Сохранить результат
        action_save = QAction("Сохранить результат...", self)
        action_save.setShortcut("Ctrl+S")
        action_save.triggered.connect(self._on_save_result)
        file_menu.addAction(action_save)

        file_menu.addSeparator()

        # Действие: Выход
        action_exit = QAction("Выход", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    def _setup_statusbar(self):
        """Настраивает статус-бар внизу окна."""
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
            return  # Пользователь отменил выбор

        try:
            # Загружаем оригинал (цветной)
            self.original_image = load_image_original(filepath)

            # Конвертируем в grayscale для морфологии
            self.grayscale_image = load_image_as_grayscale(filepath)

            # Отображаем оригинал в левой панели
            self.viewer_original.set_image(self.original_image)

            # В правой панели пока просто копируем grayscale
            # (позже здесь будет результат операций)
            self.viewer_result.set_image(self.grayscale_image)

            # Обновляем статус-бар
            h, w = self.grayscale_image.shape[:2]
            self.statusbar.showMessage(
                f"Загружено: {filepath} | Размер: {w}×{h} | "
                f"Каналов: {len(self.original_image.shape) if len(self.original_image.shape) > 2 else 1}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{e}")

    def _reset_to_original(self):
        """Возвращает результат к исходному grayscale изображению."""
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        # Восстанавливаем исходное grayscale изображение
        self.viewer_result.set_image(self.grayscale_image)
        self.statusbar.showMessage("Возврат к исходному изображению")

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
            # Используем нашу функцию с поддержкой Unicode
            success = imwrite_unicode(filepath, result_image)
            if success:
                self.statusbar.showMessage(f"Сохранено: {filepath}")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении:\n{e}")

    def _on_kernel_changed(self, kernel: np.ndarray):
        """Обработчик изменения структурного элемента."""
        self.current_kernel = kernel

    def _apply_morphology(self, operation: str):
        """
        Применяет морфологическую операцию к текущему изображению.

        Args:
            operation: Тип операции ('erosion', 'dilation', 'opening',
                      'closing', 'gradient', 'tophat', 'blackhat')
        """
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        iterations = self.spin_iterations.value()
        kernel = self.current_kernel

        try:
            # Применяем операцию
            if operation == 'erosion':
                result = cv2.erode(self.grayscale_image, kernel, iterations=iterations)
            elif operation == 'dilation':
                result = cv2.dilate(self.grayscale_image, kernel, iterations=iterations)
            elif operation == 'opening':
                result = cv2.morphologyEx(self.grayscale_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif operation == 'closing':
                result = cv2.morphologyEx(self.grayscale_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            elif operation == 'gradient':
                result = cv2.morphologyEx(self.grayscale_image, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
            elif operation == 'tophat':
                result = cv2.morphologyEx(self.grayscale_image, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
            elif operation == 'blackhat':
                result = cv2.morphologyEx(self.grayscale_image, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
            else:
                raise ValueError(f"Неизвестная операция: {operation}")

            # Отображаем результат
            self.viewer_result.set_image(result)

            # Обновляем статус-бар
            self.statusbar.showMessage(f"Применена операция: {operation} | Итерации: {iterations}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить операцию:\n{e}")

    def _apply_sharpen(self):
        """Применяет фильтр повышения резкости."""
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        try:
            kernel = get_sharpen_kernel()
            result = apply_custom_filter(self.grayscale_image, kernel)
            self.viewer_result.set_image(result)
            self.statusbar.showMessage("Применён фильтр: Повышение резкости")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить фильтр:\n{e}")

    def _apply_motion_blur(self):
        """Применяет эффект размытия в движении."""
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        try:
            size = self.spin_motion_size.value()
            angle = self.spin_motion_angle.value()
            kernel = get_motion_blur_kernel(size, angle)
            result = apply_custom_filter(self.grayscale_image, kernel)
            self.viewer_result.set_image(result)
            self.statusbar.showMessage(f"Применён фильтр: Размытие в движении (размер={size}, угол={angle}°)")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить фильтр:\n{e}")

    def _apply_emboss(self):
        """Применяет эффект тиснения."""
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        try:
            kernel = get_emboss_kernel()
            result = apply_custom_filter(self.grayscale_image, kernel)
            self.viewer_result.set_image(result)
            self.statusbar.showMessage("Применён фильтр: Тиснение")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить фильтр:\n{e}")

    def _apply_median(self):
        """Применяет медианный фильтр."""
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        try:
            size = self.spin_median_size.value()
            result = apply_median_filter(self.grayscale_image, size)
            self.viewer_result.set_image(result)
            self.statusbar.showMessage(f"Применён фильтр: Медианный (размер={size})")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить фильтр:\n{e}")

    def _apply_custom_filter(self):
        """Открывает диалог для ввода пользовательского фильтра."""
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        dialog = CustomFilterDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            try:
                kernel = dialog.get_kernel()
                result = apply_custom_filter(self.grayscale_image, kernel)
                self.viewer_result.set_image(result)
                self.statusbar.showMessage(
                    f"Применён пользовательский фильтр (размер={kernel.shape[0]}×{kernel.shape[1]})")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось применить фильтр:\n{e}")
