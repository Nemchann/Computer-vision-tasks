import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QAction, QFileDialog, QMessageBox,
    QStatusBar, QSplitter, QPushButton, QGroupBox, QLabel, QListWidget, QListWidgetItem, QApplication
)
from PyQt5.QtCore import Qt
from gui.image_viewer import ImageViewer
from gui.video_player import VideoPlayer
from core.utils import load_image, load_video
from core.image_matcher import find_most_similar_pair, get_sift_features
from core.background_subtractor import BackgroundSubtractor
from core.background_replacer import replace_background


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Состояние приложения
        self.images = []  # Список загруженных изображений
        self.video_cap = None
        self.bg_subtractor = None  # Вычитатель фона
        self.background_image = None  # Статичное изображение для замены фона

        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()

    def _setup_ui(self):
        """Настраивает основной интерфейс."""
        self.setWindowTitle("Сопоставление изображений и работа с видео")
        self.setMinimumSize(1400, 800)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Разделитель
        splitter = QSplitter(Qt.Horizontal)

        # Левая панель — список изображений / видео
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)

        self.image_list = QListWidget()
        self.image_list.setMaximumWidth(200)
        left_layout.addWidget(QLabel("Загруженные изображения:"))
        left_layout.addWidget(self.image_list)

        splitter.addWidget(self.left_panel)

        # Центральная панель — отображение
        self.viewer = ImageViewer(title="Результат")
        splitter.addWidget(self.viewer)

        # Правая панель — видео
        self.video_player = VideoPlayer(title="Видео")
        splitter.addWidget(self.video_player)

        splitter.setSizes([200, 500, 500])
        main_layout.addWidget(splitter)

        # Панель управления
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)

        # Группа "Загрузка"
        load_group = QGroupBox("Загрузка")
        load_layout = QVBoxLayout(load_group)

        btn_load_images = QPushButton("Загрузить изображения (3-10)...")
        btn_load_images.clicked.connect(self._on_load_images)
        load_layout.addWidget(btn_load_images)

        btn_load_video = QPushButton("Загрузить видео...")
        btn_load_video.clicked.connect(self._on_load_video)
        load_layout.addWidget(btn_load_video)

        controls_layout.addWidget(load_group)

        # Группа "Сопоставление"
        match_group = QGroupBox("Сопоставление изображений")
        match_layout = QVBoxLayout(match_group)

        btn_match = QPushButton("Найти два наиболее похожих")
        btn_match.clicked.connect(self._find_similar_images)
        match_layout.addWidget(btn_match)

        controls_layout.addWidget(match_group)

        # Группа "Обработка видео"
        video_group = QGroupBox("Обработка видео")
        video_layout = QVBoxLayout(video_group)

        btn_load_bg = QPushButton("Загрузить изображение для фона...")
        btn_load_bg.clicked.connect(self._on_load_background)
        video_layout.addWidget(btn_load_bg)

        btn_bg_subtract = QPushButton("Вычитание фона")
        btn_bg_subtract.clicked.connect(self._subtract_background)
        video_layout.addWidget(btn_bg_subtract)

        btn_bg_replace = QPushButton("Замена фона")
        btn_bg_replace.clicked.connect(self._replace_background)
        video_layout.addWidget(btn_bg_replace)

        btn_motion_blur = QPushButton("Размытие движущихся объектов")
        btn_motion_blur.clicked.connect(self._apply_motion_blur)
        video_layout.addWidget(btn_motion_blur)

        btn_optical_flow = QPushButton("Оптический поток")
        btn_optical_flow.clicked.connect(self._show_optical_flow)
        video_layout.addWidget(btn_optical_flow)

        controls_layout.addWidget(video_group)

        controls_layout.addStretch(1)
        main_layout.addWidget(controls_panel)

    def _setup_menu(self):
        """Создаёт меню."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Файл")

        action_exit = QAction("Выход", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    def _setup_statusbar(self):
        """Настраивает статус-бар."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Готово к работе.")

    def _on_load_images(self):
        """Загрузка нескольких изображений."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите изображения (3-10)",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)"
        )

        if not filepaths:
            return

        # строгое неравенство 3 < N < 10:
        if len(filepaths) < 4 or len(filepaths) > 9:
            QMessageBox.warning(self, "Внимание", "Необходимо выбрать от 4 до 9 изображений (3 < N < 10).")
            return

        self.images.clear()
        self.image_list.clear()

        for filepath in filepaths:
            try:
                image = load_image(filepath)
                self.images.append(image)

                # Добавляем в список
                item = QListWidgetItem(filepath.split('/')[-1])
                self.image_list.addItem(item)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить {filepath}:\n{e}")

        self.statusbar.showMessage(f"Загружено изображений: {len(self.images)}")

    def _on_load_video(self):
        """Загрузка видео."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            "",
            "Видео (*.mp4 *.avi *.mov *.mkv);;Все файлы (*)"
        )

        if not filepath:
            return

        try:
            self.video_player.load_video(filepath)
            self.statusbar.showMessage(f"Загружено видео: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить видео:\n{e}")

    def _on_load_background(self):
        """Загрузка статичного изображения для замены фона."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение для фона",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)"
        )

        if not filepath:
            return

        try:
            from core.utils import load_image
            self.background_image = load_image(filepath)
            self.statusbar.showMessage(f"Фон загружен: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить фон:\n{e}")

    def _find_similar_images(self):
        """Находит и отображает два наиболее похожих изображения."""
        # Проверка условия 3 < N < 10
        if len(self.images) < 4 or len(self.images) > 9:
            QMessageBox.warning(self, "Внимание", "Загрузите от 4 до 9 изображений.")
            return

        try:
            self.setCursor(Qt.WaitCursor)
            self.statusbar.showMessage("Анализ изображений... Это может занять несколько секунд.")

            # Находим лучшую пару
            idx1, idx2, score, matches = find_most_similar_pair(self.images)

            if score == 0:
                QMessageBox.warning(self, "Результат", "Не удалось найти совпадающие признаки между изображениями.")
                self.unsetCursor()
                return

            # Извлекаем ключевые точки для визуализации
            kp1, _ = get_sift_features(self.images[idx1])
            kp2, _ = get_sift_features(self.images[idx2])

            # Рисуем линии совпадений между двумя изображениями
            # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS убирает точки без совпадений
            result_image = cv2.drawMatches(
                self.images[idx1], kp1,
                self.images[idx2], kp2,
                matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # Отображаем результат в центральной панели
            self.viewer.set_image(result_image)

            self.statusbar.showMessage(
                f"Найдена пара: Изображение {idx1 + 1} и Изображение {idx2 + 1} "
                f"(совпадений: {score})"
            )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сопоставлении:\n{e}")
        finally:
            self.unsetCursor()

    def _subtract_background(self):
        """Демонстрация вычитания фона на видео."""
        if self.video_player.cap is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите видео.")
            return

        try:
            # Создаём новый вычитатель с более мягкими параметрами
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200,  # Уменьшили с 500 для быстрой адаптации
                varThreshold=25.0,  # Увеличили с 16.0 (более чувствительный)
                detectShadows=True
            )

            # Переходим в начало видео
            self.video_player.current_frame = 0
            self.video_player.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # "Обучаем" модель на первых кадрах
            warmup_frames = 100  # Увеличили количество
            self.statusbar.showMessage(f"Обучение модели фона ({warmup_frames} кадров)...")
            QApplication.processEvents()  # Обновляем UI

            for i in range(warmup_frames):
                ret, frame = self.video_player.cap.read()
                if not ret:
                    break
                bg_subtractor.apply(frame)
                # Показываем прогресс каждые 20 кадров
                if i % 20 == 0:
                    self.statusbar.showMessage(f"Обучение: {i}/{warmup_frames} кадров")
                    QApplication.processEvents()

            # Возвращаемся к началу
            self.video_player.current_frame = 0
            self.video_player.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Берём кадр для демонстрации (пропускаем первые 10, чтобы модель стабилизировалась)
            for i in range(10):
                ret, frame = self.video_player.cap.read()
                if not ret:
                    break

            if not ret:
                QMessageBox.warning(self, "Внимание", "Не удалось прочитать кадр из видео.")
                return

            # Применяем вычитание фона
            fg_mask = bg_subtractor.apply(frame)

            # Создаём изображение только с передним планом
            foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)

            # Показываем МАСКУ (это важнее!)
            # Маска — это чёрно-белое изображение, где белое = объект
            self.viewer.set_image(fg_mask)

            self.statusbar.showMessage(
                f"Вычитание фона выполнено. Белые пиксели = движущийся объект, "
                f"чёрные = фон, серые (127) = тень"
            )

            # Сохраняем ссылку, чтобы не создавать каждый раз
            self.bg_subtractor = bg_subtractor

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при вычитании фона:\n{e}")

    def _replace_background(self):
        """Заменяет фон в текущем кадре видео на загруженное изображение."""
        if self.video_player.cap is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите видео.")
            return
        if self.background_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение для нового фона.")
            return
        if self.bg_subtractor is None:
            QMessageBox.warning(self, "Внимание", "Сначала выполните 'Вычитание фона', чтобы обучить модель.")
            return

        try:
            # Берём текущий кадр из видеоплеера
            # (Для демо мы обрабатываем один кадр. Для всего видео нужен цикл по всем кадрам с сохранением в файл)
            self.video_player.cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_player.current_frame)
            ret, frame = self.video_player.cap.read()

            if not ret:
                QMessageBox.warning(self, "Внимание", "Не удалось прочитать кадр.")
                return

            # Получаем маску переднего плана для этого кадра
            fg_mask = self.bg_subtractor.apply(frame)

            # Заменяем фон
            result = replace_background(frame, self.background_image, fg_mask)

            # Показываем результат
            self.viewer.set_image(result)
            self.statusbar.showMessage("Фон успешно заменён на текущем кадре")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при замене фона:\n{e}")


    def _apply_motion_blur(self):
        """Размытие движущихся объектов."""
        QMessageBox.information(self, "Размытие", "Функция будет реализована на следующем шаге.")

    def _show_optical_flow(self):
        """Оптический поток."""
        QMessageBox.information(self, "Оптический поток", "Функция будет реализована на следующем шаге.")