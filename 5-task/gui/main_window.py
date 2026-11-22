from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QMenuBar, QMenu, QAction, QStatusBar, QMessageBox
)
from PyQt5.QtCore import Qt
from gui.database_tab import DatabaseTab
from gui.recognition_tab import RecognitionTab
from gui.camera_tab import CameraTab
from core.face_database import FaceDatabase


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Инициализируем базу данных (она сама создаст папку data/ при первом запуске)
        self.face_db = FaceDatabase()

        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()

    def _setup_ui(self):
        """Настраивает основной интерфейс с вкладками."""
        self.setWindowTitle("Система распознавания лиц")
        self.setMinimumSize(1200, 800)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Создаём вкладки
        self.tabs = QTabWidget()

        # Вкладка 1: База лиц
        self.database_tab = DatabaseTab(self.face_db, self)
        self.tabs.addTab(self.database_tab, "База лиц")

        # Вкладка 2: Распознавание
        self.recognition_tab = RecognitionTab(self.face_db, self)
        self.tabs.addTab(self.recognition_tab, "Распознавание")

        # Вкладка 3: Камера
        self.camera_tab = CameraTab(self.face_db, self)
        self.tabs.addTab(self.camera_tab, "📷 Камера")

        main_layout.addWidget(self.tabs)

    def _setup_menu(self):
        """Создаёт меню приложения."""
        menubar = self.menuBar()

        # Меню "Файл"
        file_menu = menubar.addMenu("Файл")

        action_exit = QAction("Выход", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        # Меню "Справка"
        help_menu = menubar.addMenu("Справка")

        action_about = QAction("О программе", self)
        action_about.triggered.connect(self._show_about)
        help_menu.addAction(action_about)

    def _setup_statusbar(self):
        """Настраивает статус-бар."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Готово к работе. Загрузите изображения в базу лиц.")

    def _show_about(self):
        """Показывает информацию о программе."""
        QMessageBox.about(
            self,
            "О программе",
            "Система распознавания лиц\n\n"
            "Версия 1.0\n\n"
            "Реализовано с использованием:\n"
            "• PyQt5 - графический интерфейс\n"
            "• OpenCV - обработка изображений\n"
            "• face_recognition - детекция и распознавание лиц"
        )

    def closeEvent(self, event):
        """Обработчик закрытия окна."""
        # Останавливаем камеру, если она активна
        if self.camera_tab.camera_worker is not None:
            self.camera_tab.camera_worker.stop()

        event.accept()