import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QGridLayout, QSpinBox, QLabel, QGroupBox, QScrollArea,
    QWidget, QDialogButtonBox, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal


class KernelCell(QPushButton):
    """
    Одна клетка структурного элемента.
    Хранит состояние (0 или 1) и меняет цвет при клике.
    """

    def __init__(self, row: int, col: int, cell_size: int, parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col
        self.value = 1  # По умолчанию все клетки активны (1)

        # Настройка внешнего вида — квадратные клетки
        self.setFixedSize(cell_size, cell_size)
        self._update_style()

        # Подключаем обработчик клика
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self):
        """Переключает значение клетки при клике."""
        self.value = 1 - self.value  # 0 → 1 или 1 → 0
        self._update_style()

    def _update_style(self):
        """Обновляет стиль кнопки в зависимости от значения."""
        if self.value == 1:
            # Активный пиксель — чёрный
            self.setStyleSheet(
                "background-color: #2b2b2b; border: 1px solid #555;"
            )
        else:
            # Неактивный пиксель — белый
            self.setStyleSheet(
                "background-color: #ffffff; border: 1px solid #ccc;"
            )

    def set_value(self, value: int):
        """Устанавливает значение клетки (0 или 1)."""
        self.value = value
        self._update_style()


class KernelEditorDialog(QDialog):
    """
    Модальное окно для интерактивного редактирования структурного элемента.
    Позволяет задать размер ядра (до 51×51) и кликать по клеткам.
    """

    # Сигнал, который испускается при подтверждении (нажатии OK)
    kernel_accepted = pyqtSignal(object)

    def __init__(self, initial_kernel: np.ndarray = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Редактор структурного элемента")
        self.setModal(True)  # Модальное окно

        # Определяем начальный размер
        if initial_kernel is not None:
            self.kernel_size = initial_kernel.shape[0]
        else:
            self.kernel_size = 3

        self.cells = []  # Список всех клеток

        self._setup_ui()
        self._build_grid()

        # Загружаем начальное ядро, если оно есть
        if initial_kernel is not None:
            self._load_kernel(initial_kernel)

    def _setup_ui(self):
        """Настраивает интерфейс редактора."""
        layout = QVBoxLayout(self)

        # Группа "Размер ядра"
        size_group = QGroupBox("Размер структурного элемента")
        size_layout = QHBoxLayout(size_group)

        size_layout.addWidget(QLabel("Сторона квадрата:"))

        self.spin_size = QSpinBox()
        self.spin_size.setRange(2, 51)  # Увеличен лимит до 51
        self.spin_size.setValue(self.kernel_size)
        self.spin_size.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.spin_size)

        layout.addWidget(size_group)

        # Группа "Сетка ядра" с прокруткой
        kernel_group = QGroupBox("Структурный элемент (клик для переключения)")
        kernel_layout = QVBoxLayout(kernel_group)

        # Создаём область прокрутки для больших размеров
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumSize(400, 400)

        # Контейнер для сетки
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(1)
        self.grid_layout.setAlignment(Qt.AlignCenter)

        scroll_area.setWidget(self.grid_container)
        kernel_layout.addWidget(scroll_area)

        layout.addWidget(kernel_group)

        # Группа "Быстрые действия"
        actions_group = QGroupBox("Быстрые действия")
        actions_layout = QHBoxLayout(actions_group)

        btn_clear = QPushButton("Очистить (все 0)")
        btn_clear.clicked.connect(self._on_clear)
        actions_layout.addWidget(btn_clear)

        btn_fill = QPushButton("Заполнить (все 1)")
        btn_fill.clicked.connect(self._on_fill)
        actions_layout.addWidget(btn_fill)

        btn_rect = QPushButton("Прямоугольник")
        btn_rect.clicked.connect(self._on_rect)
        actions_layout.addWidget(btn_rect)

        btn_cross = QPushButton("Крест")
        btn_cross.clicked.connect(self._on_cross)
        actions_layout.addWidget(btn_cross)

        layout.addWidget(actions_group)

        # Кнопки OK/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _calculate_cell_size(self) -> int:
        """
        Вычисляет размер клетки в зависимости от размера ядра.
        Для больших ядер клетки становятся меньше, но остаются квадратными.
        """
        if self.kernel_size <= 5:
            return 40
        elif self.kernel_size <= 10:
            return 30
        elif self.kernel_size <= 20:
            return 20
        elif self.kernel_size <= 35:
            return 15
        else:
            return 10  # Для 51×51 клетки будут 10×10 пикселей

    def _clear_grid(self):
        """Безопасно очищает сетку клеток."""
        if not self.cells:
            return

        # Просто удаляем ссылки, Qt сам очистит виджеты через родительский контейнер
        self.cells.clear()

        # Удаляем все виджеты из layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _build_grid(self):
        """Создаёт сетку клеток размером kernel_size × kernel_size."""
        # Очищаем старую сетку
        self._clear_grid()

        # Вычисляем размер клетки
        cell_size = self._calculate_cell_size()

        # Создаём новые клетки
        for row in range(self.kernel_size):
            row_cells = []
            for col in range(self.kernel_size):
                cell = KernelCell(row, col, cell_size)
                cell.clicked.connect(self._on_cell_changed)
                self.grid_layout.addWidget(cell, row, col)
                row_cells.append(cell)
            self.cells.append(row_cells)

    def _on_size_changed(self, new_size: int):
        """Обработчик изменения размера ядра."""
        self.kernel_size = new_size
        self._build_grid()

    def _on_cell_changed(self):
        """Обработчик изменения состояния клетки."""
        pass  # Ничего не делаем, просто ждём OK

    def _on_clear(self):
        """Устанавливает все клетки в 0."""
        for row_cells in self.cells:
            for cell in row_cells:
                cell.set_value(0)

    def _on_fill(self):
        """Устанавливает все клетки в 1."""
        for row_cells in self.cells:
            for cell in row_cells:
                cell.set_value(1)

    def _on_rect(self):
        """Создаёт прямоугольник (все 1, кроме границ)."""
        for row_idx, row_cells in enumerate(self.cells):
            for col_idx, cell in enumerate(row_cells):
                if row_idx == 0 or row_idx == self.kernel_size - 1:
                    cell.set_value(0)
                elif col_idx == 0 or col_idx == self.kernel_size - 1:
                    cell.set_value(0)
                else:
                    cell.set_value(1)

    def _on_cross(self):
        """Создаёт крест (центральные строка и столбец)."""
        center = self.kernel_size // 2
        for row_idx, row_cells in enumerate(self.cells):
            for col_idx, cell in enumerate(row_cells):
                if row_idx == center or col_idx == center:
                    cell.set_value(1)
                else:
                    cell.set_value(0)

    def _load_kernel(self, kernel: np.ndarray):
        """Загружает ядро в клетки."""
        for row_idx, row_cells in enumerate(self.cells):
            for col_idx, cell in enumerate(row_cells):
                if row_idx < kernel.shape[0] and col_idx < kernel.shape[1]:
                    cell.set_value(int(kernel[row_idx, col_idx]))

    def _on_accept(self):
        """Обработчик нажатия OK."""
        kernel = self.get_kernel()
        self.kernel_accepted.emit(kernel)
        self.accept()

    def get_kernel(self) -> np.ndarray:
        """
        Возвращает текущий структурный элемент как numpy-массив.
        Тип данных: uint8 (0 или 1).
        """
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.uint8)
        for row_idx, row_cells in enumerate(self.cells):
            for col_idx, cell in enumerate(row_cells):
                kernel[row_idx, col_idx] = cell.value
        return kernel