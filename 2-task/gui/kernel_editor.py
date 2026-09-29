import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGridLayout, QSpinBox, QLabel, QGroupBox, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal


class KernelCell(QPushButton):
    """
    Одна клетка структурного элемента.
    Хранит состояние (0 или 1) и меняет цвет при клике.
    """

    def __init__(self, row: int, col: int, parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col
        self.value = 1  # По умолчанию все клетки активны (1)

        # Настройка внешнего вида
        self.setFixedSize(30, 30)
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


class KernelEditor(QWidget):
    """
    Интерактивный редактор структурного элемента.
    Позволяет задать размер ядра и кликать по клеткам для изменения их состояния.
    """

    # Сигнал, который испускается при изменении ядра
    kernel_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kernel_size = 3
        self.cells = []  # Список всех клеток (для быстрого доступа)

        self._setup_ui()
        self._build_grid()

    def _setup_ui(self):
        """Настраивает интерфейс редактора."""
        layout = QVBoxLayout(self)

        # Группа "Размер ядра"
        size_group = QGroupBox("Размер структурного элемента")
        size_layout = QHBoxLayout(size_group)

        size_layout.addWidget(QLabel("Сторона квадрата:"))

        self.spin_size = QSpinBox()
        self.spin_size.setRange(2, 15)  # От 2×2 до 15×15
        self.spin_size.setValue(3)
        self.spin_size.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.spin_size)

        layout.addWidget(size_group)

        # Группа "Сетка ядра"
        kernel_group = QGroupBox("Структурный элемент (клик для переключения)")
        kernel_layout = QVBoxLayout(kernel_group)

        # Контейнер для сетки (центрируем)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setAlignment(Qt.AlignCenter)

        kernel_layout.addWidget(self.grid_container)

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

        # Группа "Текущая матрица" (для отладки)
        matrix_group = QGroupBox("Текущая матрица")
        matrix_layout = QVBoxLayout(matrix_group)

        self.label_matrix = QLabel()
        self.label_matrix.setFont(self.label_matrix.font())
        self.label_matrix.setStyleSheet(
            "font-family: monospace; background-color: #f5f5f5; "
            "padding: 10px; border: 1px solid #ccc;"
        )
        self.label_matrix.setAlignment(Qt.AlignCenter)
        matrix_layout.addWidget(self.label_matrix)

        layout.addWidget(matrix_group)

        # Обновляем отображение матрицы
        self._update_matrix_display()

    def _build_grid(self):
        """Создаёт сетку клеток размером kernel_size × kernel_size."""
        # Блокируем сигналы на время перестройки (чтобы избежать лишних вызовов)
        self.blockSignals(True)

        # Очищаем старую сетку (ПРАВИЛЬНАЯ итерация по вложенным спискам!)
        for row_cells in self.cells:
            for cell in row_cells:
                self.grid_layout.removeWidget(cell)
                cell.setParent(None)  # Отсоединяем от родителя
                cell.deleteLater()  # Запрашиваем удаление
        self.cells.clear()

        # Создаём новые клетки
        for row in range(self.kernel_size):
            row_cells = []
            for col in range(self.kernel_size):
                cell = KernelCell(row, col)
                cell.clicked.connect(self._on_cell_changed)
                self.grid_layout.addWidget(cell, row, col)
                row_cells.append(cell)
            self.cells.append(row_cells)

        # Разблокируем сигналы
        self.blockSignals(False)

        # Обновляем отображение матрицы
        self._update_matrix_display()

        # Испускаем сигнал один раз после полной перестройки
        self.kernel_changed.emit(self.get_kernel())

    def _on_size_changed(self, new_size: int):
        """Обработчик изменения размера ядра."""
        self.kernel_size = new_size
        self._build_grid()
        self.kernel_changed.emit(self.get_kernel())

    def _on_cell_changed(self):
        """Обработчик изменения состояния клетки."""
        self._update_matrix_display()
        self.kernel_changed.emit(self.get_kernel())

    def _on_clear(self):
        """Устанавливает все клетки в 0."""
        for row_cells in self.cells:
            for cell in row_cells:
                cell.set_value(0)
        self._update_matrix_display()
        self.kernel_changed.emit(self.get_kernel())

    def _on_fill(self):
        """Устанавливает все клетки в 1."""
        for row_cells in self.cells:
            for cell in row_cells:
                cell.set_value(1)
        self._update_matrix_display()
        self.kernel_changed.emit(self.get_kernel())

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
        self._update_matrix_display()
        self.kernel_changed.emit(self.get_kernel())

    def _on_cross(self):
        """Создаёт крест (диагонали и центральные линии)."""
        center = self.kernel_size // 2
        for row_idx, row_cells in enumerate(self.cells):
            for col_idx, cell in enumerate(row_cells):
                if row_idx == center or col_idx == center:
                    cell.set_value(1)
                else:
                    cell.set_value(0)
        self._update_matrix_display()
        self.kernel_changed.emit(self.get_kernel())

    def _update_matrix_display(self):
        """Обновляет текстовое отображение текущей матрицы."""
        kernel = self.get_kernel()
        matrix_str = "\n".join([" ".join(map(str, row)) for row in kernel])
        self.label_matrix.setText(matrix_str)

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